"""Isolated C2 recipe. Legacy labels/models/train/export remain unchanged.

Targets describe steady, human-reviewed cues in emitted units. Invert the frozen
presentation map into pre-filter reference units; transitions/unknowns never become zero labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .models import EmbeddingHeadAdapter
from .schema import EXPRESSION_NAMES

VERSION = "c2-raw-reference-v1"
RECIPE = "c2-reviewed-holds-v1"
N = 45


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def raw_target(target, context):
    r = np.asarray(context["Ranges"], np.float32)
    if r.shape != (N, 4) or not np.isfinite(r).all() or np.any(r[:, 1] <= r[:, 0]) or np.any(r[:, 3] <= r[:, 2]):
        raise ValueError("Face calibration has non-invertible ranges. Review face calibration first.")
    result = r[:, 0] + (target - r[:, 2]) / (r[:, 3] - r[:, 2]) * (r[:, 1] - r[:, 0])
    result[4] = np.clip(result[4], 0, 1) ** (1 / context["EffectiveJawExponent"])
    return result


def emit(values, context):
    r = np.asarray(context["Ranges"], np.float32)
    x = np.asarray(values, np.float32).copy()
    x[..., 4] = np.clip(x[..., 4], 0, 1) ** context["EffectiveJawExponent"]
    return np.clip(r[:, 2] + (x - r[:, 0]) / (r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2]), r[:, 2], r[:, 3])


def replay(values, times, context):
    """Recorded software-time replay. Separate state for each branch; no camera-to-photon claim."""
    x = np.asarray(values, np.float32)
    if not context["FilterEnabled"] or len(x) < 2:
        return emit(x, context)
    previous, derivative = x[0].copy(), np.zeros(N, np.float32)
    output = [previous.copy()]
    for i in range(1, len(x)):
        dt = np.float32(times[i] - times[i - 1])
        if dt <= 0:
            previous = x[i].copy()
        else:
            rd = np.float32(2 * np.pi) * dt
            ad = rd / (rd + 1)
            derivative = ad * ((x[i] - previous) / dt) + (1 - ad) * derivative
            cutoff = context["MinCutoff"] + context["Beta"] * np.abs(derivative)
            r = np.float32(2 * np.pi) * cutoff * dt
            a = r / (r + 1)
            previous = a * x[i] + (1 - a) * previous
        output.append(previous.copy())
    return emit(np.asarray(output), context)


def labels(rows, context, accepted, legacy=False):
    targets = np.zeros((len(rows), N), np.float32)
    weights = np.zeros_like(targets)
    evidence = np.full((len(rows), N), "unknown", dtype="U16")
    # Legacy all-zero/rest/correction rules are deliberately not imported.
    if not accepted or legacy:
        return targets, weights, evidence
    for i, row in enumerate(rows):
        cue = row.get("cue")
        if not cue or cue.get("phase") != "hold":
            continue
        dims = cue.get("dims", [])
        if any(not isinstance(d, int) or d < 0 or d >= N for d in dims):
            raise ValueError("Invalid cued channel index")
        command = np.asarray(cue["target"], np.float32)
        if command.shape != (N,) or not np.isfinite(command).all():
            raise ValueError("Invalid cue vector")
        converted = raw_target(command, context)
        if np.any((converted[dims] < 0) | (converted[dims] > 1)):
            raise ValueError("A reviewed cue is unreachable under this output contract")
        targets[i, dims] = converted[dims]
        weights[i, dims] = np.where(command[dims] == 0, .5, .35)
        evidence[i, dims] = np.where(command[dims] == 0, "explicit_absent", "approximate_cue")
    return targets, weights, evidence


def masked_loss(predicted, reference, targets, weights, support):
    valid = weights > 0
    safe_targets = torch.where(valid, targets, predicted.detach())
    cell = torch.nn.functional.huber_loss(predicted, safe_targets, delta=.05, reduction="none")
    human = (cell * weights).sum() / weights.sum().clamp_min(1)
    # Separate prior: unknown cells only. Never outvote an explicit correction.
    prior_mask = (~valid).float() * support
    prior = ((predicted - reference).square() * prior_mask).sum() / prior_mask.sum().clamp_min(1)
    return human + .02 * prior, human, prior


class C2Head(nn.Module):
    def __init__(self, support, embedding_dim=1280, linear=False):
        super().__init__()
        self.register_buffer("support", torch.as_tensor(support, dtype=torch.float32))
        self.linear = linear
        if linear:
            self.head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, N))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        else:
            self.head = EmbeddingHeadAdapter(embedding_dim=embedding_dim)

    def forward(self, reference, embedding):
        residual = self.head(embedding) if self.linear else self.head.residual(None, reference, embedding)
        candidate = torch.clamp(reference + residual, 0, 1)
        # Exactly preserve the compatible C reference for unsupported channels, including edges.
        return torch.where(self.support > 0, candidate, reference)


def cpu_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def load_manifest(path):
    manifest = read(path)
    if manifest.get("Version") != VERSION or manifest.get("LabelRecipe") != RECIPE:
        raise ValueError("Unsupported C2 manifest or label recipe")
    contract_path = manifest["ContractPath"]
    if sha(contract_path) != manifest["ContractSha256"]:
        raise ValueError("Output/reference contract changed; refresh C2 readiness")
    contract = read(contract_path)
    if contract["Version"] != VERSION:
        raise ValueError("Unsupported output domain")
    schema_hash = hashlib.sha256("\n".join(EXPRESSION_NAMES).encode()).hexdigest()
    if contract["SchemaSha256"] != schema_hash or contract["InputNormalization"] != "gray_div255":
        raise ValueError("C2 schema or preprocessing differs")
    for name in ("Stock", "Feature", "Reference"):
        if sha(contract[name + "Path"]) != contract[name + "Sha256"]:
            raise ValueError(f"{name} model changed; prepare a new contract")
    raw_target(np.zeros(N, np.float32), contract["Output"])
    return manifest, contract


def prepare(manifest, contract, destination):
    teacher = cpu_session(contract["ReferencePath"])
    metadata = teacher.get_modelmeta().custom_metadata_map
    if metadata.get("adapter_type") != "embedding_head_v1":
        raise ValueError("C2 currently requires the selected compatible Model C as its reference")
    if metadata.get("expression_schema_sha256") != contract["SchemaSha256"] or metadata.get("base_model_md5") != hashlib.md5(Path(contract["StockPath"]).read_bytes()).hexdigest():
        raise ValueError("Reference C schema/base model differs from the pinned producer")
    def tensor(items, name, tail):
        item = next((i for i in items if i.name == name), None)
        if item is None or item.type != "tensor(float)" or list(item.shape[1:]) != tail:
            raise ValueError(f"Incompatible named float tensor: {name}")
    tensor(teacher.get_inputs(), "stock", [N])
    tensor(teacher.get_inputs(), "embedding", [contract["EmbeddingDim"]])
    tensor(teacher.get_outputs(), "personal", [N])
    features = None
    sessions = []
    seen = {}
    origin_roles = {}
    for entry in manifest["Sessions"]:
        directory = Path(entry["Directory"])
        for file, key in (("session.json", "SessionSha256"), ("labels.jsonl", "LabelsSha256")):
            if sha(directory / file) != entry[key]:
                raise ValueError(f"{directory.name}: source changed after review")
        metadata = read(directory / "session.json")
        if not metadata.get("EndedUtc") or not metadata.get("FrameCount"):
            raise ValueError(f"{directory.name}: incomplete recording; repeat/review it first")
        rows = [json.loads(line) for line in (directory / "labels.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(rows) != metadata["FrameCount"]:
            raise ValueError(f"{directory.name}: frame count differs; source is not silently repaired")
        legacy = entry["Legacy"]
        role = "replay" if legacy else entry["Role"]
        origin = "legacy-unknown" if legacy else entry["OriginId"]
        if not origin or (role == "check" and origin == "legacy-unknown"):
            raise ValueError("Independent check needs an explicit new wearing-session identity")
        prior_role = origin_roles.setdefault(origin, role)
        if prior_role != role:
            raise ValueError("Practice and check recordings share one wearing session. Reseat, then start a new check session.")
        embeddings_file = directory / "embeddings.f32"
        live = not legacy and metadata.get("C2", {}).get("ContractSha256") == manifest["ContractSha256"]
        if not legacy and not live:
            raise ValueError(f"{directory.name}: capture contract changed; review/re-record under the current context")
        embedding_dim = contract["EmbeddingDim"]
        emb = None
        if live:
            if entry.get("EmbeddingsSha256") and sha(embeddings_file) != entry["EmbeddingsSha256"]:
                raise ValueError("Live features changed after review")
            emb = np.fromfile(embeddings_file, dtype="<f4")
            if emb.size % embedding_dim:
                raise ValueError("Truncated live embeddings; source retained, attempt excluded")
            emb = emb.reshape(-1, embedding_dim)
        stocks, visuals, refs, frame_hashes = [], [], [], []
        for row in rows:
            image_path = directory / "frames" / f"{row['i']:06d}.jpg"
            digest = sha(image_path)
            frame_hashes.append(digest)
            if digest in seen and seen[digest] != origin:
                raise ValueError("Copied/overlapping images cross wearing sessions. Keep their origins together; no split leakage allowed.")
            seen[digest] = origin
            if live:
                stock = np.asarray(row["stock"], np.float32).reshape(1, N)
                visual = emb[row["embedding_row"]].reshape(1, embedding_dim)
            else:
                if features is None:
                    features = cpu_session(contract["FeaturePath"])
                    if len(features.get_inputs()) != 1:
                        raise ValueError("Feature producer must have one image input")
                    tensor(features.get_inputs(), features.get_inputs()[0].name, [1, 224, 224])
                    tensor(features.get_outputs(), "embedding", [embedding_dim])
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != (224, 224):
                    raise ValueError("Missing/wrong-size recorded image; no guessed replacements")
                names = [o.name for o in features.get_outputs()]
                if "embedding" not in names:
                    raise ValueError("Feature graph has no named embedding")
                stock_name = next(o.name for o in features.get_outputs() if o.name != "embedding" and o.shape[-1] == N)
                stock, visual = features.run([stock_name, "embedding"], {features.get_inputs()[0].name: image[None, None].astype(np.float32) / 255})
            personal = teacher.run(["personal"], {"stock": stock, "embedding": visual})[0]
            reference = stock + (personal - stock) * contract["ReferenceBlend"]
            if not all(np.isfinite(x).all() for x in (stock, visual, reference)):
                raise ValueError("Non-finite model/sample values; excluded before training")
            stocks.append(stock[0]); visuals.append(visual[0]); refs.append(reference[0])
        target, weight, evidence = labels(rows, contract["Output"], entry["Review"]["Accepted"], legacy)
        times = np.asarray([r.get("inference_timestamp", 0) for r in rows], np.float64)
        if live:
            times /= metadata["C2"]["StopwatchFrequency"]
        else:
            times = np.asarray([r["t"] for r in rows], np.float64) / 10_000_000
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Recording clock is missing/non-monotonic; no fabricated replay times")
        item = dict(reference=np.asarray(refs), embedding=np.asarray(visuals), target=target,
                    weight=weight, evidence=evidence, times=times - times[0],
                    origin=origin, role=role, task=entry["TaskId"], frame_hashes=frame_hashes)
        sessions.append(item)
        np.savez_compressed(destination / f"session-{len(sessions):03}.npz", **item)
        print(f"[preparing] {directory.name}: {len(rows)} rows; {role}; " + ("same-frame float32 features" if live else "new JPEG replay features; no human labels"), flush=True)
    return sessions


def split(sessions):
    practice = [s for s in sessions if s["role"] == "practice"]
    origins = sorted({s["origin"] for s in practice})
    if not origins:
        raise ValueError("Record and accept practice holds first. Existing replay sessions alone cannot teach C2.")
    validation_origin = origins[-1] if len(origins) > 1 else None
    train = [s for s in sessions if s["role"] in ("practice", "replay") and s["origin"] != validation_origin]
    validation = [s for s in practice if s["origin"] == validation_origin]
    check = [s for s in sessions if s["role"] == "check"]
    return train, validation, check


def coverage(sessions, context):
    positive = np.zeros(N, bool); negative = np.zeros(N, bool)
    for s in sessions:
        canonical = emit(s["target"], context)
        positive |= ((s["weight"] > 0) & (canonical > .05)).any(axis=0)
        negative |= ((s["weight"] > 0) & (canonical < .01)).any(axis=0)
    return positive & negative


def comparison(model, sessions, context):
    reports = []
    for s in sessions:
        with torch.no_grad():
            candidate = model(torch.tensor(s["reference"]), torch.tensor(s["embedding"])).numpy()
        baseline = replay(s["reference"], s["times"], context)
        predicted = replay(candidate, s["times"], context)
        canonical = emit(s["target"], context)
        known = s["weight"] > 0
        channels = []
        for d in range(N):
            mask = known[:, d]
            if not mask.any(): continue
            b = float(np.abs(baseline[mask, d] - canonical[mask, d]).mean())
            c = float(np.abs(predicted[mask, d] - canonical[mask, d]).mean())
            zero = mask & (canonical[:, d] < .01)
            pos = mask & (canonical[:, d] > .05)
            # Descriptive deltas, not promotion thresholds selected after seeing the check.
            channels.append(dict(name=EXPRESSION_NAMES[d], reference_mae=b, candidate_mae=c,
                raw_reference_mae=float(np.abs(s["reference"][mask,d]-s["target"][mask,d]).mean()),
                raw_candidate_mae=float(np.abs(candidate[mask,d]-s["target"][mask,d]).mean()),
                reference_false_runs=false_runs(baseline[:, d], zero, s["times"]),
                candidate_false_runs=false_runs(predicted[:, d], zero, s["times"]),
                reference_clipped_fraction=float(((baseline[:, d] <= context["Ranges"][d][2]) | (baseline[:, d] >= context["Ranges"][d][3])).mean()),
                candidate_clipped_fraction=float(((predicted[:, d] <= context["Ranges"][d][2]) | (predicted[:, d] >= context["Ranges"][d][3])).mean()),
                rest_reference_peak=float(baseline[zero, d].max()) if zero.any() else None,
                rest_candidate_peak=float(predicted[zero, d].max()) if zero.any() else None,
                positive_reference_peak=float(baseline[pos, d].max()) if pos.any() else None,
                positive_candidate_peak=float(predicted[pos, d].max()) if pos.any() else None))
        reports.append(dict(task=s["task"], origin=s["origin"], role=s["role"], channels=channels,
            preservation_mean_absolute_change=float(np.abs(predicted - baseline)[~known].mean()) if (~known).any() else None,
            verdict="Not checked" if not channels else "Exploratory; no promotion verdict"))
    return reports


def false_runs(values, verified_rest, times, threshold=.15):
    """Descriptive rest metric at the legacy report's threshold; never a universal promotion gate."""
    active = (values > threshold) & verified_rest
    runs, longest, started, last = 0, 0., None, None
    for yes, time in zip(active, times):
        if last is not None and (time <= last or time-last > .5):
            started = None
        if yes:
            if started is None: started = time; runs += 1
            longest = max(longest, float(time-started))
        else:
            started = None
        last = time
    return dict(threshold=threshold, runs=runs, longest_observed_seconds=longest,
                fraction=float(active.sum()/max(1, verified_rest.sum())))


def train(manifest_path, output, epochs=40, linear=False):
    manifest, contract = load_manifest(manifest_path)
    requested_checks = {s["OriginId"] for s in manifest["Sessions"] if s["Role"] == "check" and not s["Legacy"]}
    for prior in Path(output).glob("*/summary.json"):
        if requested_checks.intersection(read(prior).get("check_origins", [])):
            raise ValueError("This check wearing session was already evaluated. Keep the report; record a fresh check before another training experiment.")
    run = Path(output) / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    write_new(run / "manifest.json", manifest)
    write_new(run / "contract.json", contract)
    print("[stage] preparing", flush=True)
    sessions = prepare(manifest, contract, run)
    train_sessions, validation, check = split(sessions)
    support = coverage(train_sessions, contract["Output"])
    if not support.any():
        raise ValueError("No channel has accepted positive AND absent evidence. Record relaxed jaw/smile plus a matching positive hold.")
    torch.set_num_threads(1)
    torch.manual_seed(20260910)
    model = C2Head(support.astype(np.float32), contract["EmbeddingDim"], linear)
    values = {key: torch.tensor(np.concatenate([s[key] for s in train_sessions]))
              for key in ("reference", "embedding", "target", "weight")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    print("[stage] training", flush=True)
    # Fixed small experiment. Check-only examples never choose epoch/head/strength.
    for epoch in range(epochs):
        model.train()
        for index in torch.randperm(len(values["reference"])).split(128):
            prediction = model(values["reference"][index], values["embedding"][index])
            loss, _, _ = masked_loss(prediction, values["reference"][index], values["target"][index],
                                     values["weight"][index], model.support)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"[training] epoch {epoch + 1}/{epochs}", flush=True)
    model.eval()
    print("[stage] exporting", flush=True)
    sample = (values["reference"][:1], values["embedding"][:1])
    model_path = run / "candidate.onnx"
    torch.onnx.export(model, sample, str(model_path), input_names=["reference", "embedding"],
        output_names=["personal"], opset_version=17, dynamo=False,
        dynamic_axes={"reference": {0: "batch"}, "embedding": {0: "batch"}, "personal": {0: "batch"}})
    graph = onnx.load(str(model_path))
    for key, value in {"c2_contract": VERSION, "c2_label_recipe": RECIPE,
                       "c2_context_sha256": manifest["ContractSha256"],
                       "expression_schema_sha256": contract["SchemaSha256"]}.items():
        entry = graph.metadata_props.add(); entry.key = key; entry.value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(model_path))
    runtime = cpu_session(model_path)
    with torch.no_grad():
        expected = model(values["reference"][:32], values["embedding"][:32]).numpy()
    actual = runtime.run(["personal"], {k: values[k][:32].numpy() for k in ("reference", "embedding")})[0]
    error = float(np.abs(expected - actual).max())
    if not np.isfinite(actual).all() or error > 1e-4:
        raise ValueError("Candidate export parity failed; active model unchanged")
    print("[stage] evaluating", flush=True)
    summary = dict(version=VERSION, status="comparison_needed", recommended=False,
        candidate_sha256=sha(model_path), context_sha256=manifest["ContractSha256"],
        export_max_abs_error=error, head="linear" if linear else "c-style",
        support=[EXPRESSION_NAMES[i] for i in np.flatnonzero(support)],
        practice_origins=sorted({s["origin"] for s in train_sessions}),
        development_origins=sorted({s["origin"] for s in validation}),
        check_origins=sorted({s["origin"] for s in check}),
        limitations=["Approximate human cues, not measured anatomy", "No automatic activation or promotion",
            "Replay uses recorded software timestamps, not camera-to-photon latency",
            "Audio excluded; hardware, avatar, throughput, saturation and transition checks remain",
            "No independent validation" if not validation else "Development validation is not a final quality claim",
            "No prospective check" if not check else "Check consumed by this report; do not tune on it"],
        development=comparison(model, validation, contract["Output"]),
        check=comparison(model, check, contract["Output"]),
        practice_diagnostic_only=comparison(model, [s for s in train_sessions if s["role"] == "practice"], contract["Output"]),
        preservation_replay_not_ground_truth=comparison(model, [s for s in train_sessions if s["role"] == "replay"], contract["Output"]))
    write_new(run / "summary.json", summary)
    print("[candidate] " + str(run), flush=True)
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--linear", action="store_true")
    args = p.parse_args()
    if not 1 <= args.epochs <= 100:
        p.error("C2 is a bounded experiment: use 1..100 epochs")
    try:
        train(args.manifest, args.out, args.epochs, args.linear)
    except (ValueError, OSError, KeyError, ort.Fail) as ex:
        raise SystemExit("C2 needs attention: " + str(ex)) from ex


if __name__ == "__main__":
    main()
