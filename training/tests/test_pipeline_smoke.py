"""End-to-end check of the training pipeline against synthetic sessions.

Runs the real code path - discover sessions, build labels, train, export, verify ORT parity - on
fabricated data with a *known* defect injected into the "stock" predictions. That makes the test
meaningful rather than merely non-crashing: the adapter is expected to reduce the error it was given
the evidence to fix.

    python -m pytest training/tests -q
    python training/tests/test_pipeline_smoke.py     (no pytest required)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babble_personal import dataset as ds  # noqa: E402
from babble_personal import evaluate, export, labels as lbl, models, schema, train  # noqa: E402

N = schema.EXPRESSION_COUNT
JAW = schema.INDEX_OF["JawOpen"]
SMILE_L = schema.INDEX_OF["MouthSmileLeft"]

TICKS_PER_SECOND = lbl.TICKS_PER_SECOND


def _write_session(
    root: Path,
    session_id: str,
    session_type: str,
    frames: list[tuple[np.ndarray, dict | None]],
    *,
    schema_hash: str | None = None,
) -> Path:
    """Create a session directory in the exact layout DatasetRecorderService produces."""
    path = root / session_id
    (path / "frames").mkdir(parents=True, exist_ok=True)

    metadata = {
        "SchemaVersion": 1,
        "SessionId": session_id,
        "SessionType": session_type,
        "StartedUtc": "2026-08-13T00:00:00Z",
        "AppVersion": "test",
        "ExpressionNames": list(schema.EXPRESSION_NAMES),
        "ExpressionSchemaSha256": schema_hash or schema.SCHEMA_SHA256,
        "ImageWidth": schema.IMAGE_SIZE,
        "ImageHeight": schema.IMAGE_SIZE,
        "JpegQuality": 95,
        "FrameCount": len(frames),
        "EffectiveFps": 30.0,
    }
    (path / "session.json").write_text(json.dumps(metadata), encoding="utf-8")

    lines = []
    for index, (stock, cue) in enumerate(frames):
        # Vary the image so nothing depends on identical content.
        image = np.full((schema.IMAGE_SIZE, schema.IMAGE_SIZE), (index * 7) % 200, dtype=np.uint8)
        cv2.imwrite(str(path / "frames" / f"{index:06d}.jpg"), image)

        record = {
            "i": index,
            "t": index * (TICKS_PER_SECOND // 30),
            "stock": [round(float(v), 6) for v in stock],
        }
        if cue is not None:
            record["cue"] = cue
        lines.append(json.dumps(record))

    (path / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _neutral_frames(count: int, jaw_bias: float) -> list[tuple[np.ndarray, dict | None]]:
    """Resting face, but the stock model wrongly reports jaw activity - the defect to be corrected."""
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(count):
        stock = np.abs(rng.normal(0.0, 0.01, N)).astype(np.float32)
        stock[JAW] = jaw_bias + float(rng.normal(0, 0.01))
        frames.append((stock, None))
    return frames


def _guided_hold_frames(count: int, level: float, underestimate: float) -> list[tuple[np.ndarray, dict | None]]:
    """A held smile that the stock model consistently under-reports."""
    rng = np.random.default_rng(2)
    target = [0.0] * N
    target[SMILE_L] = level

    frames = []
    for i in range(count):
        stock = np.abs(rng.normal(0.0, 0.01, N)).astype(np.float32)
        stock[SMILE_L] = max(0.0, level - underestimate + float(rng.normal(0, 0.01)))
        cue = {
            "id": f"SmileHold{int(level * 100)}",
            "phase": "hold",
            "dims": [SMILE_L],
            "target": target,
            "level": level,
            "rep": 0,
            "source": "avatar",
        }
        frames.append((stock, cue))
    return frames


def build_corpus(root: Path) -> None:
    """Two sessions per type so split_sessions can hold one of each out."""
    for suffix in ("a", "b"):
        _write_session(root, f"20260813_0000{suffix}_neutral", "Neutral", _neutral_frames(60, 0.30))
        _write_session(
            root,
            f"20260813_0001{suffix}_guided",
            "Guided",
            _guided_hold_frames(40, 0.8, 0.35) + _guided_hold_frames(40, 0.5, 0.20),
        )


def test_dataset_roundtrip_and_split() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_corpus(root)

        sessions = ds.discover_sessions(root)
        assert len(sessions) == 4, f"expected 4 sessions, got {len(sessions)}"

        train_s, val_s = ds.split_sessions(sessions)
        assert val_s, "one session of each type should be held out"

        train_ids = {s.session_id for s in train_s}
        val_ids = {s.session_id for s in val_s}
        assert not (train_ids & val_ids), "a session must never appear in both splits"
        print(f"  split OK: {len(train_s)} train / {len(val_s)} val, no overlap")


def test_schema_mismatch_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_session(root, "20260813_0000a_neutral", "Neutral", _neutral_frames(4, 0.1),
                       schema_hash="deadbeef")
        try:
            ds.discover_sessions(root)
        except ValueError as exc:
            assert "schema mismatch" in str(exc).lower()
            print("  schema mismatch rejected as expected")
            return
        raise AssertionError("data recorded under a different schema must be refused")


def test_labels_supervise_expected_cells() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_corpus(root)
        sessions = ds.discover_sessions(root)

        label_set = lbl.build_labels(sessions)
        assert label_set.coverage() > 0, "no supervision was produced"

        neutral = [s for s in sessions if s.is_neutral]
        neutral_labels = lbl.build_labels(neutral)
        assert np.all(neutral_labels.targets == 0), "neutral targets must be all-zero"
        assert np.all(neutral_labels.weights == lbl.W_NEUTRAL_SESSION)

        guided = [s for s in sessions if s.is_guided]
        guided_labels = lbl.build_labels(guided)
        cued_weight = guided_labels.weights[:, SMILE_L].max()
        assert cued_weight == lbl.W_GUIDED_HOLD, f"cued dim weight was {cued_weight}"

        # Settle trim must drop the first frames of a hold rather than label them.
        assert (guided_labels.weights[:, SMILE_L] == 0).any(), "settle trim did not mask anything"
        print(f"  labels OK: coverage {label_set.coverage() * 100:.1f}%")


def test_training_reduces_known_error_and_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_corpus(root)
        out = root / "runs"

        code = train.main([
            "--data", str(root), "--model", "a", "--out", str(out),
            "--epochs", "40", "--batch-size", "64", "--patience", "40",
        ])
        assert code == 0, "training returned non-zero"

        checkpoint = next(out.rglob("model.pt"))

        # Did it actually learn the injected defects?
        sessions = ds.discover_sessions(root)
        _, val_sessions = ds.split_sessions(sessions)
        val_labels = lbl.build_labels(val_sessions)

        import torch
        model = models.build_model("a")
        model.load_state_dict(torch.load(checkpoint)["state_dict"])
        model.eval()

        stock_t = torch.from_numpy(val_labels.stock)
        with torch.no_grad():
            personal = model(torch.zeros((len(val_labels), 1, 1, 1)), stock_t).numpy()

        neutral_val = [s for s in val_sessions if s.is_neutral]
        if neutral_val:
            nl = lbl.build_labels(neutral_val)
            with torch.no_grad():
                np_pred = model(torch.zeros((len(nl), 1, 1, 1)),
                                torch.from_numpy(nl.stock)).numpy()
            stock_jaw = float(nl.stock[:, JAW].mean())
            personal_jaw = float(np_pred[:, JAW].mean())
            print(f"  neutral JawOpen: stock {stock_jaw:.3f} -> personal {personal_jaw:.3f}")
            assert personal_jaw < stock_jaw * 0.5, (
                f"adapter failed to suppress the injected false jaw activation "
                f"({stock_jaw:.3f} -> {personal_jaw:.3f})"
            )

        reports = evaluate.per_expression_mae(
            val_labels.stock, personal, val_labels.targets, val_labels.weights
        )
        smile = next(r for r in reports if r.name == "MouthSmileLeft")
        if smile.supervised_frames:
            print(f"  MouthSmileLeft MAE: stock {smile.stock_mae:.3f} -> personal {smile.personal_mae:.3f}")
            assert smile.personal_mae < smile.stock_mae, "adapter failed to fix the underestimated smile"

        # Export must round-trip through ONNX Runtime identically.
        onnx_path = export.export(checkpoint, root / "personalFaceModel.onnx")
        assert onnx_path.exists()

        import onnx as onnx_mod
        proto = onnx_mod.load(str(onnx_path))
        meta = {p.key: p.value for p in proto.metadata_props}
        assert meta["expression_schema_sha256"] == schema.SCHEMA_SHA256
        assert meta["adapter_type"] == "output_mlp_v1"
        assert meta["input_size"] == str(schema.IMAGE_SIZE)
        print(f"  export OK with metadata ({len(meta)} keys)")


def test_both_models_export_identical_io_signature() -> None:
    """The C# runtime feeds image+stock and reads personal, regardless of adapter type.

    Baseline A ignores the image, and torch prunes inputs nothing consumes, so without the
    zero-gated tap in OutputMlpAdapter its graph would silently lose the `image` input and the
    runtime would need a per-model-type branch.
    """
    import onnx as onnx_mod
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        signatures = {}

        for kind in ("a", "b"):
            model = models.build_model(kind)
            checkpoint = root / f"{kind}.pt"
            torch.save({"state_dict": model.state_dict(), "adapter_type": model.adapter_type},
                       checkpoint)

            onnx_path = export.export(checkpoint, root / f"{kind}.onnx", parity_samples=4)
            proto = onnx_mod.load(str(onnx_path))
            signatures[kind] = (
                [i.name for i in proto.graph.input],
                [o.name for o in proto.graph.output],
            )

        assert signatures["a"] == signatures["b"], (
            f"adapter graphs disagree: A={signatures['a']} B={signatures['b']}"
        )
        assert signatures["a"] == (["image", "stock"], ["personal"]), signatures["a"]
        print(f"  both models expose {signatures['a'][0]} -> {signatures['a'][1]}")


def test_untrained_adapter_is_identity() -> None:
    """A freshly initialised adapter must pass stock through unchanged.

    This is the safety property behind the whole residual design: before learning anything, and
    wherever it has no evidence, the adapter should do nothing at all.
    """
    import torch

    for kind in ("a", "b"):
        model = models.build_model(kind)
        model.eval()
        stock = torch.rand((8, N))
        image = torch.rand((8, 1, schema.IMAGE_SIZE, schema.IMAGE_SIZE))

        with torch.no_grad():
            out = model(image, stock)

        max_diff = float((out - stock).abs().max())
        assert max_diff < 1e-6, f"model {kind} is not identity at init (max diff {max_diff})"
    print("  both models start as exact stock passthrough")


def test_summary_json_contract() -> None:
    """The app's results screen reads summary.json, so its shape is a contract.

    Also pins the conservative verdict rule: with no held-out session the numbers describe training
    data and cannot support a claim, so the verdict must stay 'unclear' however good they look.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_corpus(root)
        out = root / "runs"

        assert train.main([
            "--data", str(root), "--model", "a", "--out", str(out),
            "--epochs", "10", "--batch-size", "64", "--patience", "10",
        ]) == 0

        summary = json.loads(next(out.rglob("summary.json")).read_text(encoding="utf-8"))

        for key in (
            "summary_version", "adapter_type", "parameters", "checkpoint", "schema_sha256",
            "train_sessions", "val_sessions", "validated_on_held_out_sessions",
            "scored_expressions", "expressions_improved", "expressions_regressed",
            "mean_stock_mae", "mean_personal_mae", "verdict",
        ):
            assert key in summary, f"summary.json is missing '{key}'"

        assert summary["verdict"] in ("better", "unclear", "worse")
        assert summary["schema_sha256"] == schema.SCHEMA_SHA256
        assert summary["validated_on_held_out_sessions"] is True
        assert "neutral" in summary, "neutral stats drive the headline result"
        assert summary["neutral"]["personal_false_activation_rate"] <= \
               summary["neutral"]["stock_false_activation_rate"] + 1e-9

        # Single-session corpus: nothing can be held out, so no claim may be made.
        solo = root / "solo"
        _write_session(solo, "20260813_0000a_neutral", "Neutral", _neutral_frames(30, 0.3))
        assert train.main([
            "--data", str(solo), "--model", "a", "--out", str(solo / "runs"),
            "--epochs", "3", "--batch-size", "32", "--patience", "3",
        ]) == 0
        solo_summary = json.loads(next((solo / "runs").rglob("summary.json")).read_text(encoding="utf-8"))
        assert solo_summary["validated_on_held_out_sessions"] is False
        assert solo_summary["verdict"] == "unclear", \
            "without held-out data the result must not be reported as an improvement"

        print(f"  summary.json OK (verdict '{summary['verdict']}', "
              f"{summary['expressions_improved']} improved / "
              f"{summary['expressions_regressed']} regressed)")


def test_image_model_trains_and_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_corpus(root)
        out = root / "runs"

        code = train.main([
            "--data", str(root), "--model", "b", "--out", str(out),
            "--epochs", "3", "--batch-size", "32", "--patience", "3",
        ])
        assert code == 0

        checkpoint = next(out.rglob("model.pt"))
        onnx_path = export.export(checkpoint, root / "personal_b.onnx")
        assert onnx_path.exists()
        print(f"  image model exported ({models.parameter_count(models.build_model('b')):,} params)")


def main() -> int:
    tests = [
        test_dataset_roundtrip_and_split,
        test_schema_mismatch_is_rejected,
        test_labels_supervise_expected_cells,
        test_untrained_adapter_is_identity,
        test_both_models_export_identical_io_signature,
        test_training_reduces_known_error_and_exports,
        test_summary_json_contract,
        test_image_model_trains_and_exports,
    ]

    failures = 0
    for test in tests:
        print(f"\n--- {test.__name__}")
        try:
            test()
            print("  PASS")
        except Exception as exc:  # noqa: BLE001 - smoke runner reports rather than raises
            failures += 1
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
