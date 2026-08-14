"""Train a personal adapter.

    python -m babble_personal.train --data "%APPDATA%\\ProjectBabble\\PersonalDataset" --model a

Validation is always split by session (see dataset.split_sessions). Adjacent frames are
near-duplicates, so a per-frame split would report a flattering number that says nothing about how
the model behaves in a fresh session - which is the only thing that matters here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from . import augment, dataset as ds
from . import evaluate, labels as lbl, models, schema


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a personal face-expression adapter.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset root containing session folders")
    parser.add_argument("--model", default="a", help="'a' = output-only baseline, 'b' = image-conditioned")
    parser.add_argument("--out", type=Path, default=Path("runs"), help="Output directory for checkpoints")
    parser.add_argument("--val-sessions", nargs="*", default=None, help="Session ids to hold out")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--shrinkage", type=float, default=1e-2,
                        help="Residual shrinkage; higher stays closer to stock")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience in epochs")
    parser.add_argument("--no-speech-pseudo-labels", action="store_true",
                        help="Drop the weak stock-derived labels from speech sessions")
    parser.add_argument("--consistency", type=float, default=0.5,
                        help="Weight on illumination invariance (model B only; 0 disables). Stops "
                             "the image branch reading brightness as expression.")
    parser.add_argument("--temporal", type=float, default=0.5,
                        help="Weight on residual smoothness between consecutive frames (0 disables). "
                             "Damps correction jitter without slowing the output down.")
    parser.add_argument("--embedding-noise", type=float, default=0.01,
                        help="Model C's counterpart to --consistency: the residual must not swing "
                             "on small wobbles in the embedding it reads. 0 disables.")
    parser.add_argument("--dim-boost", default=None, metavar="Name=W,...",
                        help="Weight specific expressions more heavily, e.g. 'JawOpen=2.0'. "
                             "Symmetric: emphasises being right about zeros and about genuine "
                             "openings equally.")
    parser.add_argument("--fp-penalty", type=float, default=0.0,
                        help="Asymmetric cost for firing where the label says zero, on --fp-dims. "
                             "0 disables. Watch range retention in the report when raising it.")
    parser.add_argument("--fp-dims", default="JawOpen,TongueOut",
                        help="Expressions --fp-penalty applies to")
    parser.add_argument("--hard-negative-boost", type=float, default=0.0,
                        help="Oversample confidently-zero frames where the stock model is already "
                             "firing on a watched expression - the decision-boundary cases. "
                             "0 disables.")
    parser.add_argument("--hard-negative-threshold", type=float, default=0.3,
                        help="Stock value above which a confidently-zero frame counts as hard")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _previous_frame_index(sessions) -> np.ndarray:
    """For each row, the row of the frame before it *within the same session*.

    The first frame of each session points at itself and is masked out by ``_previous_valid``.
    Crossing a session boundary would pair two unrelated frames and ask the smoothness penalty to
    make a cut look continuous, which is worse than not applying it at all.
    """
    previous, offset = [], 0
    for session in sessions:
        for i in range(len(session)):
            previous.append(offset if i == 0 else offset + i - 1)
        offset += len(session)
    return np.asarray(previous, dtype=np.int64)


def _previous_valid(sessions) -> np.ndarray:
    valid, offset = [], 0
    for session in sessions:
        for i in range(len(session)):
            valid.append(0.0 if i == 0 else 1.0)
        offset += len(session)
    return np.asarray(valid, dtype=np.float32)


class _Batch:
    """Everything one training step needs, gathered by row index.

    Batching indices rather than tensors is what makes the temporal penalty possible: a shuffled
    batch of rows can still look up each row's predecessor in the full tensors. It also avoids
    duplicating the image tensor, which is the largest thing in memory for model B.
    """

    def __init__(self, sessions, label_set: lbl.LabelSet, need_images: bool,
                 need_embeddings: bool = False):
        self.stock = torch.from_numpy(label_set.stock)
        self.targets = torch.from_numpy(label_set.targets)
        self.weights = torch.from_numpy(label_set.weights)
        self.previous = torch.from_numpy(_previous_frame_index(sessions))
        self.valid = torch.from_numpy(_previous_valid(sessions))

        if need_images:
            self.images = torch.from_numpy(ds.load_images(sessions))
        else:
            # Keep the signature identical for both models without paying for image loading.
            self.images = torch.zeros((len(label_set), 1, 1, 1), dtype=torch.float32)

        self.embeddings = (torch.from_numpy(ds.load_embeddings(sessions))
                           if need_embeddings else None)

    def embedding_rows(self, index: torch.Tensor) -> torch.Tensor | None:
        return None if self.embeddings is None else self.embeddings[index]

    def __len__(self) -> int:
        return len(self.stock)

    def hard_negative_weights(self, dims: Sequence[int], boost: float,
                              threshold: float) -> torch.Tensor | None:
        """Sampling weight per row, raised on frames that sit on the decision boundary.

        A "hard negative" here is a frame we know should read zero on a watched expression, where
        the stock model is already firing anyway. Those are the frames the corrector has to get
        right to fix an intermittent false activation, and in a corpus dominated by easy resting
        frames they are rare enough to be drowned out. Returns None when nothing qualifies, so the
        caller falls back to plain shuffling rather than silently training on a degenerate sampler.
        """
        if boost <= 0 or not dims:
            return None

        index = torch.as_tensor(list(dims), dtype=torch.long)
        confident_zero = (self.weights.index_select(1, index) > 0) & \
                         (self.targets.index_select(1, index) <= 1e-6)
        firing = self.stock.index_select(1, index) > threshold
        hard = (confident_zero & firing).any(dim=1)

        if not bool(hard.any()):
            return None

        return torch.where(hard, torch.full_like(hard, 1.0 + boost, dtype=torch.float32),
                           torch.ones(len(self), dtype=torch.float32))

    def loader(self, batch_size: int, shuffle: bool,
               sample_weights: torch.Tensor | None = None) -> DataLoader:
        dataset = TensorDataset(torch.arange(len(self)))

        if sample_weights is not None:
            # Replacement sampling keeps the epoch the same length while changing what it contains.
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(self), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model,
    data: "_Batch",
    loader,
    optimizer,
    shrinkage: float,
    train: bool,
    *,
    consistency: float = 0.0,
    temporal: float = 0.0,
    fp_penalty: float = 0.0,
    fp_dims: Sequence[int] = (),
    embedding_noise: float = 0.0,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """One pass over the data.

    Beyond the supervised fit there are two regularizers, both aimed at the same complaint - "at
    rest my jaw wiggles and my mouth looks slightly open" - and both phrased as constraints on the
    *residual* rather than the output, so neither can make the face feel sluggish:

    * **consistency** - the residual must not change when only illumination changed.
    * **temporal** - the residual must not jump between consecutive frames of a session.

    Both are skipped at validation time: they are training pressure, not something to score.
    """
    model.train(train)
    totals = {"loss": 0.0, "fit": 0.0, "shrink": 0.0, "consistency": 0.0, "temporal": 0.0, "fp": 0.0}
    batches = 0

    uses_image = getattr(model, "uses_image", False)

    for (index,) in loader:
        images, stock = data.images[index], data.stock[index]
        targets, weights = data.targets[index], data.weights[index]
        embeddings = data.embedding_rows(index)

        with torch.set_grad_enabled(train):
            residual = model.residual(images, stock, embeddings)
            predicted = torch.clamp(stock + residual, 0.0, 1.0)
            loss, parts = models.masked_residual_loss(
                predicted, residual, targets, weights, shrinkage=shrinkage
            )

            consistency_value = 0.0
            if train and consistency > 0 and uses_image:
                jittered = augment.photometric_jitter(images, generator=generator)
                # Same stock vector on purpose: the model must not be able to satisfy this by
                # leaning on stock, only by making the image branch illumination-invariant.
                penalty = models.consistency_penalty(residual, model.residual(jittered, stock))
                loss = loss + consistency * penalty
                consistency_value = float(penalty.detach())
            elif train and embedding_noise > 0 and embeddings is not None:
                # Model C has no pixels to re-light, but the same idea applies: the correction
                # should not swing on small wobbles in the feature vector it is reading.
                noisy = embeddings + torch.randn(
                    embeddings.shape, generator=generator) * embedding_noise * embeddings.std()
                penalty = models.consistency_penalty(
                    residual, model.residual(images, stock, noisy))
                loss = loss + consistency * penalty
                consistency_value = float(penalty.detach())

            # Scored at validation as well as training: it is a real property of the model, and
            # seeing it on held-out data is how an over-aggressive setting gets caught.
            fp_value = 0.0
            if fp_penalty > 0 and len(fp_dims):
                penalty = models.false_positive_penalty(predicted, targets, weights, fp_dims)
                if train:
                    loss = loss + fp_penalty * penalty
                fp_value = float(penalty.detach())

            temporal_value = 0.0
            if train and temporal > 0:
                previous = data.previous[index]
                penalty = models.temporal_penalty(
                    residual,
                    model.residual(data.images[previous], data.stock[previous],
                                   data.embedding_rows(previous)),
                    data.valid[index],
                )
                loss = loss + temporal * penalty
                temporal_value = float(penalty.detach())

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        parts["consistency"] = consistency_value
        parts["temporal"] = temporal_value
        parts["fp"] = fp_value
        parts["loss"] = float(loss.detach())

        for key in totals:
            totals[key] += parts[key]
        batches += 1

    return {k: v / max(batches, 1) for k, v in totals.items()}


@torch.no_grad()
def _predict(model, images: torch.Tensor, stock: torch.Tensor,
             embeddings: torch.Tensor | None = None, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(stock), batch_size):
        stop = start + batch_size
        residual = model.residual(
            images[start:stop], stock[start:stop],
            None if embeddings is None else embeddings[start:stop])
        out.append(torch.clamp(stock[start:stop] + residual, 0.0, 1.0).numpy())
    return np.concatenate(out) if out else np.zeros((0, schema.EXPRESSION_COUNT), dtype=np.float32)


def _stage(name: str) -> None:
    """Emit a machine-readable stage marker.

    The app's one-button workflow maps these to friendly progress text. Printed rather than
    inferred from log shape so a wording change here cannot silently break the UI.
    """
    print(f"[stage] {name}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _stage("preparing")
    print(f"Loading sessions from {args.data}")
    sessions = ds.discover_sessions(args.data)
    print(ds.describe(sessions))

    train_sessions, val_sessions = ds.split_sessions(sessions, args.val_sessions)
    print(f"\nTrain sessions ({len(train_sessions)}):\n{ds.describe(train_sessions)}")
    if val_sessions:
        print(f"\nValidation sessions ({len(val_sessions)}):\n{ds.describe(val_sessions)}")
    else:
        print("\nWARNING: no validation sessions. Record more than one session per type so a whole "
              "session can be held out; metrics below are training-set only and will look better "
              "than reality.")

    use_pseudo = not args.no_speech_pseudo_labels
    dim_boost = lbl.parse_dim_boost(args.dim_boost)
    fp_dims = lbl.parse_dims(args.fp_dims) if args.fp_penalty > 0 else ()

    train_labels = lbl.build_labels(train_sessions, use_speech_pseudo_labels=use_pseudo,
                                    dim_boost=dim_boost)
    print(f"\nTraining labels:\n{train_labels.describe()}")

    # Printed before training, because a weak correlation means the guided labels are fiction and
    # nothing downstream would reveal it - the metrics are built from the same labels.
    lag_reports = lbl.cue_lag_report(sessions)
    if lag_reports:
        print()
        print(lbl.format_cue_lag_report(lag_reports))

    if dim_boost:
        described = ", ".join(f"{schema.EXPRESSION_NAMES[d]}x{w:g}" for d, w in dim_boost.items())
        print(f"  weight boost: {described}")
    if fp_dims:
        described = ", ".join(schema.EXPRESSION_NAMES[d] for d in fp_dims)
        print(f"  false-positive penalty: {args.fp_penalty:g} on {described}")

    if train_labels.coverage() == 0:
        print("\nERROR: no supervised cells. Record a neutral session (or a guided one) - speech "
              "alone with pseudo-labels disabled supervises nothing.")
        return 1

    model = models.build_model(args.model)
    need_images = model.uses_image
    need_embeddings = models.uses_embedding(model)

    if need_embeddings and not ds.has_embeddings(sessions):
        print("\nERROR: model C needs the stock visual embedding for every frame, and at least one "
              "session has none.\nCompute them first:\n"
              "  python -m babble_personal.derive_embedding --stock <faceModel.onnx>\n"
              "  python -m babble_personal.compute_embeddings --data <root> "
              "--model <faceModelWithEmbedding.onnx>")
        return 1

    # Augmentation draws from its own generator so a run stays reproducible from --seed.
    generator = torch.Generator().manual_seed(args.seed)
    print(f"\nModel: {model.adapter_type} ({models.parameter_count(model):,} parameters)")

    train_data = _Batch(train_sessions, train_labels, need_images, need_embeddings)

    sample_weights = train_data.hard_negative_weights(
        fp_dims or evaluate.WATCHED_DIMS, args.hard_negative_boost, args.hard_negative_threshold)
    if args.hard_negative_boost > 0:
        if sample_weights is None:
            print("  hard-negative boost requested, but no frame qualifies "
                  f"(no confidently-zero frame has stock > {args.hard_negative_threshold}); "
                  "sampling normally")
        else:
            hard_count = int((sample_weights > 1.0).sum())
            print(f"  hard-negative boost: {args.hard_negative_boost:g} on {hard_count} of "
                  f"{len(train_data)} frames")

    train_loader = train_data.loader(args.batch_size, shuffle=True, sample_weights=sample_weights)

    # Model C reuses the consistency weight against embedding noise rather than re-lighting.
    consistency = args.consistency if (need_images or need_embeddings) else 0.0
    if need_images and consistency > 0:
        print(f"  illumination consistency: {consistency} (model B sees pixels, so it needs this)")
    if need_embeddings and consistency > 0 and args.embedding_noise > 0:
        print(f"  embedding-noise consistency: {consistency} (sigma {args.embedding_noise})")
    if args.temporal > 0:
        print(f"  residual smoothness: {args.temporal}")

    val_labels = None
    val_data = None
    val_loader = None
    if val_sessions:
        val_labels = lbl.build_labels(val_sessions, use_speech_pseudo_labels=use_pseudo,
                                      dim_boost=dim_boost)
        val_data = _Batch(val_sessions, val_labels, need_images, need_embeddings)
        val_loader = val_data.loader(args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = Path(args.out) / f"{datetime.now():%Y%m%d_%H%M%S}_{model.adapter_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"

    best_score = float("inf")
    best_epoch = -1
    history = []

    _stage("training")
    print(f"\nTraining for up to {args.epochs} epochs (early stop patience {args.patience})")
    for epoch in range(args.epochs):
        train_stats = _run_epoch(
            model, train_data, train_loader, optimizer, args.shrinkage, train=True,
            consistency=consistency, temporal=args.temporal, generator=generator,
            fp_penalty=args.fp_penalty, fp_dims=fp_dims,
            embedding_noise=args.embedding_noise if need_embeddings else 0.0,
        )
        scheduler.step()

        if val_loader is not None:
            val_stats = _run_epoch(
                model, val_data, val_loader, optimizer, args.shrinkage, train=False,
                fp_penalty=args.fp_penalty, fp_dims=fp_dims)
            score = val_stats["fit"]
            line = (f"  epoch {epoch + 1:>3}  train {train_stats['loss']:.5f}"
                    f"  val_fit {val_stats['fit']:.5f}  shrink {train_stats['shrink']:.5f}")
        else:
            val_stats = {}
            score = train_stats["fit"]
            line = f"  epoch {epoch + 1:>3}  train {train_stats['loss']:.5f}  (no validation)"

        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})

        if score < best_score - 1e-6:
            best_score, best_epoch = score, epoch
            torch.save({"state_dict": model.state_dict(), "adapter_type": model.adapter_type},
                       checkpoint_path)
            line += "  *"

        print(line)

        if epoch - best_epoch >= args.patience:
            print(f"  early stop: no improvement for {args.patience} epochs")
            break

    _stage("evaluating")

    # Report using the best checkpoint, not the last epoch.
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    report_sessions = val_sessions if val_sessions else train_sessions
    report_labels = val_labels if val_labels is not None else train_labels
    scope = "held-out sessions" if val_sessions else "TRAINING data (no held-out sessions)"

    images = (torch.from_numpy(ds.load_images(report_sessions)) if need_images
              else torch.zeros((len(report_labels), 1, 1, 1)))
    report_embeddings = (torch.from_numpy(ds.load_embeddings(report_sessions))
                         if need_embeddings else None)
    stock_t = torch.from_numpy(report_labels.stock)
    personal = _predict(model, images, stock_t, report_embeddings)

    print(f"\n=== Evaluation on {scope} ===")
    expression_reports = evaluate.per_expression_mae(
        report_labels.stock, personal, report_labels.targets, report_labels.weights
    )

    neutral_sessions = [s for s in report_sessions if s.is_neutral]
    neutral = None
    if neutral_sessions:
        neutral_labels = lbl.build_labels(neutral_sessions, use_speech_pseudo_labels=False)
        neutral_images = (torch.from_numpy(ds.load_images(neutral_sessions)) if need_images
                          else torch.zeros((len(neutral_labels), 1, 1, 1)))
        neutral_embeddings = (torch.from_numpy(ds.load_embeddings(neutral_sessions))
                              if need_embeddings else None)
        neutral_personal = _predict(model, neutral_images, torch.from_numpy(neutral_labels.stock),
                                    neutral_embeddings)
        neutral = evaluate.neutral_report(neutral_labels.stock, neutral_personal)

    # The expressions the user still complains about, examined in detail: how badly they misfire
    # while the face is closed, how long each misfire lasts, and - the guardrail - whether the model
    # bought that improvement by refusing to move at all.
    dim_reports = evaluate.build_dim_reports(
        report_sessions, report_labels, report_labels.stock, personal)

    cued_rows, cued_dims = evaluate.collect_cued_dims(report_sessions)
    cross_talk_stock = cross_talk_personal = None
    if len(cued_rows):
        cross_talk_stock = evaluate.cross_talk(report_labels.stock[cued_rows], cued_dims)
        cross_talk_personal = evaluate.cross_talk(personal[cued_rows], cued_dims)

    hard_example_block = _hard_example_block(report_sessions, report_labels, personal)

    report_text = evaluate.format_report(
        expression_reports, neutral, dim_reports=dim_reports,
        cross_talk_stock=cross_talk_stock, cross_talk_personal=cross_talk_personal)
    print(report_text)

    (run_dir / "metrics.txt").write_text(report_text, encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({
        "adapter_type": model.adapter_type,
        "parameters": models.parameter_count(model),
        "schema_sha256": schema.SCHEMA_SHA256,
        "train_sessions": [s.session_id for s in train_sessions],
        "val_sessions": [s.session_id for s in val_sessions],
        "best_epoch": best_epoch + 1,
        "best_score": best_score,
        "args": {k: str(v) for k, v in vars(args).items()},
    }, indent=2), encoding="utf-8")

    summary = _build_summary(
        model=model,
        checkpoint_path=checkpoint_path,
        expression_reports=expression_reports,
        neutral=neutral,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        dim_reports=dim_reports,
        cross_talk_stock=cross_talk_stock,
        cross_talk_personal=cross_talk_personal,
        hard_examples=hard_example_block,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Export with: python -m babble_personal.export --checkpoint \"{checkpoint_path}\"")
    _stage("done")
    return 0


def _hard_example_block(report_sessions, report_labels, personal) -> dict | None:
    """Scores on frames the user personally flagged as wrong, when any are held out.

    These are the highest-value frames in the corpus: real failures of a real model, labelled by the
    only authority on what the face was actually doing. A model that improves everywhere except here
    has not fixed the thing that prompted the complaint.
    """
    corrections = [s for s in report_sessions if getattr(s, "is_correction", False)]
    if not corrections:
        return None

    per_dim: dict[str, dict] = {}
    total_frames = 0

    for dim in evaluate.WATCHED_DIMS:
        blocks = evaluate.collect_closed_blocks(
            corrections, report_labels, report_labels.stock, personal, dim)
        if not blocks:
            continue

        stock_values = np.concatenate([b[0] for b in blocks])
        personal_values = np.concatenate([b[1] for b in blocks])
        total_frames = max(total_frames, len(stock_values))

        per_dim[schema.EXPRESSION_NAMES[dim]] = {
            "frames": len(stock_values),
            "stock_fp_rate": float((stock_values > evaluate.ACTIVATION_THRESHOLD).mean()),
            "personal_fp_rate": float((personal_values > evaluate.ACTIVATION_THRESHOLD).mean()),
            "stock_mean": float(stock_values.mean()),
            "personal_mean": float(personal_values.mean()),
        }

    if not per_dim:
        return None

    return {
        "sessions": [s.session_id for s in corrections],
        "frames": total_frames,
        "per_expression": per_dim,
    }


def _build_summary(
    *,
    model,
    checkpoint_path: Path,
    expression_reports,
    neutral,
    train_sessions,
    val_sessions,
    dim_reports=(),
    cross_talk_stock=None,
    cross_talk_personal=None,
    hard_examples=None,
) -> dict:
    """Machine-readable outcome for the app's results screen.

    The app reads this instead of parsing printed text, so the human-facing report can be reworded
    freely without breaking the UI. Everything here is measured - there is deliberately no invented
    "quality score", and ``verdict`` stays "unclear" unless the numbers actually say something.
    """
    scored = [r for r in expression_reports if r.supervised_frames > 0]
    improved = [r for r in scored if r.improvement > 1e-4]
    regressed = [r for r in scored if r.improvement < -1e-4]

    stock_mae = float(np.mean([r.stock_mae for r in scored])) if scored else None
    personal_mae = float(np.mean([r.personal_mae for r in scored])) if scored else None

    summary: dict = {
        "summary_version": 2,
        "adapter_type": model.adapter_type,
        "parameters": models.parameter_count(model),
        "checkpoint": str(checkpoint_path),
        "schema_sha256": schema.SCHEMA_SHA256,
        "train_sessions": [s.session_id for s in train_sessions],
        "val_sessions": [s.session_id for s in val_sessions],
        "validated_on_held_out_sessions": bool(val_sessions),
        "scored_expressions": len(scored),
        "expressions_improved": len(improved),
        "expressions_regressed": len(regressed),
        "mean_stock_mae": stock_mae,
        "mean_personal_mae": personal_mae,
        "worst_regressions": [
            {"name": r.name, "stock_mae": r.stock_mae, "personal_mae": r.personal_mae}
            for r in sorted(regressed, key=lambda r: r.improvement)[:5]
        ],
        "biggest_improvements": [
            {"name": r.name, "stock_mae": r.stock_mae, "personal_mae": r.personal_mae}
            for r in sorted(improved, key=lambda r: r.improvement, reverse=True)[:5]
        ],
    }

    if neutral is not None and neutral.frames > 0:
        summary["neutral"] = {
            "frames": neutral.frames,
            "stock_false_activation_rate": neutral.stock_false_activation_rate,
            "personal_false_activation_rate": neutral.personal_false_activation_rate,
            "stock_mean_activation": neutral.stock_mean_activation,
            "personal_mean_activation": neutral.personal_mean_activation,
            "stock_jitter": neutral.stock_jitter,
            "personal_jitter": neutral.personal_jitter,
        }

    # Watched expressions get a top-level key each ("jaw_open", "tongue_out"), so the app can read
    # the one it cares about without knowing the list.
    for report in dim_reports:
        key = "".join(f"_{c.lower()}" if c.isupper() else c for c in report.name).lstrip("_")
        summary[key] = evaluate.dim_report_to_dict(report)

    if cross_talk_stock is not None and cross_talk_personal is not None:
        summary["cross_talk"] = {"stock": cross_talk_stock, "personal": cross_talk_personal}
    else:
        summary["cross_talk"] = None

    summary["hard_examples"] = hard_examples

    summary["verdict"] = _verdict(summary)
    return summary


def _verdict(summary: dict) -> str:
    """'better' | 'unclear' | 'worse', judged only on what was actually measured.

    Deliberately conservative. Without held-out sessions the numbers describe data the model was
    trained on and cannot support any claim, so the verdict is 'unclear' regardless of how good they
    look. A model is only "better" if it improves accuracy or neutral stability *without* trading one
    away for the other - suppressing everything would otherwise score as a win.
    """
    if not summary.get("validated_on_held_out_sessions"):
        return "unclear"

    stock_mae = summary.get("mean_stock_mae")
    personal_mae = summary.get("mean_personal_mae")
    neutral = summary.get("neutral")

    mae_better = stock_mae is not None and personal_mae is not None and personal_mae < stock_mae - 1e-4
    mae_worse = stock_mae is not None and personal_mae is not None and personal_mae > stock_mae + 1e-4

    neutral_better = neutral_worse = False
    if neutral:
        delta = neutral["stock_false_activation_rate"] - neutral["personal_false_activation_rate"]
        neutral_better = delta > 0.005
        neutral_worse = delta < -0.005

    if mae_worse or neutral_worse:
        return "worse" if not (mae_better or neutral_better) else "unclear"
    if mae_better or neutral_better:
        return "better"
    return "unclear"


if __name__ == "__main__":
    raise SystemExit(main())
