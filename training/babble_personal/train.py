"""Train a personal adapter.

    python -m babble_personal.train --data "%APPDATA%\\ProjectBabble\\PersonalDataset" --model a

Validation is always split by session (see dataset.split_sessions). Adjacent frames are
near-duplicates, so a per-frame split would report a flattering number that says nothing about how
the model behaves in a fresh session - which is the only thing that matters here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import dataset as ds
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
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _tensors(sessions, label_set: lbl.LabelSet, need_images: bool) -> TensorDataset:
    stock = torch.from_numpy(label_set.stock)
    targets = torch.from_numpy(label_set.targets)
    weights = torch.from_numpy(label_set.weights)

    if need_images:
        images = torch.from_numpy(ds.load_images(sessions))
    else:
        # Keep the signature identical for both models without paying for image loading.
        images = torch.zeros((len(label_set), 1, 1, 1), dtype=torch.float32)

    return TensorDataset(images, stock, targets, weights)


def _run_epoch(model, loader, optimizer, shrinkage, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "fit": 0.0, "shrink": 0.0}
    batches = 0

    for images, stock, targets, weights in loader:
        with torch.set_grad_enabled(train):
            residual = model.residual(images, stock)
            predicted = torch.clamp(stock + residual, 0.0, 1.0)
            loss, parts = models.masked_residual_loss(
                predicted, residual, targets, weights, shrinkage=shrinkage
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        for key in totals:
            totals[key] += parts[key]
        batches += 1

    return {k: v / max(batches, 1) for k, v in totals.items()}


@torch.no_grad()
def _predict(model, images: torch.Tensor, stock: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(stock), batch_size):
        out.append(model(images[start:start + batch_size], stock[start:start + batch_size]).numpy())
    return np.concatenate(out) if out else np.zeros((0, schema.EXPRESSION_COUNT), dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    train_labels = lbl.build_labels(train_sessions, use_speech_pseudo_labels=use_pseudo)
    print(f"\nTraining labels:\n{train_labels.describe()}")

    if train_labels.coverage() == 0:
        print("\nERROR: no supervised cells. Record a neutral session (or a guided one) - speech "
              "alone with pseudo-labels disabled supervises nothing.")
        return 1

    model = models.build_model(args.model)
    need_images = model.uses_image
    print(f"\nModel: {model.adapter_type} ({models.parameter_count(model):,} parameters)")

    train_loader = DataLoader(
        _tensors(train_sessions, train_labels, need_images),
        batch_size=args.batch_size, shuffle=True,
    )

    val_labels = None
    val_loader = None
    if val_sessions:
        val_labels = lbl.build_labels(val_sessions, use_speech_pseudo_labels=use_pseudo)
        val_loader = DataLoader(
            _tensors(val_sessions, val_labels, need_images),
            batch_size=args.batch_size, shuffle=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = Path(args.out) / f"{datetime.now():%Y%m%d_%H%M%S}_{model.adapter_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"

    best_score = float("inf")
    best_epoch = -1
    history = []

    print(f"\nTraining for up to {args.epochs} epochs (early stop patience {args.patience})")
    for epoch in range(args.epochs):
        train_stats = _run_epoch(model, train_loader, optimizer, args.shrinkage, train=True)
        scheduler.step()

        if val_loader is not None:
            val_stats = _run_epoch(model, val_loader, optimizer, args.shrinkage, train=False)
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

    # Report using the best checkpoint, not the last epoch.
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    report_sessions = val_sessions if val_sessions else train_sessions
    report_labels = val_labels if val_labels is not None else train_labels
    scope = "held-out sessions" if val_sessions else "TRAINING data (no held-out sessions)"

    images = (torch.from_numpy(ds.load_images(report_sessions)) if need_images
              else torch.zeros((len(report_labels), 1, 1, 1)))
    stock_t = torch.from_numpy(report_labels.stock)
    personal = _predict(model, images, stock_t)

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
        neutral_personal = _predict(model, neutral_images, torch.from_numpy(neutral_labels.stock))
        neutral = evaluate.neutral_report(neutral_labels.stock, neutral_personal)

    report_text = evaluate.format_report(expression_reports, neutral)
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

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Export with: python -m babble_personal.export --checkpoint \"{checkpoint_path}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
