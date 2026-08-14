"""Run several adapters against identical data and print one comparison table.

    python -m babble_personal.experiment --data <root> --models b c --seeds 0 1 2

Why a harness rather than running train twice
---------------------------------------------
Because "B versus C" is only a real question if nothing else differs, and by default nothing else is
held fixed. ``split_sessions`` picks the held-out sessions itself, so two runs can silently score
against different data; seeds change initialisation and augmentation draws; and a single run of each
cannot distinguish a two-point win from noise. This pins the split once, passes it to every run, and
reports spread across seeds so a difference has to survive being looked at properly.

The table is deliberately wide. A model that wins on mean error while lengthening the longest false
jaw-open has not won anything the user cares about, and printing only a headline number is how that
goes unnoticed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import dataset as ds
from . import train as trainer

#: One entry per column: (header, path to the model's value, path to the stock value, format).
#: The stock path is what lets the baseline print on the same axes as the candidates - without it
#: the reader has to remember what stock scored, which is exactly when a regression slips through.
COLUMNS: list[tuple[str, str, str | None, str]] = [
    ("params", "parameters", None, ",.0f"),
    ("train s", "_wall_seconds", None, ".0f"),
    ("onnx KB", "_onnx_kb", None, ".0f"),
    ("MAE", "mean_personal_mae", "mean_stock_mae", ".4f"),
    ("neutral FAR", "neutral.personal_false_activation_rate",
     "neutral.stock_false_activation_rate", ".4f"),
    ("jitter", "neutral.personal_jitter", "neutral.stock_jitter", ".4f"),
    ("jaw FP", "jaw_open.personal_fp_rate", "jaw_open.stock_fp_rate", ".4f"),
    ("jaw run s", "jaw_open.personal_runs.max_seconds", "jaw_open.stock_runs.max_seconds", ".2f"),
    ("jaw range", "jaw_open.range_retention", None, ".2f"),
    ("tongue FP", "tongue_out.personal_fp_rate", "tongue_out.stock_fp_rate", ".4f"),
    ("crosstalk", "cross_talk.personal", "cross_talk.stock", ".4f"),
]


def dig(payload: dict, path: str):
    """Follow a dotted path, returning None rather than raising on any miss."""
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current if isinstance(current, (int, float)) else None


def run_once(data: Path, model: str, seed: int, out: Path, val_sessions: list[str],
             extra: list[str]) -> dict | None:
    """Train one model at one seed and return its summary, enriched with wall time and file size."""
    argv = [
        "--data", str(data),
        "--model", model,
        "--out", str(out),
        "--seed", str(seed),
        "--val-sessions", *val_sessions,
        *extra,
    ]

    started = time.perf_counter()
    code = trainer.main(argv)
    elapsed = time.perf_counter() - started

    if code != 0:
        print(f"  !! {model} seed {seed} failed (exit {code})")
        return None

    run_dir = max((p for p in out.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["_wall_seconds"] = elapsed
    summary["_run_dir"] = str(run_dir)
    summary["_model"] = model
    summary["_seed"] = seed

    # Export too: file size is part of the cost, and a model that cannot export is not a candidate.
    try:
        from . import export as exporter

        onnx_path = run_dir / "personalFaceModel.onnx"
        exporter.export(run_dir / "model.pt", onnx_path, parity_samples=8)
        summary["_onnx_kb"] = onnx_path.stat().st_size / 1024
    except Exception as exc:  # noqa: BLE001 - a failed export is a result, not a crash
        print(f"  !! {model} seed {seed} exported badly: {exc}")
        summary["_onnx_kb"] = None

    return summary


def format_table(rows: list[tuple[str, dict[str, list[float]]]]) -> str:
    """One line per model, mean over seeds, with spread where more than one seed ran.

    Keyed by column header rather than by path so the stock baseline, whose values live under
    different keys in the same summary, lines up under the same headings.
    """
    lines: list[str] = []

    header = f"{'model':<22}" + "".join(f"{name:>16}" for name, _, _, _ in COLUMNS)
    lines.append(header)
    lines.append("-" * len(header))

    for label, values in rows:
        cells = []
        for name, _, _, spec in COLUMNS:
            samples = values.get(name, [])
            if not samples:
                cells.append(f"{'-':>16}")
                continue

            mean = float(np.mean(samples))
            text = format(mean, spec)
            if len(samples) > 1:
                spread = float(np.std(samples))
                # Only show spread when it is big enough to matter at the printed precision.
                # ASCII rather than a plus-minus sign: this text also lands in log files and
                # consoles that are not UTF-8.
                if spread > abs(mean) * 0.02:
                    text += f"+-{format(spread, spec).lstrip()}"
            cells.append(f"{text:>16}")
        lines.append(f"{label:<22}" + "".join(cells))

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare adapters on identical splits, seeds and labels.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["b", "c"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out", type=Path, default=Path("runs/experiment"))
    parser.add_argument("--val-sessions", nargs="*", default=None,
                        help="Held-out sessions. Default: chosen once and reused for every run.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    sessions = ds.discover_sessions(args.data)

    # Pinned once, up front. Letting each run choose its own would compare models on different data
    # and call the difference architecture.
    if args.val_sessions:
        val_sessions = list(args.val_sessions)
    else:
        _, held_out = ds.split_sessions(sessions)
        val_sessions = [s.session_id for s in held_out]

    if not val_sessions:
        raise SystemExit(
            "No held-out sessions available. Record at least two of one type - without a holdout "
            "this comparison would score every model on its own training data.")

    print(f"Held out for every run: {', '.join(val_sessions)}")
    print(f"Models: {', '.join(args.models)}   Seeds: {args.seeds}\n")

    extra = ["--epochs", str(args.epochs)] if args.epochs else []
    args.out.mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[dict]] = {}
    for model in args.models:
        for seed in args.seeds:
            print(f"--- {model} (seed {seed}) " + "-" * 40)
            summary = run_once(args.data, model, seed, args.out, val_sessions, extra)
            if summary:
                collected.setdefault(model, []).append(summary)

    if not collected:
        raise SystemExit("Every run failed; nothing to compare.")

    rows: list[tuple[str, dict[str, list[float]]]] = []

    # Stock first: the thing every adapter has to beat, read off the same runs so it is guaranteed
    # to describe the same held-out data.
    any_summary = next(iter(collected.values()))[0]
    stock_values = {name: [v] for name, _, stock_path, _ in COLUMNS
                    if stock_path and (v := dig(any_summary, stock_path)) is not None}
    rows.append(("stock (baseline)", stock_values))

    for model, summaries in collected.items():
        label = f"{model}: {summaries[0].get('adapter_type', model)}"
        values: dict[str, list[float]] = {}
        for name, path, _, _ in COLUMNS:
            samples = [v for s in summaries if (v := dig(s, path)) is not None]
            if samples:
                values[name] = samples
        rows.append((label, values))

    table = format_table(rows)
    print("\n" + "=" * len(table.splitlines()[0]))
    print(table)
    print()
    print("Lower is better everywhere except 'jaw range' (range retention), where a value well")
    print("below 1.0 means the model stopped moving the jaw rather than learning when to.")

    if args.csv:
        import csv

        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "seed", *[name for name, _, _, _ in COLUMNS]])
            for model, summaries in collected.items():
                for summary in summaries:
                    writer.writerow([model, summary["_seed"],
                                     *[dig(summary, path) for _, path, _, _ in COLUMNS]])
        print(f"\nWrote {args.csv}")

    if args.json:
        args.json.write_text(json.dumps(collected, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
