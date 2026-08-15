"""Why a guided cue scored the way it did, dimension by dimension.

    python -m babble_personal.diagnose_guided --data "%APPDATA%\\ProjectBabble\\PersonalDataset"
    python -m babble_personal.diagnose_guided --data ... --cue Grimace

The quality gate reduces each attempt to a single number: the correlation between one commanded
dimension and the stock model's reading of it. That is the right summary for training and the wrong
one for diagnosis, because when an attempt scores badly it cannot tell you *which* part went wrong.
A cue can fail because the user did not perform it, because the stock model cannot see the muscles
involved, or because the probe dimension was a poor stand-in for the expression - and the fix is
different in each case.

This prints every commanded dimension of every attempt so those three cases can be told apart:

* several dimensions correlate positively -> the user performed it and the model sees it. A bad
  overall verdict points at the probe, not the person.
* the dimension the cue is really about correlates, others do not -> normal. Expressions are not
  independent and the stock model is not equally sensitive everywhere.
* nothing correlates, in any repetition, in any session -> the stock model cannot see this
  expression on this face. More recordings will not help; the cue needs redesigning.

Read-only. It opens recordings and prints; it never writes to the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import dataset as ds
from . import labels as lbl
from . import schema


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-dimension cue-following diagnosis for guided recordings.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset root")
    parser.add_argument("--cue", default=None,
                        help="Only report cues whose id contains this (case-insensitive)")
    parser.add_argument("--session", default=None,
                        help="Only report sessions whose id contains this")
    return parser.parse_args(argv)


def attempt_dimension_report(session, cue_id: str, rep: int, attempt: int, frames: list) -> dict:
    """Correlation and lag for every dimension this attempt commands."""
    hold_frames = [f for f in frames
                   if str((f.cue or {}).get("phase", "")).lower() == "hold"]
    if not hold_frames:
        return {}

    exemplar = next((f for f in hold_frames if (f.cue or {}).get("dims")), hold_frames[0])
    dims = [int(d) for d in (exemplar.cue or {}).get("dims", [])]
    dims = [d for d in dims if 0 <= d < schema.EXPRESSION_COUNT]
    if not dims:
        return {}

    timestamps = np.asarray([f.timestamp_ticks for f in frames], dtype=np.float64)
    rows = []

    for dim in dims:
        def command(frame, d=dim) -> float:
            target = (frame.cue or {}).get("target")
            if not isinstance(target, (list, tuple)) or d >= len(target):
                return 0.0
            try:
                return float(target[d])
            except (TypeError, ValueError):
                return 0.0

        commanded = np.asarray([command(f) for f in frames], dtype=np.float64)
        observed = np.asarray([float(f.stock[dim]) for f in frames], dtype=np.float64)
        lag, correlation = lbl.estimate_cue_lag_seconds(observed, commanded, timestamps)

        rows.append({
            "dim": schema.EXPRESSION_NAMES[dim],
            "commanded_peak": float(np.max(np.abs(commanded))) if len(commanded) else 0.0,
            "stock_rest": float(np.median(observed[commanded <= 1e-6]))
            if np.any(commanded <= 1e-6) else float("nan"),
            "stock_hold": float(np.median(observed[commanded > 1e-6]))
            if np.any(commanded > 1e-6) else float("nan"),
            "correlation": float(correlation),
            "lag_seconds": float(lag),
        })

    return {
        "session": session.session_id,
        "cue": cue_id,
        "rep": rep,
        "attempt": attempt,
        "probe": schema.EXPRESSION_NAMES[lbl._dominant_cued_dim(dims, hold_frames)],
        "dimensions": rows,
    }


def collect(sessions, cue_filter: str | None = None) -> list[dict]:
    reports = []

    for session in sessions:
        if not session.is_guided:
            continue

        groups: dict[tuple[str, int, int], list] = {}
        for frame in session.frames:
            cue = frame.cue
            if not cue:
                continue
            key = (str(cue.get("id", "")),
                   int(cue.get("rep", 0) or 0),
                   int(cue.get("attempt", 0) or 0))
            groups.setdefault(key, []).append(frame)

        for (cue_id, rep, attempt), frames in sorted(groups.items()):
            if cue_filter and cue_filter.lower() not in cue_id.lower():
                continue
            report = attempt_dimension_report(session, cue_id, rep, attempt, frames)
            if report:
                reports.append(report)

    return reports


def format_reports(reports: list[dict]) -> str:
    if not reports:
        return "No guided attempts matched."

    lines: list[str] = []
    for report in reports:
        lines.append(
            f"\n{report['session']}  {report['cue']}  rep {report['rep']}  "
            f"attempt {report['attempt']}    probe: {report['probe']}")
        lines.append(f"  {'dimension':<24}{'cmd':>7}{'stock rest':>12}"
                     f"{'stock hold':>12}{'corr':>8}{'lag s':>8}")

        for row in sorted(report["dimensions"], key=lambda r: -r["commanded_peak"]):
            lines.append(
                f"  {row['dim']:<24}{row['commanded_peak']:>7.2f}"
                f"{row['stock_rest']:>12.3f}{row['stock_hold']:>12.3f}"
                f"{row['correlation']:>+8.2f}{row['lag_seconds']:>8.2f}")

    # The summary is the part that decides what to do next.
    lines.append("\n" + "=" * 72)
    lines.append("Best correlation per cue, across every attempt and session:")
    best: dict[str, tuple[str, float]] = {}
    for report in reports:
        for row in report["dimensions"]:
            current = best.get(report["cue"])
            if current is None or row["correlation"] > current[1]:
                best[report["cue"]] = (row["dim"], row["correlation"])

    for cue, (dim, correlation) in sorted(best.items()):
        verdict = ("the model follows this cue" if correlation >= lbl.GUIDED_GOOD_CORRELATION
                   else "nothing here tracks the command - suspect the cue, not the user"
                   if correlation < 0.2 else "weakly followed")
        lines.append(f"  {cue:<20}{dim:<24}{correlation:>+6.2f}   {verdict}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sessions = ds.discover_sessions(args.data)
    if args.session:
        sessions = [s for s in sessions if args.session.lower() in s.session_id.lower()]

    print(format_reports(collect(sessions, args.cue)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
