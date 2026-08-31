"""Tests for the JawOpen-focused metrics.

Two things are being pinned here.

The first is that a false activation is measured **in time**, not only as a fraction of frames. The
user's complaint is "my jaw hangs open", and a jaw that hangs for two seconds once is a completely
different experience from one that flickers for a single frame forty times - yet both can produce the
same false-activation rate. Every assertion about runs exists because that rate, on its own, was the
only number the project had and it could not tell those two apart.

The second is the anti-suppression guardrail. Driving an expression to zero wins every
false-positive metric in this file, so range retention and open/closed separation are computed from
the same data and must fall when a model cheats that way. The test that matters most here is
``test_a_suppressing_model_is_caught``: it builds a model that would look perfect by the old numbers
and asserts the new ones expose it.

    python -m pytest training/tests/test_evaluate_metrics.py -q
    python training/tests/test_evaluate_metrics.py          (no pytest required)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babble_personal import evaluate, labels as lbl, schema  # noqa: E402
from babble_personal.dataset import FrameRecord, Session  # noqa: E402

N = schema.EXPRESSION_COUNT
JAW = schema.INDEX_OF["JawOpen"]
FPS = 30
FRAME_TICKS = lbl.TICKS_PER_SECOND // FPS

#: 10,000,000 ticks per second does not divide evenly by 30 fps, so a nominal one-second fixture is
#: really 0.999999 s. Real recordings carry the same rounding, so the tolerance lives here rather
#: than being engineered away in the metric - 0.1 ms is far tighter than anything we assert about.
TICK_TOLERANCE = 1e-4


def _ticks(count: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.float64) * FRAME_TICKS


# ---------------------------------------------------------------------------
# activation_runs
# ---------------------------------------------------------------------------


def test_no_activation_means_no_runs() -> None:
    assert evaluate.activation_runs(np.zeros(90), _ticks(90)) == []


def test_single_frame_blip_counts_as_one_frame_not_zero() -> None:
    values = np.zeros(30)
    values[10] = 0.9

    runs = evaluate.activation_runs(values, _ticks(30))

    assert len(runs) == 1
    assert abs(runs[0] - 1 / FPS) < 1e-6, "a one-frame blip should last one frame, not zero seconds"


def test_run_duration_matches_wall_clock() -> None:
    """One continuous second above threshold reads as one second."""
    values = np.zeros(90)
    values[30:60] = 0.8  # 30 frames at 30 fps

    runs = evaluate.activation_runs(values, _ticks(90))

    assert len(runs) == 1
    assert abs(runs[0] - 1.0) < TICK_TOLERANCE


def test_separate_activations_are_separate_runs() -> None:
    values = np.zeros(120)
    values[10:20] = 0.9
    values[60:90] = 0.9

    runs = evaluate.activation_runs(values, _ticks(120))

    assert len(runs) == 2
    assert abs(max(runs) - 1.0) < TICK_TOLERANCE


def test_activation_still_open_at_the_end_is_counted() -> None:
    values = np.zeros(60)
    values[45:] = 0.9

    runs = evaluate.activation_runs(values, _ticks(60))

    assert len(runs) == 1, "a run that never ends before the recording does must still be reported"


def test_the_metric_that_rate_alone_cannot_express() -> None:
    """Same false-activation rate, opposite user experience. This is the whole point of run stats."""
    frames = 300

    flickery = np.zeros(frames)
    flickery[::10] = 0.9  # 30 isolated single-frame blips

    hanging = np.zeros(frames)
    hanging[100:130] = 0.9  # one continuous second

    assert abs((flickery > 0.15).mean() - (hanging > 0.15).mean()) < 1e-9, (
        "fixture is wrong: the two patterns must have the same false-activation rate"
    )

    flickery_runs = evaluate.summarize_runs(
        evaluate.activation_runs(flickery, _ticks(frames)), frames / FPS)
    hanging_runs = evaluate.summarize_runs(
        evaluate.activation_runs(hanging, _ticks(frames)), frames / FPS)

    assert hanging_runs.max_seconds > flickery_runs.max_seconds * 10
    assert flickery_runs.count > hanging_runs.count


def test_summarize_runs_rates() -> None:
    stats = evaluate.summarize_runs([1.0, 2.0, 3.0], total_seconds=60.0)

    assert stats.count == 3
    assert abs(stats.runs_per_minute - 3.0) < 1e-9
    assert abs(stats.mean_seconds - 2.0) < 1e-9
    assert abs(stats.max_seconds - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Contiguity - a filtered-out gap must break a run
# ---------------------------------------------------------------------------


def test_gaps_split_blocks() -> None:
    blocks = evaluate._contiguous_blocks(np.array([0, 1, 2, 7, 8, 20]))

    assert [b.tolist() for b in blocks] == [[0, 1, 2], [7, 8], [20]]


# ---------------------------------------------------------------------------
# Ordering / monotonicity
# ---------------------------------------------------------------------------


def test_monotonic_levels_score_perfectly() -> None:
    levels = [0.5, 1.0, 0.5, 1.0]
    means = [0.4, 0.9, 0.45, 0.95]

    assert evaluate._ordering_accuracy(levels, means) == 1.0
    assert evaluate._spearman(levels, means) > 0.9


def test_inverted_levels_are_caught() -> None:
    levels = [0.5, 1.0]
    means = [0.9, 0.2]  # commanded more, produced less

    assert evaluate._ordering_accuracy(levels, means) == 0.0


def test_ordering_accuracy_ignores_equal_levels() -> None:
    assert evaluate._ordering_accuracy([0.5, 0.5], [0.1, 0.9]) is None


# ---------------------------------------------------------------------------
# dim_report end to end
# ---------------------------------------------------------------------------


def test_dim_report_closed_and_sensitivity() -> None:
    closed_stock = np.full(120, 0.4, dtype=np.float32)     # stock hangs open at rest
    closed_personal = np.full(120, 0.02, dtype=np.float32)  # personal fixes it

    hold_stock = np.full(60, 0.5, dtype=np.float32)
    hold_personal = np.full(60, 0.9, dtype=np.float32)

    report = evaluate.dim_report(
        JAW,
        closed_blocks=[(closed_stock, closed_personal, _ticks(120))],
        hold_segments=[(1.0, hold_stock, hold_personal)],
    )

    assert report.name == "JawOpen"
    assert report.closed_frames == 120
    assert report.stock_fp_rate == 1.0
    assert report.personal_fp_rate == 0.0
    assert report.stock_runs.max_seconds > 3.0, "stock hangs open for the whole block"
    assert report.personal_runs.count == 0

    # And it still opens when asked - separation went up, not down.
    assert report.personal_open_separation > report.stock_open_separation


def test_a_suppressing_model_is_caught() -> None:
    """The cheat: output zero always. Perfect false-positive numbers, dead face."""
    closed_stock = np.full(120, 0.4, dtype=np.float32)
    hold_stock = np.full(60, 0.6, dtype=np.float32)
    speech_stock = np.linspace(0.0, 0.9, 300).astype(np.float32)

    zeros_closed = np.zeros(120, dtype=np.float32)
    zeros_hold = np.zeros(60, dtype=np.float32)
    zeros_speech = np.zeros(300, dtype=np.float32)

    report = evaluate.dim_report(
        JAW,
        closed_blocks=[(closed_stock, zeros_closed, _ticks(120))],
        hold_segments=[(1.0, hold_stock, zeros_hold)],
        speech=(speech_stock, zeros_speech),
    )

    # It wins on every false-positive metric...
    assert report.personal_fp_rate == 0.0
    assert report.personal_runs.count == 0

    # ...and the guardrails catch it anyway.
    assert report.range_retention is not None and report.range_retention < 0.01
    assert report.suppression_warning, "a model that outputs nothing must raise the warning"
    assert report.personal_open_separation < report.stock_open_separation, (
        "suppression must show up as lost open/closed separation"
    )


def test_a_quiet_dimension_does_not_trigger_the_warning() -> None:
    """Shrinking noise on a dimension that never moves is correct behaviour, not suppression.

    Without this guard the tongue expressions flag on every run - they are mostly sensor noise
    during ordinary speech - and a warning that fires constantly stops being read.
    """
    noise_stock = (np.random.default_rng(0).normal(0, 0.005, 300) + 0.01).astype(np.float32)
    quieter = (noise_stock * 0.3).astype(np.float32)

    report = evaluate.dim_report(JAW, speech=(noise_stock, quieter))

    assert report.range_retention is not None and report.range_retention < 0.8
    assert not report.suppression_warning, "a dimension with no real movement must not flag"


def test_range_retention_is_one_for_a_faithful_model() -> None:
    speech_stock = np.linspace(0.0, 0.9, 300).astype(np.float32)

    report = evaluate.dim_report(JAW, speech=(speech_stock, speech_stock.copy()))

    assert report.range_retention is not None
    assert abs(report.range_retention - 1.0) < 1e-6
    assert not report.suppression_warning


def test_empty_report_is_safe() -> None:
    report = evaluate.dim_report(JAW)

    assert report.closed_frames == 0
    assert report.range_retention is None
    assert "no frames" in evaluate.format_dim_report(report)


# ---------------------------------------------------------------------------
# Collection helpers over real Session objects
# ---------------------------------------------------------------------------


def _session(session_type: str, cues, *, session_id: str = "s", jaw: float = 0.0) -> Session:
    frames = []
    for i, cue in enumerate(cues):
        stock = np.zeros(N, dtype=np.float32)
        stock[JAW] = jaw
        frames.append(FrameRecord(index=i, timestamp_ticks=i * FRAME_TICKS, stock=stock,
                                  image_path=Path(f"{i:06d}.jpg"), cue=cue))
    return Session(session_id=session_id, path=Path(session_id),
                   metadata={"SessionType": session_type}, frames=frames)


def test_collect_closed_blocks_picks_up_neutral_frames() -> None:
    session = _session("Neutral", [None] * 90, jaw=0.4)
    label_set = lbl.build_labels([session])
    stock = label_set.stock
    personal = np.zeros_like(stock)

    blocks = evaluate.collect_closed_blocks([session], label_set, stock, personal, JAW)

    assert len(blocks) == 1, "an unbroken neutral session is one contiguous block"
    assert len(blocks[0][0]) == 90


def test_collect_closed_blocks_excludes_trimmed_frames_and_splits_there() -> None:
    """A hold-then-rest session: only the settled part of the rest is 'known closed'."""
    hold = [{"id": "JawOpen100", "phase": "hold", "dims": [JAW],
             "target": [1.0 if d == JAW else 0.0 for d in range(N)], "level": 1.0, "rep": 0}] * 60
    rest = [{"id": "JawOpen100", "phase": "rest", "dims": [JAW],
             "target": [1.0 if d == JAW else 0.0 for d in range(N)], "level": 1.0, "rep": 0}] * 60
    session = _session("Guided", hold + rest)

    label_set = lbl.build_labels([session])
    stock = label_set.stock
    personal = np.zeros_like(stock)

    blocks = evaluate.collect_closed_blocks([session], label_set, stock, personal, JAW)
    total = sum(len(b[0]) for b in blocks)

    # Roughly half a second of the 60-frame rest is trimmed. Asserted to within a frame because the
    # tick-per-frame rounding above can put the boundary frame on either side of the threshold.
    trim_frames = int(lbl.HOLD_SETTLE_TRIM_SECONDS * FPS)
    assert abs(total - (60 - trim_frames)) <= 1, (
        f"only the settled portion of the rest should count as known-closed; got {total} frames"
    )
    assert total > 0, "the whole rest was trimmed"


def test_collect_hold_segments_recovers_the_commanded_level() -> None:
    hold = [{"id": "JawOpen50", "phase": "hold", "dims": [JAW],
             "target": [0.5 if d == JAW else 0.0 for d in range(N)], "level": 0.5, "rep": 0}] * 60
    session = _session("Guided", hold)

    label_set = lbl.build_labels([session])
    segments = evaluate.collect_hold_segments(
        [session], label_set, label_set.stock, label_set.stock.copy(), JAW)

    assert len(segments) == 1
    assert abs(segments[0][0] - 0.5) < 1e-6


def test_collect_speech_values_only_from_speech_sessions() -> None:
    neutral = _session("Neutral", [None] * 30, session_id="n")
    speech = _session("Speech", [None] * 45, session_id="sp", jaw=0.3)

    label_set = lbl.build_labels([neutral, speech])
    result = evaluate.collect_speech_values(
        [neutral, speech], label_set.stock, label_set.stock.copy(), JAW)

    assert result is not None
    assert len(result[0]) == 45


def test_collect_cued_dims_only_returns_holds() -> None:
    target = [1.0 if d == JAW else 0.0 for d in range(N)]
    cues = ([{"id": "J", "phase": "hold", "dims": [JAW], "target": target, "level": 1.0, "rep": 0}] * 30
            + [{"id": "J", "phase": "transition", "dims": [JAW], "target": target,
                "level": 1.0, "rep": 0}] * 30)
    session = _session("Guided", cues)

    rows, dims = evaluate.collect_cued_dims([session])

    assert len(rows) == 30, "transitions must not be scored for cross-talk"
    assert all(d == [JAW] for d in dims)


def test_build_dim_reports_covers_the_watched_expressions() -> None:
    session = _session("Neutral", [None] * 60, jaw=0.4)
    label_set = lbl.build_labels([session])
    stock = label_set.stock
    personal = np.zeros_like(stock)

    reports = evaluate.build_dim_reports([session], label_set, stock, personal)

    assert [r.name for r in reports] == ["JawOpen", "TongueOut"]


def test_dim_report_serializes() -> None:
    import json

    session = _session("Neutral", [None] * 60, jaw=0.4)
    label_set = lbl.build_labels([session])
    reports = evaluate.build_dim_reports(
        [session], label_set, label_set.stock, np.zeros_like(label_set.stock))

    payload = evaluate.dim_report_to_dict(reports[0])
    json.dumps(payload)  # must not raise

    assert payload["name"] == "JawOpen"
    assert "personal_runs" in payload and "max_seconds" in payload["personal_runs"]


def test_format_report_includes_cross_talk_and_dim_blocks() -> None:
    session = _session("Neutral", [None] * 60, jaw=0.4)
    label_set = lbl.build_labels([session])
    personal = np.zeros_like(label_set.stock)
    reports = evaluate.build_dim_reports([session], label_set, label_set.stock, personal)
    expression_reports = evaluate.per_expression_mae(
        label_set.stock, personal, label_set.targets, label_set.weights)

    text = evaluate.format_report(expression_reports, None, dim_reports=reports,
                                  cross_talk_stock=0.2, cross_talk_personal=0.1)

    assert "Cross-talk" in text
    assert "JawOpen detail" in text
    assert "longest false run" in text


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - standalone runner reports rather than raises
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
