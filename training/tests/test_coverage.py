"""Tests for the supervision-coverage census and the expanded co-activation exclusions.

Coverage is the metric that decides what to record next, and it is the one thing no accuracy number
can show: an expression scores perfectly by being right about zero forever. The first real corpus
had 45/45 expressions "supervised" and 0/45 ever shown at a non-zero value - a distinction that was
invisible until it was measured directly.

The exclusion tests guard a subtler failure. During a cue, every expression the cue did not name is
labelled "stay at rest" at low weight. For most that is correct; for the ones that physically move
*with* the cued expression it is a confident lie. TongueOut is the clearest case: a tongue cannot
come out through a closed mouth, so every tongue hold would otherwise also teach "and the jaw was
shut" - which is precisely the false-JawOpen failure this project exists to fix.

    python -m pytest training/tests/test_coverage.py -q
    python training/tests/test_coverage.py            (no pytest required)
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
TONGUE_OUT = schema.INDEX_OF["TongueOut"]
SMILE_L = schema.INDEX_OF["MouthSmileLeft"]
SMILE_R = schema.INDEX_OF["MouthSmileRight"]
FROWN_L = schema.INDEX_OF["MouthFrownLeft"]
MOUTH_CLOSE = schema.INDEX_OF["MouthClose"]
FPS = 30
FRAME_TICKS = lbl.TICKS_PER_SECOND // FPS


def _cue(cue_id: str, phase: str, dims, level: float, rep: int = 0) -> dict:
    target = [0.0] * N
    for dim in dims:
        target[dim] = level
    return {"id": cue_id, "phase": phase, "dims": list(dims), "target": target,
            "level": level, "rep": rep, "source": "avatar"}


def _session(session_type: str, cues, *, session_id: str = "s") -> Session:
    frames = [
        FrameRecord(index=i, timestamp_ticks=i * FRAME_TICKS,
                    stock=np.zeros(N, dtype=np.float32),
                    image_path=Path(f"{i:06d}.jpg"), cue=cue)
        for i, cue in enumerate(cues)
    ]
    return Session(session_id=session_id, path=Path(session_id),
                   metadata={"SessionType": session_type}, frames=frames)


def _guided_hold(cue_id: str, dims, level: float, seconds: float = 2.0, rep: int = 0):
    """A settled hold long enough to survive the trim."""
    return [_cue(cue_id, "hold", dims, level, rep)] * int(seconds * FPS)


# ---------------------------------------------------------------------------
# Coverage census
# ---------------------------------------------------------------------------


def test_neutral_only_corpus_reports_nothing_taught() -> None:
    """The exact state of the first real corpus: fully supervised, nothing ever shown 'on'."""
    session = _session("Neutral", [None] * 90)
    result = lbl.build_labels([session])

    rows = evaluate.coverage_report(result)

    assert len(rows) == N
    assert all(not r.has_positive for r in rows), "nothing should be marked as taught"
    assert all(r.zero_frames == 90 for r in rows)

    text = evaluate.format_coverage_report(rows)
    assert "0/45" in text
    assert "only ever been labelled zero" in text


def test_a_guided_hold_registers_as_taught() -> None:
    session = _session("Guided", _guided_hold("JawOpen100", [JAW], 1.0))
    result = lbl.build_labels([session])

    rows = {r.name: r for r in evaluate.coverage_report(result)}

    assert rows["JawOpen"].has_positive
    assert rows["JawOpen"].positive_frames > 0
    assert rows["JawOpen"].levels == (1.0,)
    assert not rows["MouthSmileLeft"].has_positive


def test_distinct_levels_are_reported() -> None:
    cues = (_guided_hold("JawOpen50", [JAW], 0.5) + _guided_hold("JawOpen100", [JAW], 1.0))
    result = lbl.build_labels([_session("Guided", cues)])

    row = next(r for r in evaluate.coverage_report(result) if r.name == "JawOpen")

    assert set(row.levels) == {0.5, 1.0}


def test_weak_supervision_is_counted_separately_from_confident() -> None:
    """Speech pseudo-labels supervise, but at 0.3 - they must not read as 'taught'."""
    result = lbl.build_labels([_session("Speech", [None] * 60)])

    rows = {r.name: r for r in evaluate.coverage_report(result)}

    assert rows["JawOpen"].weak_frames == 60
    assert rows["JawOpen"].positive_frames == 0
    assert not rows["JawOpen"].has_positive


def test_coverage_report_formats_a_mixed_corpus() -> None:
    guided = _session("Guided", _guided_hold("Smile100", [SMILE_L, SMILE_R], 1.0),
                      session_id="g")
    neutral = _session("Neutral", [None] * 60, session_id="n")
    result = lbl.build_labels([guided, neutral])

    text = evaluate.format_coverage_report(evaluate.coverage_report(result))

    assert "2/45" in text, "both smile corners should count as taught"
    assert "MouthSmileLeft" in text


# ---------------------------------------------------------------------------
# Co-activation exclusions
# ---------------------------------------------------------------------------


def test_a_tongue_cue_does_not_teach_that_the_jaw_was_shut() -> None:
    """The exclusion that matters most: a tongue cannot come out through a closed mouth."""
    session = _session("Guided", _guided_hold("TongueOut100", [TONGUE_OUT], 1.0))
    result = lbl.build_labels([session])

    # The tongue itself is supervised (at the reduced tongue weight)...
    assert (result.weights[-1, TONGUE_OUT] > 0)

    # ...but the jaw is left unlabelled rather than told it was closed.
    assert result.weights[-1, JAW] == 0.0, (
        "a tongue hold must not supervise the jaw as being at rest"
    )
    assert result.weights[-1, MOUTH_CLOSE] == 0.0


def test_a_frown_cue_does_not_punish_the_lower_lip() -> None:
    session = _session("Guided", _guided_hold("Frown100", [FROWN_L], 1.0))
    result = lbl.build_labels([session])

    for name in ("MouthLowerDownLeft", "MouthLowerDownRight", "MouthShrugLower"):
        dim = schema.INDEX_OF[name]
        assert result.weights[-1, dim] == 0.0, f"{name} should be excluded during a frown"


def test_a_sideways_mouth_cue_excludes_the_matching_jaw_shift() -> None:
    session = _session("Guided", _guided_hold("MouthLeft100", [schema.INDEX_OF["MouthLeft"]], 1.0))
    result = lbl.build_labels([session])

    assert result.weights[-1, schema.INDEX_OF["JawLeft"]] == 0.0


def test_unrelated_expressions_are_still_supervised_as_resting() -> None:
    """Exclusions must stay narrow - the 'rest of the face is relaxed' prior is still valuable."""
    session = _session("Guided", _guided_hold("TongueOut100", [TONGUE_OUT], 1.0))
    result = lbl.build_labels([session])

    for name in ("MouthSmileLeft", "CheekPuffLeft", "NoseSneerLeft"):
        dim = schema.INDEX_OF[name]
        assert result.weights[-1, dim] == lbl.W_UNCUED_DIM, f"{name} lost its resting prior"


def test_every_exclusion_entry_points_at_real_expressions() -> None:
    for cued, excluded in lbl.CO_ACTIVATION_EXCLUSIONS.items():
        assert 0 <= cued < N, f"cued index {cued} out of range"
        for dim in excluded:
            assert 0 <= dim < N, f"excluded index {dim} out of range for cue {cued}"
        assert cued not in excluded, "a cue does not need to exclude itself"


def test_combination_cues_exclude_the_union_of_their_parts() -> None:
    """A smile+jaw hold should not punish either expression's natural co-activators."""
    session = _session("Guided", _guided_hold("SmileJaw100", [SMILE_L, SMILE_R, JAW], 1.0))
    result = lbl.build_labels([session])

    # From the smile entries.
    assert result.weights[-1, schema.INDEX_OF["MouthDimpleLeft"]] == 0.0
    # From the jaw entry.
    assert result.weights[-1, schema.INDEX_OF["MouthStretchLeft"]] == 0.0
    # And all three cued dims are supervised.
    for dim in (SMILE_L, SMILE_R, JAW):
        assert result.weights[-1, dim] == lbl.W_GUIDED_HOLD


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
