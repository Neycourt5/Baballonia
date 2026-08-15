"""Tests for cue-segment trimming and per-expression weight boosting.

The bug these were written against: the settle-in trim was keyed on ``(session, cue id, repetition)``
without the phase, and a ``rest`` phase normally carries the same cue id and repetition as the
``hold`` it follows. So the rest looked like a continuation of the hold, its elapsed time was already
past the trim window on its very first frame, and the frames where the user is still relaxing out of
the expression were labelled "face at rest" at full confidence. Those are exactly the frames where
the face is *not* at rest, which is the worst possible thing to teach a model whose entire job is
telling resting from moving.

    python -m pytest training/tests/test_labels_trim.py -q
    python training/tests/test_labels_trim.py            (no pytest required)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babble_personal import labels as lbl, schema  # noqa: E402
from babble_personal.dataset import FrameRecord, Session  # noqa: E402

N = schema.EXPRESSION_COUNT
JAW = schema.INDEX_OF["JawOpen"]
TONGUE_OUT = schema.INDEX_OF["TongueOut"]
TICKS = lbl.TICKS_PER_SECOND
FPS = 30

#: Frames per second of recording, as ticks. Matches the recorder's 30 fps cap.
FRAME_TICKS = TICKS // FPS


def _cue(cue_id: str, phase: str, level: float, rep: int = 0, dims=(JAW,)) -> dict:
    target = [0.0] * N
    for dim in dims:
        target[dim] = level
    return {
        "id": cue_id,
        "phase": phase,
        "dims": list(dims),
        "target": target,
        "level": level,
        "rep": rep,
        "source": "avatar",
    }


def _session(session_type: str, cues: list[dict | None], *, session_id: str = "s") -> Session:
    """An in-memory session. build_labels never touches images, so none are written."""
    frames = [
        FrameRecord(
            index=i,
            timestamp_ticks=i * FRAME_TICKS,
            stock=np.zeros(N, dtype=np.float32),
            image_path=Path(f"{i:06d}.jpg"),
            cue=cue,
        )
        for i, cue in enumerate(cues)
    ]
    return Session(
        session_id=session_id,
        path=Path(session_id),
        metadata={"SessionType": session_type},
        frames=frames,
    )


def _hold_then_rest(hold_seconds: float = 2.0, rest_seconds: float = 2.0) -> Session:
    """The exact shape the cue engine emits: a hold followed by a rest under one cue id and rep."""
    hold = [_cue("JawOpen100", "hold", 1.0)] * int(hold_seconds * FPS)
    rest = [_cue("JawOpen100", "rest", 1.0)] * int(rest_seconds * FPS)
    return _session("Guided", hold + rest)


def test_rest_after_hold_is_trimmed() -> None:
    """The regression test. Fails against the id+rep keying, passes with per-segment tracking."""
    session = _hold_then_rest()
    hold_frames = 2 * FPS
    trim_frames = int(lbl.HOLD_SETTLE_TRIM_SECONDS * FPS)

    result = lbl.build_labels([session])

    # The first half-second of the rest is the user relaxing out of a fully open jaw. Labelling that
    # as "at rest" would teach the model that a half-open mouth is neutral.
    for offset in range(trim_frames):
        row = hold_frames + offset
        assert result.weights[row].sum() == 0.0, (
            f"rest frame {offset} ({offset / FPS:.2f}s into the rest) was labelled; "
            "the settle-in trim did not restart at the phase boundary"
        )

    # Once settled, the rest supervises the whole face as genuinely neutral.
    settled = hold_frames + trim_frames + 1
    assert np.allclose(result.weights[settled], lbl.W_GUIDED_REST)
    assert np.allclose(result.targets[settled], 0.0)


def test_hold_itself_is_still_trimmed() -> None:
    """The original behaviour must survive the fix."""
    session = _hold_then_rest()
    trim_frames = int(lbl.HOLD_SETTLE_TRIM_SECONDS * FPS)

    result = lbl.build_labels([session])

    for row in range(trim_frames):
        assert result.weights[row].sum() == 0.0, f"hold frame {row} should be trimmed"

    settled = trim_frames + 1
    assert result.weights[settled, JAW] == lbl.W_GUIDED_HOLD
    assert result.targets[settled, JAW] == 1.0


def test_two_levels_under_one_id_and_rep_each_get_their_own_trim() -> None:
    """Levels are part of segment identity, so a 0.5 hold following a 1.0 hold is not a continuation."""
    first = [_cue("JawOpenSweep", "hold", 1.0)] * (2 * FPS)
    second = [_cue("JawOpenSweep", "hold", 0.5)] * (2 * FPS)
    session = _session("Guided", first + second)
    trim_frames = int(lbl.HOLD_SETTLE_TRIM_SECONDS * FPS)

    result = lbl.build_labels([session])

    boundary = len(first)
    for offset in range(trim_frames):
        assert result.weights[boundary + offset].sum() == 0.0, (
            "the second level inherited the first level's start time"
        )

    settled = boundary + trim_frames + 1
    assert result.targets[settled, JAW] == 0.5


def test_repetitions_restart_the_trim() -> None:
    """Rep 1 of a cue is a fresh movement, not a continuation of rep 0."""
    rep0 = [_cue("JawOpen100", "hold", 1.0, rep=0)] * (2 * FPS)
    rep1 = [_cue("JawOpen100", "hold", 1.0, rep=1)] * (2 * FPS)
    session = _session("Guided", rep0 + rep1)
    trim_frames = int(lbl.HOLD_SETTLE_TRIM_SECONDS * FPS)

    result = lbl.build_labels([session])

    for offset in range(trim_frames):
        assert result.weights[len(rep0) + offset].sum() == 0.0


def test_session_boundary_restarts_the_trim() -> None:
    """Two sessions whose cue metadata happens to match must not run into each other."""
    a = _session("Guided", [_cue("JawOpen100", "hold", 1.0)] * (2 * FPS), session_id="a")
    b = _session("Guided", [_cue("JawOpen100", "hold", 1.0)] * (2 * FPS), session_id="b")
    trim_frames = int(lbl.HOLD_SETTLE_TRIM_SECONDS * FPS)

    result = lbl.build_labels([a, b])

    for offset in range(trim_frames):
        assert result.weights[len(a) + offset].sum() == 0.0, (
            "session b's hold inherited session a's start time"
        )


def test_transitions_between_hold_and_rest_stay_unlabelled() -> None:
    hold = [_cue("JawOpen100", "hold", 1.0)] * (2 * FPS)
    transition = [_cue("JawOpen100", "transition", 1.0)] * FPS
    rest = [_cue("JawOpen100", "rest", 1.0)] * (2 * FPS)
    session = _session("Guided", hold + transition + rest)

    result = lbl.build_labels([session])

    for offset in range(len(transition)):
        row = len(hold) + offset
        assert result.weights[row].sum() == 0.0, "transition frames must carry no supervision"


def test_prep_countdown_frames_stay_unlabelled() -> None:
    """The countdown before an attempt teaches nothing and must supervise nothing.

    Guided capture now announces the expression and counts down before commanding it. Those frames
    show a resting face while the headset says a smile is coming, so labelling them either way -
    as the expression or as rest - would be wrong.
    """
    prep = [_cue("Smile100", "prep", 1.0)] * (3 * FPS)
    transition = [_cue("Smile100", "transition", 1.0)] * FPS
    hold = [_cue("Smile100", "hold", 1.0)] * (3 * FPS)
    session = _session("Guided", prep + transition + hold)

    result = lbl.build_labels([session])

    for row in range(len(prep)):
        assert result.weights[row].sum() == 0.0, "prep frames must carry no supervision"

    # The hold that follows is still supervised, so this is not just "everything is zero".
    assert result.weights[len(prep) + len(transition) + len(hold) - 1].sum() > 0.0


def test_settle_trim_leaves_a_usable_hold() -> None:
    """The trim has to stay well inside the hold, or guided capture records nothing usable."""
    assert lbl.HOLD_SETTLE_TRIM_SECONDS < 3.0
    assert 3.0 - lbl.HOLD_SETTLE_TRIM_SECONDS >= 2.0


def test_dim_boost_scales_weights_without_touching_targets() -> None:
    session = _session("Neutral", [None] * 60)

    plain = lbl.build_labels([session])
    boosted = lbl.build_labels([session], dim_boost={JAW: 2.0})

    assert np.allclose(boosted.weights[:, JAW], plain.weights[:, JAW] * 2.0)
    assert np.allclose(boosted.targets, plain.targets), "boosting must never move a target"

    # Every other expression is untouched.
    others = [d for d in range(N) if d != JAW]
    assert np.allclose(boosted.weights[:, others], plain.weights[:, others])


def test_dim_boost_leaves_unsupervised_cells_unsupervised() -> None:
    """Multiplying a zero weight keeps it zero - a boost cannot invent supervision."""
    session = _session("Speech", [None] * 30)

    boosted = lbl.build_labels([session], dim_boost={TONGUE_OUT: 5.0})

    # Speech supervises jaw and mouth only; the tongue has no opinion and must keep having none.
    assert boosted.weights[:, TONGUE_OUT].sum() == 0.0


def test_parse_dim_boost() -> None:
    assert lbl.parse_dim_boost("JawOpen=2.0,TongueOut=1.5") == {JAW: 2.0, TONGUE_OUT: 1.5}
    assert lbl.parse_dim_boost("") == {}
    assert lbl.parse_dim_boost(None) == {}
    assert lbl.parse_dim_boost(" JawOpen = 3 ") == {JAW: 3.0}

    for bad in ("NotAnExpression=2", "JawOpen=notanumber", "JawOpen=-1"):
        try:
            lbl.parse_dim_boost(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_parse_dims() -> None:
    assert lbl.parse_dims("JawOpen,TongueOut") == (JAW, TONGUE_OUT)
    assert lbl.parse_dims(None) == ()

    try:
        lbl.parse_dims("Nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown expression name")


# ---------------------------------------------------------------------------
# Cue-lag diagnostic - the experiment the guided design depends on
# ---------------------------------------------------------------------------


def _guided_session_with_response(delay_frames: int, *, follows: bool = True) -> Session:
    """A guided hold where the stock model's reading trails the command by `delay_frames`."""
    commanded = ([0.0] * (1 * FPS)) + ([1.0] * (2 * FPS)) + ([0.0] * (1 * FPS))
    cues = []
    for value in commanded:
        phase = "hold" if value > 0 else "rest"
        cues.append(_cue("JawOpen100", phase, value))

    session = _session("Guided", cues)

    # Shift the observed response later in time, and optionally decouple it entirely.
    rng = np.random.default_rng(0)
    for i, frame in enumerate(session.frames):
        source = i - delay_frames
        observed = commanded[source] if 0 <= source < len(commanded) else 0.0
        if not follows:
            observed = float(rng.random())
        frame.stock[JAW] = observed + float(rng.normal(0, 0.01))

    return session


def test_cue_lag_is_recovered_when_the_user_follows() -> None:
    session = _guided_session_with_response(delay_frames=6)  # 200 ms at 30 fps

    reports = lbl.cue_lag_report([session])

    assert reports, "a guided session should produce a lag report"
    best = max(reports, key=lambda r: r["correlation"])
    assert best["correlation"] > 0.6, f"correlation {best['correlation']:.2f} - labels look real"
    assert 0.1 < best["lag_seconds"] < 0.35, f"recovered lag {best['lag_seconds']:.2f}s"


def test_an_unfollowed_cue_is_flagged() -> None:
    """The failure this exists to catch: labels that describe nothing the face did."""
    session = _guided_session_with_response(delay_frames=0, follows=False)

    reports = lbl.cue_lag_report([session])
    text = lbl.format_cue_lag_report(reports)

    assert reports
    assert max(r["correlation"] for r in reports) < 0.5
    assert "weak" in text


def test_non_guided_sessions_produce_no_lag_report() -> None:
    assert lbl.cue_lag_report([_session("Neutral", [None] * 60)]) == []
    assert lbl.cue_lag_report([_session("Speech", [None] * 60)]) == []


def test_lag_report_formats_empty_input_as_empty_text() -> None:
    assert lbl.format_cue_lag_report([]) == ""


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
