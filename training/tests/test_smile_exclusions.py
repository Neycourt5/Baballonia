"""Tests for the smile cue's co-activation exclusions (fix A1).

The bug these were written against: an ordinary ``Smile`` hold supervised every tooth- and
lip-opening dimension toward **zero** at ``W_UNCUED_DIM``, because none of them appeared in
``CO_ACTIVATION_EXCLUSIONS`` for the smile dimensions. A person smiling at full intensity shows
teeth, so those frames taught "this face, and the lips stayed together" on an image visually almost
identical to the confirmed toothy-smile cue, which teaches the opposite. ``Smile`` is in every core
session at two levels; the toothy cue is one opt-in binary routine. The plain-smile frames therefore
outnumbered it and the network's least-loss answer was to regress the toothy prediction back toward
the closed-lip label -- smile fires, teeth do not, which is exactly the reported symptom.

The fix is *not* to teach teeth on during a plain smile. Excluding a dimension removes it from the
loss for those frames entirely: the cue has **no opinion**, and the dedicated toothy cue remains the
only thing that teaches the tooth-revealing pose.

    python -m pytest training/tests/test_smile_exclusions.py -q
    python training/tests/test_smile_exclusions.py            (no pytest required)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babble_personal import labels as lbl, schema  # noqa: E402
from babble_personal.dataset import FrameRecord, Session  # noqa: E402

N = schema.EXPRESSION_COUNT
I = schema.INDEX_OF
FPS = 30
FRAME_TICKS = lbl.TICKS_PER_SECOND // FPS

SMILE_L, SMILE_R = I["MouthSmileLeft"], I["MouthSmileRight"]

#: The dimensions a genuine smile may legitimately move, which A1 stopped being trained to zero.
TEETH_DIMS = (
    I["MouthUpperUpLeft"], I["MouthUpperUpRight"],
    I["MouthLowerDownLeft"], I["MouthLowerDownRight"],
    I["MouthStretchLeft"], I["MouthStretchRight"],
    I["JawOpen"], I["MouthClose"],
)

#: Already protected before A1; must not be lost.
ORIGINAL_DIMS = (
    I["MouthDimpleLeft"], I["MouthDimpleRight"],
    I["CheekPuffLeft"], I["CheekPuffRight"],
)

#: Nothing to do with smiling. These must still be held at rest, or A1 has over-reached.
UNRELATED_DIMS = (
    I["TongueOut"], I["NoseSneerLeft"], I["CheekSuckLeft"],
    I["MouthPucker"], I["MouthFrownLeft"], I["JawLeft"],
)


def _smile_session(dims=(SMILE_L, SMILE_R), level: float = 1.0) -> Session:
    """A settled smile hold, long enough that the settle-in trim leaves real frames behind."""
    cue = {
        "id": "Smile100",
        "phase": "hold",
        "dims": list(dims),
        "target": [level if d in dims else 0.0 for d in range(N)],
        "level": level,
        "rep": 0,
        "source": "avatar",
    }
    frames = [
        FrameRecord(
            index=i,
            timestamp_ticks=i * FRAME_TICKS,
            stock=np.zeros(N, dtype=np.float32),
            image_path=Path(f"{i:06d}.jpg"),
            cue=cue,
        )
        for i in range(3 * FPS)
    ]
    return Session(session_id="s", path=Path("s"), metadata={"SessionType": "Guided"}, frames=frames)


def _settled_row(result) -> int:
    """The first frame past the settle-in trim, i.e. one that is actually labelled."""
    for row in range(result.weights.shape[0]):
        if result.weights[row].sum() > 0.0:
            return row
    raise AssertionError("no frame was labelled at all")


def test_smile_still_supervises_the_smile_dimensions() -> None:
    """A1 must not weaken what the cue is actually for."""
    result = lbl.build_labels([_smile_session()])
    row = _settled_row(result)

    for dim in (SMILE_L, SMILE_R):
        assert result.weights[row, dim] > 0.0, "the cued smile dimension lost its supervision"
        assert result.targets[row, dim] == 1.0, "the cued smile dimension lost its target"


def test_teeth_dimensions_are_no_longer_pushed_to_zero() -> None:
    """The regression test. Fails before A1, passes after."""
    result = lbl.build_labels([_smile_session()])
    row = _settled_row(result)

    for dim in TEETH_DIMS:
        assert result.weights[row, dim] == 0.0, (
            f"{schema.EXPRESSION_NAMES[dim]} is still being trained toward zero on a smile hold; "
            "that is the label conflict that averaged the toothy smile away"
        )


def test_exclusion_means_no_opinion_not_teeth_on() -> None:
    """A1 must not teach teeth ON during an ordinary smile -- only remove the false 'off'."""
    result = lbl.build_labels([_smile_session()])
    row = _settled_row(result)

    for dim in TEETH_DIMS:
        assert result.weights[row, dim] == 0.0, "a zero weight is what 'no opinion' means"
        # With zero weight the target is not trained on at all, whatever it holds.
        assert result.targets[row, dim] == 0.0, (
            "an excluded dimension must not be given a positive target -- that would teach the "
            "plain smile to show teeth, which is the dedicated toothy cue's job"
        )


def test_originally_protected_dimensions_are_still_protected() -> None:
    """Dimple and cheek exclusions predate A1 and must survive it."""
    result = lbl.build_labels([_smile_session()])
    row = _settled_row(result)

    for dim in ORIGINAL_DIMS:
        assert result.weights[row, dim] == 0.0, (
            f"{schema.EXPRESSION_NAMES[dim]} lost the protection it had before A1"
        )


def test_unrelated_dimensions_are_still_held_at_rest() -> None:
    """The guard against over-reach: A1 widened the exclusion, it did not remove it."""
    result = lbl.build_labels([_smile_session()])
    row = _settled_row(result)

    for dim in UNRELATED_DIMS:
        assert result.weights[row, dim] == lbl.W_UNCUED_DIM, (
            f"{schema.EXPRESSION_NAMES[dim]} should still be supervised toward rest during a smile"
        )
        assert result.targets[row, dim] == 0.0


def test_left_and_right_smile_are_symmetrical() -> None:
    """A one-sided smile must protect the same dimensions as the other side."""
    left = set(lbl.CO_ACTIVATION_EXCLUSIONS[SMILE_L]) - {SMILE_R}
    right = set(lbl.CO_ACTIVATION_EXCLUSIONS[SMILE_R]) - {SMILE_L}

    assert left == right, "the two smile exclusion entries have drifted apart"

    for dim in TEETH_DIMS:
        assert dim in left and dim in right


def test_a_single_sided_smile_cue_also_protects_the_teeth_dimensions() -> None:
    """Exclusions are keyed per cued dimension, so one side alone must still work."""
    result = lbl.build_labels([_smile_session(dims=(SMILE_L,))])
    row = _settled_row(result)

    for dim in TEETH_DIMS:
        assert result.weights[row, dim] == 0.0


def test_the_schema_is_untouched() -> None:
    """A1 is a label-weighting change. Any schema movement invalidates every trained model."""
    assert schema.EXPRESSION_COUNT == 45
    assert schema.EXPRESSION_NAMES[SMILE_L] == "MouthSmileLeft"
    assert schema.EXPRESSION_NAMES[I["JawOpen"]] == "JawOpen"


def test_other_cues_are_unaffected() -> None:
    """A1 touched only the two smile entries."""
    jaw = set(lbl.CO_ACTIVATION_EXCLUSIONS[I["JawOpen"]])
    assert jaw == {I["MouthClose"], I["MouthStretchLeft"], I["MouthStretchRight"]}

    tongue = set(lbl.CO_ACTIVATION_EXCLUSIONS[I["TongueOut"]])
    assert I["JawOpen"] in tongue, "the tongue/jaw rule is unrelated to A1 and must survive it"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all smile-exclusion tests passed")
