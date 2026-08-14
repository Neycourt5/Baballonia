"""Tests for user-flagged correction sessions - the hard-example path.

A correction is the strongest label this project has: the user watched the tracker get something
wrong and said so, which no automatic source can match. It is also the easiest to over-read. The
button says "my mouth was closed", and that is a claim about the jaw and nothing else - the user may
have been mid-sentence, or smiling at someone. Turning that into "the whole face was neutral" would
manufacture supervision they never gave, at the highest weight in the system, on exactly the frames
where the model was already confused.

``test_only_the_corrected_dimension_is_supervised`` is the one that matters. The rest check that the
data survives the trip from C# to the trainer intact.

    python -m pytest training/tests/test_corrections.py -q
    python training/tests/test_corrections.py            (no pytest required)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babble_personal import dataset as ds  # noqa: E402
from babble_personal import evaluate, labels as lbl, schema  # noqa: E402

N = schema.EXPRESSION_COUNT
JAW = schema.INDEX_OF["JawOpen"]
TONGUE_OUT = schema.INDEX_OF["TongueOut"]
SMILE_L = schema.INDEX_OF["MouthSmileLeft"]
FPS = 30
FRAME_TICKS = lbl.TICKS_PER_SECOND // FPS


def _write_correction_session(
    root: Path,
    session_id: str = "20260814_120000_correction",
    *,
    frames: int = 30,
    stock_jaw: float = 0.7,
    personal_jaw: float = 0.55,
    smiling: float = 0.0,
    corrected_dims: list[int] | None = None,
    target: float = 0.0,
) -> Path:
    """Writes exactly what HardExampleService produces."""
    path = root / session_id
    (path / "frames").mkdir(parents=True, exist_ok=True)

    metadata = {
        "SchemaVersion": 1,
        "SessionId": session_id,
        "SessionType": "Correction",
        "StartedUtc": "2026-08-14T12:00:00Z",
        "AppVersion": "test",
        "ExpressionNames": list(schema.EXPRESSION_NAMES),
        "ExpressionSchemaSha256": schema.SCHEMA_SHA256,
        "ImageWidth": schema.IMAGE_SIZE,
        "ImageHeight": schema.IMAGE_SIZE,
        "JpegQuality": 95,
        "FrameCount": frames,
        "Notes": "User correction: My mouth was closed",
    }
    (path / "session.json").write_text(json.dumps(metadata), encoding="utf-8")

    correction = {
        "Version": 1,
        "Kind": "mouth_closed",
        "CorrectedDims": corrected_dims if corrected_dims is not None else [JAW],
        "Target": target,
        "WindowSeconds": 5.0,
        "FlaggedUtc": "2026-08-14T12:00:05Z",
        "Source": "hard_example",
        "Model": {"AdapterType": "image_residual_v1", "TrainedUtc": "2026-08-13T19:07:45Z",
                  "Blend": 1.0},
    }
    (path / "correction.json").write_text(json.dumps(correction), encoding="utf-8")

    lines = []
    for index in range(frames):
        image = np.full((schema.IMAGE_SIZE, schema.IMAGE_SIZE), (index * 7) % 200, dtype=np.uint8)
        cv2.imwrite(str(path / "frames" / f"{index:06d}.jpg"), image)

        stock = np.zeros(N, dtype=np.float32)
        stock[JAW] = stock_jaw
        stock[SMILE_L] = smiling

        personal = stock.copy()
        personal[JAW] = personal_jaw

        lines.append(json.dumps({
            "i": index,
            "t": index * FRAME_TICKS,
            "stock": [round(float(v), 6) for v in stock],
            "personal": [round(float(v), 6) for v in personal],
        }))

    (path / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_correction_session_is_recognised() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_correction_session(root)

        session = ds.discover_sessions(root)[0]

        assert session.is_correction
        assert not session.is_neutral and not session.is_speech and not session.is_guided
        assert session.corrected_dims() == [JAW]
        assert session.correction_target() == 0.0


def test_personal_values_survive_the_round_trip() -> None:
    """The personal vector is the mistake itself; it must load so it can be inspected later."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_correction_session(root, stock_jaw=0.7, personal_jaw=0.55)

        session = ds.discover_sessions(root)[0]
        frame = session.frames[0]

        assert frame.personal is not None
        assert abs(float(frame.personal[JAW]) - 0.55) < 1e-4
        assert abs(float(frame.stock[JAW]) - 0.70) < 1e-4


def test_ordinary_sessions_have_no_correction_or_personal_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "20260814_110000_neutral"
        (path / "frames").mkdir(parents=True)
        (path / "session.json").write_text(json.dumps({
            "SessionType": "Neutral",
            "ExpressionSchemaSha256": schema.SCHEMA_SHA256,
        }), encoding="utf-8")

        cv2.imwrite(str(path / "frames" / "000000.jpg"),
                    np.zeros((schema.IMAGE_SIZE, schema.IMAGE_SIZE), dtype=np.uint8))
        (path / "labels.jsonl").write_text(
            json.dumps({"i": 0, "t": 0, "stock": [0.0] * N}) + "\n", encoding="utf-8")

        session = ds.discover_sessions(root)[0]

        assert not session.is_correction
        assert session.correction is None
        assert session.frames[0].personal is None
        assert session.corrected_dims() == []


# ---------------------------------------------------------------------------
# Labelling - the part that must not over-claim
# ---------------------------------------------------------------------------


def test_only_the_corrected_dimension_is_supervised() -> None:
    """The whole design decision, in one assertion.

    The user pressed "my mouth was closed" while, in this fixture, actually smiling. Supervising the
    smile as zero would be a confident, high-weight, wrong label - and the fastest route to the dead
    face this project is trying to avoid.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_correction_session(root, frames=20, smiling=0.8)

        session = ds.discover_sessions(root)[0]
        result = lbl.build_labels([session])

        assert (result.weights[:, JAW] == lbl.W_MANUAL_CORRECTION).all()
        assert (result.targets[:, JAW] == 0.0).all()

        others = [d for d in range(N) if d != JAW]
        assert result.weights[:, others].sum() == 0.0, (
            "a correction must stay silent about every expression it did not name"
        )


def test_corrections_outrank_every_automatic_source() -> None:
    assert lbl.W_MANUAL_CORRECTION > lbl.W_NEUTRAL_SESSION
    assert lbl.W_MANUAL_CORRECTION > lbl.W_GUIDED_HOLD
    assert lbl.W_MANUAL_CORRECTION > lbl.W_SPEECH_PSEUDO


def test_a_correction_can_name_several_dimensions() -> None:
    """Nothing in the pipeline is JawOpen-specific; the dimensions travel with the data."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_correction_session(root, corrected_dims=[JAW, TONGUE_OUT])

        session = ds.discover_sessions(root)[0]
        result = lbl.build_labels([session])

        assert (result.weights[:, JAW] == lbl.W_MANUAL_CORRECTION).all()
        assert (result.weights[:, TONGUE_OUT] == lbl.W_MANUAL_CORRECTION).all()
        assert result.weights[:, SMILE_L].sum() == 0.0


def test_a_non_zero_correction_target_is_honoured() -> None:
    """"That should have been a small opening, not a wide one" is expressible."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_correction_session(root, target=0.25)

        session = ds.discover_sessions(root)[0]
        result = lbl.build_labels([session])

        assert np.allclose(result.targets[:, JAW], 0.25)


def test_a_malformed_correction_supervises_nothing_rather_than_guessing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = _write_correction_session(root)
        (path / "correction.json").write_text(json.dumps({"Version": 1, "Kind": "mouth_closed"}),
                                              encoding="utf-8")

        session = ds.discover_sessions(root)[0]
        result = lbl.build_labels([session])

        assert result.weights.sum() == 0.0, (
            "with no corrected dimensions recorded, the safe reading is 'no opinion'"
        )


def test_out_of_range_dimensions_are_ignored() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_correction_session(root, corrected_dims=[JAW, 999, -3])

        session = ds.discover_sessions(root)[0]
        result = lbl.build_labels([session])

        assert (result.weights[:, JAW] == lbl.W_MANUAL_CORRECTION).all()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_corrections_are_scored_as_known_closed_frames() -> None:
    """Corrections feed the JawOpen closed-set metrics automatically, by weight rather than by type."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_correction_session(root, frames=30, stock_jaw=0.7)

        session = ds.discover_sessions(root)[0]
        label_set = lbl.build_labels([session])
        stock = label_set.stock
        personal = np.zeros_like(stock)

        blocks = evaluate.collect_closed_blocks([session], label_set, stock, personal, JAW)
        report = evaluate.dim_report(JAW, closed_blocks=blocks)

        assert report.closed_frames == 30
        assert report.stock_fp_rate == 1.0, "the stock model was firing throughout - that is the complaint"
        assert report.personal_fp_rate == 0.0
        assert report.stock_runs.max_seconds > 0.5, "a sustained false activation, not a blip"


def test_corrections_do_not_pollute_the_speech_range_metric() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_correction_session(root)

        session = ds.discover_sessions(root)[0]
        label_set = lbl.build_labels([session])

        speech = evaluate.collect_speech_values(
            [session], label_set.stock, label_set.stock.copy(), JAW)

        assert speech is None, "a correction is not natural speech and must not set the range baseline"


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
