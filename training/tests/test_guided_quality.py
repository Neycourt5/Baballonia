"""Per-attempt Guided quality gating.

The gate must catch a missed repetition without throwing away its session, and must remain humble
about the stock model: an extremely weak signal is suppressed automatically only when a sibling
attempt proves that exact cue is observable. Explicit headset judgments always win.

    python -m pytest training/tests/test_guided_quality.py -q
    python training/tests/test_guided_quality.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babble_personal import labels as lbl, schema  # noqa: E402
from babble_personal.dataset import FrameRecord, Session  # noqa: E402

N = schema.EXPRESSION_COUNT
SMILE_L = schema.INDEX_OF["MouthSmileLeft"]
SMILE_R = schema.INDEX_OF["MouthSmileRight"]
JAW = schema.INDEX_OF["JawOpen"]
FPS = 30
FRAME_TICKS = lbl.TICKS_PER_SECOND // FPS


class _InMemorySidecarPath:
    """Path-shaped sidecar seam; avoids making unit-test correctness depend on temp permissions."""

    def __init__(self, payload: dict):
        self.payload = payload

    def __truediv__(self, name: str):
        assert name == lbl.GUIDED_QUALITY_OVERRIDES_FILE
        return self

    def exists(self) -> bool:
        return True

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8-sig"
        return json.dumps(self.payload)

    def __str__(self) -> str:
        return "<in-memory-guided-quality-sidecar>"


def _attempt(
    cue_id: str,
    dims: tuple[int, ...],
    *,
    rep: int,
    attempt: int = 0,
    follows: bool,
    level: float = 1.0,
) -> list[tuple[dict, float]]:
    """One realistic prep/transition/hold/transition/rest attempt and observed primary value."""
    phases: list[tuple[str, float]] = []
    phases += [("prep", 0.0)] * 6
    phases += [("transition", float(v)) for v in np.linspace(0.0, level, 15)]
    phases += [("hold", level)] * 60
    phases += [("transition", float(v)) for v in np.linspace(level, 0.0, 15)]
    phases += [("rest", 0.0)] * 30

    result = []
    for phase, commanded in phases:
        target = [0.0] * N
        for dim in dims:
            target[dim] = commanded
        cue = {
            "id": cue_id,
            "phase": phase,
            "dims": list(dims),
            "target": target,
            "level": level,
            "rep": rep,
            "attempt": attempt,
            "source": "avatar",
        }
        result.append((cue, commanded if follows else 0.0))
    return result


def _session(attempts: list[list[tuple[dict, float]]], path=Path("quality-session")) -> Session:
    frames: list[FrameRecord] = []
    for cue, observed in (item for attempt in attempts for item in attempt):
        stock = np.zeros(N, dtype=np.float32)
        primary = int(cue["dims"][0])
        stock[primary] = observed
        frames.append(FrameRecord(
            index=len(frames),
            timestamp_ticks=len(frames) * FRAME_TICKS,
            stock=stock,
            image_path=Path(f"{len(frames):06d}.jpg"),
            cue=cue,
        ))
    return Session("guided-quality", path, {"SessionType": "Guided"}, frames)


def _entry(reports: list[dict], cue: str, rep: int, attempt: int = 0) -> dict:
    return next(r for r in reports
                if r["cue"] == cue and r["rep"] == rep and r["attempt"] == attempt)


def _settled_row(session: Session, cue: str, rep: int, attempt: int, phase: str) -> int:
    rows = [i for i, frame in enumerate(session.frames)
            if frame.cue
            and frame.cue["id"] == cue
            and frame.cue["rep"] == rep
            and frame.cue.get("attempt", 0) == attempt
            and frame.cue["phase"] == phase]
    # Derived from the trim rather than hardcoded: picking a fixed frame index silently starts
    # sampling *inside* the discarded window the moment the trim grows.
    settled = math.ceil(lbl.HOLD_SETTLE_TRIM_SECONDS * FPS) + 1
    assert settled < len(rows), f"{cue} rep {rep} {phase} is shorter than the settle trim"
    return rows[settled]


def test_one_missed_smile_rep_is_suppressed_without_losing_other_evidence() -> None:
    session = _session([
        _attempt("Smile50", (SMILE_L, SMILE_R), rep=0, follows=False, level=0.5),
        _attempt("Smile50", (SMILE_L, SMILE_R), rep=1, follows=True, level=0.5),
        _attempt("JawOpen100", (JAW,), rep=0, follows=True),
    ])

    reports = lbl.guided_attempt_quality([session])
    missed = _entry(reports, "Smile50", 0)
    good_smile = _entry(reports, "Smile50", 1)
    good_jaw = _entry(reports, "JawOpen100", 0)

    assert missed["correlation"] <= lbl.GUIDED_SUPPRESS_CORRELATION
    assert missed["status"] == "suppressed" and missed["weight_scale"] == 0.0
    assert good_smile["status"] == "good" and good_smile["weight_scale"] == 1.0
    assert good_jaw["status"] == "good" and good_jaw["weight_scale"] == 1.0

    labels = lbl.build_labels([session], guided_quality=reports)
    missed_hold = _settled_row(session, "Smile50", 0, 0, "hold")
    good_smile_hold = _settled_row(session, "Smile50", 1, 0, "hold")
    good_jaw_hold = _settled_row(session, "JawOpen100", 0, 0, "hold")
    missed_rest = _settled_row(session, "Smile50", 0, 0, "rest")

    assert labels.weights[missed_hold].sum() == 0.0, "only the missed hold must be suppressed"
    assert labels.weights[good_smile_hold, SMILE_L] == lbl.W_GUIDED_HOLD
    assert labels.weights[good_smile_hold, SMILE_R] == lbl.W_GUIDED_HOLD
    assert labels.weights[good_jaw_hold, JAW] == lbl.W_GUIDED_HOLD
    assert np.all(labels.weights[missed_rest] == lbl.W_GUIDED_REST), \
        "quality gating must not erase valid settled-rest supervision"

    # Prep and transitions are never made trainable by a quality decision.
    for row, frame in enumerate(session.frames):
        if frame.cue and frame.cue["phase"] in ("prep", "transition"):
            assert labels.weights[row].sum() == 0.0


def test_extremely_weak_attempt_without_observable_peer_is_downweighted_not_discarded() -> None:
    session = _session([
        _attempt("Smile50", (SMILE_L, SMILE_R), rep=0, follows=False, level=0.5),
    ])
    reports = lbl.guided_attempt_quality([session])
    report = reports[0]

    assert report["status"] == "weak"
    assert report["reason"] == "extremely-weak-but-no-observable-peer"
    assert report["weight_scale"] == lbl.GUIDED_WEAK_WEIGHT_SCALE

    labels = lbl.build_labels([session], guided_quality=reports)
    hold = _settled_row(session, "Smile50", 0, 0, "hold")
    assert labels.weights[hold, SMILE_L] == \
           lbl.W_GUIDED_HOLD * lbl.GUIDED_WEAK_WEIGHT_SCALE


def test_explicit_attempt_overrides_win_and_are_deterministic() -> None:
    sidecar = _InMemorySidecarPath({
        "attempts": [
            {"cue": "Smile50", "rep": 0, "attempt": 0, "valid": False,
             "reason": "retried-in-vr"},
            {"cue": "Smile50", "rep": 0, "attempt": 1, "valid": True,
             "reason": "accepted-in-vr"},
        ]
    })
    session = _session([
        # The signal looks good but the user rejected this pre-retry attempt.
        _attempt("Smile50", (SMILE_L, SMILE_R), rep=0, attempt=0, follows=True, level=0.5),
        # The signal looks absent but the user explicitly accepted the retry; stock blindness must
        # not overrule the person who actually saw/performed it.
        _attempt("Smile50", (SMILE_L, SMILE_R), rep=0, attempt=1, follows=False, level=0.5),
    ], path=sidecar)

    first = lbl.guided_attempt_quality([session])
    second = lbl.guided_attempt_quality([session])
    assert first == second, "quality decisions and report ordering must be deterministic"

    rejected = _entry(first, "Smile50", 0, 0)
    accepted = _entry(first, "Smile50", 0, 1)
    assert rejected["status"] == "suppressed" and rejected["decision_source"] == \
           lbl.GUIDED_QUALITY_OVERRIDES_FILE
    assert accepted["status"] == "good" and accepted["decision_source"] == \
           lbl.GUIDED_QUALITY_OVERRIDES_FILE
    assert rejected["override_reason"] == "retried-in-vr"
    assert accepted["override_reason"] == "accepted-in-vr"

    labels = lbl.build_labels([session], guided_quality=first)
    rejected_hold = _settled_row(session, "Smile50", 0, 0, "hold")
    accepted_hold = _settled_row(session, "Smile50", 0, 1, "hold")
    assert labels.weights[rejected_hold].sum() == 0.0
    assert labels.weights[accepted_hold, SMILE_L] == lbl.W_GUIDED_HOLD

    artifact = lbl.guided_quality_artifact(first, train_session_ids=[session.session_id])
    assert artifact["summary"] == {"good": 1, "weak": 0, "suppressed": 1}
    assert all(item["split"] == "train" for item in artifact["attempts"])
    json.dumps(artifact)  # The immutable run artifact must be directly serializable.


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
