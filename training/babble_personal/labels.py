"""Turning recorded sessions into supervision targets and per-value confidence weights.

Supervision quality is the biggest risk in this project, so the design is deliberately conservative:
every target carries a weight, and where no trustworthy label exists the weight is zero. Combined
with the residual-shrinkage term in the loss, "no confident label" resolves to "leave the stock
prediction alone" rather than to some invented target.

Sources of supervision, strongest first:

* **Neutral sessions** - the user was resting, so every expression is genuinely ~0. High confidence.
* **Guided step-holds** - the commanded target during a settled hold, which the user was imitating
  from the avatar. High confidence, after trimming the settle-in period.
* **Guided rest between cues** - back to neutral, moderate confidence.
* **Non-cued dimensions during a cue** - "only do this one thing", low confidence, with per-cue
  exclusions so natural co-activations are not punished.
* **Speech pseudo-labels** - the stock model's own opinion on jaw/mouth only, very low weight. This
  is the one circular source, kept weak on purpose and easy to disable.

Transitions between hold levels are masked out entirely: avatar animator smoothing and human
reaction delay both make the commanded value an unreliable description of the user's face while it
is moving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import schema
from .dataset import Session, iter_frames

# ---------------------------------------------------------------------------
# Weights. Tuned to be conservative; adjust here rather than scattering magic numbers.
# ---------------------------------------------------------------------------

W_NEUTRAL_SESSION = 1.0
W_GUIDED_HOLD = 1.0
W_GUIDED_HOLD_TONGUE = 0.4  # tongue intensity cannot be modulated reliably
W_GUIDED_REST = 1.0
W_GUIDED_RAMP = 0.5
W_UNCUED_DIM = 0.25
W_SPEECH_PSEUDO = 0.3
W_MANUAL_CORRECTION = 2.0

#: Frames at the start of a hold, while the user is still moving into position, are not labelled.
HOLD_SETTLE_TRIM_SECONDS = 0.5

#: .NET DateTime ticks are 100 ns.
TICKS_PER_SECOND = 10_000_000

#: Expressions that legitimately co-activate with a cue, so "keep everything else at rest" should
#: not be enforced on them. Keyed by the primary cued dimension.
CO_ACTIVATION_EXCLUSIONS: dict[int, tuple[int, ...]] = {
    schema.INDEX_OF["JawOpen"]: (
        schema.INDEX_OF["MouthClose"],
        schema.INDEX_OF["MouthStretchLeft"],
        schema.INDEX_OF["MouthStretchRight"],
    ),
    schema.INDEX_OF["MouthSmileLeft"]: (
        schema.INDEX_OF["MouthSmileRight"],
        schema.INDEX_OF["MouthDimpleLeft"],
        schema.INDEX_OF["MouthDimpleRight"],
        schema.INDEX_OF["CheekPuffLeft"],
        schema.INDEX_OF["CheekPuffRight"],
    ),
    schema.INDEX_OF["MouthSmileRight"]: (
        schema.INDEX_OF["MouthSmileLeft"],
        schema.INDEX_OF["MouthDimpleLeft"],
        schema.INDEX_OF["MouthDimpleRight"],
        schema.INDEX_OF["CheekPuffLeft"],
        schema.INDEX_OF["CheekPuffRight"],
    ),
    schema.INDEX_OF["MouthPucker"]: (
        schema.INDEX_OF["MouthFunnel"],
        schema.INDEX_OF["MouthRollUpper"],
        schema.INDEX_OF["MouthRollLower"],
    ),
    schema.INDEX_OF["MouthFunnel"]: (
        schema.INDEX_OF["MouthPucker"],
        schema.INDEX_OF["JawOpen"],
    ),
}


@dataclass
class LabelSet:
    """Targets and weights aligned with the frame ordering of the sessions they came from."""

    targets: np.ndarray  # float32 [N, 45]
    weights: np.ndarray  # float32 [N, 45], 0 means "no opinion"
    stock: np.ndarray  # float32 [N, 45]

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def coverage(self) -> float:
        """Fraction of (frame, expression) cells carrying any supervision. Useful sanity check."""
        if self.weights.size == 0:
            return 0.0
        return float((self.weights > 0).mean())

    def describe(self) -> str:
        per_dim = (self.weights > 0).mean(axis=0)
        strong = int((per_dim > 0.05).sum())
        return (
            f"  frames: {len(self)}\n"
            f"  supervised cells: {self.coverage() * 100:.1f}%\n"
            f"  expressions with >5% coverage: {strong}/{schema.EXPRESSION_COUNT}"
        )


def estimate_cue_lag_seconds(
    stock: np.ndarray,
    cue_signal: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_lag_seconds: float = 0.7,
) -> tuple[float, float]:
    """Estimate how far the user's response trails the commanded cue.

    The avatar changes, the user reacts a few hundred milliseconds later, and the avatar's own
    animator smoothing adds more. Both show up as one roughly constant delay, which we recover by
    correlating the commanded signal against the stock model's reading of the user's face.

    Returns ``(lag_seconds, correlation)``. A low correlation means the cue was not followed, and
    the caller should drop that repetition rather than train on it.
    """
    if len(stock) < 8 or len(cue_signal) < 8:
        return 0.0, 0.0

    duration = (timestamps[-1] - timestamps[0]) / TICKS_PER_SECOND
    if duration <= 0:
        return 0.0, 0.0

    fps = len(timestamps) / duration
    max_lag_frames = max(1, int(round(max_lag_seconds * fps)))

    def zscore(x: np.ndarray) -> np.ndarray:
        sd = x.std()
        return (x - x.mean()) / sd if sd > 1e-8 else np.zeros_like(x)

    a, b = zscore(cue_signal.astype(np.float64)), zscore(stock.astype(np.float64))
    if not a.any() or not b.any():
        return 0.0, 0.0

    best_lag, best_corr = 0, -1.0
    for lag in range(0, min(max_lag_frames, len(a) - 4) + 1):
        # Compare the cue at time t against the face at time t+lag.
        corr = float(np.mean(a[: len(a) - lag] * b[lag:]))
        if corr > best_corr:
            best_lag, best_corr = lag, corr

    return best_lag / fps, best_corr


def build_labels(
    sessions: Sequence[Session],
    *,
    use_speech_pseudo_labels: bool = True,
    hold_settle_trim: float = HOLD_SETTLE_TRIM_SECONDS,
) -> LabelSet:
    """Build targets and weights for every frame in ``sessions``."""
    n_dims = schema.EXPRESSION_COUNT
    frames = list(iter_frames(sessions))
    n = len(frames)

    targets = np.zeros((n, n_dims), dtype=np.float32)
    weights = np.zeros((n, n_dims), dtype=np.float32)
    stock = np.zeros((n, n_dims), dtype=np.float32)

    # Hold phases need to know when they started, to trim the settle-in period.
    hold_started_at: dict[tuple[str, str, int], int] = {}

    for row, (session, frame) in enumerate(frames):
        stock[row] = frame.stock

        if session.is_neutral:
            # Whole face at rest: the strongest and cheapest supervision available.
            weights[row, :] = W_NEUTRAL_SESSION
            continue

        if session.is_speech:
            if use_speech_pseudo_labels:
                # Weak, and deliberately so: this is the one circular source. It exists to keep the
                # adapter from drifting on dimensions no cue covers, not to teach anything new.
                for dim in schema.SPEECH_SUPERVISED_DIMS:
                    targets[row, dim] = frame.stock[dim]
                    weights[row, dim] = W_SPEECH_PSEUDO
            continue

        cue = frame.cue
        if cue is None:
            continue

        phase = str(cue.get("phase", "")).lower()
        cued_dims = [int(d) for d in cue.get("dims", [])]
        target_vector = cue.get("target")

        if target_vector is None or len(target_vector) != n_dims:
            continue

        commanded = np.asarray(target_vector, dtype=np.float32)

        # Transitions are unusable: the avatar is still animating and the user is still moving.
        if phase in ("transition", "ramp", "ramp_up", "ramp_down", "prep"):
            if phase in ("ramp", "ramp_up", "ramp_down"):
                for dim in cued_dims:
                    targets[row, dim] = commanded[dim]
                    weights[row, dim] = W_GUIDED_RAMP
            continue

        if phase in ("hold", "rest"):
            key = (session.session_id, str(cue.get("id", "")), int(cue.get("rep", 0)))
            start = hold_started_at.setdefault(key, frame.timestamp_ticks)
            elapsed = (frame.timestamp_ticks - start) / TICKS_PER_SECOND

            # Still moving into the expression: record nothing rather than something wrong.
            if elapsed < hold_settle_trim:
                continue

            if phase == "rest":
                weights[row, :] = W_GUIDED_REST
                continue

            for dim in cued_dims:
                is_tongue = dim in schema.TONGUE_DIMS
                targets[row, dim] = commanded[dim]
                weights[row, dim] = W_GUIDED_HOLD_TONGUE if is_tongue else W_GUIDED_HOLD

            # "Keep the rest of your face relaxed", minus expressions that legitimately come along
            # for the ride with this cue.
            excluded = set(cued_dims)
            for dim in cued_dims:
                excluded.update(CO_ACTIVATION_EXCLUSIONS.get(dim, ()))

            for dim in range(n_dims):
                if dim not in excluded:
                    targets[row, dim] = 0.0
                    weights[row, dim] = W_UNCUED_DIM

    return LabelSet(targets=targets, weights=weights, stock=stock)


def apply_corrections(labels: LabelSet, corrections: Sequence[dict], frame_offset: int = 0) -> LabelSet:
    """Overlay manual corrections from the review tool at high weight.

    Each correction is ``{"frame": int, "dim": int, "value": float}``. These are the user telling us
    directly that the model is wrong on a specific frame, so they outrank every automatic source.
    """
    for correction in corrections:
        row = int(correction["frame"]) - frame_offset
        dim = int(correction["dim"])

        if not (0 <= row < len(labels)) or not (0 <= dim < schema.EXPRESSION_COUNT):
            continue

        labels.targets[row, dim] = float(correction["value"])
        labels.weights[row, dim] = W_MANUAL_CORRECTION

    return labels
