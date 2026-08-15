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

import json
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
#: Mirrored in C# as ``GuidedCaptureRoutine.HoldSettleTrimSeconds`` so the headset can show the user
#: that the trusted part of the hold has not started yet; the two must be changed together.
HOLD_SETTLE_TRIM_SECONDS = 0.75

#: A repetition at or above this correlation is trusted at the normal guided-hold weight. This is
#: the threshold the original cue-lag diagnostic already described as evidence that the commanded
#: value follows the face.
GUIDED_GOOD_CORRELATION = 0.60

#: Attempts below the good threshold still carry useful evidence, but at a deliberately smaller
#: weight. A stock-model miss can make a real expression look weak, so weak is not the same as bad.
GUIDED_WEAK_WEIGHT_SCALE = 0.25

#: Automatic suppression is reserved for an almost completely unrelated response, and only when a
#: sibling attempt proves the stock model can see this exact cue in this exact session.
GUIDED_SUPPRESS_CORRELATION = 0.10

#: Optional, immutable annotation emitted by the headset workflow when the user explicitly skips,
#: retries, accepts, or rejects an attempt. Explicit judgment outranks the automatic diagnostic.
GUIDED_QUALITY_OVERRIDES_FILE = "guided_quality_overrides.json"

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
    # Pulling the lip corners down also drags the lower lip and often the chin.
    schema.INDEX_OF["MouthFrownLeft"]: (
        schema.INDEX_OF["MouthFrownRight"],
        schema.INDEX_OF["MouthLowerDownLeft"],
        schema.INDEX_OF["MouthLowerDownRight"],
        schema.INDEX_OF["MouthShrugLower"],
    ),
    schema.INDEX_OF["MouthFrownRight"]: (
        schema.INDEX_OF["MouthFrownLeft"],
        schema.INDEX_OF["MouthLowerDownLeft"],
        schema.INDEX_OF["MouthLowerDownRight"],
        schema.INDEX_OF["MouthShrugLower"],
    ),
    # Sliding the mouth sideways carries the jaw with it to some degree.
    schema.INDEX_OF["MouthLeft"]: (
        schema.INDEX_OF["JawLeft"],
    ),
    schema.INDEX_OF["MouthRight"]: (
        schema.INDEX_OF["JawRight"],
    ),
    # A tongue cannot come out through a closed mouth. Without this exclusion every TongueOut hold
    # would also teach "and the jaw was shut", which is both false and the exact failure this
    # project is trying to fix.
    schema.INDEX_OF["TongueOut"]: (
        schema.INDEX_OF["JawOpen"],
        schema.INDEX_OF["MouthClose"],
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

    Returns ``(lag_seconds, correlation)``. A low correlation means the cue may not have been
    followed, but it can also mean the stock model failed to see a real expression. The quality
    gate below therefore suppresses only extremely weak attempts with a proven observable peer.
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


def _attempt_key(session: Session, frame) -> tuple[str, str, int, int] | None:
    """Stable identity of one user attempt, with an old-recording-compatible attempt default."""
    cue = frame.cue
    if cue is None:
        return None
    return (
        session.session_id,
        str(cue.get("id", "")),
        int(cue.get("rep", 0) or 0),
        int(cue.get("attempt", 0) or 0),
    )


def _segment_key(session: Session, frame) -> tuple | None:
    """Identity of the contiguous cue segment a frame belongs to, or None outside a cue.

    Includes **phase and level**, not just the cue id and repetition. That matters: a ``rest`` phase
    normally shares its cue id and repetition with the ``hold`` it follows, so keying on id+rep alone
    makes the rest look like a continuation of the hold and its settle-in period is never trimmed.
    Two holds at different levels under one id+rep alias the same way.
    """
    cue = frame.cue
    if cue is None:
        return None

    try:
        level = round(float(cue.get("level", 0.0)), 4)
    except (TypeError, ValueError):
        level = 0.0

    return (
        session.session_id,
        str(cue.get("id", "")),
        int(cue.get("rep", 0) or 0),
        int(cue.get("attempt", 0) or 0),
        str(cue.get("phase", "")).lower(),
        level,
    )


def build_labels(
    sessions: Sequence[Session],
    *,
    use_speech_pseudo_labels: bool = True,
    hold_settle_trim: float = HOLD_SETTLE_TRIM_SECONDS,
    dim_boost: dict[int, float] | None = None,
    guided_quality: Sequence[dict] | None = None,
) -> LabelSet:
    """Build targets and weights for every frame in ``sessions``.

    ``dim_boost`` multiplies the final weight of specific expressions, so a dimension the user
    actually complains about (JawOpen) can carry more of the loss than the other forty-four. It is
    deliberately **symmetric** - it emphasises being right about zeros and about genuine openings
    equally, and never touches a target. Asymmetric "punish false positives harder" pressure lives in
    the training loss (``--fp-penalty``), where it is visible and separately tunable.
    """
    n_dims = schema.EXPRESSION_COUNT
    frames = list(iter_frames(sessions))
    n = len(frames)

    # Quality is opt-in at this low-level API so callers that are inspecting raw label semantics do
    # not get a hidden stock-model dependency. The training entry point always computes one global
    # report across train+validation sessions and passes it to both splits.
    guided_scales: dict[tuple[str, str, int, int], float] = {}
    if guided_quality is not None:
        for entry in guided_quality:
            key = (
                str(entry.get("session", "")),
                str(entry.get("cue", "")),
                int(entry.get("rep", 0) or 0),
                int(entry.get("attempt", 0) or 0),
            )
            guided_scales[key] = float(entry.get("weight_scale", 1.0))

    targets = np.zeros((n, n_dims), dtype=np.float32)
    weights = np.zeros((n, n_dims), dtype=np.float32)
    stock = np.zeros((n, n_dims), dtype=np.float32)

    # Cue segments are contiguous in recording order, so the settle-in trim is measured from the
    # frame where the segment identity last changed rather than from a per-key memo.
    segment_key: tuple | None = None
    segment_start_ticks: int = 0

    for row, (session, frame) in enumerate(frames):
        stock[row] = frame.stock

        key = _segment_key(session, frame)
        if key != segment_key:
            segment_key = key
            segment_start_ticks = frame.timestamp_ticks

        if session.is_correction:
            # The user watched the tracker get this wrong and said so. That outranks every
            # automatic source, hence the highest weight in the table.
            #
            # Only the dimensions the correction names are supervised. "My mouth was closed" says
            # nothing about whether the user was also smiling - and in VR they may well have been -
            # so a zero prior on the rest of the face would invent supervision that was never given.
            # An unlabelled cell costs nothing; a confidently wrong one costs a lot.
            for dim in session.corrected_dims():
                if 0 <= dim < n_dims:
                    targets[row, dim] = session.correction_target()
                    weights[row, dim] = W_MANUAL_CORRECTION
            continue

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
            elapsed = (frame.timestamp_ticks - segment_start_ticks) / TICKS_PER_SECOND

            # Still moving into the expression: record nothing rather than something wrong. This
            # applies to rest phases too - relaxing out of a hold takes just as long as moving into
            # one, and the frames in between show neither the cue nor a resting face.
            if elapsed < hold_settle_trim:
                continue

            if phase == "rest":
                weights[row, :] = W_GUIDED_REST
                continue

            attempt_key = _attempt_key(session, frame)
            quality_scale = guided_scales.get(attempt_key, 1.0) if attempt_key else 1.0

            for dim in cued_dims:
                is_tongue = dim in schema.TONGUE_DIMS
                targets[row, dim] = commanded[dim]
                base_weight = W_GUIDED_HOLD_TONGUE if is_tongue else W_GUIDED_HOLD
                weights[row, dim] = base_weight * quality_scale

            # "Keep the rest of your face relaxed", minus expressions that legitimately come along
            # for the ride with this cue.
            excluded = set(cued_dims)
            for dim in cued_dims:
                excluded.update(CO_ACTIVATION_EXCLUSIONS.get(dim, ()))

            for dim in range(n_dims):
                if dim not in excluded:
                    targets[row, dim] = 0.0
                    weights[row, dim] = W_UNCUED_DIM * quality_scale

    if dim_boost:
        for dim, factor in dim_boost.items():
            if 0 <= int(dim) < n_dims:
                weights[:, int(dim)] *= float(factor)

    return LabelSet(targets=targets, weights=weights, stock=stock)


def parse_dim_boost(text: str | None) -> dict[int, float]:
    """Parse ``"JawOpen=2.0,TongueOut=1.5"`` into ``{4: 2.0, 33: 1.5}``.

    Names rather than indices on the command line: an index typo silently reweights the wrong
    expression, whereas an unknown name fails immediately.
    """
    if not text:
        return {}

    boosts: dict[int, float] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        name, _, value = chunk.partition("=")
        name = name.strip()
        if name not in schema.INDEX_OF:
            raise ValueError(f"Unknown expression '{name}' in --dim-boost. "
                             f"Expected one of the 45 canonical names, e.g. JawOpen.")

        try:
            factor = float(value)
        except ValueError:
            raise ValueError(f"--dim-boost entry '{chunk}' has no numeric weight (use Name=2.0).")

        if factor < 0:
            raise ValueError(f"--dim-boost weight for {name} must not be negative.")

        boosts[schema.INDEX_OF[name]] = factor

    return boosts


def _read_guided_quality_overrides(session: Session) -> dict[tuple[str, int, int], dict]:
    """Read explicit headset judgments for a session.

    The file is deliberately separate from ``session.json`` and ``labels.jsonl``: recordings remain
    append-only while a retry/skip decision can be written atomically when the guided run stops.
    Conflicting duplicate entries are rejected instead of letting file order silently choose which
    supervision wins.
    """
    path = session.path / GUIDED_QUALITY_OVERRIDES_FILE
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read guided quality overrides from {path}: {exc}") from exc

    attempts = payload.get("attempts", []) if isinstance(payload, dict) else None
    if not isinstance(attempts, list):
        raise ValueError(f"{path}: 'attempts' must be a JSON array")

    result: dict[tuple[str, int, int], dict] = {}
    for index, item in enumerate(attempts):
        if not isinstance(item, dict) or not isinstance(item.get("valid"), bool):
            raise ValueError(f"{path}: attempts[{index}] must be an object with boolean 'valid'")

        key = (
            str(item.get("cue", "")),
            int(item.get("rep", 0) or 0),
            int(item.get("attempt", 0) or 0),
        )
        value = {
            "valid": bool(item["valid"]),
            "reason": str(item.get("reason", "explicit-headset-judgment")),
            "source": GUIDED_QUALITY_OVERRIDES_FILE,
        }
        previous = result.get(key)
        if previous is not None and previous["valid"] != value["valid"]:
            raise ValueError(f"{path}: conflicting valid values for cue={key[0]!r}, "
                             f"rep={key[1]}, attempt={key[2]}")
        result[key] = value

    return result


def _inline_attempt_judgment(frames: Sequence) -> dict | None:
    """Read an optional boolean stamped directly into cue rows by future recorders.

    The sidecar is the current contract, but accepting these unambiguous field names keeps old
    trainer builds compatible with a recorder that later chooses to stamp the decision per frame.
    An explicit false wins if a partially written attempt contains both values.
    """
    values: list[bool] = []
    for frame in frames:
        cue = frame.cue or {}
        for name in ("valid", "quality_valid", "attempt_valid"):
            value = cue.get(name)
            if isinstance(value, bool):
                values.append(value)
                break

    if not values:
        return None
    valid = all(values)
    return {
        "valid": valid,
        "reason": "cue-metadata-valid" if valid else "cue-metadata-invalid",
        "source": "cue-metadata",
    }


def guided_attempt_quality(sessions: Sequence[Session]) -> list[dict]:
    """Classify each Guided hold attempt without ever rejecting a whole session.

    Automatic suppression has two conditions: correlation is nearly absent *and* another attempt
    of the exact cue in the same session reaches the established 0.60 observability threshold. This
    catches a genuinely missed repetition (for example one missed Smile50 among successful Smile50
    repetitions) without declaring a whole expression unusable merely because the stock model is
    blind to it. Less decisive attempts remain in the corpus at one quarter weight.

    Explicit ``valid``/``invalid`` judgments from the headset sidecar or cue metadata always win.
    Returns deterministic entries ordered by session, cue, repetition, and attempt.
    """
    reports: list[dict] = []

    for session in sessions:
        if not session.is_guided:
            continue

        overrides = _read_guided_quality_overrides(session)

        # Lag is a property of one attempt, not of a session or cue family. ``attempt`` defaults to
        # zero so recordings made before retry support keep their original grouping.
        groups: dict[tuple[str, int, int], list] = {}
        for frame in session.frames:
            cue = frame.cue
            if not cue:
                continue
            key = (
                str(cue.get("id", "")),
                int(cue.get("rep", 0) or 0),
                int(cue.get("attempt", 0) or 0),
            )
            groups.setdefault(key, []).append(frame)

        for (cue_id, rep, attempt), frames in sorted(groups.items()):
            hold_frames = [f for f in frames
                           if str((f.cue or {}).get("phase", "")).lower() == "hold"]
            # Lead-ins and rest-only groups provide known-neutral evidence, not an expression
            # attempt. Their rest semantics stay unchanged and should not pollute Good/Weak counts.
            if not hold_frames:
                continue

            exemplar = next((f for f in hold_frames if (f.cue or {}).get("dims")), hold_frames[0])
            dims = [int(d) for d in (exemplar.cue or {}).get("dims", [])]
            dims = [d for d in dims if 0 <= d < schema.EXPRESSION_COUNT]
            if not dims:
                continue

            primary = dims[0]

            def command(frame) -> float:
                target = (frame.cue or {}).get("target")
                if not isinstance(target, (list, tuple)) or primary >= len(target):
                    return 0.0
                try:
                    return float(target[primary])
                except (TypeError, ValueError):
                    return 0.0

            commanded = np.asarray([command(f) for f in frames], dtype=np.float64)
            observed = np.asarray([float(f.stock[primary]) for f in frames], dtype=np.float64)
            timestamps = np.asarray([f.timestamp_ticks for f in frames], dtype=np.float64)
            lag, correlation = estimate_cue_lag_seconds(observed, commanded, timestamps)

            explicit = overrides.get((cue_id, rep, attempt)) or _inline_attempt_judgment(frames)
            reports.append({
                "session": session.session_id,
                "cue": cue_id,
                "rep": rep,
                "attempt": attempt,
                "dim": schema.EXPRESSION_NAMES[primary],
                "dims": [schema.EXPRESSION_NAMES[d] for d in dims],
                "frames": len(frames),
                "hold_frames": len(hold_frames),
                "lag_seconds": float(lag),
                "correlation": float(correlation),
                "command_range": float(np.ptp(commanded)) if len(commanded) else 0.0,
                "observed_range": float(np.ptp(observed)) if len(observed) else 0.0,
                "explicit_valid": None if explicit is None else bool(explicit["valid"]),
                "override_reason": None if explicit is None else explicit["reason"],
                "decision_source": "automatic" if explicit is None else explicit["source"],
                "reason": "pending",
                "status": "pending",
                "weight_scale": 1.0,
            })

    # Decide only after all correlations are known, because suppression requires an observable peer
    # of the same exact cue and session. Sorted input/output makes the decision reproducible.
    for entry in reports:
        explicit = entry["explicit_valid"]
        if explicit is True:
            entry.update(status="good", weight_scale=1.0, reason="explicit-valid")
            continue
        if explicit is False:
            entry.update(status="suppressed", weight_scale=0.0, reason="explicit-invalid")
            continue

        correlation = float(entry["correlation"])
        if correlation >= GUIDED_GOOD_CORRELATION:
            entry.update(status="good", weight_scale=1.0,
                         reason="correlation-at-or-above-good-threshold")
            continue

        observable_peer = any(
            other is not entry
            and other["session"] == entry["session"]
            and other["cue"] == entry["cue"]
            and float(other["correlation"]) >= GUIDED_GOOD_CORRELATION
            for other in reports
        )
        if correlation <= GUIDED_SUPPRESS_CORRELATION and observable_peer:
            entry.update(status="suppressed", weight_scale=0.0,
                         reason="extremely-weak-with-observable-peer")
        else:
            reason = ("extremely-weak-but-no-observable-peer" if
                      correlation <= GUIDED_SUPPRESS_CORRELATION else
                      "below-good-threshold")
            entry.update(status="weak", weight_scale=GUIDED_WEAK_WEIGHT_SCALE, reason=reason)

    return reports


def cue_lag_report(sessions: Sequence[Session]) -> list[dict]:
    """Backward-compatible name for the now-actionable per-attempt quality report."""
    return guided_attempt_quality(sessions)


def guided_quality_artifact(
    reports: Sequence[dict],
    *,
    train_session_ids: Sequence[str] = (),
    val_session_ids: Sequence[str] = (),
) -> dict:
    """Build the JSON-serializable artifact written beside every immutable training run."""
    train_ids, val_ids = set(train_session_ids), set(val_session_ids)
    attempts = []
    for report in reports:
        entry = dict(report)
        session_id = str(entry.get("session", ""))
        entry["split"] = ("train" if session_id in train_ids else
                          "validation" if session_id in val_ids else "unassigned")
        attempts.append(entry)

    counts = {status: sum(1 for r in attempts if r.get("status") == status)
              for status in ("good", "weak", "suppressed")}
    return {
        "format_version": 1,
        "policy": {
            "good_correlation": GUIDED_GOOD_CORRELATION,
            "suppress_correlation": GUIDED_SUPPRESS_CORRELATION,
            "weak_weight_scale": GUIDED_WEAK_WEIGHT_SCALE,
            "automatic_suppression_requires_same_session_same_cue_good_peer": True,
            "explicit_validity_precedence": True,
            "hold_only_gating": True,
        },
        "summary": counts,
        "attempts": attempts,
    }


def format_cue_lag_report(
    reports: Sequence[dict],
    *,
    threshold: float = GUIDED_GOOD_CORRELATION,
) -> str:
    """Human-readable Good/Weak/suppressed table for the pre-training console log."""
    if not reports:
        return ""

    statuses = []
    for entry in reports:
        fallback = "good" if float(entry["correlation"]) >= threshold else "weak"
        statuses.append(str(entry.get("status", fallback)).lower())
    counts = {status: statuses.count(status) for status in ("good", "weak", "suppressed")}

    lines = [
        (f"Guided attempt quality: Good {counts['good']} / Weak {counts['weak']} / "
         f"suppressed {counts['suppressed']}"),
        "Cue tracking (did the face follow the avatar?)",
        f"  {'cue':<18}{'rep':>5}{'try':>5}{'frames':>8}{'lag s':>9}{'corr':>8}"
        f"{'weight':>9}  quality",
    ]

    for entry, status in zip(reports, statuses):
        weight_scale = float(entry.get("weight_scale", 1.0 if status == "good" else
                                       GUIDED_WEAK_WEIGHT_SCALE))
        label = {"good": "Good", "weak": "Weak", "suppressed": "suppressed"}.get(status, status)
        flag = "  <-- weak" if status == "weak" else \
               "  <-- SUPPRESSED" if status == "suppressed" else ""
        lines.append(f"  {entry['cue']:<18}{entry['rep']:>5}{entry.get('attempt', 0):>5}"
                     f"{entry['frames']:>8}{entry['lag_seconds']:>9.2f}"
                     f"{entry['correlation']:>8.2f}{weight_scale:>9.2f}  {label}{flag}")

    mean = float(np.mean([e["correlation"] for e in reports]))
    lines.append(f"  mean correlation {mean:.2f} over {len(reports)} attempts")
    if counts["weak"] or counts["suppressed"]:
        lines.append("  Weak attempts keep 25% hold weight. Automatic suppression requires an "
                     "extremely weak response plus a Good peer of the same cue; explicit headset "
                     "valid/invalid judgments take precedence.")

    return "\n".join(lines)


def parse_dims(text: str | None) -> tuple[int, ...]:
    """Parse ``"JawOpen,TongueOut"`` into ``(4, 33)``."""
    if not text:
        return ()

    dims: list[int] = []
    for chunk in text.split(","):
        name = chunk.strip()
        if not name:
            continue
        if name not in schema.INDEX_OF:
            raise ValueError(f"Unknown expression '{name}'. Expected a canonical name, e.g. JawOpen.")
        dims.append(schema.INDEX_OF[name])

    return tuple(dims)


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
