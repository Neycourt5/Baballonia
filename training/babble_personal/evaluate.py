"""Metrics that reflect what actually annoys the user, not just aggregate loss.

A single MSE number hides the two failure modes this project exists to fix: expressions firing while
the face is at rest, and expressions bleeding into each other. Worse, it rewards a model that
suppresses everything, which trades one complaint for a deader-feeling face.

So the report is three-part:

* **Per-expression MAE** on confidently-labelled frames - is it accurate where we know the answer?
* **Neutral false-activation** - how often does an expression fire while resting, and how strongly?
* **Cross-talk** - during a cue, how much do the *other* expressions light up?

Every metric is reported for stock and personal side by side. Personal is only better if it wins on
neutral stability and cross-talk *without* losing accuracy on intentional expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import schema

#: Above this, an expression is "active" for false-activation accounting. Matches the plan.
ACTIVATION_THRESHOLD = 0.15

#: .NET DateTime ticks are 100 ns. Duplicated from labels.py rather than imported, so this module
#: stays free of the label-building machinery and can score a bare pair of arrays.
TICKS_PER_SECOND = 10_000_000


@dataclass
class ExpressionReport:
    name: str
    stock_mae: float
    personal_mae: float
    supervised_frames: int

    @property
    def improvement(self) -> float:
        return self.stock_mae - self.personal_mae


@dataclass
class NeutralReport:
    frames: int
    stock_false_activation_rate: float
    personal_false_activation_rate: float
    stock_mean_activation: float
    personal_mean_activation: float
    worst_offenders: list[tuple[str, float, float]]  # name, stock rate, personal rate
    stock_jitter: float = 0.0
    personal_jitter: float = 0.0


def per_expression_mae(
    stock: np.ndarray,
    personal: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    min_weight: float = 0.9,
) -> list[ExpressionReport]:
    """MAE per expression over confidently-labelled cells only.

    Low-confidence cells (ramps, uncued 'stay relaxed' priors, speech pseudo-labels) are excluded:
    scoring against labels we do not trust would just measure how well the model copied a guess.
    """
    reports: list[ExpressionReport] = []

    for dim in range(schema.EXPRESSION_COUNT):
        mask = weights[:, dim] >= min_weight
        count = int(mask.sum())

        if count == 0:
            reports.append(ExpressionReport(schema.EXPRESSION_NAMES[dim], float("nan"), float("nan"), 0))
            continue

        target = targets[mask, dim]
        reports.append(
            ExpressionReport(
                name=schema.EXPRESSION_NAMES[dim],
                stock_mae=float(np.abs(stock[mask, dim] - target).mean()),
                personal_mae=float(np.abs(personal[mask, dim] - target).mean()),
                supervised_frames=count,
            )
        )

    return reports


def resting_jitter(values: np.ndarray) -> float:
    """Mean frame-to-frame movement while the face is still - the "wiggle" the user actually sees.

    A model can score a perfect false-activation rate and still feel bad: an expression that hovers
    just under the threshold but shivers every frame reads as a twitching face. Averaging the
    absolute change between consecutive frames catches that, and it is the one number that moved
    when users complained about wiggle while every other metric said the model was fine.

    Assumes ``values`` are consecutive frames of one session, which is how the neutral report feeds
    it. Concatenating sessions adds one meaningless step per boundary, which is negligible over
    thousands of frames.
    """
    if len(values) < 2:
        return 0.0
    return float(np.abs(np.diff(values, axis=0)).mean())


def neutral_report(
    stock: np.ndarray,
    personal: np.ndarray,
    *,
    threshold: float = ACTIVATION_THRESHOLD,
    top_n: int = 8,
) -> NeutralReport:
    """False activation while the face is at rest. Feed only frames from neutral sessions."""
    if len(stock) == 0:
        return NeutralReport(0, 0.0, 0.0, 0.0, 0.0, [])

    stock_active = stock > threshold
    personal_active = personal > threshold

    per_dim = [
        (schema.EXPRESSION_NAMES[d], float(stock_active[:, d].mean()), float(personal_active[:, d].mean()))
        for d in range(schema.EXPRESSION_COUNT)
    ]
    per_dim.sort(key=lambda row: row[1], reverse=True)

    return NeutralReport(
        frames=len(stock),
        stock_false_activation_rate=float(stock_active.mean()),
        personal_false_activation_rate=float(personal_active.mean()),
        stock_mean_activation=float(stock.mean()),
        personal_mean_activation=float(personal.mean()),
        worst_offenders=[row for row in per_dim[:top_n] if row[1] > 0 or row[2] > 0],
        stock_jitter=resting_jitter(stock),
        personal_jitter=resting_jitter(personal),
    )


def cross_talk(
    values: np.ndarray,
    cued_dims_per_frame: list[list[int]],
    *,
    threshold: float = ACTIVATION_THRESHOLD,
) -> float:
    """Mean activation of non-cued expressions during cue holds.

    This is 'I opened my jaw and the avatar smiled' expressed as a number. Lower is better, but only
    meaningful alongside the MAE table - a model that outputs nothing scores perfectly here.
    """
    if len(values) == 0:
        return 0.0

    totals: list[float] = []
    for row, cued in enumerate(cued_dims_per_frame):
        mask = np.ones(schema.EXPRESSION_COUNT, dtype=bool)
        for dim in cued:
            if 0 <= dim < schema.EXPRESSION_COUNT:
                mask[dim] = False

        if mask.any():
            active = values[row, mask] > threshold
            totals.append(float(active.mean()))

    return float(np.mean(totals)) if totals else 0.0


# ---------------------------------------------------------------------------
# Per-expression deep dive, built for the two complaints that survived P1:
# "my jaw hangs open when my mouth is closed" and "my tongue pokes out sometimes".
#
# A false-activation *rate* is not enough to describe either. One two-second hang is far more
# noticeable than twenty single-frame blips, and both can score the same rate. So the closed-face
# behaviour is measured as runs over time, not just as a fraction of frames.
#
# The other half of this report exists to stop the obvious cheat. Any model can drive a dimension to
# zero and win every false-positive metric on the board, at the cost of a mouth that no longer opens.
# So every closed-face number is reported next to a sensitivity number, and range retention is
# computed explicitly rather than left for a human to notice.
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    """How false activations are distributed in time, rather than merely how many there are."""

    count: int
    runs_per_minute: float
    mean_seconds: float
    p95_seconds: float
    max_seconds: float

    @classmethod
    def empty(cls) -> "RunStats":
        return cls(0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class DimReport:
    """Everything worth knowing about one expression, stock versus personal."""

    name: str
    dim: int

    closed_frames: int = 0
    stock_fp_rate: float = 0.0
    personal_fp_rate: float = 0.0
    stock_mean_closed: float = 0.0
    personal_mean_closed: float = 0.0
    stock_p95_closed: float = 0.0
    personal_p95_closed: float = 0.0
    stock_max_closed: float = 0.0
    personal_max_closed: float = 0.0
    stock_runs: RunStats = None  # type: ignore[assignment]
    personal_runs: RunStats = None  # type: ignore[assignment]

    # Sensitivity - the anti-suppression guardrail.
    hold_frames: int = 0
    stock_mean_at_level: dict[float, float] = None  # type: ignore[assignment]
    personal_mean_at_level: dict[float, float] = None  # type: ignore[assignment]
    stock_open_separation: float | None = None
    personal_open_separation: float | None = None

    speech_frames: int = 0
    stock_speech_range: float = 0.0
    personal_speech_range: float = 0.0
    stock_speech_std: float = 0.0
    personal_speech_std: float = 0.0
    range_retention: float | None = None

    # Ordering - does a bigger commanded jaw produce a bigger reading?
    stock_monotonicity: float | None = None
    personal_monotonicity: float | None = None
    stock_ordering_accuracy: float | None = None
    personal_ordering_accuracy: float | None = None

    def __post_init__(self) -> None:
        if self.stock_runs is None:
            self.stock_runs = RunStats.empty()
        if self.personal_runs is None:
            self.personal_runs = RunStats.empty()
        if self.stock_mean_at_level is None:
            self.stock_mean_at_level = {}
        if self.personal_mean_at_level is None:
            self.personal_mean_at_level = {}

    @property
    def suppression_warning(self) -> bool:
        """True when the model bought its false-positive win by flattening real movement.

        Requires there to have been real movement in the first place. On a dimension the stock model
        barely moves - a tongue expression during ordinary speech is mostly sensor noise - shrinking
        that noise is the adapter working correctly, and flagging it would train the reader to
        ignore this warning on the dimensions where it means something.
        """
        return (
            self.range_retention is not None
            and self.range_retention < RANGE_RETENTION_FLOOR
            and self.stock_speech_range >= RANGE_RETENTION_MIN_MOVEMENT
        )


#: Below this fraction of the stock model's speech-time range, a "quieter" dimension is more likely
#: to have been suppressed than corrected. Not a failure on its own - it is a prompt to look at the
#: sensitivity numbers next to it - but it is the number that catches a dead-looking avatar early.
RANGE_RETENTION_FLOOR = 0.8

#: Stock p05-p95 range below which a dimension is not really moving, so "it moves less now" is not
#: evidence of suppression. Chosen at the level where a change stops being visible on an avatar.
RANGE_RETENTION_MIN_MOVEMENT = 0.05


def activation_runs(
    values: np.ndarray,
    timestamps: np.ndarray,
    *,
    threshold: float = ACTIVATION_THRESHOLD,
) -> list[float]:
    """Durations, in seconds, of each maximal stretch where ``values`` stays above ``threshold``.

    ``values`` and ``timestamps`` must be consecutive frames of one continuous recording. A single
    frame above threshold counts as one median frame interval rather than zero, because a frame is
    the shortest event the tracker can express and calling it zero would make blips free.
    """
    if len(values) == 0:
        return []

    ticks = np.asarray(timestamps, dtype=np.float64)
    if len(ticks) > 1:
        intervals = np.diff(ticks)
        positive = intervals[intervals > 0]
        frame_ticks = float(np.median(positive)) if len(positive) else 0.0
    else:
        frame_ticks = 0.0

    active = np.asarray(values) > threshold
    runs: list[float] = []
    start: int | None = None

    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            runs.append((ticks[i - 1] - ticks[start] + frame_ticks) / TICKS_PER_SECOND)
            start = None

    if start is not None:
        runs.append((ticks[-1] - ticks[start] + frame_ticks) / TICKS_PER_SECOND)

    return runs


def summarize_runs(runs: list[float], total_seconds: float) -> RunStats:
    if not runs:
        return RunStats.empty()

    array = np.asarray(runs, dtype=np.float64)
    minutes = total_seconds / 60.0
    return RunStats(
        count=len(runs),
        runs_per_minute=float(len(runs) / minutes) if minutes > 0 else 0.0,
        mean_seconds=float(array.mean()),
        p95_seconds=float(np.percentile(array, 95)),
        max_seconds=float(array.max()),
    )


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation without pulling in scipy. Returns None when it is undefined."""
    if len(x) < 2 or len(x) != len(y):
        return None

    def rank(values: Sequence[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, dtype=np.float64))
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks

    a, b = rank(x), rank(y)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _ordering_accuracy(levels: Sequence[float], means: Sequence[float]) -> float | None:
    """Fraction of level pairs whose measured order matches the commanded order.

    More forgiving than a correlation and easier to read: 1.0 means a bigger commanded jaw always
    produced a bigger reading, 0.5 means the model is guessing.
    """
    pairs = correct = 0
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            if abs(levels[i] - levels[j]) < 1e-9:
                continue
            pairs += 1
            lower, higher = (i, j) if levels[i] < levels[j] else (j, i)
            if means[higher] > means[lower]:
                correct += 1

    return correct / pairs if pairs else None


def _contiguous_blocks(rows: np.ndarray) -> list[np.ndarray]:
    """Split ascending row indices into runs of consecutive values.

    Filtering a session down to "frames where this expression is confidently zero" can punch holes
    in it, and a hole must break a run - otherwise two activations either side of a gap merge into
    one long false hang that never happened.
    """
    if len(rows) == 0:
        return []
    breaks = np.nonzero(np.diff(rows) != 1)[0] + 1
    return [block for block in np.split(rows, breaks) if len(block)]


def dim_report(
    dim: int,
    *,
    closed_blocks: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]] = (),
    hold_segments: Sequence[tuple[float, np.ndarray, np.ndarray]] = (),
    speech: tuple[np.ndarray, np.ndarray] | None = None,
    threshold: float = ACTIVATION_THRESHOLD,
) -> DimReport:
    """Build the full picture for one expression.

    * ``closed_blocks`` - contiguous stretches where this expression is known to belong at zero,
      as ``(stock_values, personal_values, timestamp_ticks)``. Neutral sessions, settled guided
      rests, and user-flagged corrections all land here.
    * ``hold_segments`` - ``(commanded_level, stock_values, personal_values)`` per settled hold.
    * ``speech`` - ``(stock_values, personal_values)`` over natural speech, used for range retention.
    """
    report = DimReport(name=schema.EXPRESSION_NAMES[dim], dim=dim)

    if closed_blocks:
        stock_all = np.concatenate([b[0] for b in closed_blocks])
        personal_all = np.concatenate([b[1] for b in closed_blocks])

        report.closed_frames = len(stock_all)
        report.stock_fp_rate = float((stock_all > threshold).mean())
        report.personal_fp_rate = float((personal_all > threshold).mean())
        report.stock_mean_closed = float(stock_all.mean())
        report.personal_mean_closed = float(personal_all.mean())
        report.stock_p95_closed = float(np.percentile(stock_all, 95))
        report.personal_p95_closed = float(np.percentile(personal_all, 95))
        report.stock_max_closed = float(stock_all.max())
        report.personal_max_closed = float(personal_all.max())

        stock_runs: list[float] = []
        personal_runs: list[float] = []
        total_seconds = 0.0
        for stock_values, personal_values, ticks in closed_blocks:
            stock_runs += activation_runs(stock_values, ticks, threshold=threshold)
            personal_runs += activation_runs(personal_values, ticks, threshold=threshold)
            if len(ticks) > 1:
                total_seconds += float(ticks[-1] - ticks[0]) / TICKS_PER_SECOND

        report.stock_runs = summarize_runs(stock_runs, total_seconds)
        report.personal_runs = summarize_runs(personal_runs, total_seconds)

    if hold_segments:
        levels = [level for level, _, _ in hold_segments]
        stock_means = [float(s.mean()) for _, s, _ in hold_segments]
        personal_means = [float(p.mean()) for _, _, p in hold_segments]

        report.hold_frames = sum(len(s) for _, s, _ in hold_segments)

        for level in sorted(set(levels)):
            picked = [i for i, value in enumerate(levels) if abs(value - level) < 1e-9]
            report.stock_mean_at_level[level] = float(np.mean([stock_means[i] for i in picked]))
            report.personal_mean_at_level[level] = float(np.mean([personal_means[i] for i in picked]))

        report.stock_monotonicity = _spearman(levels, stock_means)
        report.personal_monotonicity = _spearman(levels, personal_means)
        report.stock_ordering_accuracy = _ordering_accuracy(levels, stock_means)
        report.personal_ordering_accuracy = _ordering_accuracy(levels, personal_means)

        # Separation between a full commanded opening and the resting face: the number that goes
        # down when a model "fixes" false positives by refusing to open at all.
        top = max(report.stock_mean_at_level)
        if closed_blocks:
            report.stock_open_separation = report.stock_mean_at_level[top] - report.stock_mean_closed
            report.personal_open_separation = (
                report.personal_mean_at_level[top] - report.personal_mean_closed
            )

    if speech is not None and len(speech[0]):
        stock_values, personal_values = speech
        report.speech_frames = len(stock_values)
        report.stock_speech_std = float(stock_values.std())
        report.personal_speech_std = float(personal_values.std())
        report.stock_speech_range = float(
            np.percentile(stock_values, 95) - np.percentile(stock_values, 5))
        report.personal_speech_range = float(
            np.percentile(personal_values, 95) - np.percentile(personal_values, 5))

        if report.stock_speech_range > 1e-6:
            report.range_retention = report.personal_speech_range / report.stock_speech_range

    return report


def collect_closed_blocks(
    sessions: Sequence,
    label_set,
    stock: np.ndarray,
    personal: np.ndarray,
    dim: int,
    *,
    min_weight: float = 0.9,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Contiguous stretches where ``dim`` is confidently labelled zero, per session.

    Defined off the labels rather than off session type, so neutral recordings, settled guided rests
    and user-flagged corrections are all picked up by the same rule with no source-specific code.
    """
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    offset = 0

    for session in sessions:
        count = len(session.frames)
        if count == 0:
            continue

        rows = np.arange(offset, offset + count)
        confident = label_set.weights[rows, dim] >= min_weight
        at_zero = label_set.targets[rows, dim] <= 1e-6
        local = np.nonzero(confident & at_zero)[0]

        ticks = np.asarray([f.timestamp_ticks for f in session.frames], dtype=np.float64)
        for block in _contiguous_blocks(local):
            absolute = rows[block]
            blocks.append((stock[absolute, dim], personal[absolute, dim], ticks[block]))

        offset += count

    return blocks


def collect_hold_segments(
    sessions: Sequence,
    label_set,
    stock: np.ndarray,
    personal: np.ndarray,
    dim: int,
    *,
    min_weight: float = 0.9,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Settled hold stretches for ``dim``, grouped into contiguous segments with their commanded level."""
    segments: list[tuple[float, np.ndarray, np.ndarray]] = []
    offset = 0

    for session in sessions:
        count = len(session.frames)
        if count == 0:
            continue

        rows = np.arange(offset, offset + count)
        confident = label_set.weights[rows, dim] >= min_weight
        commanded = label_set.targets[rows, dim] > 1e-6
        local = np.nonzero(confident & commanded)[0]

        for block in _contiguous_blocks(local):
            absolute = rows[block]
            level = float(np.median(label_set.targets[absolute, dim]))
            segments.append((level, stock[absolute, dim], personal[absolute, dim]))

        offset += count

    return segments


def collect_speech_values(
    sessions: Sequence,
    stock: np.ndarray,
    personal: np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """All frames from speech sessions for one dimension - the natural-movement reference."""
    picked: list[np.ndarray] = []
    offset = 0

    for session in sessions:
        count = len(session.frames)
        if count and getattr(session, "is_speech", False):
            picked.append(np.arange(offset, offset + count))
        offset += count

    if not picked:
        return None

    rows = np.concatenate(picked)
    return stock[rows, dim], personal[rows, dim]


def build_dim_reports(
    sessions: Sequence,
    label_set,
    stock: np.ndarray,
    personal: np.ndarray,
    dims: Sequence[int] = (),
    *,
    threshold: float = ACTIVATION_THRESHOLD,
) -> list[DimReport]:
    """Deep-dive reports for the expressions worth watching closely."""
    if not dims:
        dims = WATCHED_DIMS

    return [
        dim_report(
            dim,
            closed_blocks=collect_closed_blocks(sessions, label_set, stock, personal, dim),
            hold_segments=collect_hold_segments(sessions, label_set, stock, personal, dim),
            speech=collect_speech_values(sessions, stock, personal, dim),
            threshold=threshold,
        )
        for dim in dims
    ]


#: The two expressions the user reports as still wrong, in priority order.
WATCHED_DIMS: tuple[int, ...] = (
    schema.INDEX_OF["JawOpen"],
    schema.INDEX_OF["TongueOut"],
)


def format_dim_report(report: DimReport) -> str:
    """The block a human reads to answer 'did the jaw problem get better, and at what cost?'"""
    lines = [f"{report.name} detail"]

    if report.closed_frames:
        lines.append(f"  while known closed ({report.closed_frames} frames)")
        lines.append(f"    fires above {ACTIVATION_THRESHOLD}   "
                     f"stock {report.stock_fp_rate:>7.4f}   personal {report.personal_fp_rate:>7.4f}")
        lines.append(f"    mean value            stock {report.stock_mean_closed:>7.4f}   "
                     f"personal {report.personal_mean_closed:>7.4f}")
        lines.append(f"    p95 value             stock {report.stock_p95_closed:>7.4f}   "
                     f"personal {report.personal_p95_closed:>7.4f}")
        lines.append(f"    max value             stock {report.stock_max_closed:>7.4f}   "
                     f"personal {report.personal_max_closed:>7.4f}")
        lines.append(f"    false runs/min        stock {report.stock_runs.runs_per_minute:>7.2f}   "
                     f"personal {report.personal_runs.runs_per_minute:>7.2f}")
        lines.append(f"    longest false run     stock {report.stock_runs.max_seconds:>7.2f}s  "
                     f"personal {report.personal_runs.max_seconds:>7.2f}s")

    if report.hold_frames:
        lines.append(f"  while commanded open ({report.hold_frames} frames)")
        for level in sorted(report.stock_mean_at_level):
            lines.append(f"    level {level:<4.2f}            "
                         f"stock {report.stock_mean_at_level[level]:>7.4f}   "
                         f"personal {report.personal_mean_at_level[level]:>7.4f}")
        if report.personal_open_separation is not None:
            lines.append(f"    open/closed separation stock {report.stock_open_separation:>6.4f}   "
                         f"personal {report.personal_open_separation:>7.4f}")
        if report.personal_ordering_accuracy is not None:
            lines.append(f"    level ordering        stock {report.stock_ordering_accuracy:>7.2f}   "
                         f"personal {report.personal_ordering_accuracy:>7.2f}")

    if report.speech_frames:
        lines.append(f"  during speech ({report.speech_frames} frames)")
        lines.append(f"    p05-p95 range         stock {report.stock_speech_range:>7.4f}   "
                     f"personal {report.personal_speech_range:>7.4f}")
        if report.range_retention is not None:
            tag = "  <-- SUPPRESSED?" if report.suppression_warning else ""
            lines.append(f"    range retention       {report.range_retention:.2f}{tag}")

    if len(lines) == 1:
        lines.append("  (no frames with a confident label for this expression)")

    return "\n".join(lines)


def format_report(
    expression_reports: list[ExpressionReport],
    neutral: NeutralReport | None = None,
    *,
    dim_reports: Sequence[DimReport] = (),
    cross_talk_stock: float | None = None,
    cross_talk_personal: float | None = None,
    top_n: int = 12,
) -> str:
    """Human-readable comparison, printed after training and by the evaluate CLI."""
    lines: list[str] = []

    scored = [r for r in expression_reports if r.supervised_frames > 0]

    lines.append("Per-expression MAE (confident labels only, sorted by improvement)")
    lines.append(f"  {'expression':<24}{'frames':>8}{'stock':>10}{'personal':>10}{'delta':>10}")

    if not scored:
        lines.append("  (no confidently-labelled frames - record neutral or guided sessions)")
    else:
        for report in sorted(scored, key=lambda r: r.improvement, reverse=True)[:top_n]:
            lines.append(
                f"  {report.name:<24}{report.supervised_frames:>8}"
                f"{report.stock_mae:>10.4f}{report.personal_mae:>10.4f}"
                f"{report.improvement:>+10.4f}"
            )

        stock_mean = float(np.mean([r.stock_mae for r in scored]))
        personal_mean = float(np.mean([r.personal_mae for r in scored]))
        lines.append(f"  {'MEAN':<24}{'':>8}{stock_mean:>10.4f}{personal_mean:>10.4f}"
                     f"{stock_mean - personal_mean:>+10.4f}")

        regressions = [r for r in scored if r.personal_mae > r.stock_mae + 1e-4]
        if regressions:
            lines.append(f"  {len(regressions)} expression(s) got worse; worst: " +
                         ", ".join(r.name for r in sorted(
                             regressions, key=lambda r: r.improvement)[:3]))

    if neutral is not None and neutral.frames > 0:
        lines.append("")
        lines.append(f"Neutral stability ({neutral.frames} resting frames, threshold "
                     f"{ACTIVATION_THRESHOLD})")
        lines.append(f"  false activation rate  stock {neutral.stock_false_activation_rate:.4f}"
                     f"   personal {neutral.personal_false_activation_rate:.4f}")
        lines.append(f"  mean activation        stock {neutral.stock_mean_activation:.4f}"
                     f"   personal {neutral.personal_mean_activation:.4f}")
        lines.append(f"  resting jitter         stock {neutral.stock_jitter:.4f}"
                     f"   personal {neutral.personal_jitter:.4f}"
                     f"   {'(smoother)' if neutral.personal_jitter <= neutral.stock_jitter else '(WIGGLIER)'}")

        if neutral.worst_offenders:
            lines.append("  worst offenders (stock -> personal):")
            for name, stock_rate, personal_rate in neutral.worst_offenders:
                lines.append(f"    {name:<24}{stock_rate:>8.3f} -> {personal_rate:.3f}")

    if cross_talk_stock is not None and cross_talk_personal is not None:
        lines.append("")
        lines.append("Cross-talk during cue holds (non-cued expressions firing)")
        lines.append(f"  stock {cross_talk_stock:.4f}   personal {cross_talk_personal:.4f}"
                     f"   {'(better)' if cross_talk_personal <= cross_talk_stock else '(WORSE)'}")

    for report in dim_reports:
        lines.append("")
        lines.append(format_dim_report(report))

    return "\n".join(lines)


def collect_cued_dims(sessions: Sequence) -> tuple[np.ndarray, list[list[int]]]:
    """Rows that are settled cue holds, and the dimensions each of them commanded.

    Returns ``(row_indices, cued_dims_per_row)`` ready for :func:`cross_talk`. Only ``hold`` phases
    count: during a transition the face is mid-movement, so "everything else should be quiet" is not
    a fair expectation.
    """
    rows: list[int] = []
    cued: list[list[int]] = []
    offset = 0

    for session in sessions:
        for i, frame in enumerate(session.frames):
            cue = frame.cue
            if cue and str(cue.get("phase", "")).lower() == "hold":
                dims = [int(d) for d in cue.get("dims", [])]
                if dims:
                    rows.append(offset + i)
                    cued.append(dims)
        offset += len(session.frames)

    return np.asarray(rows, dtype=np.int64), cued


def dim_report_to_dict(report: DimReport) -> dict:
    """Serializable form for summary.json, consumed by the app's results screen."""
    def runs(stats: RunStats) -> dict:
        return {
            "count": stats.count,
            "runs_per_minute": stats.runs_per_minute,
            "mean_seconds": stats.mean_seconds,
            "p95_seconds": stats.p95_seconds,
            "max_seconds": stats.max_seconds,
        }

    return {
        "name": report.name,
        "dim": report.dim,
        "closed_frames": report.closed_frames,
        "stock_fp_rate": report.stock_fp_rate,
        "personal_fp_rate": report.personal_fp_rate,
        "stock_mean_closed": report.stock_mean_closed,
        "personal_mean_closed": report.personal_mean_closed,
        "stock_p95_closed": report.stock_p95_closed,
        "personal_p95_closed": report.personal_p95_closed,
        "stock_max_closed": report.stock_max_closed,
        "personal_max_closed": report.personal_max_closed,
        "stock_runs": runs(report.stock_runs),
        "personal_runs": runs(report.personal_runs),
        "hold_frames": report.hold_frames,
        "stock_mean_at_level": {str(k): v for k, v in report.stock_mean_at_level.items()},
        "personal_mean_at_level": {str(k): v for k, v in report.personal_mean_at_level.items()},
        "stock_open_separation": report.stock_open_separation,
        "personal_open_separation": report.personal_open_separation,
        "speech_frames": report.speech_frames,
        "stock_speech_range": report.stock_speech_range,
        "personal_speech_range": report.personal_speech_range,
        "stock_speech_std": report.stock_speech_std,
        "personal_speech_std": report.personal_speech_std,
        "range_retention": report.range_retention,
        "suppression_warning": report.suppression_warning,
        "stock_monotonicity": report.stock_monotonicity,
        "personal_monotonicity": report.personal_monotonicity,
        "stock_ordering_accuracy": report.stock_ordering_accuracy,
        "personal_ordering_accuracy": report.personal_ordering_accuracy,
    }


# ---------------------------------------------------------------------------
# CLI - score an already-exported model without retraining it.
#
# This is what makes A/B/C comparisons cheap: point it at a session folder and an .onnx and get the
# same report training prints. It also scores the model that is actually installed, against sessions
# recorded after it was trained, which is the only honest way to ask "is my current model still good
# on today's lighting?"
# ---------------------------------------------------------------------------


def run_onnx(model_path, images: np.ndarray, stock: np.ndarray,
             embeddings: np.ndarray | None = None) -> np.ndarray:
    """Run an exported adapter over prepared inputs, adapting to its declared signature.

    v1 adapters (A and B) take ``image`` + ``stock``; v2 embedding adapters (C) take ``stock`` +
    ``embedding``. The graph is asked what it wants rather than the caller assuming, so one code path
    scores every model this project produces.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}

    if "embedding" in names and embeddings is None:
        raise ValueError(
            f"{model_path} needs a 1280-d embedding per frame, but none were supplied.\n"
            "Backfill them first:  python -m babble_personal.compute_embeddings --data <root>"
        )

    outputs = []
    for i in range(len(stock)):
        feed: dict[str, np.ndarray] = {"stock": stock[i : i + 1].astype(np.float32)}
        if "image" in names:
            feed["image"] = images[i : i + 1].astype(np.float32)
        if "embedding" in names:
            feed["embedding"] = embeddings[i : i + 1].astype(np.float32)  # type: ignore[index]
        outputs.append(session.run(None, feed)[0])

    return np.concatenate(outputs) if outputs else np.zeros_like(stock)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from . import dataset as ds, labels as lbl

    parser = argparse.ArgumentParser(
        description="Score a personal adapter (or stock alone) on recorded sessions.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset root containing sessions")
    parser.add_argument("--onnx", type=Path, default=None,
                        help="Exported adapter to score. Omitted: report stock behaviour only.")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="Session ids to score. Default: every session under --data.")
    parser.add_argument("--json", type=Path, default=None, help="Also write the metrics as JSON")
    parser.add_argument("--threshold", type=float, default=ACTIVATION_THRESHOLD)
    parser.add_argument("--no-speech-pseudo-labels", action="store_true")
    args = parser.parse_args(argv)

    sessions = ds.discover_sessions(args.data)
    if args.sessions:
        wanted = set(args.sessions)
        sessions = [s for s in sessions if s.session_id in wanted]
        missing = wanted - {s.session_id for s in sessions}
        if missing:
            raise SystemExit(f"Unknown session ids: {sorted(missing)}")
    if not sessions:
        raise SystemExit("No sessions selected.")

    print(ds.describe(sessions))

    label_set = lbl.build_labels(
        sessions, use_speech_pseudo_labels=not args.no_speech_pseudo_labels)
    stock = label_set.stock

    if args.onnx:
        import onnxruntime as ort

        needs_image = "image" in {i.name for i in ort.InferenceSession(
            str(args.onnx), providers=["CPUExecutionProvider"]).get_inputs()}
        images = ds.load_images(sessions) if needs_image else np.zeros((len(stock), 1, 1, 1), np.float32)
        embeddings = ds.load_embeddings(sessions) if hasattr(ds, "load_embeddings") else None
        personal = run_onnx(args.onnx, images, stock, embeddings)
        print(f"\nScored: {args.onnx}")
    else:
        # No model: personal == stock, so every "improvement" is zero and the report is a pure
        # description of stock behaviour. Useful as the baseline half of a comparison.
        personal = stock.copy()
        print("\nNo --onnx given: reporting stock behaviour only.")

    expression_reports = per_expression_mae(
        stock, personal, label_set.targets, label_set.weights)

    neutral = None
    neutral_sessions = [s for s in sessions if s.is_neutral]
    if neutral_sessions:
        rows, offset = [], 0
        for session in sessions:
            if session.is_neutral:
                rows.append(np.arange(offset, offset + len(session.frames)))
            offset += len(session.frames)
        picked = np.concatenate(rows)
        neutral = neutral_report(stock[picked], personal[picked], threshold=args.threshold)

    cued_rows, cued_dims = collect_cued_dims(sessions)
    ct_stock = ct_personal = None
    if len(cued_rows):
        ct_stock = cross_talk(stock[cued_rows], cued_dims, threshold=args.threshold)
        ct_personal = cross_talk(personal[cued_rows], cued_dims, threshold=args.threshold)

    reports = build_dim_reports(sessions, label_set, stock, personal, threshold=args.threshold)

    text = format_report(expression_reports, neutral, dim_reports=reports,
                         cross_talk_stock=ct_stock, cross_talk_personal=ct_personal)
    print()
    print(text)

    if args.json:
        payload = {
            "sessions": [s.session_id for s in sessions],
            "model": str(args.onnx) if args.onnx else None,
            "threshold": args.threshold,
            "watched": [dim_report_to_dict(r) for r in reports],
            "cross_talk": ({"stock": ct_stock, "personal": ct_personal}
                           if ct_stock is not None else None),
        }
        if neutral is not None:
            payload["neutral"] = {
                "frames": neutral.frames,
                "stock_false_activation_rate": neutral.stock_false_activation_rate,
                "personal_false_activation_rate": neutral.personal_false_activation_rate,
                "stock_jitter": neutral.stock_jitter,
                "personal_jitter": neutral.personal_jitter,
            }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
