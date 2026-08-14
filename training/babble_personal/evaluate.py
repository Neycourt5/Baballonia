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

import numpy as np

from . import schema

#: Above this, an expression is "active" for false-activation accounting. Matches the plan.
ACTIVATION_THRESHOLD = 0.15


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


def format_report(
    expression_reports: list[ExpressionReport],
    neutral: NeutralReport | None = None,
    *,
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

    return "\n".join(lines)
