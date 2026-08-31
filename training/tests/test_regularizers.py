"""Tests for the two regularizers added to fix "at rest my jaw wiggles and my mouth looks open".

Both are constraints on the *residual*, never on the output, which is the property that lets them be
pushed hard without making the face feel laggy. These tests pin that distinction down, because it is
the thing that would be easy to break later while every headline metric still looked fine.

    python -m pytest training/tests/test_regularizers.py -q
    python training/tests/test_regularizers.py            (no pytest required)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babble_personal import augment, evaluate, models, schema  # noqa: E402

N = schema.EXPRESSION_COUNT
SIZE = schema.IMAGE_SIZE


def _images(n: int = 4, value: float = 0.5) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand((n, 1, SIZE, SIZE)) * 0.2 + value


def test_jitter_stays_in_range_and_actually_changes_the_image() -> None:
    images = _images()
    out = augment.photometric_jitter(images, generator=torch.Generator().manual_seed(1))

    assert out.shape == images.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, "images must stay in [0,1]"
    assert not torch.allclose(out, images), "augmentation did nothing"


def test_jitter_preserves_geometry() -> None:
    """Only appearance may change. Moving the mouth would break correspondence with the stock vector.

    Brightness, contrast and gamma are all monotonic per-pixel maps, so a ramp stays a ramp and
    nothing moves. Sensor noise is excluded because it is deliberately not monotonic - reordering
    near-equal pixels is its whole purpose - and the pixel values are well separated so that float
    rounding cannot collapse two of them into a tie.
    """
    ramp = torch.linspace(0.05, 0.95, 64).reshape(1, 1, 8, 8)
    out = augment.photometric_jitter(ramp, generator=torch.Generator().manual_seed(2), noise=0.0)

    flat = out.flatten()
    assert torch.all(flat[1:] >= flat[:-1]),         "an ascending ramp came back out of order; augmentation must only re-light, never move"
    assert not torch.allclose(out, ramp), "the re-lighting did nothing"


def test_jitter_is_reproducible_from_a_seed() -> None:
    images = _images()
    a = augment.photometric_jitter(images, generator=torch.Generator().manual_seed(7))
    b = augment.photometric_jitter(images, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(a, b), "same seed must give the same augmentation"


def test_consistency_penalty_is_zero_when_the_model_ignores_appearance() -> None:
    """Model A cannot see the image, so it is illumination-invariant by construction."""
    model = models.build_model("a")
    images = _images()
    stock = torch.rand((len(images), N))

    residual = model.residual(images, stock)
    jittered = model.residual(augment.photometric_jitter(images, generator=torch.Generator().manual_seed(3)), stock)

    assert float(models.consistency_penalty(residual, jittered)) == 0.0


def test_consistency_penalty_is_positive_for_the_image_model() -> None:
    """Model B does see pixels, so an untrained one reacts to lighting - that is what we penalize."""
    model = models.build_model("b")
    # A zero-initialized output layer would make every residual identical, hiding the effect.
    torch.nn.init.normal_(model.head[-1].weight, std=0.1)

    images = _images()
    stock = torch.rand((len(images), N))
    jittered = augment.photometric_jitter(images, generator=torch.Generator().manual_seed(4))

    penalty = float(models.consistency_penalty(model.residual(images, stock),
                                               model.residual(jittered, stock)))
    assert penalty > 0.0, "an image-conditioned model should react to a re-lit frame"


def test_temporal_penalty_scores_only_valid_pairs() -> None:
    residual = torch.tensor([[0.0] * N, [1.0] * N, [0.0] * N])
    previous = torch.tensor([[0.0] * N, [0.0] * N, [0.0] * N])

    none_valid = models.temporal_penalty(residual, previous, torch.tensor([0.0, 0.0, 0.0]))
    assert float(none_valid) == 0.0, "with no valid predecessors the penalty must vanish"

    # Only row 1 differs, so masking it out must drop the penalty to zero.
    without_jump = models.temporal_penalty(residual, previous, torch.tensor([1.0, 0.0, 1.0]))
    assert float(without_jump) == 0.0

    with_jump = models.temporal_penalty(residual, previous, torch.tensor([0.0, 1.0, 0.0]))
    assert float(with_jump) > 0.9, "a full-scale jump should be penalized"


def test_temporal_penalty_does_not_punish_output_movement_only_residual_movement() -> None:
    """The key property: a fast-moving face is free, a jumpy *correction* is not.

    Two frames where the stock prediction moves a lot but the correction is identical must cost
    nothing - otherwise the penalty would blunt real expression speed.
    """
    steady_residual = torch.tensor([[0.2] * N, [0.2] * N])
    penalty = models.temporal_penalty(steady_residual, steady_residual, torch.tensor([1.0, 1.0]))
    assert float(penalty) == 0.0, "identical residuals must cost nothing however fast stock moved"


def test_resting_jitter_measures_frame_to_frame_movement() -> None:
    still = np.zeros((100, N), dtype=np.float32)
    assert evaluate.resting_jitter(still) == 0.0

    # Alternating +-0.1 on one dim: mean |diff| is 0.2 on that dim, spread over N dims.
    shivering = np.zeros((100, N), dtype=np.float32)
    shivering[1::2, 0] = 0.2
    jitter = evaluate.resting_jitter(shivering)
    assert jitter > 0.0
    assert abs(jitter - 0.2 / N) < 1e-6, f"expected {0.2 / N}, got {jitter}"

    assert evaluate.resting_jitter(np.zeros((1, N), dtype=np.float32)) == 0.0


def test_neutral_report_carries_jitter_for_both_models() -> None:
    stock = np.zeros((50, N), dtype=np.float32)
    stock[1::2, 0] = 0.4          # stock shivers
    personal = np.zeros((50, N), dtype=np.float32)  # personal is steady

    report = evaluate.neutral_report(stock, personal)
    assert report.stock_jitter > report.personal_jitter
    assert "steadier" not in evaluate.format_report([], report)  # label lives on the jitter line
    assert "resting jitter" in evaluate.format_report([], report)


def main() -> int:
    tests = [
        test_jitter_stays_in_range_and_actually_changes_the_image,
        test_jitter_preserves_geometry,
        test_jitter_is_reproducible_from_a_seed,
        test_consistency_penalty_is_zero_when_the_model_ignores_appearance,
        test_consistency_penalty_is_positive_for_the_image_model,
        test_temporal_penalty_scores_only_valid_pairs,
        test_temporal_penalty_does_not_punish_output_movement_only_residual_movement,
        test_resting_jitter_measures_frame_to_frame_movement,
        test_neutral_report_carries_jitter_for_both_models,
    ]

    failures = 0
    for test in tests:
        print(f"\n--- {test.__name__}")
        try:
            test()
            print("  PASS")
        except Exception as exc:  # noqa: BLE001 - standalone runner reports rather than raises
            failures += 1
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
