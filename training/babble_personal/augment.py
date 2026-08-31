"""Photometric augmentation for the image-conditioned adapter.

Model B is the only model that sees pixels, and that is exactly why it needs this. Measured on a
held-out neutral session, the trained adapter's JawOpen output correlated **-0.56** with mean frame
brightness, worse than the stock model's own -0.36: the image branch had learned to read
illumination as if it were expression. With one recording session per lighting condition there is
nothing in the data to tell it otherwise.

The fix is to state the invariance directly. A resting face is a resting face whether the room got
slightly brighter, the IR illuminator drifted, or the sensor gained a little noise - so the adapter's
*residual* must not move when only those things change. That is enforced two ways in ``train``:

* the augmented view is fed alongside the clean one and their residuals are pulled together
  (:func:`models.consistency_penalty`), and
* nothing here touches the stock vector, so the model cannot satisfy the constraint by leaning on
  stock instead - it has to make the image branch itself illumination-invariant.

Magnitudes are deliberately far wider than the within-session brightness spread actually observed
(std 0.0017), because the failure mode is *between* sessions - a shifted headset, a different time
of day - not within one.
"""

from __future__ import annotations

import torch

#: Additive brightness shift, in [0,1] image units.
BRIGHTNESS = 0.05

#: Multiplicative contrast around the frame's own mean.
CONTRAST = 0.10

#: Gamma exponent range, as a +/- fraction of 1.0. Models non-linear sensor/illuminator response.
GAMMA = 0.18

#: Per-pixel Gaussian noise sigma. Roughly the frame-to-frame sensor noise of the recorded camera.
NOISE = 0.008


def photometric_jitter(
    images: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    brightness: float = BRIGHTNESS,
    contrast: float = CONTRAST,
    gamma: float = GAMMA,
    noise: float = NOISE,
) -> torch.Tensor:
    """Return a photometrically perturbed copy of ``images`` (float32 [N,1,H,W] in [0,1]).

    Geometry is left alone on purpose. Shifting or rotating the frame would change where the mouth
    is, and the recorded frame is already the exact post-transform tensor the stock model consumed -
    so a geometric change would break the correspondence with the stock vector we pair it with.
    Only appearance is varied.
    """
    if images.numel() == 0:
        return images

    n = images.shape[0]
    shape = (n, 1, 1, 1)
    device = images.device

    def uniform(low: float, high: float) -> torch.Tensor:
        raw = torch.rand(shape, generator=generator, device=device, dtype=images.dtype)
        return raw * (high - low) + low

    out = images

    if gamma > 0:
        # Clamp away from zero first: 0 ** 0.8 is fine, but the gradient there is not.
        out = out.clamp_min(1e-4) ** uniform(1.0 - gamma, 1.0 + gamma)

    if contrast > 0:
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        out = (out - mean) * uniform(1.0 - contrast, 1.0 + contrast) + mean

    if brightness > 0:
        out = out + uniform(-brightness, brightness)

    if noise > 0:
        out = out + torch.randn(out.shape, generator=generator, device=device, dtype=out.dtype) * noise

    return out.clamp(0.0, 1.0)
