"""Personal adapter models.

Both models predict a **residual** on top of the stock prediction::

    personal = clip(stock + f(...), 0, 1)

Residual form matters more than the specific architecture. Combined with the shrinkage penalty in
the loss it makes "change nothing" the cheapest answer, so the adapter only moves an expression
where the data actually argues for it. That is what keeps a model trained on a few thousand frames
from wrecking the expressions it has no evidence about.

Two models, deliberately in this order:

* :class:`OutputMlpAdapter` (baseline A) sees only the stock vector. It answers "how much of this is
  fixable from relationships between the stock outputs alone?" - systematic cross-talk, ranges,
  left/right confusion.
* :class:`ImageResidualAdapter` (model B) also sees the camera frame, so it can fix errors that need
  visual information the stock outputs no longer contain.

If B does not beat A by a worthwhile margin on held-out sessions, ship A: it is smaller, faster and
has nothing to overfit to.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import schema

N = schema.EXPRESSION_COUNT


class OutputMlpAdapter(nn.Module):
    """Baseline A: stock[45] -> residual[45]. ~29k parameters.

    Accepts an image argument and ignores it, so the exported graph has the same signature as model
    B and the C# runtime path is identical for both.
    """

    adapter_type = "output_mlp_v1"
    uses_image = False

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, N),
        )
        # Start as an exact identity: the untrained adapter must be a no-op, not noise.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Permanently zero, and never trained. See residual() for why it exists.
        self.register_buffer("image_gate", torch.zeros(1))

    def residual(self, image: torch.Tensor, stock: torch.Tensor) -> torch.Tensor:
        # This model ignores the image, but the exported graph must still declare an `image` input:
        # torch.onnx.export prunes inputs nothing consumes, which would give baseline A a different
        # signature from the image-conditioned model and force the C# runtime to branch per model
        # type. Touching a single pixel through a zero buffer keeps the input live at O(1) cost and
        # contributes exactly 0.0 to the result.
        keep_input_alive = image[:, 0, 0, 0].unsqueeze(1) * self.image_gate
        return self.net(stock) + keep_input_alive

    def forward(self, image: torch.Tensor, stock: torch.Tensor) -> torch.Tensor:
        return torch.clamp(stock + self.residual(image, stock), 0.0, 1.0)


class ImageResidualAdapter(nn.Module):
    """Model B: (image[1,224,224], stock[45]) -> residual[45]. ~48k parameters.

    The 224 input is average-pooled to 112 **inside the graph**, so C# can feed the exact tensor the
    stock model already consumed with no extra preprocessing, while the convolutions run at a
    quarter of the cost.
    """

    adapter_type = "image_residual_v1"
    uses_image = True

    def __init__(self, hidden: int = 128, embed: int = 64):
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
            )

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)  # 224 -> 112
        self.trunk = nn.Sequential(
            block(1, 8),    # 112 -> 56
            block(8, 16),   # 56  -> 28
            block(16, 32),  # 28  -> 14
            block(32, embed),  # 14 -> 7
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed + N, hidden),
            nn.SiLU(),
            nn.Linear(hidden, N),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def residual(self, image: torch.Tensor, stock: torch.Tensor) -> torch.Tensor:
        features = self.trunk(self.downsample(image))
        return self.head(torch.cat([features, stock], dim=1))

    def forward(self, image: torch.Tensor, stock: torch.Tensor) -> torch.Tensor:
        return torch.clamp(stock + self.residual(image, stock), 0.0, 1.0)


def build_model(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind in ("a", "output", "output_mlp"):
        return OutputMlpAdapter()
    if kind in ("b", "image", "image_residual"):
        return ImageResidualAdapter()
    raise ValueError(f"Unknown model kind '{kind}'. Use 'a' (output-only) or 'b' (image-conditioned).")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def consistency_penalty(residual: torch.Tensor, residual_augmented: torch.Tensor) -> torch.Tensor:
    """How much the residual moves when only the image's *appearance* changed.

    Zero is the goal: the correction the adapter applies should depend on the shape of the face, not
    on how bright the room is. Measured on real data, an unregularized model B read illumination as
    expression (brightness/JawOpen correlation -0.56 on a held-out neutral session), which is the
    "my mouth looks slightly open sometimes" failure exactly.

    Squared rather than absolute, so the rare large disagreements - the visible glitches - are what
    gets punished, not the constant small ones.
    """
    return (residual - residual_augmented).pow(2).mean()


def temporal_penalty(residual: torch.Tensor, previous_residual: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
    """How much the residual jumps between consecutive frames of the same session.

    This penalizes jitter in the *correction*, never in the output. The adapter's job is to undo a
    per-user bias, which changes slowly if at all; fast movement should keep coming from the stock
    model. So this can be pushed fairly hard without making the face feel laggy - it constrains the
    thing that should be smooth and leaves the thing that should be quick alone.

    ``valid`` masks out the first frame of each session, which has no predecessor.
    """
    if valid.sum() == 0:
        return residual.new_zeros(())

    delta = (residual - previous_residual).pow(2).mean(dim=1)
    return (delta * valid).sum() / valid.sum().clamp_min(1.0)


def masked_residual_loss(
    predicted: torch.Tensor,
    residual: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    shrinkage: float = 1e-2,
    huber_delta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted Huber fit plus residual shrinkage.

    The mask is what lets a frame supervise three expressions and stay silent about the other
    forty-two. Shrinkage is what makes that silence mean "leave stock alone" instead of "anything
    goes": without it the adapter is free to drift wildly wherever no label objects, which is
    exactly how a small personal model destroys expressions it was never taught.
    """
    per_cell = nn.functional.huber_loss(predicted, targets, reduction="none", delta=huber_delta)
    weighted = per_cell * weights
    denominator = weights.sum().clamp_min(1.0)

    fit = weighted.sum() / denominator
    shrink = residual.pow(2).mean()
    total = fit + shrinkage * shrink

    return total, {
        "loss": float(total.detach()),
        "fit": float(fit.detach()),
        "shrink": float(shrink.detach()),
    }
