"""Export a trained adapter to ONNX, with the metadata the C# runtime validates.

    python -m babble_personal.export --checkpoint runs/.../model.pt

The exported graph takes the tensor Baballonia already has in hand and returns the corrected vector::

    image [1,1,224,224] float32   (grayscale /255 - exactly the stock model's input)
    stock [1,45]        float32   (raw pre-filter stock output)
        -> personal [1,45] float32 = clip(stock + residual, 0, 1)

Clipping lives in the graph so the runtime cannot forget it. Blending between stock and personal
stays in C#, where it is an evaluation control rather than part of the model.

Metadata is not decoration. ``expression_schema_sha256`` is what stops a model trained against one
expression ordering from being silently applied under another, which would map every learned
correction onto the wrong expression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import torch

from . import models, schema

PERSONAL_ADAPTER_VERSION = 1

#: Models A and B take (image, stock); model C takes (stock, embedding) and is useless without a
#: runtime that can supply the stock network's internal features. The version is what makes an
#: older build refuse it cleanly instead of failing at the first inference.
EMBEDDING_ADAPTER_VERSION = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a personal adapter to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="model.pt from train.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output .onnx path (default: personalFaceModel.onnx beside the checkpoint)")
    parser.add_argument("--base-model", type=Path, default=None,
                        help="Optional faceModel.onnx, recorded as base_model_md5 for provenance")
    parser.add_argument("--roi", type=str, default=None,
                        help="Optional JSON of the camera/ROI settings the data was recorded with")
    parser.add_argument("--parity-samples", type=int, default=32)
    parser.add_argument("--parity-tolerance", type=float, default=1e-4)
    return parser.parse_args(argv)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(
    checkpoint_path: Path,
    output_path: Path,
    *,
    base_model: Path | None = None,
    roi: str | None = None,
    parity_samples: int = 32,
    parity_tolerance: float = 1e-4,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    adapter_type = checkpoint.get("adapter_type", "output_mlp_v1")

    if adapter_type.startswith("embedding"):
        kind = "c"
    elif adapter_type.startswith("image"):
        kind = "b"
    else:
        kind = "a"

    model = models.build_model(kind)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    size = schema.IMAGE_SIZE
    dummy_image = torch.zeros((1, 1, size, size), dtype=torch.float32)
    dummy_stock = torch.zeros((1, schema.EXPRESSION_COUNT), dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_model = models.uses_embedding(model)

    if embedding_model:
        # A different graph signature, and therefore a different adapter version. Model C consumes
        # the stock network's internal features instead of pixels, so it has no `image` input at
        # all - and an older runtime that cannot supply an embedding must refuse it outright rather
        # than feed it something plausible. Bumping the version is what makes that refusal happen.
        dummy_embedding = torch.zeros((1, model.embedding_dim), dtype=torch.float32)
        torch.onnx.export(
            model,
            (dummy_stock, dummy_embedding),
            str(output_path),
            input_names=["stock", "embedding"],
            output_names=["personal"],
            opset_version=17,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            model,
            (dummy_image, dummy_stock),
            str(output_path),
            input_names=["image", "stock"],
            output_names=["personal"],
            opset_version=17,
            dynamo=False,
        )

    metadata = {
        "personal_adapter_version": str(
            EMBEDDING_ADAPTER_VERSION if embedding_model else PERSONAL_ADAPTER_VERSION),
        "adapter_type": adapter_type,
        "expression_names": json.dumps(list(schema.EXPRESSION_NAMES)),
        "expression_schema_sha256": schema.SCHEMA_SHA256,
        "expression_schema_version": str(schema.SCHEMA_VERSION),
        "input_normalization": schema.INPUT_NORMALIZATION,
        "input_size": str(size),
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": str(models.parameter_count(model)),
    }
    if embedding_model:
        metadata["requires_embedding"] = "1"
        metadata["embedding_dim"] = str(model.embedding_dim)
    if base_model is not None and base_model.exists():
        metadata["base_model_md5"] = _md5(base_model)
    if roi:
        metadata["roi_settings"] = roi

    proto = onnx.load(str(output_path))
    for key, value in metadata.items():
        entry = proto.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(proto, str(output_path))

    _verify_parity(model, output_path, parity_samples, parity_tolerance)
    return output_path


def _verify_parity(model, onnx_path: Path, samples: int, tolerance: float) -> None:
    """Confirm ONNX Runtime reproduces PyTorch before anything trusts this file.

    A silent export discrepancy would show up as "the personal model behaves differently in the app
    than in training", which is painful to diagnose later and trivial to catch here.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    size = schema.IMAGE_SIZE
    worst = 0.0
    embedding_model = models.uses_embedding(model)

    for _ in range(samples):
        stock = rng.random((1, schema.EXPRESSION_COUNT), dtype=np.float32)

        if embedding_model:
            # Standard-normal rather than uniform: the real embedding is a post-activation feature
            # vector spanning both signs, and testing parity only on positive inputs would leave
            # half the LayerNorm's behaviour unexercised.
            embedding = rng.standard_normal((1, model.embedding_dim)).astype(np.float32)
            with torch.no_grad():
                expected = model(torch.from_numpy(stock), torch.from_numpy(embedding)).numpy()
            actual = session.run(["personal"], {"stock": stock, "embedding": embedding})[0]
        else:
            image = rng.random((1, 1, size, size), dtype=np.float32)
            with torch.no_grad():
                expected = model(torch.from_numpy(image), torch.from_numpy(stock)).numpy()
            actual = session.run(["personal"], {"image": image, "stock": stock})[0]

        worst = max(worst, float(np.abs(expected - actual).max()))

    if worst > tolerance:
        raise RuntimeError(
            f"ONNX export does not match PyTorch (max abs diff {worst:.2e} > {tolerance:.0e}). "
            "Do not ship this model."
        )

    print(f"  torch/ORT parity OK (max abs diff {worst:.2e} over {samples} samples)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.out or (args.checkpoint.parent / "personalFaceModel.onnx")

    print(f"Exporting {args.checkpoint} -> {output}")
    export(
        args.checkpoint,
        output,
        base_model=args.base_model,
        roi=args.roi,
        parity_samples=args.parity_samples,
        parity_tolerance=args.parity_tolerance,
    )

    print(f"  schema sha256: {schema.SCHEMA_SHA256}")
    print(f"\nInstall by copying to:\n  %APPDATA%\\ProjectBabble\\Models\\personalFaceModel.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
