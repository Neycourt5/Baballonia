"""Expose the stock face model's internal visual embedding as a second graph output.

    python -m babble_personal.derive_embedding --stock src/Baballonia/faceModel.onnx

Why this exists
---------------
The stock model is an EfficientNet-B0-class network that ends in
``conv_head -> GlobalAveragePool -> Flatten -> Gemm[45x1280]``. That flatten output is a 1280-d
description of the face, learned from a corpus far larger than any personal dataset, and it is
already computed on every frame - the final layer is a single matrix multiply away from it.

Model B has to learn visual features from scratch from a few thousand of the user's own frames.
Model C reuses these instead, and only has to learn how to *interpret* them for this face. The
appeal is not elegance, it is that the features are free: no second forward pass, no extra weights,
nothing to overfit.

What this script does
---------------------
Adds one output to a *copy* of the graph. No weight is touched, no node is added, and the stock
45-value output must come out bit-identical - which is asserted here rather than assumed, because
silently changing what the stock model predicts would invalidate every recording ever made.

The stock file itself is never modified. It ships with the app, it is the fallback the whole design
depends on, and a derived model that disagrees with it is worse than no derived model at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

#: The flatten immediately before the classifier Gemm. Named rather than discovered so a graph that
#: does not match this architecture fails loudly instead of exporting some other tensor.
DEFAULT_EMBEDDING_TENSOR = "/model/global_pool/flatten/Flatten_output_0"

#: Name the derived output carries. The C# runtime and the trainer both look for exactly this.
EMBEDDING_OUTPUT_NAME = "embedding"

EMBEDDING_DIM = 1280

TOOL_VERSION = 1


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(model_path: Path) -> Path:
    """Where the provenance record lives - beside the model, same stem."""
    return model_path.with_suffix(".json")


def derive(
    stock_path: Path,
    out_path: Path,
    *,
    tensor_name: str = DEFAULT_EMBEDDING_TENSOR,
    parity_samples: int = 8,
) -> dict:
    """Write a copy of ``stock_path`` that also outputs the visual embedding.

    Returns the provenance record, which is also written as a sidecar so the runtime can check
    freshness without opening an inference session.
    """
    stock_path = Path(stock_path)
    out_path = Path(out_path)

    if not stock_path.exists():
        raise FileNotFoundError(f"Stock model not found: {stock_path}")

    if out_path.resolve() == stock_path.resolve():
        raise ValueError(
            "Refusing to overwrite the stock model. The derived graph must be a separate file - "
            "the stock one is the fallback everything else depends on."
        )

    model = onnx.load(str(stock_path))
    graph = model.graph

    produced = {output for node in graph.node for output in node.output}
    if tensor_name not in produced:
        raise ValueError(
            f"{stock_path.name} has no tensor named '{tensor_name}'.\n"
            "The stock face model architecture has changed, so this script's assumption about where "
            "the visual embedding lives no longer holds. Re-inspect the graph before trusting any "
            "derived model."
        )

    if any(output.name == EMBEDDING_OUTPUT_NAME for output in graph.output):
        raise ValueError(f"{stock_path.name} already exposes an '{EMBEDDING_OUTPUT_NAME}' output.")

    if EMBEDDING_OUTPUT_NAME in produced:
        raise ValueError(
            f"{stock_path.name} already has a tensor called '{EMBEDDING_OUTPUT_NAME}'; "
            "the tap would collide with it.")

    original_outputs = [output.name for output in graph.output]

    # Exposed through an Identity node rather than by naming the internal tensor directly, so the
    # output has a stable name of our choosing. The internal one is an artefact of how PyTorch
    # happened to export the graph ("/model/global_pool/flatten/Flatten_output_0"); making the C#
    # runtime depend on that string would tie it to an export detail that a future upstream model
    # could change for no reason. Identity is a copy at worst and usually folded away entirely.
    #
    # It also leaves the original tensor untouched, so the classifier Gemm still consumes exactly
    # what it consumed before - which is what keeps the stock output bit-identical.
    graph.node.append(helper.make_node(
        "Identity",
        inputs=[tensor_name],
        outputs=[EMBEDDING_OUTPUT_NAME],
        name="personal_embedding_tap",
    ))

    # Appended, never inserted: the existing output keeps index 0 so any reader that indexes
    # positionally still finds the expressions where it expects them.
    graph.output.append(
        helper.make_tensor_value_info(EMBEDDING_OUTPUT_NAME, TensorProto.FLOAT, [1, EMBEDDING_DIM]))

    onnx.checker.check_model(model)

    base_md5 = file_md5(stock_path)
    record = {
        "tool_version": TOOL_VERSION,
        "base_model": stock_path.name,
        "base_model_md5": base_md5,
        "embedding_source_tensor": tensor_name,
        "embedding_output_name": EMBEDDING_OUTPUT_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "stock_output_name": original_outputs[0] if original_outputs else "",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))

    verify_parity(stock_path, out_path, samples=parity_samples)

    sidecar_path(out_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def verify_parity(stock_path: Path, derived_path: Path, *, samples: int = 8) -> float:
    """Assert the derived graph predicts exactly what the stock one does.

    Bit-identical is the bar, not "close enough". Adding a graph output changes no computation, so
    any difference at all means something else moved - and a personal model trained against drifted
    stock outputs is silently wrong in a way no metric would surface.

    Returns the maximum absolute difference observed (expected: 0.0).
    """
    import onnxruntime as ort

    stock = ort.InferenceSession(str(stock_path), providers=["CPUExecutionProvider"])
    derived = ort.InferenceSession(str(derived_path), providers=["CPUExecutionProvider"])

    input_name = stock.get_inputs()[0].name
    shape = [d if isinstance(d, int) else 1 for d in stock.get_inputs()[0].shape]

    output_names = [o.name for o in derived.get_outputs()]
    if EMBEDDING_OUTPUT_NAME not in output_names and DEFAULT_EMBEDDING_TENSOR not in output_names:
        raise RuntimeError(f"Derived model does not expose an embedding output: {output_names}")

    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(samples):
        sample = rng.random(shape, dtype=np.float32)

        expected = stock.run(None, {input_name: sample})[0]
        actual = derived.run(None, {input_name: sample})

        worst = max(worst, float(np.abs(actual[0] - expected).max()))

        embedding = actual[1]
        if embedding.shape[-1] != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding has {embedding.shape[-1]} values, expected {EMBEDDING_DIM}")
        if not np.isfinite(embedding).all():
            raise RuntimeError("Embedding contains non-finite values")

    if worst != 0.0:
        raise RuntimeError(
            f"Derived model changed the stock prediction (max |diff| {worst:.3e}).\n"
            "Adding an output must not alter any computation. Do not ship this model."
        )

    return worst


def is_current(derived_path: Path, stock_path: Path) -> bool:
    """Whether a derived model was built from exactly this stock file.

    Cheap enough to call before every load: it hashes the stock model and compares against the
    sidecar, with no ONNX session involved. An embedding taken from a mismatched base model would be
    describing a different feature space than the personal head was trained on - which produces
    plausible-looking nonsense rather than an error, so it must never be allowed to happen quietly.
    """
    derived_path, stock_path = Path(derived_path), Path(stock_path)
    sidecar = sidecar_path(derived_path)

    if not (derived_path.exists() and sidecar.exists() and stock_path.exists()):
        return False

    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    return record.get("base_model_md5") == file_md5(stock_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expose the stock face model's 1280-d visual embedding as a second output.")
    parser.add_argument("--stock", type=Path, required=True, help="Path to the stock faceModel.onnx")
    parser.add_argument("--out", type=Path, default=None,
                        help="Destination (default: faceModelWithEmbedding.onnx beside --stock)")
    parser.add_argument("--tensor", default=DEFAULT_EMBEDDING_TENSOR,
                        help="Graph tensor to expose")
    parser.add_argument("--parity-samples", type=int, default=8)
    args = parser.parse_args(argv)

    out_path = args.out or args.stock.parent / "faceModelWithEmbedding.onnx"

    print(f"Stock model:   {args.stock}")
    print(f"Derived model: {out_path}")

    record = derive(args.stock, out_path, tensor_name=args.tensor,
                    parity_samples=args.parity_samples)

    print("\nStock output verified bit-identical over "
          f"{args.parity_samples} random inputs.")
    print(f"Embedding:     {record['embedding_dim']}-d from {record['embedding_source_tensor']}")
    print(f"Base MD5:      {record['base_model_md5']}")
    print(f"Sidecar:       {sidecar_path(out_path)}")
    print("\nNext: python -m babble_personal.compute_embeddings --data <dataset root> "
          f"--model \"{out_path}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
