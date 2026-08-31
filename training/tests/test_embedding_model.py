"""Tests for model C: the adapter that reuses the stock network's own visual features.

The idea is only worth anything if two things hold, and both are asserted here against the real
faceModel.onnx rather than a mock.

The first is that exposing the embedding does not change what the stock model predicts. Adding a
graph output computes nothing new, so the 45 values must come out *bit-identical* - not merely
close. If they drifted, every recording ever made would describe a slightly different model than the
one now running, and no metric in this project would show it.

The second is that a model C file cannot be loaded by a runtime that has no embedding to give it.
It takes (stock, embedding) instead of (image, stock), and a runtime that fed it something else
would get plausible-looking nonsense rather than an error. The adapter version is what makes that
refusal happen, so it is pinned.

    python -m pytest training/tests/test_embedding_model.py -q
    python training/tests/test_embedding_model.py          (no pytest required)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from babble_personal import compute_embeddings as ce  # noqa: E402
from babble_personal import derive_embedding as de  # noqa: E402
from babble_personal import export, models, schema  # noqa: E402

N = schema.EXPRESSION_COUNT

#: The stock model ships in the repo, so these tests exercise the real graph.
STOCK_MODEL = Path(__file__).resolve().parents[2] / "src" / "Baballonia" / "faceModel.onnx"


def _skip_without_stock_model() -> bool:
    if not STOCK_MODEL.exists():
        print(f"  (skipped: {STOCK_MODEL} not present)")
        return True
    return False


# ---------------------------------------------------------------------------
# Deriving the embedding graph
# ---------------------------------------------------------------------------


def test_derived_model_keeps_the_stock_prediction_bit_identical() -> None:
    """The assertion the whole model-C idea rests on."""
    if _skip_without_stock_model():
        return

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "faceModelWithEmbedding.onnx"
        de.derive(STOCK_MODEL, out, parity_samples=4)

        # derive() asserts parity internally; re-run it explicitly so the guarantee is visible here.
        worst = de.verify_parity(STOCK_MODEL, out, samples=4)

        assert worst == 0.0, f"stock output drifted by {worst:.3e}; it must be exactly unchanged"


def test_derived_model_exposes_a_1280_embedding() -> None:
    if _skip_without_stock_model():
        return

    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "derived.onnx"
        record = de.derive(STOCK_MODEL, out, parity_samples=2)

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        outputs = session.get_outputs()

        assert len(outputs) == 2, "derived graph should have the expressions plus the embedding"
        assert record["embedding_dim"] == 1280

        sample = np.random.default_rng(0).random((1, 1, 224, 224), dtype=np.float32)
        expressions, embedding = session.run(None, {session.get_inputs()[0].name: sample})

        assert expressions.shape == (1, N)
        assert embedding.shape == (1, 1280)
        assert np.isfinite(embedding).all()


def test_expressions_are_recoverable_from_the_embedding() -> None:
    """Confirms the exposed tensor really is the input to the final classifier, not some other layer.

    If a future stock model moved this tensor, the shape check alone would still pass while the
    features meant something entirely different.
    """
    if _skip_without_stock_model():
        return

    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "derived.onnx"
        de.derive(STOCK_MODEL, out, parity_samples=1)

        model = onnx.load(str(STOCK_MODEL))
        weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        classifier_w = weights["model.classifier.weight"]
        classifier_b = weights["model.classifier.bias"]

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        sample = np.random.default_rng(1).random((1, 1, 224, 224), dtype=np.float32)
        expressions, embedding = session.run(None, {session.get_inputs()[0].name: sample})

        recomputed = embedding @ classifier_w.T + classifier_b

        assert np.abs(recomputed - expressions).max() < 1e-4, (
            "the exposed tensor is not the classifier's input"
        )


def test_deriving_refuses_to_overwrite_the_stock_model() -> None:
    """The stock file is the fallback everything depends on."""
    if _skip_without_stock_model():
        return

    try:
        de.derive(STOCK_MODEL, STOCK_MODEL)
    except ValueError as exc:
        assert "stock" in str(exc).lower()
    else:
        raise AssertionError("expected a refusal to overwrite the stock model")


def test_missing_tensor_fails_loudly() -> None:
    if _skip_without_stock_model():
        return

    with tempfile.TemporaryDirectory() as tmp:
        try:
            de.derive(STOCK_MODEL, Path(tmp) / "x.onnx", tensor_name="/not/a/real/tensor")
        except ValueError as exc:
            assert "no tensor named" in str(exc)
        else:
            raise AssertionError("a missing embedding tensor must fail, not export something else")


def test_staleness_is_detected_from_the_sidecar() -> None:
    """A derived model built from a different stock file must never be used silently."""
    if _skip_without_stock_model():
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        derived = tmp_path / "derived.onnx"
        de.derive(STOCK_MODEL, derived, parity_samples=1)

        assert de.is_current(derived, STOCK_MODEL)

        # Pretend the stock model was replaced by an upstream update.
        impostor = tmp_path / "other.onnx"
        impostor.write_bytes(STOCK_MODEL.read_bytes() + b"\x00")

        assert not de.is_current(derived, impostor), (
            "an embedding from a different base model describes a different feature space"
        )

        de.sidecar_path(derived).unlink()
        assert not de.is_current(derived, STOCK_MODEL), "no sidecar means no provenance, so no trust"


# ---------------------------------------------------------------------------
# Embedding storage
# ---------------------------------------------------------------------------


def test_embeddings_round_trip_through_float16() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        session_path = Path(tmp)
        values = np.random.default_rng(0).standard_normal((25, 1280)).astype(np.float32)

        ce.write_embeddings(session_path, values, source="backfill", model_md5="abc")
        loaded = ce.load_embeddings(session_path, expected_frames=25)

        assert loaded is not None
        assert loaded.shape == (25, 1280)
        assert loaded.dtype == np.float32
        # float16 keeps ~3 decimal digits, which is far more than a LayerNorm'd feature needs.
        assert np.abs(loaded - values).max() < 0.01

        meta = json.loads((session_path / ce.EMBEDDINGS_META).read_text(encoding="utf-8"))
        assert meta["source"] == "backfill"
        assert meta["count"] == 25


def test_a_frame_count_mismatch_is_reported_not_silently_used() -> None:
    """A session that grew after its embeddings were computed must not train against stale rows."""
    with tempfile.TemporaryDirectory() as tmp:
        session_path = Path(tmp)
        ce.write_embeddings(session_path, np.zeros((10, 1280), dtype=np.float32), source="backfill")

        assert ce.load_embeddings(session_path, expected_frames=10) is not None
        assert ce.load_embeddings(session_path, expected_frames=12) is None


def test_absent_embeddings_return_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert ce.load_embeddings(Path(tmp)) is None


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def test_untrained_model_c_is_an_exact_passthrough() -> None:
    """Same invariant as A and B: an untrained adapter changes nothing."""
    model = models.build_model("c")
    model.eval()

    stock = torch.rand((8, N))
    embedding = torch.randn((8, 1280))

    with torch.no_grad():
        out = model(stock, embedding)

    assert torch.allclose(out, stock, atol=1e-6)


def test_model_c_without_an_embedding_makes_no_change() -> None:
    """Refusing to guess is the safe failure: no features means no opinion."""
    model = models.build_model("c")

    with torch.no_grad():
        residual = model.residual(None, torch.rand((4, N)), None)

    assert torch.count_nonzero(residual) == 0


def test_model_c_is_declared_as_needing_an_embedding() -> None:
    assert models.uses_embedding(models.build_model("c"))
    assert not models.uses_embedding(models.build_model("a"))
    assert not models.uses_embedding(models.build_model("b"))


def test_model_c_reads_the_embedding() -> None:
    """With a trained head, different features must produce different corrections."""
    model = models.build_model("c")
    torch.manual_seed(0)
    with torch.no_grad():
        model.head[-1].weight.normal_(0, 0.1)

    stock = torch.rand((1, N))
    a = model.residual(None, stock, torch.randn((1, 1280)))
    b = model.residual(None, stock, torch.randn((1, 1280)) * 3)

    assert not torch.allclose(a, b), "the embedding is not reaching the output"


def test_model_c_variants_differ_in_size() -> None:
    full = models.parameter_count(models.build_model("c"))
    small = models.parameter_count(models.build_model("c-small"))

    assert small < full
    assert full < 500_000, "this is a personal adapter, not another face network"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export_model_c(directory: Path) -> Path:
    model = models.build_model("c")
    torch.manual_seed(1)
    with torch.no_grad():
        model.head[-1].weight.normal_(0, 0.05)
        model.head[-1].bias.normal_(0, 0.02)

    checkpoint = directory / "model.pt"
    torch.save({"state_dict": model.state_dict(), "adapter_type": model.adapter_type}, checkpoint)

    out = directory / "personalFaceModel.onnx"
    export.export(checkpoint, out, parity_samples=8)
    return out


def test_model_c_exports_with_the_embedding_signature() -> None:
    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = _export_model_c(Path(tmp))

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs = {i.name for i in session.get_inputs()}
        outputs = {o.name for o in session.get_outputs()}

        assert inputs == {"stock", "embedding"}, f"unexpected inputs: {inputs}"
        assert outputs == {"personal"}
        assert "image" not in inputs, "model C has no image branch at all"


def test_model_c_carries_a_higher_adapter_version() -> None:
    """This is what makes an older runtime refuse it instead of feeding it the wrong inputs."""
    import onnx

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = _export_model_c(Path(tmp))
        meta = {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}

        assert meta["personal_adapter_version"] == "2"
        assert meta["adapter_type"] == "embedding_head_v1"
        assert meta["requires_embedding"] == "1"
        assert meta["embedding_dim"] == "1280"
        assert meta["expression_schema_sha256"] == schema.SCHEMA_SHA256


def test_models_a_and_b_keep_version_one() -> None:
    """Existing models must not be invalidated by model C's arrival."""
    import onnx

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind in ("a", "b"):
            model = models.build_model(kind)
            checkpoint = directory / f"{kind}.pt"
            torch.save({"state_dict": model.state_dict(), "adapter_type": model.adapter_type},
                       checkpoint)

            out = directory / f"{kind}.onnx"
            export.export(checkpoint, out, parity_samples=4)

            meta = {p.key: p.value for p in onnx.load(str(out)).metadata_props}
            assert meta["personal_adapter_version"] == "1", f"model {kind} version changed"
            assert "requires_embedding" not in meta


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
