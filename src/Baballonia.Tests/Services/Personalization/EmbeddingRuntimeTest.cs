using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The runtime half of model C: capturing the stock network's visual embedding, and refusing to use
/// one that came from the wrong network.
///
/// The staleness check is the point of <see cref="EmbeddingModelStore"/>. A derived model built from
/// a different stock file still loads, still runs, and still emits 1280 numbers - they just describe
/// a different feature space than the personal head was trained against. There is no exception and
/// no obviously broken output, only a face that is subtly wrong, so the mismatch has to be caught
/// before the model is ever used rather than diagnosed afterwards.
///
/// Model C is off by default and stays that way until the home-PC comparison says it earns its
/// place; these tests cover the plumbing, not the claim that it is better.
/// </summary>
[TestClass]
public class EmbeddingModelStoreTest
{
    private string _directory = null!;

    [TestInitialize]
    public void Initialize()
    {
        _directory = Path.Combine(Path.GetTempPath(), "babble-embed-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_directory);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(_directory))
            Directory.Delete(_directory, recursive: true);
    }

    private string WriteFile(string name, string content)
    {
        var path = Path.Combine(_directory, name);
        File.WriteAllText(path, content);
        return path;
    }

    [TestMethod]
    public void Md5MatchesForIdenticalContent()
    {
        var a = WriteFile("a.onnx", "same bytes");
        var b = WriteFile("b.onnx", "same bytes");
        var c = WriteFile("c.onnx", "other bytes");

        Assert.AreEqual(EmbeddingModelStore.ComputeMd5(a), EmbeddingModelStore.ComputeMd5(b));
        Assert.AreNotEqual(EmbeddingModelStore.ComputeMd5(a), EmbeddingModelStore.ComputeMd5(c));
    }

    [TestMethod]
    public void MissingStockModelIsReportedRatherThanAssumedFine()
    {
        var result = EmbeddingModelStore.TryGetValid(Path.Combine(_directory, "nope.onnx"));

        Assert.IsFalse(result.Valid);
    }
}

/// <summary>
/// The derived model running through the real inference runner, end to end.
/// </summary>
/// <remarks>
/// Skipped unless <c>faceModelWithEmbedding.onnx</c> has been generated next to the stock model.
/// Building it is a Python step (<c>babble_personal.derive_embedding</c>), so this asserts the
/// runtime contract when the artefact is present and stays quiet when it is not, rather than
/// failing on a machine that has simply not built it yet.
/// </remarks>
[TestClass]
public class EmbeddingRunnerTest
{
    private static string StockModelPath =>
        Path.Combine(AppContext.BaseDirectory, "faceModel.onnx");

    private static string DerivedModelPath =>
        Path.Combine(AppContext.BaseDirectory, "faceModelWithEmbedding.onnx");

    private static bool DerivedModelAvailable => File.Exists(DerivedModelPath);

    [TestMethod]
    public void PlainStockModelExposesNoEmbedding()
    {
        // The default path must be completely unaffected by model C existing.
        if (!File.Exists(StockModelPath))
        {
            Assert.Inconclusive("Stock face model not present in the test output.");
            return;
        }

        var runner = new Baballonia.Services.DefaultInferenceRunner(NullLoggerFactory.Instance);
        runner.Setup(StockModelPath, useGpu: false);

        Assert.IsNull(runner.GetEmbedding(), "a runner with no secondary output must report none");

        var result = runner.Run();
        Assert.IsNotNull(result);
        Assert.AreEqual(PersonalizationSchema.ExpressionCount, result.Length);
    }

    [TestMethod]
    public void DerivedModelYieldsBothExpressionsAndEmbedding()
    {
        if (!DerivedModelAvailable)
        {
            Assert.Inconclusive("Run babble_personal.derive_embedding to generate the derived model.");
            return;
        }

        var runner = new Baballonia.Services.DefaultInferenceRunner(NullLoggerFactory.Instance)
        {
            SecondaryOutputName = EmbeddingModelStore.EmbeddingOutputName
        };
        runner.Setup(DerivedModelPath, useGpu: false);

        var expressions = runner.Run();
        var embedding = runner.GetEmbedding();

        Assert.IsNotNull(expressions);
        Assert.AreEqual(PersonalizationSchema.ExpressionCount, expressions.Length);
        Assert.IsNotNull(embedding, "the derived model should have produced an embedding");
        Assert.AreEqual(1280, embedding.Length);
        Assert.IsTrue(embedding.ToArray().All(float.IsFinite));
    }

    [TestMethod]
    public void DerivedModelPredictsExactlyWhatTheStockModelDoes()
    {
        // The property everything else rests on. Adding a graph output computes nothing new, so any
        // difference at all would mean the recordings no longer describe the running model.
        if (!DerivedModelAvailable || !File.Exists(StockModelPath))
        {
            Assert.Inconclusive("Both the stock and derived models are needed for this comparison.");
            return;
        }

        var stock = new Baballonia.Services.DefaultInferenceRunner(NullLoggerFactory.Instance);
        stock.Setup(StockModelPath, useGpu: false);

        var derived = new Baballonia.Services.DefaultInferenceRunner(NullLoggerFactory.Instance)
        {
            SecondaryOutputName = EmbeddingModelStore.EmbeddingOutputName
        };
        derived.Setup(DerivedModelPath, useGpu: false);

        var random = new Random(0);
        for (var i = 0; i < stock.InputTensor.Length; i++)
        {
            var value = (float)random.NextDouble();
            stock.InputTensor.SetValue(i, value);
            derived.InputTensor.SetValue(i, value);
        }

        var expected = stock.Run();
        var actual = derived.Run();

        Assert.IsNotNull(expected);
        Assert.IsNotNull(actual);
        for (var i = 0; i < expected.Length; i++)
            Assert.AreEqual(expected[i], actual[i], 0f,
                $"expression {i} differs between the stock and derived models");
    }

    [TestMethod]
    public void RequestingAnAbsentSecondaryOutputFallsBackQuietly()
    {
        // A derived model replaced by an ordinary one must degrade to stock, not fail to start.
        if (!File.Exists(StockModelPath))
        {
            Assert.Inconclusive("Stock face model not present in the test output.");
            return;
        }

        var runner = new Baballonia.Services.DefaultInferenceRunner(NullLoggerFactory.Instance)
        {
            SecondaryOutputName = "not_in_this_graph"
        };
        runner.Setup(StockModelPath, useGpu: false);

        Assert.IsNull(runner.GetEmbedding());
        Assert.AreEqual(PersonalizationSchema.ExpressionCount, runner.Run()!.Length,
            "expressions must still come out normally");
    }
}

/// <summary>The embedding-aware corrector's failure behaviour.</summary>
[TestClass]
public class EmbeddingModelCorrectorTest
{
    private static float[] Stock(float jaw)
    {
        var values = new float[PersonalizationSchema.ExpressionCount];
        values[PersonalizationSchema.IndexOf("JawOpen")] = jaw;
        return values;
    }

    private static string AssetPath(string name) =>
        Path.Combine(AppContext.BaseDirectory, "Assets", "PersonalModels", name);

    [TestMethod]
    public void NoEmbeddingMeansStockPassthrough()
    {
        // Feeding zeros instead would ship whatever the model made of them; returning stock is the
        // only honest answer when the features the model needs are not available.
        var path = AssetPath("validAdapter.onnx");
        if (!File.Exists(path))
        {
            Assert.Inconclusive("Personal model fixtures are not present.");
            return;
        }

        using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(path);
        var metadata = PersonalModelMetadata.FromSession(session);
        using var corrector = new EmbeddingModelCorrector(session, metadata, NullLogger.Instance);

        var stock = Stock(0.42f);
        var result = corrector.Correct(null!, stock, embedding: null);

        CollectionAssert.AreEqual(stock, result);
        Assert.AreNotSame(stock, result, "the corrector must never hand back the caller's array");
    }

    [TestMethod]
    public void ZeroBlendIsExactlyStock()
    {
        var path = AssetPath("validAdapter.onnx");
        if (!File.Exists(path))
        {
            Assert.Inconclusive("Personal model fixtures are not present.");
            return;
        }

        using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(path);
        var metadata = PersonalModelMetadata.FromSession(session);
        using var corrector = new EmbeddingModelCorrector(session, metadata, NullLogger.Instance)
        {
            Blend = 0f
        };

        var stock = Stock(0.42f);
        var embedding = new DenseTensor<float>([1, 1280]);

        CollectionAssert.AreEqual(stock, corrector.Correct(null!, stock, embedding));
    }

    [TestMethod]
    public void BlendIsClamped()
    {
        var path = AssetPath("validAdapter.onnx");
        if (!File.Exists(path))
        {
            Assert.Inconclusive("Personal model fixtures are not present.");
            return;
        }

        using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(path);
        var metadata = PersonalModelMetadata.FromSession(session);
        using var corrector = new EmbeddingModelCorrector(session, metadata, NullLogger.Instance);

        corrector.Blend = 5f;
        Assert.AreEqual(1f, corrector.Blend);

        corrector.Blend = -2f;
        Assert.AreEqual(0f, corrector.Blend);
    }
}
