using System;
using System.IO;
using System.Linq;
using Baballonia.Services;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The Stage C face-parity gate: the combined build must reproduce the personalized face tracking
/// of the v22 build, not merely compile the code that used to do it.
/// </summary>
/// <remarks>
/// <para>These run against the user's <em>real</em> artefacts in <c>%APPDATA%\ProjectBabble</c> -
/// the stock model, the derived embedding model and the trained personal adapter - because the
/// thing being verified is that this build loads and drives exactly what the old one did. Fixture
/// models would prove the plumbing and nothing about parity.</para>
///
/// <para>Every test is <c>Inconclusive</c> rather than failing when an artefact is absent, so the
/// suite still passes on a machine that has never trained a personal model. On the user's machine
/// they must all run and pass.</para>
///
/// <para>What these cannot cover: the actual per-frame numbers coming off a real camera. That
/// comparison against the running v22 build is the manual step recorded in WORK_PROGRESS.md.</para>
/// </remarks>
[TestClass]
public class FaceParityTest
{
    private const int N = PersonalizationSchema.ExpressionCount;

    private static string StockModelPath => Path.Combine(AppContext.BaseDirectory, "faceModel.onnx");

    /// <summary>The stock face model the v22 build shipped and every personal model was trained against.</summary>
    private const string ExpectedStockMd5 = "4d7240eb056c867ae11e5a4f095a5ccc";

    private static string? PersonalAdapterPath
    {
        get
        {
            var configured = PersonalizationPaths.PersonalModelPath("c");
            return File.Exists(configured) ? configured : null;
        }
    }

    private static void RequireStockModel()
    {
        if (!File.Exists(StockModelPath))
            Assert.Inconclusive("faceModel.onnx is not in the test output directory.");
    }

    [TestMethod]
    public void StockFaceModel_IsTheOneEveryPersonalModelWasTrainedAgainst()
    {
        RequireStockModel();

        Assert.AreEqual(ExpectedStockMd5, EmbeddingModelStore.ComputeMd5(StockModelPath),
            "The stock face model changed. Every trained personal adapter and every recorded " +
            "dataset frame describes the old network's output, so they would all be invalid.");
    }

    [TestMethod]
    public void RunningModelLayout_MatchesThePersonalizationSchema()
    {
        RequireStockModel();

        var runner = new DefaultInferenceRunner(NullLoggerFactory.Instance);
        runner.Setup(StockModelPath, useGpu: false);

        var output = runner.Run();
        Assert.IsNotNull(output, "The stock face model produced no output.");

        // The real assertion of the whole Stage B merge: the keyed pipeline output and the
        // positional personalization schema describe the same 45 values in the same order.
        PersonalizationSchemaBinding.Bind(output!);
        Assert.AreEqual(N, output!.Count);
    }

    [TestMethod]
    public void DerivedEmbeddingModel_IsPresentValidAndProduces1280Features()
    {
        RequireStockModel();

        var validation = EmbeddingModelStore.TryGetValid(StockModelPath);
        if (!validation.Valid)
            Assert.Inconclusive($"No usable embedding model: {validation.Message}");

        var runner = new DefaultInferenceRunner(NullLoggerFactory.Instance)
        {
            SecondaryOutputName = EmbeddingModelStore.EmbeddingOutputName
        };
        runner.Setup(validation.Path!, useGpu: false);

        var expressions = runner.Run();
        var embedding = runner.GetEmbedding();

        Assert.IsNotNull(expressions);
        Assert.AreEqual(N, expressions!.Count);
        Assert.IsNotNull(embedding, "The embedding runner produced no features.");
        Assert.AreEqual(1280, embedding!.Length,
            "The personal head was trained on 1280-d features; any other width is a different " +
            "feature space wearing the same name.");
        Assert.IsTrue(embedding.ToArray().All(float.IsFinite));
    }

    [TestMethod]
    public void DerivedModel_PredictsExactlyWhatTheStockModelDoes()
    {
        RequireStockModel();

        var validation = EmbeddingModelStore.TryGetValid(StockModelPath);
        if (!validation.Valid)
            Assert.Inconclusive($"No usable embedding model: {validation.Message}");

        var stock = new DefaultInferenceRunner(NullLoggerFactory.Instance);
        stock.Setup(StockModelPath, useGpu: false);

        var derived = new DefaultInferenceRunner(NullLoggerFactory.Instance)
        {
            SecondaryOutputName = EmbeddingModelStore.EmbeddingOutputName
        };
        derived.Setup(validation.Path!, useGpu: false);

        FillDeterministic(stock.InputTensor, derived.InputTensor);

        var expected = stock.Run()?.Values.ToArray();
        var actual = derived.Run()?.Values.ToArray();

        Assert.IsNotNull(expected);
        Assert.IsNotNull(actual);
        for (var i = 0; i < expected!.Length; i++)
        {
            Assert.AreEqual(expected[i], actual![i], 0f,
                $"expression {i} differs between the stock and derived models - adding a graph " +
                "output must compute nothing new, or the recordings stop describing the runtime");
        }
    }

    [TestMethod]
    public void PersonalAdapter_LoadsAndPassesEveryValidationGate()
    {
        var path = PersonalAdapterPath;
        if (path == null)
            Assert.Inconclusive("No trained model-C adapter installed.");

        using var options = new SessionOptions();
        options.AppendExecutionProvider_CPU();
        using var session = new InferenceSession(path!, options);
        var metadata = PersonalModelMetadata.FromSession(session);

        Assert.AreEqual("embedding_head_v1", metadata.AdapterType);
        Assert.AreEqual(PersonalModelManager.SupportedAdapterVersion, metadata.AdapterVersion);
        Assert.AreEqual(PersonalizationSchema.Sha256, metadata.SchemaSha256.ToLowerInvariant(),
            "The adapter was trained against a different expression order.");
        Assert.IsTrue(metadata.RequiresEmbedding || metadata.AdapterVersion >= PersonalModelManager.EmbeddingAdapterVersion);
        Assert.AreEqual(ExpectedStockMd5, metadata.BaseModelMd5?.ToLowerInvariant(),
            "The adapter was trained on a different stock face model.");

        CollectionAssert.AreEquivalent(
            new[] { EmbeddingModelCorrector.StockInputName, EmbeddingModelCorrector.EmbeddingInputName },
            session.InputMetadata.Keys.ToArray());
        CollectionAssert.Contains(session.OutputMetadata.Keys.ToArray(), EmbeddingModelCorrector.OutputName);
    }

    /// <summary>
    /// The whole chain the user actually runs: derived stock inference produces expressions plus an
    /// embedding, and the trained head turns them into the personalized vector.
    /// </summary>
    [TestMethod]
    public void FullPersonalizedChain_RunsAndActuallyChangesTheOutput()
    {
        RequireStockModel();

        var path = PersonalAdapterPath;
        if (path == null)
            Assert.Inconclusive("No trained model-C adapter installed.");

        var validation = EmbeddingModelStore.TryGetValid(StockModelPath);
        if (!validation.Valid)
            Assert.Inconclusive($"No usable embedding model: {validation.Message}");

        var runner = new DefaultInferenceRunner(NullLoggerFactory.Instance)
        {
            SecondaryOutputName = EmbeddingModelStore.EmbeddingOutputName
        };
        runner.Setup(validation.Path!, useGpu: false);
        FillDeterministic(runner.InputTensor);

        using var options = new SessionOptions();
        options.AppendExecutionProvider_CPU();
        using var session = new InferenceSession(path!, options);
        using var corrector = new EmbeddingModelCorrector(
            session, PersonalModelMetadata.FromSession(session), NullLogger.Instance)
        {
            Blend = 1f
        };

        var stockMap = runner.Run();
        Assert.IsNotNull(stockMap);
        PersonalizationSchemaBinding.Bind(stockMap!);

        var stockVector = stockMap!.Values.ToArray();
        var personal = corrector.Correct(
            runner.GetInputTensor(), stockVector, runner.GetEmbedding());

        Assert.IsFalse(corrector.HasFailed, "The personal head failed and fell back to stock.");
        Assert.AreEqual(N, personal.Length);
        Assert.IsTrue(personal.All(float.IsFinite), "Personalized output contained NaN or Infinity.");
        Assert.IsTrue(personal.All(v => v is >= -0.01f and <= 1.01f),
            "Personalized output left the [0,1] expression range.");

        var maxDelta = stockVector.Zip(personal, (s, p) => Math.Abs(s - p)).Max();
        Assert.IsTrue(maxDelta > 1e-5f,
            "The personal head changed nothing at all, which means it is not actually running - " +
            $"max |personal - stock| was {maxDelta}.");
    }

    /// <summary>
    /// Blend 0 must be exact stock. This is the escape hatch the user relies on to A/B the model,
    /// so "almost stock" is not good enough.
    /// </summary>
    [TestMethod]
    public void ZeroBlend_IsExactlyStock()
    {
        RequireStockModel();

        var path = PersonalAdapterPath;
        if (path == null)
            Assert.Inconclusive("No trained model-C adapter installed.");

        var validation = EmbeddingModelStore.TryGetValid(StockModelPath);
        if (!validation.Valid)
            Assert.Inconclusive($"No usable embedding model: {validation.Message}");

        var runner = new DefaultInferenceRunner(NullLoggerFactory.Instance)
        {
            SecondaryOutputName = EmbeddingModelStore.EmbeddingOutputName
        };
        runner.Setup(validation.Path!, useGpu: false);
        FillDeterministic(runner.InputTensor);

        using var options = new SessionOptions();
        options.AppendExecutionProvider_CPU();
        using var session = new InferenceSession(path!, options);
        using var corrector = new EmbeddingModelCorrector(
            session, PersonalModelMetadata.FromSession(session), NullLogger.Instance)
        {
            Blend = 0f
        };

        var stockVector = runner.Run()!.Values.ToArray();
        var blended = corrector.Correct(runner.GetInputTensor(), stockVector, runner.GetEmbedding());

        CollectionAssert.AreEqual(stockVector, blended,
            "Blend 0 must return the stock vector unchanged.");
    }

    /// <summary>
    /// The corrector must never mutate the array it is handed: the pipeline reuses that buffer for
    /// the raw-expressions event that dataset recording listens to.
    /// </summary>
    [TestMethod]
    public void Corrector_DoesNotMutateTheStockVectorItIsGiven()
    {
        RequireStockModel();

        var path = PersonalAdapterPath;
        if (path == null)
            Assert.Inconclusive("No trained model-C adapter installed.");

        var validation = EmbeddingModelStore.TryGetValid(StockModelPath);
        if (!validation.Valid)
            Assert.Inconclusive($"No usable embedding model: {validation.Message}");

        var runner = new DefaultInferenceRunner(NullLoggerFactory.Instance)
        {
            SecondaryOutputName = EmbeddingModelStore.EmbeddingOutputName
        };
        runner.Setup(validation.Path!, useGpu: false);
        FillDeterministic(runner.InputTensor);

        using var options = new SessionOptions();
        options.AppendExecutionProvider_CPU();
        using var session = new InferenceSession(path!, options);
        using var corrector = new EmbeddingModelCorrector(
            session, PersonalModelMetadata.FromSession(session), NullLogger.Instance);

        var stockVector = runner.Run()!.Values.ToArray();
        var before = (float[])stockVector.Clone();

        var personal = corrector.Correct(runner.GetInputTensor(), stockVector, runner.GetEmbedding());

        CollectionAssert.AreEqual(before, stockVector, "The corrector mutated its input buffer.");
        Assert.AreNotSame(stockVector, personal, "The corrector must return a fresh array.");
    }

    private static void FillDeterministic(params DenseTensor<float>[] tensors)
    {
        // A fixed seed, so a regression shows up as a changed number rather than as flakiness.
        foreach (var tensor in tensors)
        {
            var random = new Random(20260829);
            for (var i = 0; i < tensor.Length; i++)
                tensor.SetValue(i, (float)random.NextDouble());
        }
    }
}
