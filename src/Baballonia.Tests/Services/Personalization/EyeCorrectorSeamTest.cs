using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.Eye;
using JetBrains.Annotations;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>The twelve-channel contract a trained eye adapter is written against.</summary>
[TestClass]
[TestSubject(typeof(EyePersonalizationSchema))]
public class EyePersonalizationSchemaTest
{
    [TestMethod]
    public void TheSchemaIsTheTunedModelsOwnOutputOrder()
    {
        // Read directly from the model's blendshape_names metadata. A trained adapter maps twelve
        // numbers to twelve numbers with no idea what any of them mean, so this order is the only
        // thing keeping a learned correction on the channel it was learned for.
        CollectionAssert.AreEqual(
            new[]
            {
                "rightEyeY", "rightEyeX", "rightEyeLid", "rightEyeWiden", "rightEyeSquint", "rightEyeBrow",
                "leftEyeY", "leftEyeX", "leftEyeLid", "leftEyeWiden", "leftEyeSquint", "leftEyeBrow",
            },
            EyePersonalizationSchema.ExpressionNames.ToArray());

        Assert.AreEqual(EyePersonalizationSchema.ExpressionCount,
            EyePersonalizationSchema.ExpressionNames.Count);
    }

    [TestMethod]
    public void KeysAreTheNamesAsThePipelineSpellsThem()
    {
        Assert.AreEqual("/rightEyeY", EyePersonalizationSchema.ExpressionKeys[0]);
        Assert.AreEqual("/leftEyeBrow", EyePersonalizationSchema.ExpressionKeys[11]);
    }

    [TestMethod]
    public void TheHashRecipeMatchesTheFaceSchemasSoOneImplementationServesBoth()
    {
        var expected = Convert.ToHexStringLower(SHA256.HashData(
            Encoding.UTF8.GetBytes(string.Join("\n", EyePersonalizationSchema.ExpressionNames))));

        Assert.AreEqual(expected, EyePersonalizationSchema.Sha256);
        Assert.AreEqual(64, EyePersonalizationSchema.Sha256.Length);
        Assert.IsTrue(EyePersonalizationSchema.Sha256.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f'));
    }

    [TestMethod]
    public void IndexOfResolvesKnownChannelsAndRejectsUnknown()
    {
        Assert.AreEqual(2, EyePersonalizationSchema.IndexOf("rightEyeLid"));
        Assert.AreEqual(8, EyePersonalizationSchema.IndexOf("leftEyeLid"));
        Assert.AreEqual(-1, EyePersonalizationSchema.IndexOf("notAChannel"));
    }
}

/// <summary>Whether a given running model can be personalized at all.</summary>
[TestClass]
[TestSubject(typeof(EyePersonalizationSchemaBinding))]
public class EyePersonalizationSchemaBindingTest
{
    [TestMethod]
    public void TheTunedTwelveOutputModelBinds()
    {
        var map = new OrderedFloatMap(EyePersonalizationSchema.ExpressionKeys.ToArray());

        Assert.IsTrue(EyePersonalizationSchemaBinding.TryBind(map, out var error), error);
        Assert.IsNull(error);
    }

    [TestMethod]
    public void TheStockSixOutputModelIsRefusedWithAReasonRatherThanAnError()
    {
        // Running the stock model is a normal thing to do; it simply cannot be personalized by an
        // adapter trained on twelve channels.
        var map = new OrderedFloatMap([
            "/rightEyeY", "/rightEyeX", "/rightEyeLid",
            "/leftEyeY", "/leftEyeX", "/leftEyeLid",
        ]);

        Assert.IsFalse(EyePersonalizationSchemaBinding.TryBind(map, out var error));
        StringAssert.Contains(error, "6 values");
        StringAssert.Contains(error, "stock eye model");
    }

    [TestMethod]
    public void AReorderedModelIsRefusedAndNamesThePosition()
    {
        var keys = EyePersonalizationSchema.ExpressionKeys.ToArray();
        (keys[3], keys[4]) = (keys[4], keys[3]); // widen <-> squint

        Assert.IsFalse(EyePersonalizationSchemaBinding.TryBind(new OrderedFloatMap(keys), out var error));
        StringAssert.Contains(error, "position 3");
        StringAssert.Contains(error, "/rightEyeWiden");
    }

    [TestMethod]
    public void NullIsRefusedRatherThanThrowing()
    {
        Assert.IsFalse(EyePersonalizationSchemaBinding.TryBind(null, out var error));
        Assert.IsNotNull(error);
    }
}

/// <summary>
/// The corrector seam in the live eye pipeline.
/// </summary>
/// <remarks>
/// The load-bearing test here is the first one. Everything else in this feature is optional; the
/// promise that installing nothing changes nothing is what makes it safe to ship at all.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeProcessingPipeline))]
public class EyeCorrectorSeamTest
{
    private PipelineMetrics _metrics = null!;
    private EyePipelineEventBus _bus = null!;
    private EyeProcessingPipeline _pipeline = null!;

    [TestInitialize]
    public void Initialize()
    {
        _metrics = new PipelineMetrics();
        _bus = new EyePipelineEventBus();
        _pipeline = new EyeProcessingPipeline(_bus, _metrics)
        {
            VideoSource = new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = new FakeRunner(),
        };
    }

    [TestCleanup]
    public void Cleanup() => _pipeline.Dispose();

    private OrderedFloatMap? RunUntilInference()
    {
        OrderedFloatMap? result = null;
        for (var i = 0; i < 8 && result is null; i++)
            result = _pipeline.RunUpdate();
        return result;
    }

    [TestMethod]
    public void WithNoCorrectorTheOutputIsUnchanged()
    {
        var baseline = RunUntilInference()!.Values.ToArray();

        // Same fake inputs through a second pipeline that has never heard of a corrector.
        _pipeline.Dispose();
        _pipeline = new EyeProcessingPipeline(new EyePipelineEventBus(), new PipelineMetrics())
        {
            VideoSource = new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = new FakeRunner(),
        };
        var again = RunUntilInference()!.Values.ToArray();

        CollectionAssert.AreEqual(baseline, again,
            "installing nothing must change nothing; this is the promise the feature rests on");
        Assert.AreEqual(0f, _metrics.EyeCorrectorDelta, 1e-9);
    }

    [TestMethod]
    public void TheBaseModelOutputIsPublishedEveryFrame()
    {
        var seen = new List<float[]>();
        _bus.Subscribe<EyePipelineEvents.NewRawEyeExpressionsEvent>(e => seen.Add(e.rawResult.ToArray()));

        Assert.IsNotNull(RunUntilInference());

        Assert.AreEqual(1, seen.Count, "exactly one raw publication per inference");
        Assert.AreEqual(EyePersonalizationSchema.ExpressionCount, seen[0].Length);
        // Raw model space: lid is closedness and gaze has not been remapped yet.
        CollectionAssert.AreEqual(FakeRunner.Values, seen[0]);
    }

    [TestMethod]
    public void NoCorrectedEventWithoutACorrector()
    {
        var corrected = 0;
        _bus.Subscribe<EyePipelineEvents.NewCorrectedEyeExpressionsEvent>(_ => corrected++);

        RunUntilInference();

        Assert.AreEqual(0, corrected);
    }

    [TestMethod]
    public void ACorrectorChangesWhatReachesTheRestOfThePipeline()
    {
        _pipeline.Corrector = new AddConstCorrector(0.05f);

        var result = RunUntilInference();

        Assert.IsNotNull(result);
        // The geometry pass has since run, so compare against a pipeline given the same shifted
        // values by the model itself.
        var expectedRunner = new FakeRunner(FakeRunner.Values.Select(v => v + 0.05f).ToArray());
        _pipeline.Dispose();
        _pipeline = new EyeProcessingPipeline(new EyePipelineEventBus(), new PipelineMetrics())
        {
            VideoSource = new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = expectedRunner,
        };
        var expected = RunUntilInference()!.Values.ToArray();

        CollectionAssert.AreEqual(expected, result!.Values.ToArray(),
            "correcting the model output must be indistinguishable from the model producing it");
    }

    [TestMethod]
    public void CorrectedValuesAlsoReachTheNativeDfrSnapshot()
    {
        // DFR wants the truest gaze available; feeding it the uncorrected bias would defeat the
        // point of correcting at all.
        _pipeline.Corrector = new AddConstCorrector(0.05f);

        Assert.IsNotNull(RunUntilInference());
        Assert.IsNotNull(_pipeline.RawEyeResult);

        var uncorrected = new EyeProcessingPipeline(new EyePipelineEventBus(), new PipelineMetrics())
        {
            VideoSource = new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = new FakeRunner(),
        };
        OrderedFloatMap? plain = null;
        for (var i = 0; i < 8 && plain is null; i++)
            plain = uncorrected.RunUpdate();

        CollectionAssert.AreNotEqual(
            plain!.Values.ToArray(), _pipeline.RawEyeResult!.Values.ToArray());
        uncorrected.Dispose();
    }

    [TestMethod]
    public void BothVectorsArePublishedForTheComparisonView()
    {
        EyePipelineEvents.NewCorrectedEyeExpressionsEvent? seen = null;
        _bus.Subscribe<EyePipelineEvents.NewCorrectedEyeExpressionsEvent>(e => seen = e);
        _pipeline.Corrector = new AddConstCorrector(0.05f);

        RunUntilInference();

        Assert.IsNotNull(seen);
        CollectionAssert.AreEqual(FakeRunner.Values, seen!.rawResult.ToArray());
        CollectionAssert.AreEqual(
            FakeRunner.Values.Select(v => v + 0.05f).ToArray(), seen.correctedResult.ToArray());
    }

    [TestMethod]
    public void ACorrectorThatThrowsIsRemovedRatherThanBreakingTracking()
    {
        // The eye worker's catch tears down every camera, so a bad adapter must never reach it.
        _pipeline.Corrector = new ThrowingCorrector();

        var result = RunUntilInference();

        Assert.IsNotNull(result, "tracking must survive a broken corrector");
        Assert.IsNull(_pipeline.Corrector, "and the corrector must not get a second chance");
    }

    [TestMethod]
    public void ACorrectorReturningNonFiniteValuesIsIgnoredForThatFrame()
    {
        _pipeline.Corrector = new NonFiniteCorrector();

        var result = RunUntilInference();

        Assert.IsNotNull(result);
        foreach (var value in result!.Values)
            Assert.IsTrue(float.IsFinite(value), "no NaN may escape the corrector");
    }

    [TestMethod]
    public void ACorrectorOfTheWrongWidthIsRemoved()
    {
        _pipeline.Corrector = new WrongWidthCorrector();

        Assert.IsNotNull(RunUntilInference());
        Assert.IsNull(_pipeline.Corrector);
    }

    [TestMethod]
    public void AStockSixOutputModelIsLeftAloneEntirely()
    {
        _pipeline.InferenceService = new FakeRunner([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f], sixOutput: true);
        _pipeline.ResetTemporalHistory();
        var raw = 0;
        _bus.Subscribe<EyePipelineEvents.NewRawEyeExpressionsEvent>(_ => raw++);
        _pipeline.Corrector = new AddConstCorrector(0.5f);

        var result = RunUntilInference();

        Assert.IsNotNull(result, "the stock model must keep working");
        Assert.AreEqual(0, raw, "there is nothing a twelve-channel adapter could be told about it");
    }

    [TestMethod]
    public void CorrectorTimingAndDeltaAreReported()
    {
        _pipeline.Corrector = new AddConstCorrector(0.05f);

        Assert.IsNotNull(RunUntilInference());

        Assert.AreEqual(0.05f, _metrics.EyeCorrectorDelta, 1e-5,
            "mean absolute change is what tells 'off' apart from 'doing nothing'");
        Assert.IsTrue(_metrics.EyeCorrectMs >= 0);
    }

    private sealed class AddConstCorrector(float delta) : IExpressionCorrector
    {
        public float Blend { get; set; } = 1f;
        public float[] Correct(DenseTensor<float> image, float[] stock) =>
            stock.Select(v => v + delta).ToArray();
    }

    private sealed class ThrowingCorrector : IExpressionCorrector
    {
        public float Blend { get; set; } = 1f;
        public float[] Correct(DenseTensor<float> image, float[] stock) =>
            throw new InvalidOperationException("broken adapter");
    }

    private sealed class NonFiniteCorrector : IExpressionCorrector
    {
        public float Blend { get; set; } = 1f;
        public float[] Correct(DenseTensor<float> image, float[] stock)
        {
            var output = (float[])stock.Clone();
            output[0] = float.NaN;
            return output;
        }
    }

    private sealed class WrongWidthCorrector : IExpressionCorrector
    {
        public float Blend { get; set; } = 1f;
        public float[] Correct(DenseTensor<float> image, float[] stock) => new float[3];
    }

    private sealed class FakeVideoSource : IVideoSource
    {
        private int _issued;
        public Mat GetFrame(ColorType? color = null) =>
            new(8, 16, MatType.CV_8UC1, Scalar.All(_issued++ + 20));
        public bool Start() => true;
        public bool Stop() => true;
        public WaitHandle[] GetFrameWaitHandles() => [];
        public void Dispose() { }
    }

    private sealed class FakeTransformer : IImageTransformer
    {
        private int _produced;
        public Mat? Apply(Mat image) => new(8, 8, MatType.CV_8UC2, Scalar.All(_produced++ + 30));
    }

    private sealed class FakeConverter : IImageConverter
    {
        public void Convert(Mat input, DenseTensor<float> outTensor) { }
    }

    private sealed class FakeRunner : IInferenceRunner
    {
        /// <summary>Plausible raw model output: sigmoids, lid as closedness, gaze around centre.</summary>
        public static readonly float[] Values =
        [
            0.50f, 0.48f, 0.12f, 0.30f, 0.05f, 0.40f,
            0.52f, 0.55f, 0.14f, 0.28f, 0.07f, 0.42f,
        ];

        private readonly string[] _keys;
        private readonly float[] _values;
        private readonly DenseTensor<float> _input = new([1, 8, 8, 8]);

        public FakeRunner() : this(Values) { }

        public FakeRunner(float[] values, bool sixOutput = false)
        {
            _values = values;
            _keys = sixOutput
                ? ["/rightEyeY", "/rightEyeX", "/rightEyeLid", "/leftEyeY", "/leftEyeX", "/leftEyeLid"]
                : EyePersonalizationSchema.ExpressionKeys.ToArray();
        }

        public DenseTensor<float> GetInputTensor() => _input;

        public OrderedFloatMap? Run()
        {
            var output = new OrderedFloatMap(_keys);
            _values.AsSpan(0, _keys.Length).CopyTo(output.ValuesSpan);
            return output;
        }
    }
}
