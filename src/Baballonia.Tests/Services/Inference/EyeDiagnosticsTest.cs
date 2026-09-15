using System;
using System.Collections.Generic;
using System.Threading;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Enums;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Inference;

/// <summary>
/// The live eye values the Debug page reads.
/// </summary>
/// <remarks>
/// These exist so a bad calibration or an unexpected widen/squint source is visible as a number
/// rather than as "tracking feels off". That only works if the numbers are actually populated, and
/// a silently-zero diagnostic is worse than none - it reads as a working readout showing nothing
/// wrong. Hence testing that they are published at all.
/// </remarks>
[TestClass]
public class EyeDiagnosticsTest
{
    private PipelineMetrics _metrics = null!;
    private EyeProcessingPipeline _pipeline = null!;

    [TestInitialize]
    public void Initialize()
    {
        _metrics = new PipelineMetrics();
        _pipeline = new EyeProcessingPipeline(new EyePipelineEventBus(), _metrics)
        {
            VideoSource = new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = new FakeRunner(),
            PostProcessor = new EyeOutputPostProcessor(),
        };
    }

    [TestCleanup]
    public void Cleanup() => _pipeline.Dispose();

    /// <summary>The temporal collector needs several frames before the first inference happens.</summary>
    private OrderedFloatMap? RunUntilInference()
    {
        OrderedFloatMap? result = null;
        for (var i = 0; i < 8 && result is null; i++)
            result = _pipeline.RunUpdate();
        return result;
    }

    [TestMethod]
    public void ATickPublishesTheLiveEyeValues()
    {
        Assert.IsNotNull(RunUntilInference(), "the pipeline never reached inference");

        // The six-output fake has no widen/squint, so the fallback supplies them.
        Assert.IsTrue(_metrics.EyeWidenSquintDerived);

        Assert.IsTrue(_metrics.EyeLeftRawOpenness is >= 0f and <= 1f);
        Assert.IsTrue(_metrics.EyeRightRawOpenness is >= 0f and <= 1f);
        Assert.IsTrue(_metrics.EyeLeftOpenness is >= 0f and <= 1f);
        Assert.IsTrue(_metrics.EyeRightOpenness is >= 0f and <= 1f);
        Assert.IsTrue(_metrics.EyeLeftGazeX is >= -1f and <= 1f);
        Assert.IsTrue(_metrics.EyeRightGazeY is >= -1f and <= 1f);
        Assert.IsTrue(float.IsFinite(_metrics.EyeLeftWiden));
        Assert.IsTrue(float.IsFinite(_metrics.EyeRightSquint));
    }

    [TestMethod]
    public void PublishedOpennessMatchesWhatIsSent()
    {
        var result = RunUntilInference();
        Assert.IsNotNull(result);

        // Whatever the Debug page shows must be the value that actually went out, not an
        // intermediate from an earlier stage.
        Assert.AreEqual(result!["/leftEyeLid"], _metrics.EyeLeftOpenness, 1e-6);
        Assert.AreEqual(result["/rightEyeLid"], _metrics.EyeRightOpenness, 1e-6);
        Assert.AreEqual(result["/leftEyeX"], _metrics.EyeLeftGazeX, 1e-6);
        Assert.AreEqual(result["/rightEyeY"], _metrics.EyeRightGazeY, 1e-6);
    }

    [TestMethod]
    public void AModelWithItsOwnShapesIsReportedAsSuch()
    {
        _pipeline.InferenceService = new FakeRunner(withShapes: true);
        _pipeline.ResetTemporalHistory();

        Assert.IsNotNull(RunUntilInference());

        Assert.IsFalse(_metrics.EyeWidenSquintDerived,
            "the twelve-output model supplies these; reporting them as derived would be a lie");
    }

    [TestMethod]
    public void CorruptFrameCounterStartsAtZeroAndIsReadable()
    {
        Assert.AreEqual(0, Interlocked.Read(ref _metrics.EyeCorruptFrames));

        Assert.IsNotNull(RunUntilInference());

        Assert.AreEqual(0, Interlocked.Read(ref _metrics.EyeCorruptFrames),
            "clean frames must not be counted as corrupt");
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

    /// <summary>Two channels, one per eye: what the collector's temporal stack expects.</summary>
    private sealed class FakeTransformer : IImageTransformer
    {
        private int _produced;

        public Mat? Apply(Mat image) =>
            new(8, 8, MatType.CV_8UC2, Scalar.All(_produced++ + 30));
    }

    private sealed class FakeConverter : IImageConverter
    {
        public void Convert(Mat input, DenseTensor<float> outTensor) { }
    }

    private sealed class FakeRunner(bool withShapes = false) : IInferenceRunner
    {
        private static readonly string[] SixOutputs =
        [
            "/rightEyeY", "/rightEyeX", "/rightEyeLid",
            "/leftEyeY", "/leftEyeX", "/leftEyeLid",
        ];

        private static readonly string[] TwelveOutputs =
        [
            "/rightEyeY", "/rightEyeX", "/rightEyeLid", "/rightEyeWiden", "/rightEyeSquint", "/rightEyeBrow",
            "/leftEyeY", "/leftEyeX", "/leftEyeLid", "/leftEyeWiden", "/leftEyeSquint", "/leftEyeBrow",
        ];

        private readonly string[] _keys = withShapes ? TwelveOutputs : SixOutputs;
        private readonly DenseTensor<float> _input = new([1, 8, 8, 8]);

        public DenseTensor<float> GetInputTensor() => _input;

        public OrderedFloatMap? Run()
        {
            var output = new OrderedFloatMap(_keys);
            for (var i = 0; i < _keys.Length; i++)
                output.ValuesSpan[i] = 0.1f * (i + 1) % 1f;
            return output;
        }
    }
}
