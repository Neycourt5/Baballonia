using System;
using System.Collections.Generic;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Enums;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Inference;

[TestClass]
public class EyeProcessingPipelineTickTest
{
    private EyeProcessingPipeline _pipeline = null!;

    [TestInitialize]
    public void Initialize()
    {
        _pipeline = new EyeProcessingPipeline(new EyePipelineEventBus(), new PipelineMetrics())
        {
            VideoSource = new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = new FakeRunner(),
        };
    }

    [TestCleanup]
    public void Cleanup() => _pipeline.Dispose();

    [TestMethod]
    public void EarlyTicksReleaseCameraAndTransformedFrames()
    {
        var source = (FakeVideoSource)_pipeline.VideoSource!;
        var transformer = (FakeTransformer)_pipeline.ImageTransformer!;

        Assert.IsNull(_pipeline.RunUpdate());
        Assert.IsNull(_pipeline.RunUpdate());

        Assert.AreEqual(2, source.Issued.Count);
        Assert.AreEqual(2, transformer.Produced.Count);
        Assert.IsTrue(source.Issued.TrueForAll(frame => frame.IsDisposed));
        Assert.IsTrue(transformer.Produced.TrueForAll(frame => frame.IsDisposed));
    }

    [TestMethod]
    public void CompleteTickReleasesEveryOwnedMatrix()
    {
        var source = (FakeVideoSource)_pipeline.VideoSource!;
        var transformer = (FakeTransformer)_pipeline.ImageTransformer!;
        var converter = (FakeConverter)_pipeline.ImageConverter!;

        Assert.IsNotNull(RunUntilInference());

        Assert.IsTrue(source.Issued.TrueForAll(frame => frame.IsDisposed));
        Assert.IsTrue(transformer.Produced.TrueForAll(frame => frame.IsDisposed));
        Assert.IsTrue(converter.Received.TrueForAll(frame => frame.IsDisposed));
    }

    [TestMethod]
    public void PublishedTransformedFrameLivesOnlyForTheSynchronousEvent()
    {
        var bus = new EyePipelineEventBus();
        Mat? published = null;
        var aliveInHandler = false;
        bus.Subscribe<EyePipelineEvents.NewTransformedFrameEvent>(message =>
        {
            published = message.image;
            aliveInHandler = !message.image.IsDisposed;
        });
        _pipeline.Dispose();
        _pipeline = new EyeProcessingPipeline(bus, new PipelineMetrics())
        {
            VideoSource = new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = new FakeRunner(),
        };

        _pipeline.RunUpdate();

        Assert.IsTrue(aliveInHandler);
        Assert.IsNotNull(published);
        Assert.IsTrue(published.IsDisposed);
    }

    [TestMethod]
    public void ResetDropsTemporalFramesAndRawResult()
    {
        Assert.IsNotNull(RunUntilInference());
        Assert.IsNotNull(_pipeline.RawEyeResult);

        _pipeline.ResetTemporalHistory();

        Assert.IsNull(_pipeline.RawEyeResult);
        Assert.IsNull(_pipeline.RunUpdate(), "the temporal queue should refill after a reset");
    }

    [TestMethod]
    public void FullBlinkHoldsLastVerticalGazeWithoutProducingNaN()
    {
        var runner = (FakeRunner)_pipeline.InferenceService!;
        runner.Values = [0.75f, 0.2f, 0f, 0.75f, 0.8f, 0f];
        var open = RunUntilInference()!;
        Assert.AreEqual(0.5f, open["/leftEyeY"], 1e-6f);

        runner.Values = [0.1f, 0.2f, 1f, 0.9f, 0.8f, 1f];
        var blink = _pipeline.RunUpdate()!;

        Assert.AreEqual(0.5f, blink["/leftEyeY"], 1e-6f);
        Assert.AreEqual(0.5f, blink["/rightEyeY"], 1e-6f);
        Assert.IsTrue(float.IsFinite(blink["/leftEyeY"]));
        Assert.IsTrue(float.IsFinite(blink["/rightEyeY"]));
    }

    [TestMethod]
    public void PostProcessorSanitizesFilteredOutputWithoutChangingNativeRawOutput()
    {
        var runner = (FakeRunner)_pipeline.InferenceService!;
        runner.Values = [0.5f, 2f, 0f, 0.5f, 2f, 0f];
        _pipeline.StabilizeEyes = false;
        _pipeline.PostProcessor = new EyeOutputPostProcessor();

        var filtered = RunUntilInference()!;
        var raw = _pipeline.RawEyeResult!;

        Assert.AreNotSame(filtered, raw);
        Assert.AreEqual(1f, filtered["/leftEyeX"]);
        Assert.AreEqual(3f, raw["/leftEyeX"], 1e-6f,
            "the native/DFR snapshot must remain geometry-corrected but un-post-processed");
    }

    private OrderedFloatMap? RunUntilInference()
    {
        OrderedFloatMap? result = null;
        for (var i = 0; i < 10 && result == null; i++)
            result = _pipeline.RunUpdate();
        return result;
    }

    private sealed class FakeVideoSource : IVideoSource
    {
        public List<Mat> Issued { get; } = [];

        public Mat? GetFrame(ColorType? color = null)
        {
            var frame = new Mat(8, 16, MatType.CV_8UC1, Scalar.All(Issued.Count + 20));
            Issued.Add(frame);
            return frame;
        }

        public bool Start() => true;
        public bool Stop() => true;
        public System.Threading.WaitHandle[] GetFrameWaitHandles() => [];
        public void Dispose() { }
    }

    private sealed class FakeTransformer : IImageTransformer
    {
        public List<Mat> Produced { get; } = [];

        public Mat? Apply(Mat image)
        {
            var frame = new Mat(8, 8, MatType.CV_8UC2, Scalar.All(Produced.Count + 30));
            Produced.Add(frame);
            return frame;
        }
    }

    private sealed class FakeConverter : IImageConverter
    {
        public List<Mat> Received { get; } = [];
        public void Convert(Mat input, DenseTensor<float> outTensor) => Received.Add(input);
    }

    private sealed class FakeRunner : IInferenceRunner
    {
        private static readonly string[] Keys =
        [
            "/rightEyeY", "/rightEyeX", "/rightEyeLid",
            "/leftEyeY", "/leftEyeX", "/leftEyeLid",
        ];

        private readonly DenseTensor<float> _input = new([1, 8, 8, 8]);
        public float[] Values { get; set; } = [0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f];

        public DenseTensor<float> GetInputTensor() => _input;

        public OrderedFloatMap? Run()
        {
            var output = new OrderedFloatMap(Keys);
            Values.AsSpan().CopyTo(output.ValuesSpan);
            return output;
        }
    }
}
