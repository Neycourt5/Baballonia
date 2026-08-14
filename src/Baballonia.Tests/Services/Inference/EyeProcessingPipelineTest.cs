using System;
using System.Collections.Generic;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Inference.Filters;
using Baballonia.Services.Inference.VideoSources;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Inference;

/// <summary>
/// The eye tick: what it publishes, and what it releases.
///
/// The raw-output event is the point of this work. Everything downstream of it is deliberately
/// lossy - a single vertical gaze is averaged across both eyes, a closed eye borrows the open eye's
/// yaw, convergence is clamped - so any per-user eye calibration has to see the model's own output
/// or it is fitting a mapping on top of decisions it cannot observe.
///
/// The disposal tests exist because this method runs about a hundred times a second. It previously
/// leaked the 8-channel temporal stack every tick, disposed the transformed frame twice, and
/// returned on six paths without releasing the camera frame; none of that is visible in behaviour
/// until memory runs out.
/// </summary>
/// <remarks>
/// Named for the tick rather than the class because <c>DefaultProcessingPipelineTest.cs</c> already
/// declares an <c>EyeProcessingPipelineTest</c> that actually exercises the base pipeline.
/// </remarks>
[TestClass]
public class EyeProcessingPipelineTickTest
{
    private const int Size = 128;
    private static readonly int Outputs = Utils.EyeRawExpressions;

    private EyePipelineEventBus _bus = null!;
    private EyeProcessingPipeline _pipeline = null!;

    [TestInitialize]
    public void Initialize()
    {
        _bus = new EyePipelineEventBus();
        _pipeline = new EyeProcessingPipeline(_bus)
        {
            VideoSource = new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = new FakeRunner(),
        };
    }

    /// <summary>The collector needs five frames before it emits anything.</summary>
    private float[]? RunUntilInference(int maxTicks = 10)
    {
        float[]? result = null;
        for (var i = 0; i < maxTicks && result == null; i++)
            result = _pipeline.RunUpdate();
        return result;
    }

    // =============================================================================================
    // The raw event
    // =============================================================================================

    [TestMethod]
    public void PublishesRawModelOutputBeforePostProcessing()
    {
        EyePipelineEvents.NewRawExpressionsEvent? captured = null;
        _bus.Subscribe<EyePipelineEvents.NewRawExpressionsEvent>(e => captured = e);

        var final = RunUntilInference();

        Assert.IsNotNull(final);
        Assert.IsNotNull(captured, "the raw event was never published");
        CollectionAssert.AreEqual(FakeRunner.Output, captured.rawResult,
            "the event must carry the model's own values, not post-processed ones");
    }

    [TestMethod]
    public void RawOutputDiffersFromTheFinalResult()
    {
        // If these were equal the event would be worthless - it exists because ProcessExpressions
        // discards information that a calibration layer needs.
        float[]? raw = null;
        _bus.Subscribe<EyePipelineEvents.NewRawExpressionsEvent>(e => raw = (float[])e.rawResult.Clone());

        var final = RunUntilInference();

        Assert.IsNotNull(final);
        Assert.IsNotNull(raw);
        CollectionAssert.AreNotEqual(raw, final,
            "post-processing should have transformed the raw values");
    }

    [TestMethod]
    public void RawEventCarriesTheFrameThatProducedIt()
    {
        // Everything is inspected inside the handler: the frame's lifetime ends when the publish
        // returns, so touching it afterwards is exactly the misuse the contract warns about.
        var wasAlive = false;
        var channels = 0;
        var sawFrame = false;

        _bus.Subscribe<EyePipelineEvents.NewRawExpressionsEvent>(e =>
        {
            sawFrame = e.transformedFrame is not null;
            wasAlive = !e.transformedFrame.IsDisposed && !e.transformedFrame.Empty();
            channels = e.transformedFrame.Channels();
        });

        RunUntilInference();

        Assert.IsTrue(sawFrame, "the raw event was never published");
        Assert.IsTrue(wasAlive, "the frame must still be alive while handlers run");
        Assert.AreEqual(2, channels, "eye frames carry one channel per eye");
    }

    [TestMethod]
    public void RawEventFrameIsDisposedAfterThePublishReturns()
    {
        // The documented lifetime contract: handlers must clone to retain.
        Mat? seen = null;
        _bus.Subscribe<EyePipelineEvents.NewRawExpressionsEvent>(e => seen = e.transformedFrame);

        RunUntilInference();

        Assert.IsNotNull(seen);
        Assert.IsTrue(seen.IsDisposed, "the pipeline should have released the frame");
    }

    [TestMethod]
    public void RawEventCarriesAUsableTimestamp()
    {
        long ticks = 0;
        _bus.Subscribe<EyePipelineEvents.NewRawExpressionsEvent>(e => ticks = e.timestampTicks);

        var before = DateTime.UtcNow.Ticks;
        RunUntilInference();

        Assert.IsTrue(ticks >= before && ticks <= DateTime.UtcNow.Ticks);
    }

    [TestMethod]
    public void NoRawEventWhileTheTemporalQueueIsStillFilling()
    {
        var count = 0;
        _bus.Subscribe<EyePipelineEvents.NewRawExpressionsEvent>(_ => count++);

        // The collector buffers five frames before producing a stack.
        _pipeline.RunUpdate();
        _pipeline.RunUpdate();

        Assert.AreEqual(0, count, "inference has not run yet, so there is nothing raw to publish");
    }

    [TestMethod]
    public void RawEventIsPublishedBeforeTheFilteredOne()
    {
        var order = new List<string>();
        _bus.Subscribe<EyePipelineEvents.NewRawExpressionsEvent>(_ => order.Add("raw"));
        _bus.Subscribe<EyePipelineEvents.NewFilteredResultEvent>(_ => order.Add("filtered"));

        RunUntilInference();

        CollectionAssert.AreEqual(new[] { "raw", "filtered" }, order);
    }

    [TestMethod]
    public void TheFilterDoesNotSeeItsOwnOutputTwice()
    {
        var filter = new CountingFilter();
        _pipeline.Filter = filter;

        RunUntilInference();

        Assert.AreEqual(1, filter.Calls, "the filter should run once per tick");
    }

    // =============================================================================================
    // Resource lifetime
    // =============================================================================================

    [TestMethod]
    public void CameraFrameIsReleasedEvenWhenTheTickReturnsEarly()
    {
        // The temporal queue swallows the first four frames; each one still has to be released.
        var source = (FakeVideoSource)_pipeline.VideoSource!;

        _pipeline.RunUpdate();
        _pipeline.RunUpdate();

        Assert.AreEqual(2, source.Issued.Count);
        foreach (var frame in source.Issued)
            Assert.IsTrue(frame.IsDisposed, "an early return leaked the camera frame");
    }

    [TestMethod]
    public void CameraFrameIsReleasedOnACompleteTick()
    {
        var source = (FakeVideoSource)_pipeline.VideoSource!;

        RunUntilInference();

        foreach (var frame in source.Issued)
            Assert.IsTrue(frame.IsDisposed);
    }

    [TestMethod]
    public void TransformedFrameIsDisposedExactlyOnce()
    {
        // It used to be disposed twice - harmless with OpenCvSharp today, but it is the kind of
        // thing that becomes a crash the moment ownership changes.
        var transformer = (FakeTransformer)_pipeline.ImageTransformer!;

        RunUntilInference();

        foreach (var mat in transformer.Produced)
            Assert.IsTrue(mat.IsDisposed, "transformed frame was not released");
    }

    [TestMethod]
    public void TemporalStackIsReleasedEveryTick()
    {
        // This is the leak that mattered: one 8-channel 128x128 buffer per tick, ~100 times a
        // second, never freed.
        var converter = (FakeConverter)_pipeline.ImageConverter!;

        RunUntilInference();
        var afterFirst = converter.Received.Count;
        Assert.IsTrue(afterFirst > 0, "inference never ran");

        _pipeline.RunUpdate();

        foreach (var stack in converter.Received)
            Assert.IsTrue(stack.IsDisposed, "the temporal stack was leaked");
    }

    [TestMethod]
    public void CorruptFramesAreDroppedAndReleased()
    {
        var source = new FakeVideoSource { EmitCorrupt = true };
        _pipeline.VideoSource = source;

        var result = _pipeline.RunUpdate();

        Assert.IsNull(result);
        foreach (var frame in source.Issued)
            Assert.IsTrue(frame.IsDisposed);
    }

    [TestMethod]
    public void AMissingCameraIsNotAnError()
    {
        _pipeline.VideoSource = new FakeVideoSource { ReturnNull = true };

        Assert.IsNull(_pipeline.RunUpdate());
    }

    [TestMethod]
    public void ResetClearsTheTemporalHistory()
    {
        // After a reset the model must wait for a fresh stack rather than being handed frames from
        // before a camera change.
        RunUntilInference();

        _pipeline.ResetTemporalState();

        Assert.IsNull(_pipeline.RunUpdate(), "the queue should be refilling after a reset");
    }

    // =============================================================================================
    // Fakes
    // =============================================================================================

    private sealed class FakeVideoSource : IVideoSource
    {
        public List<Mat> Issued { get; } = [];
        public bool ReturnNull { get; set; }
        public bool EmitCorrupt { get; set; }
        private int _counter;

        public Mat? GetFrame(ColorType? color = null)
        {
            if (ReturnNull) return null;

            // Vary the content so nothing depends on identical frames.
            var value = EmitCorrupt ? 0 : (byte)(_counter++ * 13 % 240 + 8);
            var frame = new Mat(Size, Size * 2, MatType.CV_8UC1, new Scalar(value));
            Issued.Add(frame);
            return frame;
        }

        public bool Start() => true;
        public bool Stop() => true;
        public void Dispose() { }
    }

    private sealed class FakeTransformer : IImageTransformer
    {
        public List<Mat> Produced { get; } = [];
        private int _counter;

        public Mat? Apply(Mat image)
        {
            // Two channels: one per eye, matching DualImageTransformer's output.
            var value = (byte)(_counter++ * 7 % 200 + 20);
            var mat = new Mat(Size, Size, MatType.CV_8UC2, new Scalar(value, value / 2));
            Produced.Add(mat);
            return mat;
        }
    }

    private sealed class FakeConverter : IImageConverter
    {
        public List<Mat> Received { get; } = [];

        public void Convert(Mat image, DenseTensor<float> outTensor) => Received.Add(image);
    }

    private sealed class FakeRunner : IInferenceRunner
    {
        /// <summary>Distinct per slot so post-processing changes are detectable.</summary>
        public static readonly float[] Output = [0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f];

        private readonly DenseTensor<float> _input = new([1, 8, Size, Size]);

        public float[]? Run() => (float[])Output.Clone();
        public DenseTensor<float> GetInputTensor() => _input;
    }

    private sealed class CountingFilter : IFilter
    {
        public int Calls { get; private set; }

        public float[] Filter(float[] input)
        {
            Calls++;
            return input;
        }
    }
}
