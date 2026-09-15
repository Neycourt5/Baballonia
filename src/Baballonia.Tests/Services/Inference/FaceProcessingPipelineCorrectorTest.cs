using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.Audio;
using JetBrains.Annotations;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Inference;

/// <summary>
/// Covers the personalization hooks added to the face pipeline: the raw-result event that dataset
/// recording taps, the optional corrector stage, and - most importantly - that installing nothing
/// leaves stock behavior byte-for-byte unchanged.
///
/// Uses fakes rather than the real 24 MB ONNX model so the assertions are about pipeline wiring
/// (ordering, event payloads, passthrough) and not about model numerics.
/// </summary>
[TestClass]
[TestSubject(typeof(FaceProcessingPipeline))]
public class FaceProcessingPipelineCorrectorTest
{
    private const int N = PersonalizationSchema.ExpressionCount;

    private sealed class EmbeddedInference : IInferenceRunner, IEmbeddingSource
    {
        private readonly FakeInference _runner = new();
        private readonly DenseTensor<float> _embedding = new([1,1280]);
        public OrderedFloatMap? Run() => _runner.Run();
        public DenseTensor<float> GetInputTensor() => _runner.GetInputTensor();
        public DenseTensor<float>? GetEmbedding() => _embedding;
    }

    [TestMethod]
    public void C2RunsAfterCBeforeFilterAndContextFailurePermanentlyRetainsC()
    {
        var pipeline = BuildPipeline(new FacePipelineEventBus(),new EmbeddedInference());
        pipeline.Corrector = new AddConstCorrector(.2f);
        pipeline.Filter = new DoublingFilter();
        using var candidate = new Baballonia.Services.Personalization.C2.C2CandidateCorrector(
            System.IO.Path.Combine(AppContext.BaseDirectory,"Assets","PersonalModels","c2Adapter.onnx"),"fixture-context");
        var contextMatches = true;
        pipeline.SwapC2(candidate,()=>contextMatches);
        Assert.AreEqual(.68f,ValuesOf(pipeline.RunUpdate())[4],1e-6f);
        contextMatches = false;
        Assert.AreEqual(.48f,ValuesOf(pipeline.RunUpdate())[4],1e-6f);
        Assert.IsNotNull(candidate.Failure);
        contextMatches = true;
        Assert.AreEqual(.48f,ValuesOf(pipeline.RunUpdate())[4],1e-6f);
        Assert.AreSame(candidate,pipeline.SwapC2(null));
        Assert.AreEqual(.48f,ValuesOf(pipeline.RunUpdate())[4],1e-6f);
    }

    [TestMethod]
    public async Task C2SwapWaitsUntilTheInFlightFrameNoLongerUsesTheCandidate()
    {
        var pipeline = BuildPipeline(new FacePipelineEventBus(),new EmbeddedInference());
        var blocker = new BlockingCorrector();
        pipeline.Corrector = blocker;
        using var candidate = new Baballonia.Services.Personalization.C2.C2CandidateCorrector(
            System.IO.Path.Combine(AppContext.BaseDirectory,"Assets","PersonalModels","c2Adapter.onnx"),"fixture-context");
        pipeline.SwapC2(candidate);
        var tick = Task.Run(pipeline.RunUpdate);
        await blocker.Entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var swap = Task.Run(()=>pipeline.SwapC2(null));
        try
        {
            await Task.Delay(30);
            Assert.IsFalse(swap.IsCompleted,"Publication/disposal must wait for the in-flight correction.");
        }
        finally { blocker.Release(); }
        Assert.AreEqual(.14f,ValuesOf(await tick.WaitAsync(TimeSpan.FromSeconds(5)))[4],1e-6f);
        Assert.AreSame(candidate,await swap.WaitAsync(TimeSpan.FromSeconds(5)));
    }

    private sealed class GainEnhancer : IExpressionEnhancer
    {
        public float[] Enhance(float[] expressions) => expressions.Select(value => value * 1.25f).ToArray();
    }

    [TestMethod]
    public void KeptC2FeedsAudioAssistBeforeSmoothingAndAllowsAudioToggles()
    {
        var pipeline = BuildPipeline(new FacePipelineEventBus(), new EmbeddedInference());
        pipeline.Corrector = new AddConstCorrector(.2f);
        pipeline.Filter = new DoublingFilter();
        using var candidate = new Baballonia.Services.Personalization.C2.C2CandidateCorrector(
            System.IO.Path.Combine(AppContext.BaseDirectory, "Assets", "PersonalModels", "c2Adapter.onnx"), "fixture-context");
        pipeline.SwapC2(candidate, () => true);
        Assert.AreEqual(.68f, ValuesOf(pipeline.RunUpdate())[4], 1e-6f);
        pipeline.Enhancer = new GainEnhancer();
        Assert.AreEqual(.85f, ValuesOf(pipeline.RunUpdate())[4], 1e-6f,
            "Audio must amplify the C2-corrected visual output before smoothing.");
        pipeline.Enhancer = null;
        Assert.AreEqual(.68f, ValuesOf(pipeline.RunUpdate())[4], 1e-6f);
        Assert.IsNull(candidate.Failure);
        pipeline.Corrector = new AddConstCorrector(.1f);
        Assert.AreEqual(.28f, ValuesOf(pipeline.RunUpdate())[4], 1e-6f);
        StringAssert.Contains(candidate.Failure!, "Reference Model C was reloaded or replaced");
    }

    // The pipeline speaks OrderedFloatMap keyed by OSC address; personalization speaks a positional
    // float[45]. These two helpers are the test-side mirror of the pipeline's own boundary adapter.
    private static OrderedFloatMap NewMap() =>
        new(PersonalizationSchemaBinding.ExpectedKeys.ToArray());

    private static OrderedFloatMap MapOf(float[] values)
    {
        var map = NewMap();
        values.AsSpan(0, N).CopyTo(map.ValuesSpan);
        return map;
    }

    private static float[] ValuesOf(OrderedFloatMap? map) => map?.Values.ToArray() ?? [];

    private sealed class FakeVideoSource : IVideoSource
    {
        public Mat? LastFrame { get; private set; }
        public Mat? GetFrame(ColorType? color = null) =>
            LastFrame = new Mat(8, 8, MatType.CV_8UC1, new Scalar(120));
        public bool Start() => true;
        public bool Stop() => true;
        public WaitHandle[] GetFrameWaitHandles() => [];
        public void Dispose() { }
    }

    /// <summary>Passes the frame through as a distinct Mat, mimicking a real transform allocation.</summary>
    private sealed class FakeTransformer : IImageTransformer
    {
        public Mat? Apply(Mat image) => image.Clone();
    }

    private sealed class FakeConverter : IImageConverter
    {
        public void Convert(Mat input, DenseTensor<float> outTensor) { }
    }

    private sealed class FakeInference : IInferenceRunner
    {
        private readonly DenseTensor<float> _tensor = new([1, 1, 224, 224]);
        public float[] Output = CreateVector(i => i / 100f);
        public OrderedFloatMap? Run() => MapOf(Output);
        public DenseTensor<float> GetInputTensor() => _tensor;
    }

    /// <summary>Adds a fixed delta so corrected output is trivially distinguishable from stock.</summary>
    private sealed class AddConstCorrector(float delta) : IExpressionCorrector
    {
        public float Blend { get; set; } = 1f;
        public DenseTensor<float>? LastImage;
        public float[]? LastStock;

        public float[] Correct(DenseTensor<float> image, float[] stock)
        {
            LastImage = image;
            LastStock = stock;
            return stock.Select(v => v + delta).ToArray();
        }
    }

    private sealed class BlockingCorrector : IExpressionCorrector
    {
        private readonly ManualResetEventSlim _release = new();
        public TaskCompletionSource Entered { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);
        public float Blend { get; set; } = 1f;

        public float[] Correct(DenseTensor<float> image, float[] stock)
        {
            Entered.SetResult();
            _release.Wait();
            return (float[])stock.Clone();
        }

        public void Release() => _release.Set();
    }

    private sealed class BlockingInference : IInferenceRunner, IDisposable
    {
        private readonly DenseTensor<float> _tensor = new([1, 1, 224, 224]);
        private readonly ManualResetEventSlim _release = new();
        public TaskCompletionSource Entered { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);
        public bool Disposed { get; private set; }

        public OrderedFloatMap? Run()
        {
            Entered.SetResult();
            _release.Wait();
            return MapOf(CreateVector(i => i / 100f));
        }

        public DenseTensor<float> GetInputTensor() => _tensor;
        public void Release() => _release.Set();
        public void Dispose() => Disposed = true;
    }

    private sealed class DoublingFilter : IFilter
    {
        public float[]? LastInput;
        public OrderedFloatMap Filter(OrderedFloatMap input)
        {
            // Snapshot: the pipeline reuses the map, so holding the reference would not prove what
            // the filter actually saw.
            LastInput = ValuesOf(input);
            var doubled = MapOf(LastInput.Select(v => v * 2f).ToArray());
            return doubled;
        }
    }

    private static float[] CreateVector(System.Func<int, float> f) =>
        Enumerable.Range(0, N).Select(f).ToArray();

    private static FaceProcessingPipeline BuildPipeline(
        FacePipelineEventBus bus,
        IInferenceRunner inference,
        FakeVideoSource? source = null) =>
        new(bus, new PipelineMetrics())
        {
            VideoSource = source ?? new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = inference
        };

    [TestMethod]
    public void NoCorrector_ReturnsStockOutputUnchanged()
    {
        var bus = new FacePipelineEventBus();
        var inference = new FakeInference();
        var pipeline = BuildPipeline(bus, inference);

        var result = pipeline.RunUpdate();

        CollectionAssert.AreEqual(inference.Output, ValuesOf(result),
            "With no corrector installed the pipeline must return the stock vector untouched.");
    }

    [TestMethod]
    public void NoCorrector_PublishesNoCorrectedEvent()
    {
        var bus = new FacePipelineEventBus();
        var corrected = 0;
        bus.Subscribe<FacePipelineEvents.NewCorrectedExpressionsEvent>(_ => corrected++);

        BuildPipeline(bus, new FakeInference()).RunUpdate();

        Assert.AreEqual(0, corrected,
            "The corrected event is personalization-only and must stay silent in stock configuration.");
    }

    [TestMethod]
    public void RawEvent_CarriesPreFilterValuesAndTheFrameThatProducedThem()
    {
        var bus = new FacePipelineEventBus();
        var inference = new FakeInference();
        FacePipelineEvents.NewRawExpressionsEvent? seen = null;
        var frameWasUsable = false;
        long ticks = 0;

        bus.Subscribe<FacePipelineEvents.NewRawExpressionsEvent>(e =>
        {
            seen = e;
            ticks = e.timestampTicks;
            // Handlers get a live Mat; recording clones it here.
            frameWasUsable = !e.transformedFrame.IsDisposed && e.transformedFrame.Width == 8;
        });

        var pipeline = BuildPipeline(bus, inference);
        pipeline.Filter = new DoublingFilter(); // must NOT affect the raw payload
        pipeline.RunUpdate();

        Assert.IsNotNull(seen, "Raw event was never published.");
        CollectionAssert.AreEqual(inference.Output, seen.rawResult,
            "Raw event must carry pre-filter stock values; recording depends on this.");
        Assert.IsTrue(frameWasUsable, "Frame must still be alive while handlers run.");
        Assert.IsTrue(ticks > 0, "Timestamp must be populated for dataset synchronization.");
    }

    [TestMethod]
    public void RawEvent_FrameIsDisposedAfterPublishSoHandlersMustClone()
    {
        var bus = new FacePipelineEventBus();
        Mat? captured = null;
        bus.Subscribe<FacePipelineEvents.NewRawExpressionsEvent>(e => captured = e.transformedFrame);

        BuildPipeline(bus, new FakeInference()).RunUpdate();

        Assert.IsNotNull(captured);
        Assert.IsTrue(captured.IsDisposed,
            "Pipeline owns the transformed Mat and must dispose it after publishing; this pins the " +
            "contract that handlers clone rather than retain.");
    }

    [TestMethod]
    public void CameraFrameIsDisposedAfterEveryFaceTick()
    {
        var source = new FakeVideoSource();
        var pipeline = BuildPipeline(new FacePipelineEventBus(), new FakeInference(), source);

        pipeline.RunUpdate();

        Assert.IsNotNull(source.LastFrame);
        Assert.IsTrue(source.LastFrame.IsDisposed,
            "the camera-owned native Mat leaked after a successful face inference tick");
    }

    [TestMethod]
    public async Task CorrectorSwapWaitsForInFlightCorrectionBeforeReturning()
    {
        var pipeline = BuildPipeline(new FacePipelineEventBus(), new FakeInference());
        var blocking = new BlockingCorrector();
        pipeline.Corrector = blocking;
        var tick = Task.Run(pipeline.RunUpdate);
        await blocking.Entered.Task.WaitAsync(TimeSpan.FromSeconds(5));

        var swapAttempted = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var replacement = new AddConstCorrector(0.1f);
        var swap = Task.Run(() =>
        {
            swapAttempted.SetResult();
            pipeline.Corrector = replacement;
        });

        try
        {
            await swapAttempted.Task;
            await Task.Delay(50);
            Assert.IsFalse(swap.IsCompleted,
                "the outgoing corrector could be disposed while Correct was still executing");
        }
        finally
        {
            blocking.Release();
        }

        await Task.WhenAll(tick, swap).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.AreSame(replacement, pipeline.Corrector);
    }

    [TestMethod]
    public async Task InferenceSwapReturnsOldRunnerOnlyAfterItsTickFinishes()
    {
        var blocking = new BlockingInference();
        var pipeline = BuildPipeline(new FacePipelineEventBus(), blocking);
        var tick = Task.Run(pipeline.RunUpdate);
        await blocking.Entered.Task.WaitAsync(TimeSpan.FromSeconds(5));

        var swapAttempted = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var swap = Task.Run(() =>
        {
            swapAttempted.SetResult();
            var outgoing = pipeline.SwapInferenceService(new FakeInference());
            ((IDisposable)outgoing!).Dispose();
        });

        try
        {
            await swapAttempted.Task;
            await Task.Delay(50);
            Assert.IsFalse(swap.IsCompleted);
            Assert.IsFalse(blocking.Disposed,
                "the old native runner was disposed while Run was still active");
        }
        finally
        {
            blocking.Release();
        }

        await Task.WhenAll(tick, swap).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.IsTrue(blocking.Disposed);
    }

    [TestMethod]
    public void Corrector_OutputIsFilteredAndReturned()
    {
        var bus = new FacePipelineEventBus();
        var inference = new FakeInference();
        var filter = new DoublingFilter();
        var pipeline = BuildPipeline(bus, inference);
        pipeline.Corrector = new AddConstCorrector(0.5f);
        pipeline.Filter = filter;

        var result = pipeline.RunUpdate();

        // Corrector runs before the filter, so the filter sees stock+0.5 and output is (stock+0.5)*2.
        CollectionAssert.AreEqual(inference.Output.Select(v => v + 0.5f).ToArray(), filter.LastInput,
            "The filter must smooth the corrected signal, i.e. the corrector runs first.");
        CollectionAssert.AreEqual(inference.Output.Select(v => (v + 0.5f) * 2f).ToArray(), ValuesOf(result));
    }

    [TestMethod]
    public void Corrector_ReceivesRawStockAndInputTensor()
    {
        var bus = new FacePipelineEventBus();
        var inference = new FakeInference();
        var corrector = new AddConstCorrector(0.1f);
        var pipeline = BuildPipeline(bus, inference);
        pipeline.Corrector = corrector;
        pipeline.Filter = new DoublingFilter();

        pipeline.RunUpdate();

        CollectionAssert.AreEqual(inference.Output, corrector.LastStock,
            "Corrector must see raw pre-filter stock values, matching what training recorded.");
        Assert.AreSame(inference.GetInputTensor(), corrector.LastImage,
            "Corrector should reuse the tensor the stock model just consumed (zero copy).");
    }

    [TestMethod]
    public void CorrectedEvent_CarriesBothVectors()
    {
        var bus = new FacePipelineEventBus();
        var inference = new FakeInference();
        var events = new List<FacePipelineEvents.NewCorrectedExpressionsEvent>();
        bus.Subscribe<FacePipelineEvents.NewCorrectedExpressionsEvent>(events.Add);

        var pipeline = BuildPipeline(bus, inference);
        pipeline.Corrector = new AddConstCorrector(0.25f);
        pipeline.RunUpdate();

        Assert.AreEqual(1, events.Count);
        CollectionAssert.AreEqual(inference.Output, events[0].rawResult);
        CollectionAssert.AreEqual(inference.Output.Select(v => v + 0.25f).ToArray(), events[0].correctedResult);
    }

    [TestMethod]
    public void RemovingCorrector_RestoresStockImmediately()
    {
        var bus = new FacePipelineEventBus();
        var inference = new FakeInference();
        var pipeline = BuildPipeline(bus, inference);

        pipeline.Corrector = new AddConstCorrector(0.5f);
        var personalized = pipeline.RunUpdate();

        pipeline.Corrector = null;
        var stock = pipeline.RunUpdate();

        CollectionAssert.AreNotEqual(inference.Output, ValuesOf(personalized));
        CollectionAssert.AreEqual(inference.Output, ValuesOf(stock),
            "Clearing the corrector must restore stock behavior on the very next frame.");
    }

    [TestMethod]
    public void NullInferenceResult_DoesNotPublishOrThrow()
    {
        var bus = new FacePipelineEventBus();
        var raw = 0;
        bus.Subscribe<FacePipelineEvents.NewRawExpressionsEvent>(_ => raw++);

        var pipeline = new FaceProcessingPipeline(bus, new PipelineMetrics())
        {
            VideoSource = new FakeVideoSource(),
            ImageTransformer = new FakeTransformer(),
            ImageConverter = new FakeConverter(),
            InferenceService = new NullInference()
        };

        Assert.IsNull(pipeline.RunUpdate());
        Assert.AreEqual(0, raw, "No raw event should fire when inference produced nothing.");
    }

    private sealed class NullInference : IInferenceRunner
    {
        private readonly DenseTensor<float> _tensor = new([1, 1, 224, 224]);
        public OrderedFloatMap? Run() => null;
        public DenseTensor<float> GetInputTensor() => _tensor;
    }
}
