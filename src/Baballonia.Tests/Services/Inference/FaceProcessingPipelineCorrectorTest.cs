using System.Collections.Generic;
using System.Linq;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Personalization;
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

    private sealed class FakeVideoSource : IVideoSource
    {
        public Mat? GetFrame(ColorType? color = null) => new(8, 8, MatType.CV_8UC1, new Scalar(120));
        public bool Start() => true;
        public bool Stop() => true;
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
        public float[]? Run() => (float[])Output.Clone();
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

    private sealed class DoublingFilter : IFilter
    {
        public float[]? LastInput;
        public float[] Filter(float[] input)
        {
            LastInput = input;
            return input.Select(v => v * 2f).ToArray();
        }
    }

    private static float[] CreateVector(System.Func<int, float> f) =>
        Enumerable.Range(0, N).Select(f).ToArray();

    private static FaceProcessingPipeline BuildPipeline(FacePipelineEventBus bus, FakeInference inference) =>
        new(bus)
        {
            VideoSource = new FakeVideoSource(),
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

        CollectionAssert.AreEqual(inference.Output, result,
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
        CollectionAssert.AreEqual(inference.Output.Select(v => (v + 0.5f) * 2f).ToArray(), result);
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

        CollectionAssert.AreNotEqual(inference.Output, personalized);
        CollectionAssert.AreEqual(inference.Output, stock,
            "Clearing the corrector must restore stock behavior on the very next frame.");
    }

    [TestMethod]
    public void NullInferenceResult_DoesNotPublishOrThrow()
    {
        var bus = new FacePipelineEventBus();
        var raw = 0;
        bus.Subscribe<FacePipelineEvents.NewRawExpressionsEvent>(_ => raw++);

        var pipeline = new FaceProcessingPipeline(bus)
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
        public float[]? Run() => null;
        public DenseTensor<float> GetInputTensor() => _tensor;
    }
}
