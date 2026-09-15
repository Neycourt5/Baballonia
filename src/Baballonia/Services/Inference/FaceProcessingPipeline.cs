using Baballonia.Contracts;
using Baballonia.Services.events;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.Audio;
using Baballonia.Services.Personalization.C2;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Baballonia.Services.Inference;

public class FaceProcessingPipeline(IFacePipelineEventBus facePipelineEventBus, PipelineMetrics metrics) : DefaultProcessingPipeline
{
    private IExpressionCorrector? _corrector;
    private IExpressionEnhancer? _enhancer;
    private C2CandidateCorrector? _c2;
    private IExpressionCorrector? _c2Reference;
    private IInferenceRunner? _c2Inference;
    private IFilter? _c2Filter;
    private float _c2ReferenceBlend;
    private Func<bool>? _c2ContextMatches;

    public C2CandidateCorrector? SwapC2(C2CandidateCorrector? replacement, Func<bool>? contextMatches = null)
    {
        lock (SyncRoot)
        {
            var previous = _c2;
            _c2 = replacement;
            _c2Reference = _corrector;
            _c2Inference = InferenceService;
            _c2Filter = Filter;
            _c2ReferenceBlend = _corrector?.Blend ?? 0;
            _c2ContextMatches = contextMatches;
            return previous;
        }
    }

    // Reusable positional views of the keyed model output, so the per-frame personalization path
    // allocates nothing. Only ever touched under SyncRoot, i.e. by the face worker or by whoever
    // is reconfiguring the pipeline - never concurrently.
    private float[]? _stockVector;
    private OrderedFloatMap? _vectorSourceMap;

    /// <summary>
    /// Optional user-specific correction stage, applied between stock inference and the filter.
    /// Null (the default) means stock behavior. Declared here rather than on
    /// <see cref="DefaultProcessingPipeline"/> so the eye pipeline is unaffected.
    /// The setter waits for an in-flight tick so the owner can safely dispose the outgoing ONNX
    /// session once assignment returns.
    /// </summary>
    public IExpressionCorrector? Corrector
    {
        get { lock (SyncRoot) return _corrector; }
        set { lock (SyncRoot) _corrector = value; }
    }

    /// <summary>
    /// Optional audio-driven enhancement, applied after correction and before the filter.
    /// </summary>
    /// <remarks>
    /// Placed here rather than further downstream for three reasons. The One Euro filter then
    /// smooths the gain changes, so a microphone glitch cannot put a step on the wire. The recording
    /// tap upstream is unaffected, so training data can never be contaminated by audio. And the
    /// calibration remap stays last, so the user's own output ranges still apply.
    ///
    /// Null is the default and means the audio path does not exist, producing byte-identical output
    /// to a build without this field.
    /// </remarks>
    public IExpressionEnhancer? Enhancer
    {
        get { lock (SyncRoot) return _enhancer; }
        set { lock (SyncRoot) _enhancer = value; }
    }

    /// <summary>
    /// Atomically replaces the stock runner after any active inference/correction has finished.
    /// The returned runner is no longer reachable by a tick and is therefore safe to dispose.
    /// </summary>
    public IInferenceRunner? SwapInferenceService(IInferenceRunner? replacement)
    {
        lock (SyncRoot)
        {
            var previous = InferenceService;
            InferenceService = replacement;
            // The new runner may expose a different key set; drop the cached positional view so it
            // is rebuilt (and re-validated against the schema) on the next frame.
            _vectorSourceMap = null;
            return previous;
        }
    }

    public OrderedFloatMap? RunUpdate()
    {
        var sw = Stopwatch.StartNew();

        // `frame` is owned by us; `using` frees it on every exit path. Subscribers copy the Mat
        // synchronously during Publish, so disposing afterwards is safe.
        using var frame = VideoSource?.GetFrame(ColorType.Gray8);
        if(frame == null)
            return null;

        facePipelineEventBus.Publish(new FacePipelineEvents.NewFrameEvent(frame));
        metrics.FaceCaptureMs = PipelineMetrics.Ewma(metrics.FaceCaptureMs, sw.Elapsed.TotalMilliseconds);

        sw.Restart();
        var transformed = ImageTransformer?.Apply(frame);

        if(transformed == null)
            return null;

        facePipelineEventBus.Publish(new FacePipelineEvents.NewTransformedFrameEvent(transformed));
        metrics.FaceTransformMs = PipelineMetrics.Ewma(metrics.FaceTransformMs, sw.Elapsed.TotalMilliseconds);

        if (InferenceService == null)
        {
            transformed.Dispose();
            return null;
        }

        // `transformed` must outlive inference: the personalization dataset recorder pairs the exact
        // frame that produced a stock vector with that vector, so it is disposed in the finally
        // block below rather than immediately after conversion.
        try
        {
            OrderedFloatMap? inferenceResult;

            // Held for the whole inference + correction block so that swapping the runner or the
            // corrector waits for the in-flight frame, and the caller can then safely dispose the
            // outgoing ONNX session. The processing worker already holds this lock around
            // RunUpdate; Monitor is re-entrant, so taking it again here costs nothing and makes the
            // guarantee hold for any other caller too.
            lock (SyncRoot)
            {
                sw.Restart();
                // One runner for conversion, inference and embedding access. A hot reload waits at
                // this lock and then receives a runner no tick can still be using.
                var inference = InferenceService;
                if (inference == null)
                    return null;

                ImageConverter?.Convert(transformed, inference.GetInputTensor());

                inferenceResult = inference.Run();
                if (inferenceResult == null)
                    return null;
                metrics.FaceInferenceMs = PipelineMetrics.Ewma(metrics.FaceInferenceMs, sw.Elapsed.TotalMilliseconds);

                sw.Restart();

                var corrector = _corrector;
                var enhancer = _enhancer;
                if (_c2 != null && corrector == null)
                    _c2.Invalidate("The working reference is no longer active.");

                // Recording is upstream of every optional stage. Publish even if a corrector later
                // fails, while the transformed frame that produced this stock vector is alive.
                var stockVector = ToVector(inferenceResult);
                facePipelineEventBus.Publish(
                    new FacePipelineEvents.NewRawExpressionsEvent(
                        transformed, stockVector, DateTime.UtcNow.Ticks,
                        (inference as IEmbeddingSource)?.GetEmbedding(), Stopwatch.GetTimestamp(),
                        VideoSource?.LastFrameIdentity));

                if (corrector != null || enhancer != null)
                {
                    // The corrector contract requires the stock array not be mutated and a *new*
                    // array returned, so `stockVector` stays valid for the corrected event below.
                    var result = stockVector;

                    if (corrector != null)
                    {
                        // Model C reads the stock network's internal features instead of the frame.
                        // A missing embedding is an explicit stock passthrough in that corrector.
                        result = corrector is IEmbeddingAwareCorrector embeddingAware
                            ? embeddingAware.Correct(inference.GetInputTensor(), stockVector,
                                (inference as IEmbeddingSource)?.GetEmbedding())
                            : corrector.Correct(inference.GetInputTensor(), stockVector);

                        facePipelineEventBus.Publish(
                            new FacePipelineEvents.NewCorrectedExpressionsEvent(stockVector, result));
                    }

                    if (_c2 != null)
                    {
                        if (!ReferenceEquals(_c2Reference, corrector))
                            _c2.Invalidate("Reference Model C was reloaded or replaced. Choose Keep using this C2 again after selecting the matching Model C.");
                        else if (!ReferenceEquals(_c2Inference, inference))
                            _c2.Invalidate("The face feature runner was reloaded. Choose Keep using this C2 again once the matching runner is ready.");
                        else if (!ReferenceEquals(_c2Filter, Filter))
                            _c2.Invalidate("The face smoothing filter was replaced. Restore the recorded smoothing settings, then choose Keep using this C2.");
                        else if (corrector?.Blend != _c2ReferenceBlend)
                            _c2.Invalidate("Model C blend strength changed. Restore its training strength, then choose Keep using this C2.");
                        else if (_c2ContextMatches != null && !_c2ContextMatches())
                            _c2.Invalidate("Camera or output settings changed. Restore the recorded settings, then choose Keep using this C2.");
                        result = _c2.Correct(result, (inference as IEmbeddingSource)?.GetEmbedding());
                    }

                    if (enhancer != null)
                        result = enhancer.Enhance(result);

                    if (!ReferenceEquals(result, stockVector))
                        FromVector(result, inferenceResult);
                }
            }

            if (Filter != null)
                inferenceResult = Filter.Filter(inferenceResult);

            facePipelineEventBus.Publish(new FacePipelineEvents.NewFilteredResultEvent(inferenceResult));
            metrics.FacePostMs = PipelineMetrics.Ewma(metrics.FacePostMs, sw.Elapsed.TotalMilliseconds);

            return inferenceResult;
        }
        finally
        {
            if (!ReferenceEquals(transformed, frame))
                transformed.Dispose();
        }
    }

    /// <summary>
    /// The boundary between the keyed pipeline and the positional personalization world.
    /// </summary>
    /// <remarks>
    /// Everything downstream of the model in this app speaks <see cref="OrderedFloatMap"/>, keyed by
    /// OSC address. Everything in personalization - the recorded dataset, the Python trainer, the
    /// exported ONNX adapters and their <c>expression_schema_sha256</c> - speaks a positional
    /// <c>float[45]</c> in <see cref="PersonalizationSchema"/> order. Converting here, at the single
    /// point where the two meet, is what lets the trained models and the 20k+ recorded frames stay
    /// valid without retyping the whole personalization chain.
    ///
    /// The index table is built once per model and validated by
    /// <see cref="PersonalizationSchemaBinding.Bind"/>, which throws if the running model's key set
    /// does not match the schema. A silent mismatch would feed every expression to the wrong slot.
    /// </remarks>
    private float[] ToVector(OrderedFloatMap map)
    {
        if (!ReferenceEquals(_vectorSourceMap, map))
        {
            PersonalizationSchemaBinding.Bind(map);
            _vectorSourceMap = map;
            _stockVector = new float[PersonalizationSchema.ExpressionCount];
        }

        var values = map.ValuesSpan;
        var vector = _stockVector!;
        // Bind() proved the key order is identical to the schema order, so this is a straight copy.
        values[..PersonalizationSchema.ExpressionCount].CopyTo(vector);
        return vector;
    }

    private static void FromVector(float[] vector, OrderedFloatMap map)
    {
        var count = Math.Min(vector.Length, PersonalizationSchema.ExpressionCount);
        vector.AsSpan(0, count).CopyTo(map.ValuesSpan);
    }

    public void Dispose()
    {
        TryDisposeObject(VideoSource);
        TryDisposeObject(ImageTransformer);
        TryDisposeObject(ImageConverter);
        TryDisposeObject(SwapInferenceService(null));
        TryDisposeObject(Filter);
    }

    private void TryDisposeObject(object? obj)
    {
        (obj as IDisposable)?.Dispose();
    }
}
