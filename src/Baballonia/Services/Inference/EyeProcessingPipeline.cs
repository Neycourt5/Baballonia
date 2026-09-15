using System;
using System.Collections.Generic;
using System.Diagnostics;
using Baballonia.Services.events;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.Eye;

namespace Baballonia.Services.Inference;

public class EyeProcessingPipeline(IEyePipelineEventBus eyePipelineEventBus, PipelineMetrics metrics,
    EyeStageTrace? stageTrace = null) : DefaultProcessingPipeline, IDisposable
{
    private long _traceFrame;
    private readonly FastCorruptionDetector.FastCorruptionDetector _fastCorruptionDetector = new();
    private readonly ImageCollector _imageCollector = new();
    private float _lastEyeY;
    private long _lastPostProcessTimestamp;

    private IExpressionCorrector? _corrector;
    private float[]? _stockVector;
    private float[]? _correctedScratch;
    private OrderedFloatMap? _boundMap;
    private bool _bindingUsable;
    private string? _reportedBindingError;

    /// <summary>
    /// Optional personalized correction, applied to the model's own output before filtering and
    /// geometry. Null (the default) is exactly stock behaviour.
    /// </summary>
    /// <remarks>
    /// Raw model space is the only place per-eye correction is meaningful: the geometry pass
    /// downstream shares one vertical gaze between the eyes and cross-blends horizontal by lid
    /// openness, so a per-eye correction applied there could not be inverted back to one eye.
    ///
    /// Assignment takes <see cref="DefaultProcessingPipeline.SyncRoot"/>, which the inference worker
    /// holds for the whole tick, so the caller may dispose the outgoing corrector once the setter
    /// returns.
    /// </remarks>
    public IExpressionCorrector? Corrector
    {
        get { lock (SyncRoot) return _corrector; }
        set { lock (SyncRoot) _corrector = value; }
    }

    public bool StabilizeEyes { get; set; } = true;

    /// <summary>
    /// How strongly the two eyes are forced to point the same way, 0..1. Zero preserves whatever
    /// vergence survives <see cref="StabilizeEyes"/>; one makes them fully conjugate. Only has an
    /// effect while <see cref="StabilizeEyes"/> is on, since that is where vergence is computed.
    /// </summary>
    public float GazeConjugateAmount { get; set; }
    public EyeOutputPostProcessor? PostProcessor { get; set; }

    /// <summary>
    /// For single-camera (split) eye feeds, swap which half drives which eye. Off by default — most
    /// split devices (e.g. BSB2E) want the unswapped orientation. User-controlled via the
    /// "Split Eye Video Swap" advanced setting; has no effect on dual-camera setups.
    /// </summary>
    public bool SwapSplitEyes { get; set; }

    /// <summary>
    /// The raw (un-smoothed) result of the most recent <see cref="RunUpdate"/> — geometry-corrected
    /// exactly like the returned map but with the OneEuroFilter skipped. Native eye tracking (DFR /
    /// VRChat native) reads this for lowest latency. Shares RunUpdate's reused-buffer lifetime: only
    /// valid until the next RunUpdate on this pipeline.
    /// </summary>
    public OrderedFloatMap? RawEyeResult { get; private set; }

    /// <summary>Drops temporal frames whenever the camera source or eye assignment changes.</summary>
    public void ResetTemporalHistory()
    {
        _imageCollector.Reset();
        RawEyeResult = null;
        _lastEyeY = 0f;
        _lastPostProcessTimestamp = 0;
        PostProcessor?.Reset();
        _boundMap = null;
    }

    public OrderedFloatMap? RunUpdate()
    {
        var sw = Stopwatch.StartNew();

        // `frame` is owned by us (AcquireRawMat contract); `using` frees it on every exit path.
        // Subscribers to the published events copy the Mat synchronously during Publish, so it is
        // safe to dispose afterwards.
        using var frame = VideoSource?.GetFrame(ColorType.Gray8);
        if(frame == null)
            return null;

        if (_fastCorruptionDetector.IsCorrupted(frame).isCorrupted)
        {
            // Counted rather than dropped silently, so "tracking went quiet in dim light" can be
            // told apart from "the camera stalled" on the Debug page.
            System.Threading.Interlocked.Increment(ref metrics.EyeCorruptFrames);
            return null;
        }

        eyePipelineEventBus.Publish(new EyePipelineEvents.NewFrameEvent(frame));
        metrics.EyeCaptureMs = PipelineMetrics.Ewma(metrics.EyeCaptureMs, sw.Elapsed.TotalMilliseconds);

        // A single-camera (split-eye) feed drives both eyes from one sensor. Whether to swap which
        // half feeds which eye is user-controlled (SwapSplitEyes / "Split Eye Video Swap"), defaulting
        // to unswapped. Dual-camera setups assign each eye explicitly, so they are always left as-is.
        if (ImageTransformer is DualImageTransformer splitTransformer)
            splitTransformer.SwapEyes = VideoSource is VideoSources.SingleCameraSource && SwapSplitEyes;

        sw.Restart();
        using var transformed = ImageTransformer?.Apply(frame);
        if(transformed == null)
            return null;

        eyePipelineEventBus.Publish(new EyePipelineEvents.NewTransformedFrameEvent(transformed));

        // `collected` (the 8-channel temporal stack) is owned by us; `using` frees it on all paths.
        using var collected = _imageCollector.Apply(transformed);
        if (collected == null)
            return null;
        metrics.EyeTransformMs = PipelineMetrics.Ewma(metrics.EyeTransformMs, sw.Elapsed.TotalMilliseconds);

        if (InferenceService == null)
            return null;

        sw.Restart();
        ImageConverter?.Convert(collected, InferenceService.GetInputTensor());

        var inferenceResult = InferenceService?.Run();
        if(inferenceResult == null)
            return null;
        metrics.EyeInferenceMs = PipelineMetrics.Ewma(metrics.EyeInferenceMs, sw.Elapsed.TotalMilliseconds);

        var traceFrame = ++_traceFrame;
        var sourceIdentity = VideoSource?.LastFrameIdentity;
        if (stageTrace?.Enabled == true)
            stageTrace.Add(traceFrame, "alpha_raw", inferenceResult, sourceIdentity,
                $"split_swap={SwapSplitEyes}; stabilize={StabilizeEyes}; conjugate={GazeConjugateAmount}");
        Personalize(inferenceResult);
        stageTrace?.Add(traceFrame, "personal_raw", inferenceResult, sourceIdentity);

        sw.Restart();
        // OneEuroFilter returns its own buffer and leaves the input untouched, so the runner's map
        // still holds the raw values. Process both: the raw map feeds native eye tracking (DFR), the
        // filtered map feeds VRCFT/UI as before.
        OrderedFloatMap? rawForDfr = null;
        if (Filter != null)
        {
            var filtered = Filter.Filter(inferenceResult);
            stageTrace?.Add(traceFrame, "filtered_raw", filtered, sourceIdentity);
            ProcessExpressions(ref inferenceResult);
            rawForDfr = inferenceResult;
            inferenceResult = filtered;
        }

        ProcessExpressions(ref inferenceResult);
        stageTrace?.Add(traceFrame, "geometry", inferenceResult, sourceIdentity);

        // The post-processor belongs only to the filtered VRCFT/UI path. When filtering is disabled,
        // preserve a snapshot for native/DFR before sanitizing the same runner-owned map in place.
        if (PostProcessor != null)
        {
            RawEyeResult = rawForDfr ?? Clone(inferenceResult);
            // Apply may return a wider map than it was given: a model with no widen/squint channels
            // gets them derived and appended. The raw/DFR snapshot above is taken first and is
            // deliberately never widened - native tracking wants the model, not our inference.
            PostProcessor.Trace = stageTrace;
            PostProcessor.TraceFrame = traceFrame;
            inferenceResult = PostProcessor.Apply(inferenceResult, NextPostProcessDeltaSeconds());
            stageTrace?.Add(traceFrame, "post_processing", inferenceResult, sourceIdentity);
            PublishEyeDiagnostics(inferenceResult, PostProcessor);
        }
        else
        {
            RawEyeResult = rawForDfr ?? inferenceResult; // filter off: raw == filtered
            _lastPostProcessTimestamp = 0;
        }

        eyePipelineEventBus.Publish(new EyePipelineEvents.NewFilteredResultEvent(inferenceResult));
        metrics.EyePostMs = PipelineMetrics.Ewma(metrics.EyePostMs, sw.Elapsed.TotalMilliseconds);

        return inferenceResult;
    }

    /// <summary>
    /// Publishes the model's own output and applies the personalized corrector, in place.
    /// </summary>
    /// <remarks>
    /// Runs before the filter so smoothing sees the corrected signal and a correction can never put
    /// a step on the wire, and before the geometry pass because that pass couples the two eyes.
    /// Nothing here may throw: this is inside the worker's try, whose catch tears down every eye
    /// camera.
    /// </remarks>
    private void Personalize(OrderedFloatMap result)
    {
        var corrector = _corrector;

        if (!ReferenceEquals(_boundMap, result))
        {
            _boundMap = result;
            _bindingUsable = EyePersonalizationSchemaBinding.TryBind(result, out var error);
            _stockVector = _bindingUsable ? new float[EyePersonalizationSchema.ExpressionCount] : null;
            _correctedScratch = null;

            // Once per model, not once per frame: the stock six-output model is a normal thing to
            // be running and must not fill the log.
            if (!_bindingUsable && error is not null && error != _reportedBindingError)
            {
                _reportedBindingError = error;
                eyePipelineEventBus.Publish(new EyePipelineEvents.ExceptionEvent(
                    new InvalidOperationException(error)));
            }
        }

        if (!_bindingUsable || _stockVector is null)
            return;

        var stock = _stockVector;
        result.ValuesSpan.CopyTo(stock);
        eyePipelineEventBus.Publish(
            new EyePipelineEvents.NewRawEyeExpressionsEvent(stock, DateTime.UtcNow.Ticks));

        if (corrector is null)
        {
            metrics.EyeCorrectorDelta = 0f;
            return;
        }

        var sw = Stopwatch.StartNew();
        float[] corrected;
        try
        {
            corrected = corrector.Correct(InferenceService!.GetInputTensor(), stock);
        }
        catch
        {
            // A corrector that throws is a corrector that stops existing, not a camera fault.
            _corrector = null;
            metrics.EyeCorrectorDelta = 0f;
            return;
        }

        if (corrected is null || corrected.Length != stock.Length)
        {
            _corrector = null;
            metrics.EyeCorrectorDelta = 0f;
            return;
        }

        var delta = 0f;
        for (var i = 0; i < corrected.Length; i++)
        {
            if (!float.IsFinite(corrected[i]))
                return; // Leave the stock values in place; publish nothing that is not real.
            delta += Math.Abs(corrected[i] - stock[i]);
        }

        _correctedScratch ??= new float[stock.Length];
        corrected.AsSpan().CopyTo(_correctedScratch);
        eyePipelineEventBus.Publish(
            new EyePipelineEvents.NewCorrectedEyeExpressionsEvent(stock, _correctedScratch));

        corrected.AsSpan().CopyTo(result.ValuesSpan);
        metrics.EyeCorrectorDelta = delta / corrected.Length;
        metrics.EyeCorrectMs = PipelineMetrics.Ewma(metrics.EyeCorrectMs, sw.Elapsed.TotalMilliseconds);
    }

    private bool ProcessExpressions(ref OrderedFloatMap arKitExpressions)
    {

        
        const float mulV = 2.0f;
        const float mulY = 2.0f;

        var leftX = arKitExpressions["/leftEyeX"] * mulY - mulY / 2;
        var leftY = arKitExpressions["/leftEyeY"] * mulV - mulV / 2;
        var leftLid = 1 - arKitExpressions["/leftEyeLid"];

        var rightX = arKitExpressions["/rightEyeX"] * mulY - mulY / 2;
        var rightY = arKitExpressions["/rightEyeY"] * mulV - mulV / 2;
        var rightLid = 1 - arKitExpressions["/rightEyeLid"];

        var lidSum = leftLid + rightLid;
        var eyeY = _lastEyeY;
        if (lidSum > 0.02f)
        {
            var weightedEyeY = (leftY * leftLid + rightY * rightLid) / lidSum;
            if (float.IsFinite(weightedEyeY))
                eyeY = weightedEyeY;
        }
        if (float.IsFinite(eyeY))
            _lastEyeY = eyeY;

        var leftEyeXCorrected = rightX * (1 - leftLid) + leftX * leftLid;
        var rightEyeXCorrected = leftX * (1 - rightLid) + rightX * rightLid;

        if (StabilizeEyes)
        {
            var rawConvergence = (leftEyeXCorrected - rightEyeXCorrected) / 2.0f;
            var convergence = Math.Max(rawConvergence, 0.0f); // We clamp the value here to avoid accidental divergence, as the model sometimes decides that's a thing

            // How much of the remaining left/right disagreement to keep. The eyes are already
            // averaged and prevented from diverging; what is left is vergence, which is real when
            // you look at something close and noise when the two cameras merely disagree. Zero
            // keeps all of it (the long-standing behaviour); one makes the eyes fully conjugate.
            convergence *= 1f - GazeConjugateAmount;

            var averagedX = (rightEyeXCorrected + leftEyeXCorrected) / 2.0f;

            leftEyeXCorrected = averagedX + convergence;
            rightEyeXCorrected = averagedX - convergence;
        }

        // update the dict
        arKitExpressions["/leftEyeX"] = leftEyeXCorrected;
        arKitExpressions["/leftEyeY"] = eyeY;

        arKitExpressions["/rightEyeX"] = rightEyeXCorrected;
        arKitExpressions["/rightEyeY"] = eyeY;

        arKitExpressions["/leftEyeLid"] = leftLid;
        arKitExpressions["/rightEyeLid"] = rightLid;

        //try{

        //arKitExpressions["/leftEyeWiden"] = arKitExpressions["/rightEyeWiden"] = (arKitExpressions["/leftEyeWiden"] = arKitExpressions["/rightEyeWiden"]) / 2;
        //arKitExpressions["/leftEyeSquint"] = arKitExpressions["/rightEyeSquint"] = (arKitExpressions["/leftEyeSquint"] = arKitExpressions["/rightEyeSquint"]) / 2;

        //}catch{}

        return true;
    }

    /// <summary>
    /// Copies the frame's final eye values into the shared metrics for the Debug page. Cheap enough
    /// to run unconditionally: twelve float stores and no allocation.
    /// </summary>
    private void PublishEyeDiagnostics(OrderedFloatMap map, EyeOutputPostProcessor post)
    {
        static float Read(OrderedFloatMap m, string key) => m.TryGetValue(key, out var v) ? v : 0f;

        metrics.EyeLeftRawOpenness = post.LeftRawOpenness;
        metrics.EyeLeftOpenness = Read(map, "/leftEyeLid");
        metrics.EyeLeftWiden = Read(map, "/leftEyeWiden");
        metrics.EyeLeftSquint = Read(map, "/leftEyeSquint");
        metrics.EyeLeftGazeX = Read(map, "/leftEyeX");
        metrics.EyeLeftGazeY = Read(map, "/leftEyeY");

        metrics.EyeRightRawOpenness = post.RightRawOpenness;
        metrics.EyeRightOpenness = Read(map, "/rightEyeLid");
        metrics.EyeRightWiden = Read(map, "/rightEyeWiden");
        metrics.EyeRightSquint = Read(map, "/rightEyeSquint");
        metrics.EyeRightGazeX = Read(map, "/rightEyeX");
        metrics.EyeRightGazeY = Read(map, "/rightEyeY");

        metrics.EyeWidenSquintDerived = post.IsDerivingWidenSquint;
        metrics.EyeSyncReleased = post.IsWinking;

        if (post.BlinkGuard is { } guard)
        {
            metrics.EyeBlinkGuardEnabled = guard.Settings.Enabled;
            metrics.EyeBlinkGuardIntervening = guard.IsIntervening;
            metrics.EyeBlinkGuardGlitches = guard.Diagnostics.Counters.GlitchesPrevented;
        }
    }

    private float NextPostProcessDeltaSeconds()
    {
        var now = Stopwatch.GetTimestamp();
        if (_lastPostProcessTimestamp == 0)
        {
            _lastPostProcessTimestamp = now;
            return 0f;
        }

        var dt = (float)Stopwatch.GetElapsedTime(_lastPostProcessTimestamp, now).TotalSeconds;
        _lastPostProcessTimestamp = now;
        return Math.Max(dt, 0f);
    }

    private static OrderedFloatMap Clone(OrderedFloatMap source)
    {
        var result = new OrderedFloatMap(System.Linq.Enumerable.ToArray(source.Keys));
        foreach (var pair in source)
            result[pair.Key] = pair.Value;
        return result;
    }


    public void Dispose()
    {
        TryDisposeObject(VideoSource);
        TryDisposeObject(ImageTransformer);
        TryDisposeObject(ImageConverter);
        TryDisposeObject(InferenceService);
        TryDisposeObject(Filter);
        TryDisposeObject(_fastCorruptionDetector);
        TryDisposeObject(_imageCollector);
    }

    private void TryDisposeObject(object? obj)
    {
        (obj as IDisposable)?.Dispose();
    }
}
