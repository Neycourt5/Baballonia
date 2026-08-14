using Baballonia.Contracts;
using Baballonia.Services.events;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.EyeV2;
using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace Baballonia.Services.Inference;

public class EyeProcessingPipeline(IEyePipelineEventBus eyePipelineEventBus) : DefaultProcessingPipeline, IDisposable
{
    private readonly FastCorruptionDetector.FastCorruptionDetector _fastCorruptionDetector = new();
    private readonly ImageCollector _imageCollector = new();

    public bool StabilizeEyes { get; set; } = true;

    /// <summary>
    /// Optional V2 stage. Null is intentionally the exact pre-V2 path; no copy, allocation or
    /// alternate postprocessing occurs in that case.
    /// </summary>
    public volatile IEyeStateMapper? Mapper;

    /// <summary>
    /// Runs one eye inference tick.
    /// </summary>
    /// <remarks>
    /// Every native buffer this method creates is released in the finally block. It previously
    /// leaked the 8-channel temporal stack on every tick and disposed the transformed frame twice,
    /// while returning early on six paths without releasing the camera frame at all - which at
    /// ~100 ticks a second is a large amount of native memory churn for a method that is supposed
    /// to be the cheap half of the pipeline.
    /// </remarks>
    public float[]? RunUpdate()
    {
        var frame = VideoSource?.GetFrame(ColorType.Gray8);
        if (frame == null)
            return null;

        Mat? transformed = null;
        Mat? collected = null;

        try
        {
            if (_fastCorruptionDetector.IsCorrupted(frame).isCorrupted)
                return null;

            eyePipelineEventBus.Publish(new EyePipelineEvents.NewFrameEvent(frame));

            transformed = ImageTransformer?.Apply(frame);
            if (transformed == null)
                return null;

            eyePipelineEventBus.Publish(new EyePipelineEvents.NewTransformedFrameEvent(transformed));

            var timestampTicks = DateTime.UtcNow.Ticks;
            var frameMapper = Mapper;
            if (frameMapper is IEyeFrameAwareMapper frameAware)
            {
                try
                {
                    frameAware.ObserveFrame(transformed, timestampTicks);
                }
                catch
                {
                    // An experimental frame consumer may never take down stock eye tracking.
                    Mapper = null;
                }
            }

            collected = _imageCollector.Apply(transformed);
            if (collected == null)
                return null;   // still filling the temporal queue

            if (InferenceService == null)
                return null;

            ImageConverter?.Convert(collected, InferenceService.GetInputTensor());

            var modelResult = InferenceService.Run();
            if (modelResult == null)
                return null;

            // Newer trained eye models may expose expression values in addition to gaze/lid. Their
            // metadata describes a 12-value right-eye-first layout, while the rest of Baballonia
            // intentionally retains the six-value legacy contract. Project by name before the
            // six-slot OneEuro filter and stock post-processing; simply truncating would mistake
            // right-eye widen/squint/brow for the left eye and used to crash when filtering was on.
            var inferenceResult = ProjectLegacyEyeOutput(modelResult, InferenceService);

            // Published before the filter and before ProcessExpressions, so subscribers see the
            // model's raw gaze/lid values in the stable legacy order. The processing below is
            // deliberately lossy - it fuses the two eyes' vertical gaze into one value and lets a
            // closed eye borrow the other's yaw, none of which can be undone from the outside.
            eyePipelineEventBus.Publish(new EyePipelineEvents.NewRawExpressionsEvent(
                transformed, inferenceResult, timestampTicks));

            if (Filter != null)
            {
                inferenceResult = Filter.Filter(inferenceResult);
            }

            var mapper = Mapper;
            var filteredRawForMapper = mapper == null ? null : (float[])inferenceResult.Clone();
            ProcessExpressions(ref inferenceResult);

            if (mapper != null && filteredRawForMapper != null)
            {
                try
                {
                    // The stage is after stock postprocessing exactly as designed. It also receives
                    // the same tick's filtered raw values because per-eye Y/lid information has
                    // already been fused by ProcessExpressions and cannot be reconstructed later.
                    inferenceResult = mapper.Map(inferenceResult, filteredRawForMapper, timestampTicks);
                }
                catch
                {
                    // A bad experimental mapper must never take down eye tracking. Clear it and
                    // return the already-computed stock state on this same tick.
                    Mapper = null;
                }
            }

            eyePipelineEventBus.Publish(new EyePipelineEvents.NewFilteredResultEvent(inferenceResult));

            return inferenceResult;
        }
        finally
        {
            frame.Dispose();
            transformed?.Dispose();
            collected?.Dispose();
        }
    }

    /// <summary>
    /// Drops the temporal frame history. Call when the camera changes.
    /// </summary>
    /// <remarks>
    /// Without this, frames from the previous camera stay in the queue and get stacked with new
    /// ones, so the model is handed four "consecutive" frames spanning a camera switch.
    /// </remarks>
    public void ResetTemporalState() => _imageCollector.Reset();

    private static float[] ProjectLegacyEyeOutput(float[] modelResult, IInferenceRunner runner)
    {
        if (modelResult.Length == Utils.EyeRawExpressions)
            return modelResult;

        if (runner is not INamedInferenceOutput { OutputNames: { } outputNames } ||
            outputNames.Count != modelResult.Length)
        {
            throw new InvalidOperationException(
                $"Eye model emits {modelResult.Length} values, but has no matching named output layout.");
        }

        var indices = new[]
        {
            FindOutput(outputNames, "rightEyeY", "rightEyePitch"),
            FindOutput(outputNames, "rightEyeX", "rightEyeYaw"),
            FindOutput(outputNames, "rightEyeLid"),
            FindOutput(outputNames, "leftEyeY", "leftEyePitch"),
            FindOutput(outputNames, "leftEyeX", "leftEyeYaw"),
            FindOutput(outputNames, "leftEyeLid"),
        };

        var projected = new float[Utils.EyeRawExpressions];
        for (var i = 0; i < projected.Length; i++)
            projected[i] = modelResult[indices[i]];

        return projected;
    }

    private static int FindOutput(IReadOnlyList<string> outputNames, params string[] candidates)
    {
        for (var i = 0; i < outputNames.Count; i++)
        {
            var actual = outputNames[i].TrimStart('/');
            foreach (var candidate in candidates)
                if (string.Equals(actual, candidate, StringComparison.OrdinalIgnoreCase))
                    return i;
        }

        throw new InvalidOperationException(
            $"Eye model output metadata is missing '{string.Join("' or '", candidates)}'.");
    }

    private bool ProcessExpressions(ref float[] arKitExpressions)
    {
        if (arKitExpressions.Length < Utils.EyeRawExpressions)
            return false;

        const float mulV = 2.0f;
        const float mulY = 2.0f;

        var leftPitch = arKitExpressions[0] * mulY - mulY / 2;
        var leftYaw = arKitExpressions[1] * mulV - mulV / 2;
        var leftLid = 1 - arKitExpressions[2];

        var rightPitch = arKitExpressions[3] * mulY - mulY / 2;
        var rightYaw = arKitExpressions[4] * mulV - mulV / 2;
        var rightLid = 1 - arKitExpressions[5];

        var eyeY = (leftPitch * leftLid + rightPitch * rightLid) / (leftLid + rightLid);

        var leftEyeYawCorrected = rightYaw * (1 - leftLid) + leftYaw * leftLid;
        var rightEyeYawCorrected = leftYaw * (1 - rightLid) + rightYaw * rightLid;

        if (StabilizeEyes)
        {
            var rawConvergence = (rightEyeYawCorrected - leftEyeYawCorrected) / 2.0f;
            var convergence = Math.Max(rawConvergence, 0.0f); // We clamp the value here to avoid accidental divergence, as the model sometimes decides that's a thing

            var averagedYaw = (rightEyeYawCorrected + leftEyeYawCorrected) / 2.0f;

            leftEyeYawCorrected = averagedYaw - convergence;
            rightEyeYawCorrected = averagedYaw + convergence;
        }

        // [left pitch, left yaw, left lid...
        float[] convertedExpressions = new float[Utils.EyeRawExpressions];

        convertedExpressions[0] = rightEyeYawCorrected; // left pitch
        convertedExpressions[1] = eyeY;                   // left yaw
        convertedExpressions[2] = rightLid;               // left lid
        convertedExpressions[3] = leftEyeYawCorrected;  // right pitch
        convertedExpressions[4] = eyeY;                   // right yaw
        convertedExpressions[5] = leftLid;                // right lid

        arKitExpressions = convertedExpressions;

        return true;
    }


    public void Dispose()
    {
        TryDisposeObject(VideoSource);
        TryDisposeObject(ImageTransformer);
        TryDisposeObject(ImageConverter);
        TryDisposeObject(InferenceService);
        TryDisposeObject(Filter);
        TryDisposeObject(Mapper);
        TryDisposeObject(_fastCorruptionDetector);
        TryDisposeObject(_imageCollector);
    }

    private void TryDisposeObject(object? obj)
    {
        (obj as IDisposable)?.Dispose();
    }
}
