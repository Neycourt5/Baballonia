using Baballonia.Services.events;
using Baballonia.Services.Inference.Enums;
using OpenCvSharp;
using System;

namespace Baballonia.Services.Inference;

public class EyeProcessingPipeline(IEyePipelineEventBus eyePipelineEventBus) : DefaultProcessingPipeline, IDisposable
{
    private readonly FastCorruptionDetector.FastCorruptionDetector _fastCorruptionDetector = new();
    private readonly ImageCollector _imageCollector = new();

    public bool StabilizeEyes { get; set; } = true;

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

            collected = _imageCollector.Apply(transformed);
            if (collected == null)
                return null;   // still filling the temporal queue

            if (InferenceService == null)
                return null;

            ImageConverter?.Convert(collected, InferenceService.GetInputTensor());

            var inferenceResult = InferenceService.Run();
            if (inferenceResult == null)
                return null;

            // Published before the filter and before ProcessExpressions, so subscribers see what
            // the model actually said. The post-processing below is deliberately lossy - it fuses
            // the two eyes' vertical gaze into one value and lets a closed eye borrow the other's
            // yaw - and none of that can be undone from the outside.
            eyePipelineEventBus.Publish(new EyePipelineEvents.NewRawExpressionsEvent(
                transformed, inferenceResult, DateTime.UtcNow.Ticks));

            if (Filter != null)
            {
                inferenceResult = Filter.Filter(inferenceResult);
            }

            ProcessExpressions(ref inferenceResult);

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
        TryDisposeObject(_fastCorruptionDetector);
        TryDisposeObject(_imageCollector);
    }

    private void TryDisposeObject(object? obj)
    {
        (obj as IDisposable)?.Dispose();
    }
}
