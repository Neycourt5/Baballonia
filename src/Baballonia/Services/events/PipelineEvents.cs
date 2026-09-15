using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace Baballonia.Services.events;

public class FacePipelineEvents
{
    public record NewFrameEvent(Mat image);

    public record NewTransformedFrameEvent(Mat image);

    /// <summary>
    /// Raw stock model output, published immediately after inference and before any personal
    /// correction or filtering. <paramref name="transformedFrame"/> is the exact 224x224 Gray8 Mat
    /// that produced <paramref name="rawResult"/>, which is what makes this event the correct tap
    /// point for personalization dataset recording.
    ///
    /// LIFETIME: <paramref name="transformedFrame"/> is disposed by the pipeline as soon as this
    /// publish returns. Handlers must Clone() it if they need to retain it.
    /// COST: the event bus invokes handlers synchronously while holding its lock, on the processing
    /// tick. Handlers must only copy/enqueue and return - no encoding, no disk I/O.
    ///
    /// <paramref name="rawResult"/> is a positional vector in
    /// <see cref="Personalization.PersonalizationSchema"/> order, length
    /// <see cref="Personalization.PersonalizationSchema.ExpressionCount"/>. It is the pipeline's
    /// reusable scratch buffer - handlers must copy it, not retain it.
    /// </summary>
    public record NewRawExpressionsEvent(Mat transformedFrame, float[] rawResult, long timestampTicks,
        Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>? embedding = null,
        long inferenceTimestamp = 0, Inference.VideoFrameIdentity? sourceFrame = null);

    /// <summary>
    /// Published only while a personal corrector is installed, carrying both vectors so the debug
    /// comparison view needs no pairing logic. Neither array may be mutated by handlers, and
    /// neither may be retained past the call.
    /// </summary>
    public record NewCorrectedExpressionsEvent(float[] rawResult, float[] correctedResult);

    public record NewFilteredResultEvent(OrderedFloatMap result);

    public record ExceptionEvent(Exception exception);
}
public class EyePipelineEvents
{
    public record NewFrameEvent(Mat image);

    public record NewTransformedFrameEvent(Mat image);

    /// <summary>
    /// The eye model's own output, published every frame immediately after inference and before
    /// correction, filtering and the geometry pass.
    /// </summary>
    /// <remarks>
    /// This is the tap guided eye-calibration recording uses, and the "base" column of the
    /// personalization A/B view. Raw model space: all twelve values are sigmoids in [0,1], lid is
    /// closedness rather than openness, and gaze has not yet been remapped to +/-1 or coupled
    /// between the eyes.
    ///
    /// <paramref name="rawResult"/> is the pipeline's reusable scratch vector in
    /// <see cref="Personalization.Eye.EyePersonalizationSchema"/> order. Handlers run synchronously
    /// on the inference thread under the bus lock: copy and return.
    /// </remarks>
    public record NewRawEyeExpressionsEvent(float[] rawResult, long timestampTicks);

    /// <summary>
    /// Published only while an eye corrector is installed, carrying both vectors so the comparison
    /// view needs no pairing logic. Neither array may be retained or mutated.
    /// </summary>
    public record NewCorrectedEyeExpressionsEvent(float[] rawResult, float[] correctedResult);

    public record NewFilteredResultEvent(OrderedFloatMap result);

    public record ExceptionEvent(Exception exception);
}
