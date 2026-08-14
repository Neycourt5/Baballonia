using OpenCvSharp;
using System;

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
    /// COST: the event bus invokes handlers synchronously while holding its lock, on the 10 ms
    /// processing tick. Handlers must only copy/enqueue and return - no encoding, no disk I/O.
    /// </summary>
    public record NewRawExpressionsEvent(Mat transformedFrame, float[] rawResult, long timestampTicks);

    /// <summary>
    /// Published only while a personal corrector is installed, carrying both vectors so the debug
    /// comparison view needs no pairing logic. Neither array may be mutated by handlers.
    /// </summary>
    public record NewCorrectedExpressionsEvent(float[] rawResult, float[] correctedResult);

    public record NewFilteredResultEvent(float[] result);

    public record ExceptionEvent(Exception exception);
}
public class EyePipelineEvents
{
    public record NewFrameEvent(Mat image);

    public record NewTransformedFrameEvent(Mat image);

    /// <summary>
    /// Raw eye model output, published immediately after inference and before the filter and
    /// <c>ProcessExpressions</c>.
    /// </summary>
    /// <remarks>
    /// <para>The eye counterpart of <see cref="FacePipelineEvents.NewRawExpressionsEvent"/>, and the
    /// tap any per-user eye work needs. It matters that this is *raw*: the post-processing that
    /// follows is lossy in ways that cannot be undone downstream - it averages a single vertical
    /// gaze across both eyes, lets a closed eye borrow the open eye's yaw, and clamps convergence.
    /// Those are reasonable output behaviours and terrible calibration inputs.</para>
    ///
    /// <para>Layout of <paramref name="rawResult"/> is the model's own, which is right-eye-first:
    /// <c>[rightY, rightX, rightLid, leftY, leftX, leftLid]</c>, each a sigmoid in [0,1]. That is
    /// the order the shipped graph emits; the pipeline's local variable names disagree, but two
    /// sign errors cancel downstream. Consumers of this event should use the layout, not the
    /// names.</para>
    ///
    /// <para>LIFETIME: <paramref name="transformedFrame"/> is the 2-channel (left, right) 128x128
    /// Mat that produced the result, and is disposed as soon as this publish returns. Handlers must
    /// Clone() to retain it.
    /// COST: handlers run synchronously under the bus lock on the 10 ms tick - copy/enqueue only.</para>
    /// </remarks>
    public record NewRawExpressionsEvent(Mat transformedFrame, float[] rawResult, long timestampTicks);

    public record NewFilteredResultEvent(float[] result);

    public record ExceptionEvent(Exception exception);
}
