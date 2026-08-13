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

    public record NewFilteredResultEvent(float[] result);

    public record ExceptionEvent(Exception exception);
}
