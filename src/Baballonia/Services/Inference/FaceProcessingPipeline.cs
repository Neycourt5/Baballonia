using Baballonia.Services.events;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Personalization;
using System;

namespace Baballonia.Services.Inference;

public class FaceProcessingPipeline(IFacePipelineEventBus facePipelineEventBus) : DefaultProcessingPipeline
{
    /// <summary>
    /// Optional user-specific correction stage, applied between stock inference and the filter.
    /// Null (the default) means stock behavior. Declared here rather than on
    /// <see cref="DefaultProcessingPipeline"/> so the eye pipeline is unaffected.
    /// Volatile because <see cref="FacePipelineManager.SetCorrector"/> may swap it from a
    /// background model load while the processing tick is reading it.
    /// </summary>
    public volatile IExpressionCorrector? Corrector;

    public float[]? RunUpdate()
    {
        var frame = VideoSource?.GetFrame(ColorType.Gray8);
        if(frame == null)
            return null;

        facePipelineEventBus.Publish(new FacePipelineEvents.NewFrameEvent(frame));

        var transformed = ImageTransformer?.Apply(frame);

        if(transformed == null)
            return null;

        facePipelineEventBus.Publish(new FacePipelineEvents.NewTransformedFrameEvent(transformed));

        if (InferenceService == null)
            return null;

        ImageConverter?.Convert(transformed, InferenceService.GetInputTensor());

        var rawResult = InferenceService?.Run();
        if(rawResult == null)
        {
            transformed.Dispose();
            return null;
        }

        // Published before correction/filtering so personalization records exactly what the stock
        // model saw and said. The Mat is disposed immediately after; handlers must clone to retain.
        facePipelineEventBus.Publish(
            new FacePipelineEvents.NewRawExpressionsEvent(transformed, rawResult, DateTime.UtcNow.Ticks));
        transformed.Dispose();

        var result = rawResult;

        // Single read: the corrector can be swapped from a background load at any time.
        var corrector = Corrector;
        if (corrector != null)
        {
            result = corrector.Correct(InferenceService.GetInputTensor(), rawResult);
            facePipelineEventBus.Publish(
                new FacePipelineEvents.NewCorrectedExpressionsEvent(rawResult, result));
        }

        if(Filter != null)
            result = Filter.Filter(result);

        facePipelineEventBus.Publish(new FacePipelineEvents.NewFilteredResultEvent(result));


        return result;
    }

    public void Dispose()
    {
        TryDisposeObject(VideoSource);
        TryDisposeObject(ImageTransformer);
        TryDisposeObject(ImageConverter);
        TryDisposeObject(InferenceService);
        TryDisposeObject(Filter);
    }

    private void TryDisposeObject(object? obj)
    {
        (obj as IDisposable)?.Dispose();
    }
}
