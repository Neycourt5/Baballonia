using Baballonia.Services.events;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.Audio;
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
    public volatile IExpressionEnhancer? Enhancer;

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
            // Model C reads the stock network's internal features instead of the frame. The type
            // check keeps models A and B on exactly the path they have always taken, and costs
            // nothing when no embedding-aware model is installed.
            result = corrector is IExpressionCorrector and IEmbeddingAwareCorrector embeddingAware
                ? embeddingAware.Correct(InferenceService.GetInputTensor(), rawResult,
                    (InferenceService as IEmbeddingSource)?.GetEmbedding())
                : corrector.Correct(InferenceService.GetInputTensor(), rawResult);

            facePipelineEventBus.Publish(
                new FacePipelineEvents.NewCorrectedExpressionsEvent(rawResult, result));
        }

        // Single read, same reason as the corrector: it can be swapped from the settings page while
        // the tick is running.
        var enhancer = Enhancer;
        if (enhancer != null)
            result = enhancer.Enhance(result);

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
