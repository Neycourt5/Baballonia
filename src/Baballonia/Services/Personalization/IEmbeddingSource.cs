using System;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Baballonia.Services.Personalization;

/// <summary>
/// What <see cref="PersonalModelManager"/> needs from a loaded personal model, whatever kind it is.
/// </summary>
/// <remarks>
/// Exists so the manager can hold either corrector without caring which: model A/B read the camera
/// frame, model C reads the stock network's features, and everything else about managing them -
/// metadata for the status line, the permanent-failure flag, disposal - is identical.
/// </remarks>
public interface IPersonalCorrector : IExpressionCorrector, IDisposable
{
    /// <summary>Metadata read from the model file, for logging and the status line.</summary>
    PersonalModelMetadata Metadata { get; }

    /// <summary>True once inference has thrown; the corrector is inert from that point on.</summary>
    bool HasFailed { get; }
}

/// <summary>
/// An inference runner that can also hand back the stock network's internal visual features.
/// </summary>
/// <remarks>
/// Declared here rather than alongside <c>IInferenceRunner</c> deliberately. This is a
/// personalization concern, and keeping it out of the shared contract means the eye pipeline and
/// every other consumer are untouched by model C existing at all - which keeps the fork rebaseable.
/// </remarks>
public interface IEmbeddingSource
{
    /// <summary>
    /// Features from the most recent <c>Run()</c>, or null when this runner has none to give.
    /// </summary>
    /// <remarks>
    /// The tensor is owned by the runner and overwritten on the next inference, exactly like the
    /// input tensor. Callers read it within the tick and must not retain it.
    /// </remarks>
    DenseTensor<float> GetEmbedding();
}

/// <summary>
/// A corrector that consumes the stock model's visual embedding instead of the camera image.
/// </summary>
/// <remarks>
/// Separate from <see cref="IExpressionCorrector"/> so models A and B keep working through the exact
/// code path they always have. The pipeline type-checks for this interface, which means a build that
/// has never heard of embeddings still runs A and B correctly.
/// </remarks>
public interface IEmbeddingAwareCorrector : IExpressionCorrector
{
    /// <summary>
    /// Corrects using the stock features. A null <paramref name="embedding"/> must return stock
    /// unchanged - no features means no opinion, which is the safe failure.
    /// </summary>
    float[] Correct(DenseTensor<float> image, float[] stock, DenseTensor<float> embedding);
}
