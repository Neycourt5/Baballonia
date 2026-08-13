using Microsoft.ML.OnnxRuntime.Tensors;

namespace Baballonia.Services.Personalization;

/// <summary>
/// A user-specific correction stage applied to the stock face model's raw output.
///
/// Sits between stock inference and the One Euro filter in
/// <see cref="Inference.FaceProcessingPipeline"/>: the corrector consumes raw pre-filter values
/// (matching what dataset recording captured, so training and serving see the same distribution)
/// and the filter then smooths the corrected signal that actually reaches VRChat.
///
/// Implementations must be safe to call from the processing tick and must never throw - a broken
/// personal model has to degrade to stock behavior, never break tracking.
/// </summary>
public interface IExpressionCorrector
{
    /// <summary>
    /// Blend between stock and personalized output: 0 = pure stock, 1 = full personalization.
    /// Primarily an evaluation/debugging control for A/B comparison; it is not a substitute for
    /// training. Reads and writes may race with the processing tick, so implementations should
    /// treat this as a volatile scalar.
    /// </summary>
    float Blend { get; set; }

    /// <summary>
    /// Returns the corrected expression vector.
    /// </summary>
    /// <param name="image">
    /// The input tensor the stock model just consumed ([1,1,224,224], /255 grayscale). Borrowed for
    /// the duration of the call only; implementations must not retain or mutate it.
    /// </param>
    /// <param name="stock">
    /// Raw stock output (length <see cref="PersonalizationSchema.ExpressionCount"/>). Must not be
    /// mutated - downstream consumers and the corrected-event payload alias this array.
    /// </param>
    /// <returns>
    /// A new array, never <paramref name="stock"/> itself, since the One Euro filter keeps internal
    /// buffers keyed to the array it is handed. Implementations that fail internally should return
    /// <paramref name="stock"/>'s values in a fresh array (i.e. stock passthrough).
    /// </returns>
    float[] Correct(DenseTensor<float> image, float[] stock);
}
