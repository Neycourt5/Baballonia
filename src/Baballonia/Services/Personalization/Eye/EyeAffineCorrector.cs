using System;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Baballonia.Services.Personalization.Eye;

/// <summary>
/// Applies a fitted per-eye gaze correction to the model's raw output.
/// </summary>
/// <remarks>
/// <para>Gaze, widen and squint. Deliberately <em>not</em> lid: the Quick Eye Setup profile already
/// maps lid openness downstream, and two systems correcting one channel is how a correction ends up
/// fighting a calibration. Brow is never corrected either, because no pose in the routine
/// demonstrates a brow position.</para>
///
/// <para>Same safety rails the ONNX adapter will carry: the correction is clamped so no fit can
/// move a value more than <see cref="EyeAffineFitter.MaximumCorrection"/>, the result is clamped to
/// the raw range, and <see cref="Blend"/> allows instant A/B against the base model. Blend 0
/// returns the input exactly.</para>
/// </remarks>
public sealed class EyeAffineCorrector(EyeAffineProfile profile) : IExpressionCorrector
{
    // Schema order is Y before X, which reads oddly next to the (x, y) maths below.
    private const int RightY = 0, RightX = 1, RightWiden = 3, RightSquint = 4;
    private const int LeftY = 6, LeftX = 7, LeftWiden = 9, LeftSquint = 10;

    private volatile float _blend = 1f;

    public EyeAffineProfile Profile { get; } = profile;

    public float Blend
    {
        get => _blend;
        set => _blend = Math.Clamp(value, 0f, 1f);
    }

    public float[] Correct(DenseTensor<float> image, float[] stock)
    {
        // A fresh array every time: the One Euro filter keys its buffers to array identity, and the
        // caller publishes `stock` as the "base" half of the A/B comparison.
        var output = (float[])stock.Clone();

        var blend = _blend;
        if (blend <= 0f || stock.Length != EyePersonalizationSchema.ExpressionCount)
            return output;

        ApplyEye(output, Profile.Right, RightX, RightY, blend);
        ApplyEye(output, Profile.Left, LeftX, LeftY, blend);

        ApplyCurves(output, Profile.RightCurves, RightWiden, RightSquint, blend);
        ApplyCurves(output, Profile.LeftCurves, LeftWiden, LeftSquint, blend);
        return output;
    }

    /// <summary>
    /// Reshapes widen and squint so the user's own strongest pose reaches full scale.
    /// </summary>
    /// <remarks>
    /// Unclamped by <see cref="EyeAffineFitter.MaximumCorrection"/>, unlike gaze. That limit exists
    /// because a large gaze correction means the dots were not followed; here a large change is the
    /// entire point — someone whose hard squint only ever reads 0.35 needs it roughly tripled, and
    /// capping that would leave the complaint unfixed. The curve is monotone and its outputs are
    /// bounded by the anchors, so it cannot run away.
    /// </remarks>
    private static void ApplyCurves(
        float[] values, EyeExpressionCurves? curves, int widenIndex, int squintIndex, float blend)
    {
        if (curves is null)
            return;

        Apply(values, widenIndex, curves.Widen, blend);
        Apply(values, squintIndex, curves.Squint, blend);

        static void Apply(float[] values, int index, EyeResponseCurve curve, float blend)
        {
            var original = values[index];
            if (!float.IsFinite(original) || !curve.IsUsable)
                return;

            var shaped = curve.Apply(original);
            if (!float.IsFinite(shaped))
                return;

            values[index] = Math.Clamp(original + (shaped - original) * blend, 0f, 1f);
        }
    }

    private static void ApplyEye(float[] values, EyeAffineMap map, int xIndex, int yIndex, float blend)
    {
        var x = values[xIndex];
        var y = values[yIndex];
        if (!float.IsFinite(x) || !float.IsFinite(y))
            return;

        var (correctedX, correctedY) = map.Apply(x, y);
        if (!float.IsFinite(correctedX) || !float.IsFinite(correctedY))
            return;

        values[xIndex] = Blended(x, correctedX, blend);
        values[yIndex] = Blended(y, correctedY, blend);
    }

    private static float Blended(float original, float corrected, float blend)
    {
        var delta = Math.Clamp(
            corrected - original,
            -EyeAffineFitter.MaximumCorrection,
            EyeAffineFitter.MaximumCorrection);

        return Math.Clamp(original + delta * blend, 0f, 1f);
    }
}
