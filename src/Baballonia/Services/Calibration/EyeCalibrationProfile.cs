using System;

namespace Baballonia.Services.Calibration;

public static class EyeCalibrationSettings
{
    public const string LeftKey = "EyeCalibration_Left";
    public const string RightKey = "EyeCalibration_Right";
}

/// <summary>
/// Persisted mapping from the eye model's geometry-corrected output into normalized eye output.
/// Left and right profiles are stored independently so camera and eyelid asymmetry is preserved.
/// </summary>
public sealed record EyeCalibrationProfile(
    int Version,
    float OpennessClosed,
    float OpennessNeutral,
    float OpennessWide,
    float GazeCenterX,
    float GazeCenterY,
    float GazeGainX,
    float GazeGainY)
{
    public const int CurrentVersion = 1;
    public const float RestOpenness = 0.75f;
    public const float MinimumClosedToNeutralSpan = 0.10f;
    public const float MinimumNeutralToWideSpan = 0.05f;

    /// <summary>
    /// Slack on the span checks, to absorb float rounding.
    /// </summary>
    /// <remarks>
    /// Load-bearing, not cosmetic. The estimator synthesizes a missing wide anchor as
    /// <c>neutral + MinimumNeutralToWideSpan</c>, but in float32 <c>(n + 0.05f) - n</c> is
    /// 0.04999995, so an exact <c>&gt;=</c> rejected the very profile that was constructed to pass —
    /// silently, leaving <see cref="MapOpenness"/> as the identity and the user's saved calibration
    /// doing nothing at all. These are "is there enough range to map" sanity checks, and a
    /// ten-thousandth of a unit is far below anything that distinguishes a usable capture from an
    /// unusable one.
    /// </remarks>
    public const float SpanTolerance = 1e-4f;

    public static EyeCalibrationProfile Default { get; } = new(
        CurrentVersion, 0.05f, 0.75f, 0.95f, 0f, 0f, 1f, 1f);

    public bool HasUsableOpennessRange =>
        float.IsFinite(OpennessClosed) &&
        float.IsFinite(OpennessNeutral) &&
        float.IsFinite(OpennessWide) &&
        OpennessNeutral - OpennessClosed >= MinimumClosedToNeutralSpan - SpanTolerance &&
        OpennessWide - OpennessNeutral >= MinimumNeutralToWideSpan - SpanTolerance;

    /// <summary>
    /// The wide anchor that can actually be reached, given <see cref="MapOpenness"/> clamps its
    /// input to [0,1]. A synthetic anchor above 1 is outside the input domain entirely.
    /// </summary>
    public float ReachableWide => Math.Min(OpennessWide, 1f);

    /// <summary>
    /// Whether the lid channel has real room above relaxed to express opening wider.
    /// </summary>
    /// <remarks>
    /// False for a saturated lid channel, where relaxed already reads near 1. Stage N0 synthesizes a
    /// wide anchor for those so the profile stays valid, but the synthetic anchor sits above 1 and
    /// can never be reached.
    /// </remarks>
    public bool HasWideHeadroom =>
        ReachableWide - OpennessNeutral >= MinimumNeutralToWideSpan - SpanTolerance;

    public float MapOpenness(float openness)
    {
        openness = Math.Clamp(openness, 0f, 1f);
        if (!HasUsableOpennessRange)
            return openness;

        if (openness <= OpennessClosed)
            return 0f;

        // Reserving the top of the output range for a "wider than relaxed" state only makes sense
        // if the channel can report one. On a saturated lid channel it cannot, and reserving it
        // anyway pins the avatar's eyes at three quarters open forever - which reads as permanently
        // half-lidded and leaves wide-eye unreachable no matter how hard the user opens their eyes.
        // Wide-eye expression comes from the Widen channel in that case, which is exactly what the
        // synthetic anchor's note says.
        var restOutput = HasWideHeadroom ? RestOpenness : 1f;

        if (openness <= OpennessNeutral)
            return restOutput * (openness - OpennessClosed) /
                   (OpennessNeutral - OpennessClosed);
        if (openness <= ReachableWide)
            return restOutput + (1f - restOutput) *
                (openness - OpennessNeutral) / (ReachableWide - OpennessNeutral);
        return 1f;
    }

    public float MapGazeX(float gaze) => MapGaze(gaze, GazeCenterX, GazeGainX);
    public float MapGazeY(float gaze) => MapGaze(gaze, GazeCenterY, GazeGainY);

    private static float MapGaze(float gaze, float center, float gain)
    {
        center = float.IsFinite(center) ? center : 0f;
        gain = float.IsFinite(gain) ? gain : 1f;
        return Math.Clamp((gaze - center) * gain, -1f, 1f);
    }
}
