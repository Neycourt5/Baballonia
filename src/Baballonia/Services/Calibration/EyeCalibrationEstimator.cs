using System;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Calibration;

public readonly record struct EyeCalibrationSample(
    float LeftLid,
    float LeftX,
    float LeftY,
    float RightLid,
    float RightX,
    float RightY);

/// <param name="Note">
/// Non-fatal observations worth telling the user about — currently, that an eye's lid channel
/// barely distinguishes wide-open from relaxed and a synthetic wide anchor was substituted.
/// Present only on valid estimates.
/// </param>
public sealed record EyeCalibrationEstimate(
    EyeCalibrationProfile Left,
    EyeCalibrationProfile Right,
    bool IsValid,
    string? Error,
    string? Note = null);

/// <summary>Pure robust statistics used by the guided eye-calibration recorder.</summary>
public static class EyeCalibrationEstimator
{
    public const int MinimumSamplesPerStep = 30;

    public static EyeCalibrationEstimate Estimate(
        IReadOnlyList<EyeCalibrationSample> relaxed,
        IReadOnlyList<EyeCalibrationSample> closed,
        IReadOnlyList<EyeCalibrationSample> wide)
    {
        if (relaxed.Count < MinimumSamplesPerStep ||
            closed.Count < MinimumSamplesPerStep ||
            wide.Count < MinimumSamplesPerStep)
            return Invalid($"Each eye-calibration step needs at least {MinimumSamplesPerStep} valid samples.");

        var left = Profile(
            Percentile(closed.Select(sample => sample.LeftLid), 0.20f),
            Percentile(relaxed.Select(sample => sample.LeftLid), 0.50f),
            Percentile(wide.Select(sample => sample.LeftLid), 0.80f),
            Percentile(relaxed.Select(sample => sample.LeftX), 0.50f),
            Percentile(relaxed.Select(sample => sample.LeftY), 0.50f));

        var right = Profile(
            Percentile(closed.Select(sample => sample.RightLid), 0.20f),
            Percentile(relaxed.Select(sample => sample.RightLid), 0.50f),
            Percentile(wide.Select(sample => sample.RightLid), 0.80f),
            Percentile(relaxed.Select(sample => sample.RightX), 0.50f),
            Percentile(relaxed.Select(sample => sample.RightY), 0.50f));

        var (leftFinal, leftError, leftNote) = Finalize(left, "Left");
        var (rightFinal, rightError, rightNote) = Finalize(right, "Right");

        var error = leftError ?? rightError;
        var note = (leftNote, rightNote) switch
        {
            (null, null) => null,
            (not null, null) => leftNote,
            (null, not null) => rightNote,
            _ => leftNote + " " + rightNote,
        };

        return new EyeCalibrationEstimate(
            leftFinal, rightFinal, error is null, error, error is null ? note : null);
    }

    public static float Percentile(IEnumerable<float> source, float percentile)
    {
        var values = source.Where(float.IsFinite).Order().ToArray();
        if (values.Length == 0)
            return float.NaN;

        percentile = Math.Clamp(percentile, 0f, 1f);
        var rank = percentile * (values.Length - 1);
        var lower = (int)Math.Floor(rank);
        var upper = (int)Math.Ceiling(rank);
        if (lower == upper)
            return values[lower];

        var fraction = rank - lower;
        return values[lower] + (values[upper] - values[lower]) * fraction;
    }

    private static EyeCalibrationProfile Profile(
        float closed, float neutral, float wide, float centerX, float centerY) =>
        new(EyeCalibrationProfile.CurrentVersion, closed, neutral, wide,
            centerX, centerY, 1f, 1f);

    /// <summary>
    /// Turns a measured profile into a saveable one, an error, or a saveable one with a note.
    /// </summary>
    /// <remarks>
    /// The closed-to-neutral span is a hard requirement: without it the openness mapping has no
    /// usable bottom segment, and no substitute anchor could be honest.
    ///
    /// The neutral-to-wide span is NOT a hard requirement any more. On a 12-output eye model,
    /// "open wide" is expressed mainly through the Widen channel — the lid channel is often already
    /// near its ceiling at relaxed, so demanding lid headroom above neutral rejected perfectly good
    /// captures with an unactionable message. When the measured wide anchor adds nothing, a
    /// synthetic one at the minimum span is substituted: openness above neutral then ramps to 1
    /// quickly, and the wide-eye *expression* still comes from the Widen channel.
    ///
    /// Every message carries the measured numbers. "Samples overlap" with no data was undiagnosable
    /// — the user could not tell whether the close step failed to register or the lid saturated.
    /// </remarks>
    private static (EyeCalibrationProfile Profile, string? Error, string? Note) Finalize(
        EyeCalibrationProfile profile, string side)
    {
        var closed = profile.OpennessClosed;
        var neutral = profile.OpennessNeutral;
        var wide = profile.OpennessWide;
        var measured = $"closed {closed:F2}, relaxed {neutral:F2}, wide {wide:F2}";

        if (!float.IsFinite(closed) || !float.IsFinite(neutral) || !float.IsFinite(wide) ||
            neutral - closed < EyeCalibrationProfile.MinimumClosedToNeutralSpan)
        {
            return (profile,
                $"{side} eye samples overlap ({measured}): relaxed must be at least " +
                $"{EyeCalibrationProfile.MinimumClosedToNeutralSpan:F2} above closed. " +
                "Repeat the relax and gentle-close steps.",
                null);
        }

        if (!float.IsFinite(profile.GazeCenterX) || !float.IsFinite(profile.GazeCenterY))
            return (profile, $"{side} eye gaze center was not captured. Repeat the relax step.", null);

        if (wide - neutral < EyeCalibrationProfile.MinimumNeutralToWideSpan -
                             EyeCalibrationProfile.SpanTolerance)
        {
            // Deliberately allowed to exceed 1.0 when neutral is very high: the anchors are just
            // mapping breakpoints, and MapOpenness clamps its input, so nothing downstream breaks.
            var synthetic = neutral + EyeCalibrationProfile.MinimumNeutralToWideSpan;
            return (profile with { OpennessWide = synthetic },
                null,
                $"{side} eye: the lid barely distinguishes wide-open from relaxed ({measured}); " +
                "wide-eye expression comes from the Widen channel, so a synthetic wide anchor was used.");
        }

        return (profile, null, null);
    }

    private static EyeCalibrationEstimate Invalid(string error) =>
        new(EyeCalibrationProfile.Default, EyeCalibrationProfile.Default, false, error);
}
