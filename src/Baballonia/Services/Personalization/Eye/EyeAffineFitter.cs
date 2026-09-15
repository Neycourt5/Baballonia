using System;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Personalization.Eye;

/// <summary>One fixation: where the model said the eye was looking, and where it actually was.</summary>
public readonly record struct EyeGazeObservation(
    float MeasuredX, float MeasuredY, float TargetX, float TargetY);

public sealed record EyeAffineFitResult(EyeAffineMap Map, bool Fitted, string? Reason)
{
    public static EyeAffineFitResult Refused(string reason) =>
        new(EyeAffineMap.Identity, false, reason);
}

/// <summary>
/// Fits the per-eye gaze correction by least squares over the recorded fixations.
/// </summary>
/// <remarks>
/// <para>This is the no-machine-learning half of eye personalization, and it earns its place twice:
/// it turns a recorded session into an improvement immediately, without Python or a training run,
/// and it gives the trained residual model something to be measured against. A learned model that
/// cannot beat six coefficients is not worth shipping.</para>
///
/// <para>Every refusal returns identity rather than a bad fit. A gaze correction that is merely
/// wrong is worse than none: it does not look broken, it looks like the tracking drifted.</para>
/// </remarks>
public static class EyeAffineFitter
{
    /// <summary>Nine points is what the routine records; below five a 2x3 is not worth solving.</summary>
    public const int MinimumObservations = 5;

    /// <summary>
    /// Fixations must span both axes. Points along a line leave the cross terms unconstrained, and
    /// the solver would happily return enormous ones.
    /// </summary>
    public const float MinimumSpan = 0.10f;

    /// <summary>
    /// Largest correction the fit may apply anywhere it was fitted, in raw units.
    /// 0.25 is about 22 degrees; beyond that the fit is describing a mistake, not an eye.
    /// </summary>
    public const float MaximumCorrection = 0.25f;

    /// <summary>Below this the normal equations are too ill-conditioned to trust.</summary>
    private const double MinimumDeterminant = 1e-9;

    public static EyeAffineFitResult Fit(IReadOnlyList<EyeGazeObservation> observations)
    {
        if (observations.Count < MinimumObservations)
        {
            return EyeAffineFitResult.Refused(
                $"Only {observations.Count} usable fixations; {MinimumObservations} are needed.");
        }

        if (observations.Any(o => !float.IsFinite(o.MeasuredX) || !float.IsFinite(o.MeasuredY) ||
                                  !float.IsFinite(o.TargetX) || !float.IsFinite(o.TargetY)))
        {
            return EyeAffineFitResult.Refused("Some fixations contain invalid numbers.");
        }

        var spanX = observations.Max(o => o.MeasuredX) - observations.Min(o => o.MeasuredX);
        var spanY = observations.Max(o => o.MeasuredY) - observations.Min(o => o.MeasuredY);
        if (spanX < MinimumSpan || spanY < MinimumSpan)
        {
            return EyeAffineFitResult.Refused(
                "The recorded fixations do not span both directions, so a correction cannot be " +
                "separated from a guess. Repeat the capture and look fully at each dot.");
        }

        if (!TrySolve(observations, o => o.TargetX, out var x) ||
            !TrySolve(observations, o => o.TargetY, out var y))
        {
            return EyeAffineFitResult.Refused(
                "The recorded fixations are too close together to fit a correction.");
        }

        var map = new EyeAffineMap(x.A, x.B, x.C, y.A, y.B, y.C);
        if (!map.IsFinite)
            return EyeAffineFitResult.Refused("The fit produced invalid numbers.");

        // Judge the fit where it was actually fitted. A map that has to move a fixation by more
        // than MaximumCorrection is describing a bad capture, not a miscalibrated eye.
        foreach (var observation in observations)
        {
            var (fittedX, fittedY) = map.Apply(observation.MeasuredX, observation.MeasuredY);
            if (Math.Abs(fittedX - observation.MeasuredX) > MaximumCorrection ||
                Math.Abs(fittedY - observation.MeasuredY) > MaximumCorrection)
            {
                return EyeAffineFitResult.Refused(
                    "The correction needed is implausibly large. The dots were probably not " +
                    "followed accurately; repeat the capture.");
            }
        }

        return new EyeAffineFitResult(map, true, null);
    }

    /// <summary>Mean absolute error in degrees, before and after, for reporting a fit honestly.</summary>
    public static (float BeforeDegrees, float AfterDegrees) Evaluate(
        IReadOnlyList<EyeGazeObservation> observations, EyeAffineMap map)
    {
        if (observations.Count == 0)
            return (0f, 0f);

        var before = 0f;
        var after = 0f;
        foreach (var observation in observations)
        {
            before += Math.Abs(observation.MeasuredX - observation.TargetX) +
                      Math.Abs(observation.MeasuredY - observation.TargetY);

            var (x, y) = map.Apply(observation.MeasuredX, observation.MeasuredY);
            after += Math.Abs(x - observation.TargetX) + Math.Abs(y - observation.TargetY);
        }

        // Two axes per observation, and raw-to-degrees is the full 2x range.
        var scale = 2f * EyePersonalizationSchema.GazeRangeDegrees / (observations.Count * 2f);
        return (before * scale, after * scale);
    }

    /// <summary>
    /// Solves <c>target ≈ A·x + B·y + C</c> by normal equations.
    /// </summary>
    /// <remarks>
    /// Three unknowns over nine points; a 3×3 symmetric system solved by cofactor expansion is
    /// simpler to read and to test than pulling in a matrix library, and the determinant falls out
    /// of it as the conditioning check.
    /// </remarks>
    private static bool TrySolve(
        IReadOnlyList<EyeGazeObservation> observations,
        Func<EyeGazeObservation, float> target,
        out (float A, float B, float C) solution)
    {
        solution = default;

        double sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0, n = observations.Count;
        double txx = 0, txy = 0, tx = 0;

        foreach (var observation in observations)
        {
            double x = observation.MeasuredX, y = observation.MeasuredY, t = target(observation);
            sxx += x * x; sxy += x * y; sx += x;
            syy += y * y; sy += y;
            txx += x * t; txy += y * t; tx += t;
        }

        // [ sxx sxy sx ] [A]   [txx]
        // [ sxy syy sy ] [B] = [txy]
        // [ sx  sy  n  ] [C]   [tx ]
        var determinant =
            sxx * (syy * n - sy * sy) -
            sxy * (sxy * n - sy * sx) +
            sx * (sxy * sy - syy * sx);

        if (Math.Abs(determinant) < MinimumDeterminant)
            return false;

        double Det(double a1, double a2, double a3, double b1, double b2, double b3,
                   double c1, double c2, double c3) =>
            a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1);

        var a = Det(txx, sxy, sx, txy, syy, sy, tx, sy, n) / determinant;
        var b = Det(sxx, txx, sx, sxy, txy, sy, sx, tx, n) / determinant;
        var c = Det(sxx, sxy, txx, sxy, syy, txy, sx, sy, tx) / determinant;

        if (!double.IsFinite(a) || !double.IsFinite(b) || !double.IsFinite(c))
            return false;

        solution = ((float)a, (float)b, (float)c);
        return true;
    }
}
