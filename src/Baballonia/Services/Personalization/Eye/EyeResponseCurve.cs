using System;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Personalization.Eye;

/// <summary>
/// A monotone piecewise-linear map from what the model reports to what the expression should read.
/// </summary>
/// <remarks>
/// <para>This is what makes a squint actually register. The model may only ever emit 0.35 for a hard
/// squint; downstream that is a faint squint, and no amount of threshold tuning turns "the strongest
/// squint this person makes" into "full squint" without also turning noise into one. A curve
/// anchored on <em>measured</em> poses does exactly that and nothing else: your normal face reads
/// zero, your hard squint reads nearly one, and everything between is interpolated.</para>
///
/// <para>Deliberately not a gain. A single multiplier that makes a hard squint reach 1.0 also
/// multiplies the resting value, so a face at rest starts squinting slightly. Anchoring rest at zero
/// separately from the extreme is the whole point.</para>
///
/// <para>Outside the measured range the curve holds flat rather than extrapolating. Extrapolation
/// past the strongest pose someone performed is invention, and it is invention in the direction of
/// overshoot.</para>
/// </remarks>
public sealed record EyeResponseCurve(float[] Inputs, float[] Outputs)
{
    /// <summary>Two anchors closer than this cannot be told apart from noise.</summary>
    public const float MinimumSeparation = 0.03f;

    public static EyeResponseCurve Identity { get; } = new([0f, 1f], [0f, 1f]);

    public bool IsIdentity =>
        Inputs.Length == 2 && Outputs.Length == 2 &&
        Inputs[0] == 0f && Inputs[1] == 1f && Outputs[0] == 0f && Outputs[1] == 1f;

    /// <summary>
    /// Whether this curve is safe to apply: at least two anchors, strictly increasing inputs with
    /// real separation, and finite throughout.
    /// </summary>
    public bool IsUsable
    {
        get
        {
            if (Inputs is null || Outputs is null ||
                Inputs.Length < 2 || Inputs.Length != Outputs.Length)
                return false;

            for (var i = 0; i < Inputs.Length; i++)
            {
                if (!float.IsFinite(Inputs[i]) || !float.IsFinite(Outputs[i]))
                    return false;
                if (i > 0 && Inputs[i] - Inputs[i - 1] < MinimumSeparation)
                    return false;
                if (i > 0 && Outputs[i] < Outputs[i - 1])
                    return false;
            }

            return true;
        }
    }

    public float Apply(float value)
    {
        if (!float.IsFinite(value) || !IsUsable)
            return value;

        if (value <= Inputs[0])
            return Outputs[0];

        for (var i = 1; i < Inputs.Length; i++)
        {
            if (value > Inputs[i])
                continue;

            var span = Inputs[i] - Inputs[i - 1];
            var t = (value - Inputs[i - 1]) / span;
            return Outputs[i - 1] + (Outputs[i] - Outputs[i - 1]) * t;
        }

        return Outputs[^1];
    }
}

/// <summary>The per-eye expression curves fitted from a guided capture.</summary>
/// <remarks>
/// Widen and squint only. Lid openness is already owned by the Quick Eye Setup profile downstream,
/// and two systems correcting the same channel is how a correction ends up fighting a calibration.
/// Brow is never corrected: no pose in the routine demonstrates a brow position.
/// </remarks>
public sealed record EyeExpressionCurves(EyeResponseCurve Widen, EyeResponseCurve Squint)
{
    public static EyeExpressionCurves Identity { get; } =
        new(EyeResponseCurve.Identity, EyeResponseCurve.Identity);

    public bool IsIdentity => Widen.IsIdentity && Squint.IsIdentity;
}

/// <summary>One measured pose: what the model reported, and what it should have reported.</summary>
public readonly record struct EyeResponseAnchor(float Measured, float Intended);

public sealed record EyeCurveFitResult(EyeResponseCurve Curve, bool Fitted, string? Reason)
{
    public static EyeCurveFitResult Refused(string reason) =>
        new(EyeResponseCurve.Identity, false, reason);
}

/// <summary>Builds response curves from measured poses.</summary>
public static class EyeResponseCurveFitter
{
    /// <summary>
    /// Fits a curve through the anchors, or refuses.
    /// </summary>
    /// <remarks>
    /// Refusing is the common case for someone whose model already behaves, and it is the right
    /// answer: an identity curve leaves them exactly as they were. The failure worth avoiding is a
    /// curve fitted through anchors that were not actually distinguishable, which amplifies noise
    /// into expression.
    /// </remarks>
    public static EyeCurveFitResult Fit(IReadOnlyList<EyeResponseAnchor> anchors)
    {
        if (anchors.Count < 2)
            return EyeCurveFitResult.Refused("Not enough poses were recorded for this expression.");

        var ordered = anchors.OrderBy(anchor => anchor.Intended).ToList();

        if (ordered.Any(a => !float.IsFinite(a.Measured) || !float.IsFinite(a.Intended)))
            return EyeCurveFitResult.Refused("Some recorded poses contain invalid numbers.");

        for (var i = 1; i < ordered.Count; i++)
        {
            if (ordered[i].Measured - ordered[i - 1].Measured >= EyeResponseCurve.MinimumSeparation)
                continue;

            return EyeCurveFitResult.Refused(
                "The recorded poses were too similar for the model to tell apart, so no curve " +
                "could be fitted. Try making the difference between them more pronounced.");
        }

        // Pin the ends so the curve spans the full range. Below the resting pose there is nothing
        // to express; above the strongest pose the value is already at its maximum.
        //
        // Both pins are skipped when the anchor already sits near that end, because a pin closer
        // than MinimumSeparation would make the whole curve unusable - which would refuse exactly
        // the people whose model reports most honestly. Apply() holds flat outside the range
        // regardless, so skipping a pin costs nothing.
        var inputs = new List<float>();
        var outputs = new List<float>();

        if (ordered[0].Measured >= EyeResponseCurve.MinimumSeparation)
        {
            inputs.Add(0f);
            outputs.Add(ordered[0].Intended);
        }

        foreach (var anchor in ordered)
        {
            inputs.Add(anchor.Measured);
            outputs.Add(anchor.Intended);
        }

        if (ordered[^1].Measured <= 1f - EyeResponseCurve.MinimumSeparation)
        {
            inputs.Add(1f);
            outputs.Add(1f);
        }

        var curve = new EyeResponseCurve(inputs.ToArray(), outputs.ToArray());
        return curve.IsUsable
            ? new EyeCurveFitResult(curve, true, null)
            : EyeCurveFitResult.Refused("The recorded poses did not form a usable curve.");
    }

    /// <summary>
    /// How much stronger the strongest pose now reads, as a multiple. Reported so the user can see
    /// what the fit did rather than being told only that it worked.
    /// </summary>
    public static float StrongestGain(IReadOnlyList<EyeResponseAnchor> anchors, EyeResponseCurve curve)
    {
        if (anchors.Count == 0)
            return 1f;

        var strongest = anchors.MaxBy(anchor => anchor.Intended);
        if (strongest.Measured <= 1e-4f)
            return 1f;

        return curve.Apply(strongest.Measured) / strongest.Measured;
    }
}
