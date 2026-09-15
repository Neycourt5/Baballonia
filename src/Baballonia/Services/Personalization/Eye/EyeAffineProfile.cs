using System;

namespace Baballonia.Services.Personalization.Eye;

/// <summary>
/// A per-eye 2×3 affine map on raw gaze: <c>x' = Ax·x + Bx·y + Cx</c>, <c>y' = Ay·x + By·y + Cy</c>.
/// </summary>
/// <remarks>
/// <para>Six coefficients rather than two gains and two offsets, because the cross terms
/// (<c>Bx</c>, <c>Ay</c>) are what correct a camera that sits slightly rotated — the single most
/// common thing to change when a headset is reseated. Nine fixation points do not justify anything
/// higher-order, and a quadratic fitted to nine points would mostly model the noise.</para>
///
/// <para>Identity is <c>(1,0,0, 0,1,0)</c>, which is what an unusable fit falls back to.</para>
/// </remarks>
public sealed record EyeAffineMap(float Ax, float Bx, float Cx, float Ay, float By, float Cy)
{
    public static EyeAffineMap Identity { get; } = new(1f, 0f, 0f, 0f, 1f, 0f);

    public bool IsIdentity => this == Identity;

    public (float X, float Y) Apply(float x, float y) =>
        (Ax * x + Bx * y + Cx, Ay * x + By * y + Cy);

    public bool IsFinite =>
        float.IsFinite(Ax) && float.IsFinite(Bx) && float.IsFinite(Cx) &&
        float.IsFinite(Ay) && float.IsFinite(By) && float.IsFinite(Cy);
}

/// <summary>
/// A saved gaze fit, gated on the model and schema it was derived from.
/// </summary>
/// <remarks>
/// The gates are not bureaucracy. The coefficients are corrections to <em>one specific model's</em>
/// gaze output; applied to a different model they are arbitrary numbers being added to real
/// tracking, and nothing about the result would look like an error.
/// </remarks>
public sealed record EyeAffineProfile(
    int Version,
    string EyeSchemaSha256,
    string BaseEyeModelMd5,
    string CreatedUtc,
    string SourceSessionId,
    EyeAffineMap Left,
    EyeAffineMap Right)
{
    /// <summary>
    /// Per-eye widen and squint response, fitted from the same capture as the gaze map.
    /// </summary>
    /// <remarks>
    /// Added in version 2. A version 1 profile deserializes with these null and is treated as
    /// gaze-only, so an existing fit keeps working rather than being invalidated by the upgrade.
    /// </remarks>
    public EyeExpressionCurves? LeftCurves { get; init; }
    public EyeExpressionCurves? RightCurves { get; init; }

    public const int CurrentVersion = 2;

    /// <summary>Settings key holding the active fit.</summary>
    public const string SettingsKey = "EyePersonalAffine";

    public static EyeAffineProfile Identity { get; } = new(
        CurrentVersion, EyePersonalizationSchema.Sha256, string.Empty,
        DateTime.UnixEpoch.ToString("o"), string.Empty,
        EyeAffineMap.Identity, EyeAffineMap.Identity);

    /// <summary>
    /// Whether this fit may be applied to the model currently running.
    /// </summary>
    public bool AppliesTo(string baseEyeModelMd5, out string? reason)
    {
        if (Version > CurrentVersion)
        {
            reason = $"The saved gaze fit is version {Version}; this build understands {CurrentVersion}.";
            return false;
        }

        if (!string.Equals(EyeSchemaSha256, EyePersonalizationSchema.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            reason = "The saved gaze fit was made against a different eye output layout.";
            return false;
        }

        if (!string.Equals(BaseEyeModelMd5, baseEyeModelMd5, StringComparison.OrdinalIgnoreCase))
        {
            reason = "The saved gaze fit was made against a different eye model. " +
                     "Record and fit again to use it with this one.";
            return false;
        }

        if (!Left.IsFinite || !Right.IsFinite)
        {
            reason = "The saved gaze fit contains invalid numbers.";
            return false;
        }

        reason = null;
        return true;
    }
}
