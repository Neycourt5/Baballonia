using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Baballonia.Contracts;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Baballonia.Services.Personalization.Eye;

public sealed record EyePersonalizationResult(bool Success, string Message);

/// <summary>
/// Owns the personalized eye correction: what is installed, whether it may be, and at what strength.
/// </summary>
/// <remarks>
/// <para>Deliberately much smaller than <see cref="PersonalModelManager"/>. There is one slot, no
/// architecture catalogue and no training-run browser, so what remains is the part that actually
/// matters: prepare before publishing, gate on the model the correction was derived from, and keep
/// the base model reachable at all times.</para>
///
/// <para>The base-model gate is re-checked whenever the eye model is swapped. Retraining the base
/// model in VR replaces it underneath a fitted correction, and those coefficients would then be
/// arbitrary numbers added to real tracking — which does not look like an error, it looks like the
/// tracking drifted.</para>
/// </remarks>
public sealed class EyePersonalizationManager
{
    public const string EnabledSetting = "EyePersonalization_Enabled";
    public const string BlendSetting = "EyePersonalization_Blend";

    /// <summary>
    /// Set when the saved gaze correction changes, cleared when Quick Eye Setup is next captured.
    /// </summary>
    /// <remarks>
    /// <para>Both systems correct gaze centre, at different points: this corrector works in raw
    /// model space before geometry, while <see cref="EyeCalibrationProfile.MapGazeX"/> subtracts a
    /// per-eye centre afterwards. That composes correctly only when the centre was measured
    /// <em>through</em> whatever correction is installed.</para>
    ///
    /// <para>A centre captured before a fit is applied on top of a signal that has since been
    /// re-centred, so it double-corrects. For the reporting user that is about 10 degrees of gaze
    /// error introduced by installing an otherwise good fit — which would look like the fit made
    /// tracking worse.</para>
    /// </remarks>
    public const string CalibrationStaleSetting = "EyePersonalization_CalibrationStale";

    private readonly ILocalSettingsService _settings;
    private readonly EyePipelineManager _pipelineManager;
    private readonly ILogger _logger;
    private readonly object _gate = new();

    private EyeAffineCorrector? _installed;

    public EyePersonalizationManager(
        ILocalSettingsService settings,
        EyePipelineManager pipelineManager,
        ILogger<EyePersonalizationManager>? logger = null)
    {
        _settings = settings;
        _pipelineManager = pipelineManager;
        _logger = logger ?? NullLogger<EyePersonalizationManager>.Instance;

        _pipelineManager.InferenceReloaded += OnInferenceReloaded;
    }

    /// <summary>Whether a correction is currently driving the pipeline.</summary>
    public bool IsActive
    {
        get { lock (_gate) return _installed is not null; }
    }

    /// <summary>What happened at the last load, for the UI to show without guessing.</summary>
    public string Status { get; private set; } = "No personalized eye correction saved yet.";

    public bool Enabled
    {
        get => _settings.ReadSetting(EnabledSetting, true);
        set
        {
            _settings.SaveSetting(EnabledSetting, value);
            Reload();
        }
    }

    /// <summary>
    /// 0 is the base model, 1 the full correction. Applied live, so A/B is instant.
    /// </summary>
    /// <remarks>
    /// Read with an explicit default so a deliberately persisted 0 survives a restart rather than
    /// silently springing back to full strength.
    /// </remarks>
    public float Blend
    {
        get => Math.Clamp(_settings.ReadSetting(BlendSetting, 1f), 0f, 1f);
        set
        {
            var blend = Math.Clamp(value, 0f, 1f);
            _settings.SaveSetting(BlendSetting, blend);
            lock (_gate)
            {
                if (_installed is { } corrector)
                    corrector.Blend = blend;
            }
        }
    }

    /// <summary>
    /// True when Quick Eye Setup was last captured against a different correction than the one now
    /// installed, so its gaze centre no longer describes the signal it is applied to.
    /// </summary>
    public bool IsCalibrationStale => _settings.ReadSetting(CalibrationStaleSetting, false);

    /// <summary>Called once Quick Eye Setup has been captured through the current correction.</summary>
    public void MarkCalibrationFresh() => _settings.SaveSetting(CalibrationStaleSetting, false);

    public EyeAffineProfile? SavedProfile =>
        _settings.ReadSetting<EyeAffineProfile?>(EyeAffineProfile.SettingsKey);

    /// <summary>Md5 of the eye model actually running, which every saved fit is gated against.</summary>
    public string ActiveBaseModelMd5()
    {
        try
        {
            var path = _pipelineManager.ResolveActiveEyeModelPath();
            if (!File.Exists(path))
                return string.Empty;

            using var stream = File.OpenRead(path);
            return Convert.ToHexString(MD5.HashData(stream)).ToLowerInvariant();
        }
        catch (IOException)
        {
            return string.Empty;
        }
    }

    /// <summary>
    /// Fits a gaze correction from a recorded session, saves it, and installs it.
    /// </summary>
    /// <remarks>
    /// Nothing is saved unless the fit succeeds and beats doing nothing. A correction that makes
    /// the recorded fixations worse is not a correction.
    /// </remarks>
    public EyePersonalizationResult FitFromSession(string sessionId)
    {
        var clusters = EyeGazeClusterExtractor.FromSession(sessionId);
        if (!clusters.IsUsable)
            return new EyePersonalizationResult(false, clusters.Error!);

        var activeMd5 = ActiveBaseModelMd5();
        if (!string.Equals(clusters.BaseEyeModelMd5, activeMd5, StringComparison.OrdinalIgnoreCase))
        {
            return new EyePersonalizationResult(false,
                "That recording was made with a different eye model than the one running now. " +
                "Record again to fit against this model.");
        }

        var left = EyeAffineFitter.Fit(clusters.Left);
        var right = EyeAffineFitter.Fit(clusters.Right);

        var leftCurves = FitCurves(clusters.LeftExpressions);
        var rightCurves = FitCurves(clusters.RightExpressions);

        var gazeFitted = left.Fitted || right.Fitted;
        var curvesFitted = !leftCurves.Curves.IsIdentity || !rightCurves.Curves.IsIdentity;

        if (!gazeFitted && !curvesFitted)
        {
            return new EyePersonalizationResult(false,
                left.Reason ?? "Nothing in the recording could be fitted.");
        }

        var (leftBefore, leftAfter) = EyeAffineFitter.Evaluate(clusters.Left, left.Map);
        var (rightBefore, rightAfter) = EyeAffineFitter.Evaluate(clusters.Right, right.Map);

        // A gaze fit that does not beat the base model is not saved. Expression curves are judged
        // separately, because someone whose gaze is already good may still need their squint
        // reshaped - refusing the whole fit for that would throw away the part that helps.
        if (gazeFitted && leftAfter + rightAfter >= leftBefore + rightBefore)
        {
            left = EyeAffineFitResult.Refused("gaze did not improve");
            right = EyeAffineFitResult.Refused("gaze did not improve");
            gazeFitted = false;

            if (!curvesFitted)
            {
                return new EyePersonalizationResult(false,
                    "The fit did not improve on the base model, so it was not saved. " +
                    "This usually means the dots were not followed closely.");
            }
        }

        var profile = new EyeAffineProfile(
            EyeAffineProfile.CurrentVersion,
            EyePersonalizationSchema.Sha256,
            activeMd5,
            DateTime.UtcNow.ToString("o"),
            sessionId,
            left.Fitted ? left.Map : EyeAffineMap.Identity,
            right.Fitted ? right.Map : EyeAffineMap.Identity)
        {
            LeftCurves = leftCurves.Curves,
            RightCurves = rightCurves.Curves,
        };

        _settings.SaveSetting(EyeAffineProfile.SettingsKey, profile);
        _settings.SaveSetting(EnabledSetting, true);
        // The saved gaze centre was measured without this fit; applied on top of it, it
        // double-corrects. Only flag it when gaze actually moved - a curves-only fit leaves the
        // centre exactly where it was.
        if (gazeFitted)
            _settings.SaveSetting(CalibrationStaleSetting, true);
        Reload();

        var parts = new List<string>();
        if (gazeFitted)
        {
            parts.Add($"Gaze error {(leftBefore + rightBefore) / 2:F1}° → " +
                      $"{(leftAfter + rightAfter) / 2:F1}°.");
        }
        else
        {
            parts.Add("Gaze left unchanged.");
        }

        // Reported as a multiple because that is what the user will feel: "your hard squint now
        // reads three times as strongly" is actionable in a way that a curve is not.
        AddGain(parts, "Widen", clusters.LeftExpressions.Widen, clusters.RightExpressions.Widen,
            leftCurves.Curves.Widen, rightCurves.Curves.Widen);
        AddGain(parts, "Squint", clusters.LeftExpressions.Squint, clusters.RightExpressions.Squint,
            leftCurves.Curves.Squint, rightCurves.Curves.Squint);

        if (leftCurves.Reason is { } leftReason) parts.Add($"Left: {leftReason}");
        if (rightCurves.Reason is { } rightReason) parts.Add($"Right: {rightReason}");

        var message = string.Join(" ", parts);
        _logger.LogInformation("Eye fit from {Session}: {Message}", sessionId, message);
        return new EyePersonalizationResult(true, message);
    }

    private static void AddGain(
        List<string> parts, string label,
        IReadOnlyList<EyeResponseAnchor> leftAnchors, IReadOnlyList<EyeResponseAnchor> rightAnchors,
        EyeResponseCurve leftCurve, EyeResponseCurve rightCurve)
    {
        if (leftCurve.IsIdentity && rightCurve.IsIdentity)
            return;

        var gain = (EyeResponseCurveFitter.StrongestGain(leftAnchors, leftCurve) +
                    EyeResponseCurveFitter.StrongestGain(rightAnchors, rightCurve)) / 2f;

        if (gain > 1.05f)
            parts.Add($"{label} now reads {gain:F1}× stronger at your strongest pose.");
        else
            parts.Add($"{label} was already well scaled.");
    }

    /// <summary>Fits both expression curves for one eye, tolerating either failing on its own.</summary>
    private static (EyeExpressionCurves Curves, string? Reason) FitCurves(EyeExpressionAnchors anchors)
    {
        var widen = EyeResponseCurveFitter.Fit(anchors.Widen);
        var squint = EyeResponseCurveFitter.Fit(anchors.Squint);

        var reasons = new List<string>();
        if (!widen.Fitted && anchors.Widen.Count > 0) reasons.Add($"widen — {widen.Reason}");
        if (!squint.Fitted && anchors.Squint.Count > 0) reasons.Add($"squint — {squint.Reason}");

        return (new EyeExpressionCurves(widen.Curve, squint.Curve),
                reasons.Count == 0 ? null : string.Join("; ", reasons));
    }

    /// <summary>Removes any saved correction and returns to the base model.</summary>
    public void Clear()
    {
        var hadGazeFit = SavedProfile is { } profile &&
                         (!profile.Left.IsIdentity || !profile.Right.IsIdentity);

        _settings.SaveSetting<EyeAffineProfile?>(EyeAffineProfile.SettingsKey, null);

        // Removing a gaze fit is the same problem in reverse: a centre captured through the fit is
        // now applied to an uncorrected signal.
        if (hadGazeFit)
            _settings.SaveSetting(CalibrationStaleSetting, true);

        Reload();
    }

    /// <summary>Re-reads the saved fit and installs or removes it accordingly.</summary>
    public void Reload()
    {
        EyeAffineCorrector? corrector = null;
        string status;

        var profile = SavedProfile;
        if (profile is null)
        {
            status = "No personalized eye correction saved yet.";
        }
        else if (!Enabled)
        {
            status = "Personalized eye correction is turned off.";
        }
        else if (!profile.AppliesTo(ActiveBaseModelMd5(), out var reason))
        {
            status = reason!;
            _logger.LogWarning("Personalized eye correction not applied: {Reason}", reason);
        }
        else
        {
            corrector = new EyeAffineCorrector(profile) { Blend = Blend };
            status = $"Personalized eye correction active — {Covers(profile)} " +
                     $"(fitted {Format(profile.CreatedUtc)}).";
        }

        lock (_gate)
            _installed = corrector;

        // Publishing under the pipeline's own lock means the swap waits out any in-flight frame.
        _pipelineManager.SetCorrector(corrector);
        Status = status;
    }

    /// <summary>Names the parts of the fit that are actually doing something.</summary>
    private static string Covers(EyeAffineProfile profile)
    {
        var parts = new List<string>();
        if (!profile.Left.IsIdentity || !profile.Right.IsIdentity)
            parts.Add("gaze");
        if (profile.LeftCurves is { IsIdentity: false } || profile.RightCurves is { IsIdentity: false })
            parts.Add("squint and wide-eye");

        return parts.Count == 0 ? "nothing was changed" : string.Join(" plus ", parts);
    }

    private static string Format(string isoUtc) =>
        DateTime.TryParse(isoUtc, out var parsed)
            ? parsed.ToLocalTime().ToString("d MMM, HH:mm")
            : "previously";

    private void OnInferenceReloaded()
    {
        // The base model may have just been replaced - retraining in VR does exactly that - and a
        // fit derived from the old one would now be arbitrary numbers added to real tracking.
        Reload();
    }
}
