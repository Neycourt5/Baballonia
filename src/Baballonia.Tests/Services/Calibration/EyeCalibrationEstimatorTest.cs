using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Baballonia.Models;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Calibration;

[TestClass]
public class EyeCalibrationEstimatorTest
{
    [TestMethod]
    public void OpennessMappingUsesTwoSegmentsAndRestAnchor()
    {
        var profile = Profile(closed: 0.10f, neutral: 0.60f, wide: 0.90f);

        Assert.AreEqual(0f, profile.MapOpenness(0.05f), 1e-6f);
        Assert.AreEqual(0.375f, profile.MapOpenness(0.35f), 1e-6f);
        Assert.AreEqual(0.75f, profile.MapOpenness(0.60f), 1e-6f);
        Assert.AreEqual(0.875f, profile.MapOpenness(0.75f), 1e-6f);
        Assert.AreEqual(1f, profile.MapOpenness(0.95f), 1e-6f);
    }

    [TestMethod]
    public void DegenerateOpennessRangeFallsBackToIdentityClamp()
    {
        var profile = Profile(closed: 0.50f, neutral: 0.55f, wide: 0.58f);

        Assert.AreEqual(0.42f, profile.MapOpenness(0.42f), 1e-6f);
        Assert.AreEqual(0f, profile.MapOpenness(-2f), 1e-6f);
        Assert.AreEqual(1f, profile.MapOpenness(2f), 1e-6f);
    }

    [TestMethod]
    public void GazeMappingRecentersAppliesGainAndClamps()
    {
        var profile = Profile(centerX: 0.20f, centerY: -0.10f, gainX: 2f, gainY: 0.5f);

        Assert.AreEqual(0f, profile.MapGazeX(0.20f), 1e-6f);
        Assert.AreEqual(0.4f, profile.MapGazeX(0.40f), 1e-6f);
        Assert.AreEqual(0.1f, profile.MapGazeY(0.10f), 1e-6f);
        Assert.AreEqual(1f, profile.MapGazeX(5f), 1e-6f);
    }

    [TestMethod]
    public void EstimatorUsesPerEyePercentilesAndIgnoresOutliers()
    {
        var relaxed = Samples(0.70f, 0.80f, 0.12f, -0.08f);
        var closed = Samples(0.08f, 0.15f, 0f, 0f);
        var wide = Samples(0.95f, 0.98f, 0f, 0f);

        // One extreme frame cannot pull a percentile the way it would pull a mean.
        relaxed.Add(new EyeCalibrationSample(0f, 1f, 1f, 0f, -1f, -1f));
        closed.Add(new EyeCalibrationSample(1f, 0f, 0f, 1f, 0f, 0f));
        wide.Add(new EyeCalibrationSample(0f, 0f, 0f, 0f, 0f, 0f));

        var estimate = EyeCalibrationEstimator.Estimate(relaxed, closed, wide);

        Assert.IsTrue(estimate.IsValid, estimate.Error);
        Assert.AreEqual(0.70f, estimate.Left.OpennessNeutral, 1e-6f);
        Assert.AreEqual(0.80f, estimate.Right.OpennessNeutral, 1e-6f);
        Assert.AreEqual(0.08f, estimate.Left.OpennessClosed, 1e-6f);
        Assert.AreEqual(0.15f, estimate.Right.OpennessClosed, 1e-6f);
        Assert.AreEqual(0.95f, estimate.Left.OpennessWide, 1e-6f);
        Assert.AreEqual(0.98f, estimate.Right.OpennessWide, 1e-6f);
        Assert.AreEqual(0.12f, estimate.Left.GazeCenterX, 1e-6f);
        Assert.AreEqual(-0.08f, estimate.Right.GazeCenterX, 1e-6f);
    }

    [TestMethod]
    public void EstimatorRejectsOverlappingEyeRanges()
    {
        // Left: closed 0.45 vs relaxed 0.50 — the bottom segment of the openness mapping would be
        // meaningless. This one is still a hard failure; no substitute anchor could be honest.
        var relaxed = Samples(0.50f, 0.75f);
        var closed = Samples(0.45f, 0.05f);
        var wide = Samples(0.53f, 0.95f);

        var estimate = EyeCalibrationEstimator.Estimate(relaxed, closed, wide);

        Assert.IsFalse(estimate.IsValid);
        StringAssert.Contains(estimate.Error, "Left eye samples overlap");
    }

    /// <summary>
    /// Every failure message must name the numbers that caused it.
    /// </summary>
    /// <remarks>
    /// The original message said only "samples overlap. Repeat relax, gentle close, and wide-open
    /// steps." — which cannot tell the user whether the close step failed to register, the lid
    /// saturated, or the camera dropped out. A user hit exactly this and could not act on it.
    /// </remarks>
    [TestMethod]
    public void FailureMessageReportsTheMeasuredAnchors()
    {
        var estimate = EyeCalibrationEstimator.Estimate(
            Samples(0.50f, 0.75f), Samples(0.45f, 0.05f), Samples(0.53f, 0.95f));

        Assert.IsFalse(estimate.IsValid);
        StringAssert.Contains(estimate.Error, "0.45");  // measured closed
        StringAssert.Contains(estimate.Error, "0.50");  // measured relaxed
        StringAssert.Contains(estimate.Error, "0.53");  // measured wide
    }

    /// <summary>
    /// A lid that cannot express "wide" must not block calibration.
    /// </summary>
    /// <remarks>
    /// On the 12-output eye model, opening wide shows up mainly in the Widen channel; the lid is
    /// often already near its ceiling when relaxed. Demanding lid headroom above neutral rejected
    /// good captures. A synthetic wide anchor is substituted and the user is told why.
    /// </remarks>
    [TestMethod]
    public void SaturatedLidGetsASyntheticWideAnchorRatherThanAFailure()
    {
        var relaxed = Samples(0.79f, 0.62f);
        var closed = Samples(0.12f, 0.08f);
        var wide = Samples(0.81f, 0.94f); // left barely moves; right is healthy

        var estimate = EyeCalibrationEstimator.Estimate(relaxed, closed, wide);

        Assert.IsTrue(estimate.IsValid, estimate.Error);
        Assert.AreEqual(
            0.79f + EyeCalibrationProfile.MinimumNeutralToWideSpan,
            estimate.Left.OpennessWide, 1e-6f,
            "the left eye should have been given a synthetic wide anchor");
        Assert.AreEqual(0.94f, estimate.Right.OpennessWide, 1e-6f,
            "a healthy eye must keep its measured wide anchor");
        Assert.IsNotNull(estimate.Note);
        StringAssert.Contains(estimate.Note, "Left");
        StringAssert.Contains(estimate.Note, "Widen channel");
    }

    [TestMethod]
    public void ASyntheticWideAnchorStillProducesAUsableMapping()
    {
        var estimate = EyeCalibrationEstimator.Estimate(
            Samples(0.79f, 0.79f), Samples(0.12f, 0.12f), Samples(0.80f, 0.80f));

        Assert.IsTrue(estimate.IsValid, estimate.Error);
        var profile = estimate.Left;

        // The contract the rest of the pipeline depends on: closed reads 0, relaxed reads 0.75.
        Assert.IsTrue(profile.HasUsableOpennessRange);
        Assert.AreEqual(0f, profile.MapOpenness(0.12f), 1e-5f);
        Assert.AreEqual(EyeCalibrationProfile.RestOpenness, profile.MapOpenness(0.79f), 1e-5f);
        Assert.AreEqual(1f, profile.MapOpenness(1f), 1e-5f);
    }

    [TestMethod]
    public void AHealthyCaptureCarriesNoNote()
    {
        var estimate = EyeCalibrationEstimator.Estimate(
            Samples(0.70f, 0.72f), Samples(0.10f, 0.11f), Samples(0.92f, 0.95f));

        Assert.IsTrue(estimate.IsValid, estimate.Error);
        Assert.IsNull(estimate.Note);
    }

    [TestMethod]
    public void BothEyesSaturatedProducesOneNoteMentioningEach()
    {
        var estimate = EyeCalibrationEstimator.Estimate(
            Samples(0.78f, 0.80f), Samples(0.10f, 0.12f), Samples(0.79f, 0.81f));

        Assert.IsTrue(estimate.IsValid, estimate.Error);
        StringAssert.Contains(estimate.Note, "Left");
        StringAssert.Contains(estimate.Note, "Right");
    }

    [TestMethod]
    public void EstimatorRequiresEnoughSamplesForRobustPercentiles()
    {
        var tooFew = Samples(0.70f, 0.70f).Take(
            EyeCalibrationEstimator.MinimumSamplesPerStep - 1).ToArray();

        var estimate = EyeCalibrationEstimator.Estimate(tooFew, tooFew, tooFew);

        Assert.IsFalse(estimate.IsValid);
        StringAssert.Contains(estimate.Error, "at least 30");
    }

    [TestMethod]
    public void LeftAndRightProfilesPersistIndependentlyAcrossSettingsReload()
    {
        var directory = Path.Combine(Path.GetTempPath(), $"baballonia-eye-cal-{Guid.NewGuid():N}");
        var path = Path.Combine(directory, "LocalSettings.json");
        try
        {
            var options = Options.Create(new LocalSettingsOptions { LocalSettingsFile = path });
            var settings = new LocalSettingsService(options, NullLogger<LocalSettingsService>.Instance);
            var left = Profile(closed: 0.08f, neutral: 0.68f, wide: 0.92f, centerX: 0.12f);
            var right = Profile(closed: 0.14f, neutral: 0.78f, wide: 0.98f, centerX: -0.09f);
            settings.SaveSetting(EyeCalibrationSettings.LeftKey, left);
            settings.SaveSetting(EyeCalibrationSettings.RightKey, right);
            settings.ForceSave();

            var reloaded = new LocalSettingsService(options, NullLogger<LocalSettingsService>.Instance);

            Assert.AreEqual(left, reloaded.ReadSetting<EyeCalibrationProfile>(EyeCalibrationSettings.LeftKey));
            Assert.AreEqual(right, reloaded.ReadSetting<EyeCalibrationProfile>(EyeCalibrationSettings.RightKey));
        }
        finally
        {
            if (Directory.Exists(directory))
                Directory.Delete(directory, recursive: true);
        }
    }

    private static List<EyeCalibrationSample> Samples(
        float leftLid,
        float rightLid,
        float leftX = 0f,
        float rightX = 0f) =>
        Enumerable.Range(0, 100)
            .Select(_ => new EyeCalibrationSample(
                leftLid, leftX, 0.03f, rightLid, rightX, -0.04f))
            .ToList();

    private static EyeCalibrationProfile Profile(
        float closed = 0.05f,
        float neutral = 0.75f,
        float wide = 0.95f,
        float centerX = 0f,
        float centerY = 0f,
        float gainX = 1f,
        float gainY = 1f) =>
        new(EyeCalibrationProfile.CurrentVersion, closed, neutral, wide,
            centerX, centerY, gainX, gainY);
}
