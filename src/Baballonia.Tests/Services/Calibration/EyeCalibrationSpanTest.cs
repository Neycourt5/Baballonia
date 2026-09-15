using Baballonia.Services.Calibration;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Calibration;

/// <summary>
/// The span gates that decide whether a saved calibration does anything at all.
/// </summary>
/// <remarks>
/// A profile that fails <see cref="EyeCalibrationProfile.HasUsableOpennessRange"/> is not rejected
/// or reported — <see cref="EyeCalibrationProfile.MapOpenness"/> quietly becomes the identity. That
/// makes an off-by-a-rounding-error gate invisible: the user completes the setup, sees "saved and
/// applied", and gets no calibration.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeCalibrationProfile))]
public class EyeCalibrationSpanTest
{
    private static EyeCalibrationProfile Profile(float closed, float neutral, float wide) =>
        EyeCalibrationProfile.Default with
        {
            OpennessClosed = closed,
            OpennessNeutral = neutral,
            OpennessWide = wide,
        };

    [TestMethod]
    public void TheSyntheticWideAnchorSatisfiesItsOwnGate()
    {
        // The bug this exists for. The estimator builds the wide anchor as
        // neutral + MinimumNeutralToWideSpan precisely so the profile is usable, but in float32
        // (n + 0.05f) - n can be 0.04999995, so an exact >= rejected a usable profile.
        // The deliberately synthetic anchors cover both sides of float32 rounding.
        foreach (var neutral in new[] { 0.65f, 0.80f, 0.97f, 0.99f })
        {
            var wide = neutral + EyeCalibrationProfile.MinimumNeutralToWideSpan;
            var profile = Profile(neutral - 0.30f, neutral, wide);

            Assert.IsTrue(profile.HasUsableOpennessRange,
                $"a synthetic wide anchor at neutral {neutral} must produce a usable profile");
        }
    }

    [TestMethod]
    public void SyntheticSerializedAnchorsRemainUsableAfterFloatRounding()
    {
        // Decimal literals model serialization without copying anyone's calibration.
        foreach (var profile in new[] { Profile(0.55f, 0.97f, 1.02f), Profile(0.60f, 0.99f, 1.04f) })
        {
            Assert.IsTrue(profile.OpennessWide - profile.OpennessNeutral <
                          EyeCalibrationProfile.MinimumNeutralToWideSpan,
                "the fixture must reproduce the old strict comparison failure");
            Assert.IsTrue(profile.HasUsableOpennessRange);
        }
    }

    [TestMethod]
    public void AnInertProfileMapsNothing()
    {
        // Why the bug was silent rather than loud: the fallback is the identity, which looks like
        // working tracking rather than like a failure.
        var inert = Profile(0.70f, 0.72f, 0.73f);

        Assert.IsFalse(inert.HasUsableOpennessRange);
        foreach (var value in new[] { 0f, 0.3f, 0.71f, 1f })
            Assert.AreEqual(value, inert.MapOpenness(value), 1e-6);
    }

    [TestMethod]
    public void AUsableProfileStillMapsTheAnchors()
    {
        var profile = Profile(0.20f, 0.80f, 0.95f);

        Assert.AreEqual(0f, profile.MapOpenness(0.20f), 1e-5, "closed maps to shut");
        Assert.AreEqual(EyeCalibrationProfile.RestOpenness, profile.MapOpenness(0.80f), 1e-5);
        Assert.AreEqual(1f, profile.MapOpenness(0.95f), 1e-5, "wide maps to fully open");
    }

    [TestMethod]
    public void TheToleranceDoesNotAdmitAGenuinelyFlatCapture()
    {
        // The slack absorbs rounding, not a bad recording. A lid channel that cannot tell closed
        // from relaxed must still be refused.
        var flat = Profile(0.78f, 0.80f, 0.86f);

        Assert.IsFalse(flat.HasUsableOpennessRange,
            "a closed-to-relaxed span of 0.02 is not a rounding error");
    }

    [TestMethod]
    public void TheToleranceIsFarSmallerThanTheSpansItGuards()
    {
        // If this ever stops holding, the tolerance has grown into a real relaxation of the gate.
        Assert.IsTrue(
            EyeCalibrationProfile.SpanTolerance < EyeCalibrationProfile.MinimumNeutralToWideSpan / 100f);
    }
}
