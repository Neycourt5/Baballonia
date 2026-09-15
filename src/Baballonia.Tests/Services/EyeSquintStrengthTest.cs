using System.Reflection;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services;

/// <summary>
/// Scaling how strongly a detected squint is expressed, without touching what is detected.
/// </summary>
/// <remarks>
/// <para>Applied at the last point the value is mutable — after the calibration remap and
/// immediately before the OSC enqueue. Everything downstream is a verbatim copy into
/// <c>UnifiedTracking.Data</c>.</para>
///
/// <para>That placement is the whole safety argument. The eyelid channel is an <em>input</em> to the
/// gaze fusion, so scaling anything lid-derived earlier would move where the eyes appear to look;
/// and scaling before the One Euro filter would change the smoothing itself, because its adaptive
/// cutoff depends on the signal's own derivative. Both are avoided by construction here.</para>
/// </remarks>
[TestClass]
[TestSubject(typeof(ParameterSenderService))]
public class EyeSquintStrengthTest
{
    private const string LeftSquint = "/leftEyeSquint";
    private const string RightSquint = "/rightEyeSquint";

    // Reached by reflection, matching ParameterSenderEyeSafetyTest — these stay internal.
    private static float Scale(string key, float value, float strength) =>
        (float)typeof(ParameterSenderService)
            .GetMethod("ScaleSquint", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, [key, value, strength])!;

    [TestMethod]
    public void FullStrengthIsExactlyUnchanged()
    {
        // The default. Short-circuited rather than relying on a float multiply by one.
        foreach (var v in new[] { 0f, 0.17f, 0.5f, 0.83f, 1f })
            Assert.AreEqual(v, Scale(LeftSquint, v, 1f), 0f, "1.0 must be bit-for-bit identity");
    }

    [TestMethod]
    public void ZeroStrengthSuppressesSquintEntirely()
    {
        foreach (var v in new[] { 0.2f, 0.6f, 1f })
            Assert.AreEqual(0f, Scale(LeftSquint, v, 0f), 1e-6);
    }

    [TestMethod]
    public void HalfStrengthHalvesTheExpression()
    {
        Assert.AreEqual(0.30f, Scale(LeftSquint, 0.60f, 0.5f), 1e-6);
        Assert.AreEqual(0.10f, Scale(RightSquint, 0.20f, 0.5f), 1e-6);
    }

    [TestMethod]
    public void DoubleStrengthDoublesThenClamps()
    {
        Assert.AreEqual(0.60f, Scale(LeftSquint, 0.30f, 2f), 1e-6);
        Assert.AreEqual(1f, Scale(LeftSquint, 0.80f, 2f), 1e-6, "must clamp rather than overshoot");
    }

    [TestMethod]
    public void TheTwoEyesAreScaledIndependently()
    {
        // Never averaged or coupled — that would destroy the asymmetry the model detects.
        Assert.AreEqual(0.45f, Scale(LeftSquint, 0.90f, 0.5f), 1e-6);
        Assert.AreEqual(0.05f, Scale(RightSquint, 0.10f, 0.5f), 1e-6);
    }

    [TestMethod]
    public void OnlyTheSquintChannelsAreTouched()
    {
        // The lid channel especially: it feeds the gaze fusion, so scaling it would move gaze.
        foreach (var key in new[]
                 {
                     "/leftEyeLid", "/rightEyeLid", "/leftEyeX", "/rightEyeX",
                     "/leftEyeY", "/rightEyeY", "/leftEyeWiden", "/rightEyeWiden",
                     "/leftEyeBrow", "/rightEyeBrow",
                 })
        {
            Assert.AreEqual(0.42f, Scale(key, 0.42f, 0f), 0f,
                $"{key} must be untouched even at strength 0");
            Assert.AreEqual(0.42f, Scale(key, 0.42f, 2f), 0f,
                $"{key} must be untouched even at strength 2");
        }
    }

    [TestMethod]
    public void NegativeGazeIsNotClampedByThisPath()
    {
        // Gaze is signed and lives in [-1,1]. If this ever started touching gaze it would clamp the
        // negative half to zero, so the assertion above is load-bearing, not decorative.
        Assert.AreEqual(-0.7f, Scale("/leftEyeX", -0.7f, 0.5f), 0f);
    }

    [TestMethod]
    public void GarbageIsPassedThroughUntouched()
    {
        Assert.IsTrue(float.IsNaN(Scale(LeftSquint, float.NaN, 2f)));
        Assert.AreEqual(0.5f, Scale(LeftSquint, 0.5f, float.NaN), 1e-6);
    }

    [TestMethod]
    public void TheKeysMatchWhatThePipelineActuallyEmits()
    {
        // Named explicitly so a rename upstream fails here rather than silently disabling the
        // setting — a scalar that quietly stops applying is worse than one that never existed.
        var left = (string)typeof(ParameterSenderService)
            .GetField("LeftSquintKey", BindingFlags.NonPublic | BindingFlags.Static)!
            .GetRawConstantValue()!;
        var right = (string)typeof(ParameterSenderService)
            .GetField("RightSquintKey", BindingFlags.NonPublic | BindingFlags.Static)!
            .GetRawConstantValue()!;

        Assert.AreEqual(LeftSquint, left);
        Assert.AreEqual(RightSquint, right);
    }

    [TestMethod]
    public void CalibrationStillRunsBeforeTheStrengthScalar()
    {
        // Order matters: the user's calibrated range is preserved and only its expression is
        // scaled. Reproduces the send path's two steps in order.
        var settings = new CalibrationParameter { Lower = 0f, Upper = 0.5f, Min = 0f, Max = 1f };

        var calibrated = (float)typeof(ParameterSenderService)
            .GetMethod("CalibrateAndClampEye", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, [LeftSquint, 0.25f, settings])!;

        Assert.AreEqual(0.5f, calibrated, 1e-5, "the calibration remap should have doubled it");
        Assert.AreEqual(0.25f, Scale(LeftSquint, calibrated, 0.5f), 1e-5);
    }
}
