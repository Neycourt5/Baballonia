using Baballonia.Services.Calibration;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Calibration;

/// <summary>
/// Quantization is what lets the headset presenter skip work, so these tests are really about the
/// overlay staying stable: guided capture publishes twenty frames a second while the panel changes
/// perhaps twice, and redrawing plus re-uploading a megabyte-wide texture for identical pixels is
/// what made it visibly flicker. Two frames that would rasterize the same must compare equal.
/// </summary>
[TestClass]
[TestSubject(typeof(VrCalibrationFrame))]
public class VrCalibrationFrameQuantizeTest
{
    private static VrCalibrationFrame Frame(
        double overall = 0.5, double phase = 0.5, double? countdown = null, float? intensity = null) =>
        new("SMILE", "Smile naturally.", VrCalibrationPhase.Hold, overall, phase, countdown,
            Intensity: intensity);

    [TestMethod]
    public void IdenticalFramesAreEqual_SoNothingIsRedrawn()
    {
        Assert.AreEqual(Frame().Quantize(), Frame().Quantize());
    }

    [TestMethod]
    public void FloatDriftBelowOneStepIsAbsorbed()
    {
        // A 3-second hold ticked at 20 Hz moves phase progress by ~1.7% per tick. Without
        // quantization every one of those ticks is "new content".
        var a = Frame(phase: 0.500).Quantize();
        var b = Frame(phase: 0.5049).Quantize();

        Assert.AreEqual(a, b, "Sub-step progress drift must not count as a change.");
    }

    [TestMethod]
    public void MovementBeyondAStepStillRegisters()
    {
        var a = Frame(phase: 0.50).Quantize();
        var b = Frame(phase: 0.56).Quantize();

        Assert.AreNotEqual(a, b, "Real progress must still redraw, or the bar would freeze.");
    }

    [TestMethod]
    public void CountdownCollapsesToWholeSeconds()
    {
        // The panel renders ceil(seconds), so anything finer is invisible - but it changed the
        // float on every single tick.
        var a = Frame(countdown: 2.9).Quantize();
        var b = Frame(countdown: 2.1).Quantize();
        var c = Frame(countdown: 1.9).Quantize();

        Assert.AreEqual(a, b, "Both render as '3'.");
        Assert.AreNotEqual(b, c, "3 and 2 are different displays.");
    }

    [TestMethod]
    public void CountdownNeverRendersZero()
    {
        Assert.AreEqual(1d, Frame(countdown: 0.05).Quantize().CountdownSeconds);
    }

    [TestMethod]
    public void TextAndPhaseChangesAlwaysRegister()
    {
        var baseline = Frame().Quantize();

        Assert.AreNotEqual(baseline, (Frame() with { Title = "FROWN" }).Quantize());
        Assert.AreNotEqual(baseline, (Frame() with { Instruction = "Frown." }).Quantize());
        Assert.AreNotEqual(baseline, (Frame() with { Phase = VrCalibrationPhase.Relax }).Quantize());
        Assert.AreNotEqual(baseline, (Frame() with { AllowRetry = true }).Quantize());
    }

    [TestMethod]
    public void IntensityCollapsesToTheWholePercentItDisplays()
    {
        Assert.AreEqual(Frame(intensity: 0.500f).Quantize(), Frame(intensity: 0.5004f).Quantize());
        Assert.AreNotEqual(Frame(intensity: 0.50f).Quantize(), Frame(intensity: 0.52f).Quantize());
    }

    [TestMethod]
    public void QuantizeIsIdempotent()
    {
        var once = Frame(overall: 0.333333, phase: 0.777777, countdown: 2.4).Quantize();
        Assert.AreEqual(once, once.Quantize());
    }

    [TestMethod]
    public void ClampStillAppliesIndependently()
    {
        var clamped = Frame(overall: 1.8, phase: -0.4).Clamp();
        Assert.AreEqual(1d, clamped.OverallProgress);
        Assert.AreEqual(0d, clamped.PhaseProgress);
    }
}
