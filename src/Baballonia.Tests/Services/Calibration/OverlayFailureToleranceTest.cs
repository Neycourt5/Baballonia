using Baballonia.Desktop.Calibration;
using Baballonia.Services.Calibration;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Calibration;

/// <summary>
/// SteamVR can refuse a single overlay upload across a compositor hiccup, and tearing the panel down
/// for that makes it vanish mid-calibration. But tolerating the wrong failure is worse than
/// tolerating none: if an update carrying a NEW instruction is silently dropped, the headset keeps
/// showing the old phase while the recorder labels frames with the new one, and the session is
/// confidently mislabelled.
///
/// So the rule is content-based, and this pins it down.
/// </summary>
[TestClass]
[TestSubject(typeof(OpenVrCalibrationPresenter))]
public class OverlayFailureToleranceTest
{
    private static VrCalibrationFrame OnScreen() => new(
        "SMILE", "Smile naturally.", VrCalibrationPhase.Hold,
        OverallProgress: 0.4, PhaseProgress: 0.5, Intensity: 0.5f,
        Repetition: 1, RepetitionCount: 2, AllowRetry: true, AllowSkip: true);

    [TestMethod]
    public void ProgressAndCountdownDriftIsCosmetic()
    {
        var onScreen = OnScreen();

        Assert.IsTrue(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { PhaseProgress = 0.9, OverallProgress = 0.45 }, onScreen));
        Assert.IsTrue(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { CountdownSeconds = 2 }, onScreen));
        Assert.IsTrue(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { Intensity = 1.0f }, onScreen));
    }

    [TestMethod]
    public void AnyChangeTheUserMustActOnIsNotCosmetic()
    {
        var onScreen = OnScreen();

        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { Phase = VrCalibrationPhase.Relax }, onScreen),
            "HOLD becoming RELAX is the whole point of the display.");
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { Title = "FROWN" }, onScreen),
            "A different expression is being asked for.");
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { Instruction = "Pucker your lips." }, onScreen));
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { Repetition = 2 }, onScreen),
            "A new repetition is a new attempt.");
    }

    [TestMethod]
    public void ButtonChangesAreNotCosmetic()
    {
        var onScreen = OnScreen();

        // Offering or withdrawing Retry/Skip/Cancel changes where a controller click lands, so a
        // dropped update would make the strip disagree with the hit test.
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { AllowRetry = false }, onScreen));
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { AllowSkip = false }, onScreen));
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { AllowCancel = false }, onScreen));
    }

    [TestMethod]
    public void MovingAGazeTargetIsNotCosmetic()
    {
        var onScreen = OnScreen() with { TargetX = 0f, TargetY = 0f };

        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { TargetX = -0.7f }, onScreen),
            "The user is told to look somewhere else; a dropped move would sample the wrong angle.");
        Assert.IsTrue(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { PhaseProgress = 0.8 }, onScreen));
    }

    [TestMethod]
    public void NothingOnScreenYetIsNeverCosmetic()
    {
        // The first frame of a session has no predecessor to fall back to, so it must land.
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(OnScreen(), null));
    }
}
