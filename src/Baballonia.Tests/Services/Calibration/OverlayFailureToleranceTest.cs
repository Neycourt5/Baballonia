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

/// <summary>
/// What the overlay is allowed to delay, now that cosmetic-only updates are rate-limited.
/// </summary>
/// <remarks>
/// The overlay was re-uploading a three-megabyte texture about sixteen times a second to nudge a
/// progress bar, and the user reported it as visible flashing. Cosmetic updates are now held to a
/// minimum interval — which is only safe because "cosmetic" excludes everything the user has to
/// act on. These pin the classification the throttle depends on, and the gaze dot in particular,
/// since a dot that moved late would be recorded against the position it moved *from*.
/// </remarks>
[TestClass]
[TestSubject(typeof(OpenVrCalibrationPresenter))]
public class OverlayUpdateThrottleTest
{
    private static VrCalibrationFrame Target() => new(
        "EYE PERSONALIZATION", "Look at the dot and hold still.", VrCalibrationPhase.Target,
        OverallProgress: 0.3, PhaseProgress: 0.2, TargetX: 0.4f, TargetY: -0.2f);

    [TestMethod]
    public void AMovingDotIsNeverTreatedAsCosmetic()
    {
        var onScreen = Target();

        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { TargetX = -0.4f }, onScreen),
            "a delayed dot would be recorded against the position it moved from");
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { TargetY = 0.3f }, onScreen));
    }

    [TestMethod]
    public void ADotAppearingOrVanishingIsNeverCosmetic()
    {
        var onScreen = Target();

        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { TargetX = null, TargetY = null }, onScreen));
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen, onScreen with { TargetX = null, TargetY = null }));
    }

    [TestMethod]
    public void OnlyTheProgressBarMovingIsCosmetic()
    {
        // This is the case the throttle exists for: during a hold, nothing changes but the bar.
        var onScreen = Target();

        Assert.IsTrue(OpenVrCalibrationPresenter.IsCosmeticUpdate(
            onScreen with { PhaseProgress = 0.9, OverallProgress = 0.35 }, onScreen));
    }

    [TestMethod]
    public void NothingIsCosmeticAgainstABlankScreen()
    {
        // On the first frame there is nothing on screen to compare against, so it must go up
        // immediately rather than being held back as a "small change".
        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(Target(), null));
    }
}
