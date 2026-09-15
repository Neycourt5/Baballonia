using System;
using Baballonia.Desktop.Calibration;
using Baballonia.Services.Calibration;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Calibration;

/// <summary>
/// Where the VR panel sits, and why it is two different panels.
/// </summary>
/// <remarks>
/// <para>The user reported the calibration overlay covering their view: guided face calibration
/// asks you to watch your avatar in a mirror and copy its expression, and the panel was directly in
/// front of the face doing exactly the thing that makes that impossible. The old geometry spanned
/// roughly ±28° horizontally and −24° to +20° vertically at 1.25 m — the whole central field of
/// view.</para>
///
/// <para>So instructions now use a small translucent panel parked below the line of sight, while
/// the gaze dot — which <em>is</em> the thing you look at — keeps the full panel. These pin the
/// arithmetic, because it cannot be checked without a headset on.</para>
/// </remarks>
[TestClass]
[TestSubject(typeof(OpenVrCalibrationPresenter))]
public class OverlayLayoutTest
{
    // Mirrors the private constants. If these drift, the assertions below stop describing the
    // shipped panel, so they are stated once here and derived from.
    private const float InstructionWidthMeters = 0.75f;
    private const float InstructionVisibleRows = 360f;
    private const float InstructionCentreY = -0.42f;
    private const float InstructionDistance = 1.20f;
    private const float TargetWidthMeters = 1.35f;
    private const float TargetDistance = 1.25f;
    private const int TextureWidth = 1024;
    private const int TextureHeight = 768;

    private static double Degrees(double metres, double distance) =>
        Math.Atan2(metres, distance) * 180.0 / Math.PI;

    private static (double Top, double Bottom, double HalfWidth) InstructionExtent()
    {
        var halfWidth = InstructionWidthMeters / 2;
        var height = InstructionWidthMeters * (InstructionVisibleRows / TextureWidth);
        var top = InstructionCentreY + height / 2;
        var bottom = InstructionCentreY - height / 2;
        return (Degrees(top, InstructionDistance),
                Degrees(bottom, InstructionDistance),
                Degrees(halfWidth, InstructionDistance));
    }

    [TestMethod]
    public void TheInstructionPanelSitsEntirelyBelowTheLineOfSight()
    {
        var (top, _, _) = InstructionExtent();

        // The mirror, and the avatar face being copied, are straight ahead. Anything at or above
        // eye level is in the way.
        Assert.IsTrue(top < -8,
            $"the instruction panel reaches {top:F1}° above centre; it must stay clear of the view");
    }

    [TestMethod]
    public void TheInstructionPanelIsStillCloseEnoughToGlanceAt()
    {
        var (top, bottom, _) = InstructionExtent();

        // Parked below the view is only useful if a glance reaches it. Much past 30° and the user
        // is craning rather than glancing.
        Assert.IsTrue(top > -20, $"the panel starts {top:F1}° down, which is a long way to look");
        Assert.IsTrue(bottom > -35, $"the panel ends {bottom:F1}° down, which is off the bottom");
    }

    [TestMethod]
    public void TheInstructionPanelIsNarrowEnoughToLeaveTheMirrorVisible()
    {
        var (_, _, halfWidth) = InstructionExtent();

        Assert.IsTrue(halfWidth < 20,
            $"the panel spans ±{halfWidth:F1}°, which is most of the central view");
    }

    [TestMethod]
    public void InstructionTextStaysBigEnoughToRead()
    {
        // The panel got smaller, so the text got smaller with it. The phase label is the one thing
        // that must be readable from the corner of the eye.
        const float phaseLabelPixels = 62f;
        var metresPerPixel = InstructionWidthMeters / TextureWidth;
        var degrees = Degrees(phaseLabelPixels * metresPerPixel, InstructionDistance);

        Assert.IsTrue(degrees > 1.5,
            $"the phase label subtends {degrees:F2}°, which is too small to glance at");
    }

    [TestMethod]
    public void TheGazePanelStillReachesEveryDot()
    {
        // The dot grid goes to ±18°; a panel that cannot reach it would have the user looking at
        // the panel edge while we recorded it as a fixation on the target.
        var halfWidth = Degrees(TargetWidthMeters / 2, TargetDistance);

        Assert.IsTrue(halfWidth >= 18,
            $"the gaze panel only reaches ±{halfWidth:F1}°, short of the ±18° dot grid");
    }

    [TestMethod]
    public void TheGazePanelReachesTheLowestDot()
    {
        var height = TargetWidthMeters * ((float)TextureHeight / TextureWidth);
        var bottom = Degrees(-0.05 - height / 2, TargetDistance);

        Assert.IsTrue(bottom <= -13,
            $"the gaze panel bottoms out at {bottom:F1}°, above the -13° dot");
    }
}

/// <summary>
/// The pointer targets, which move with the layout.
/// </summary>
/// <remarks>
/// A button that is drawn but not clickable is a worse failure than one that is missing: the user
/// keeps pressing it. So the rectangles are shared between the renderer and the hit test, and the
/// hit test accepts every coordinate convention OpenVR might report.
/// </remarks>
[TestClass]
[TestSubject(typeof(OpenVrCalibrationPresenter))]
public class OverlayHitTestTest
{
    private static VrCalibrationFrame Instructions() => new(
        "SMILE", "Smile naturally.", VrCalibrationPhase.Hold,
        OverallProgress: 0.4, PhaseProgress: 0.5,
        Repetition: 1, RepetitionCount: 2,
        AllowRetry: true, AllowSkip: true, AllowCancel: true);

    private static VrCalibrationFrame GazeTarget() => new(
        "EYE", "Look at the dot.", VrCalibrationPhase.Target,
        OverallProgress: 0.3, TargetX: 0.4f, TargetY: -0.2f, AllowCancel: true);

    [TestMethod]
    public void TheInstructionButtonsAreClickableWhereTheyAreDrawn()
    {
        var frame = Instructions();

        Assert.AreEqual(VrCalibrationAction.Retry,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 140, 265));
        Assert.AreEqual(VrCalibrationAction.Skip,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 340, 265));
        Assert.AreEqual(VrCalibrationAction.Cancel,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 530, 265));
    }

    [TestMethod]
    public void TheInstructionButtonsWorkUnderTheReflectedYConvention()
    {
        // Bindings differ on whether overlay mouse Y counts from the top or the bottom.
        var frame = Instructions();

        Assert.AreEqual(VrCalibrationAction.Retry,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 140, 768 - 265));
    }

    [TestMethod]
    public void TheInstructionButtonsWorkWhenYIsScaledToTheVisibleStrip()
    {
        // The instruction layout crops the texture, so Y may arrive spanning only the visible part.
        var frame = Instructions();
        const float scaled = 265f / (360f / 768f);

        Assert.AreEqual(VrCalibrationAction.Retry,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 140, scaled));
    }

    [TestMethod]
    public void EmptySpaceIsNotAButton()
    {
        var frame = Instructions();

        Assert.AreEqual(VrCalibrationAction.None,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 140, 100),
            "the heading area must not be a hidden retry button");
        Assert.AreEqual(VrCalibrationAction.None,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 900, 265),
            "there is no button that far right in the compact layout");
    }

    [TestMethod]
    public void ADisabledButtonIsNotClickable()
    {
        var frame = Instructions() with { AllowRetry = false };

        Assert.AreEqual(VrCalibrationAction.None,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 140, 265));
    }

    [TestMethod]
    public void TheGazeLayoutKeepsItsOriginalCancelTarget()
    {
        var frame = GazeTarget();

        Assert.AreEqual(VrCalibrationAction.Cancel,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 800, 700));
        Assert.AreEqual(VrCalibrationAction.None,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 800, 265),
            "the gaze layout must not inherit the compact layout's button band");
    }
}
