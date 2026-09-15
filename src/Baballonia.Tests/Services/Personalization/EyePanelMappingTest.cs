using Baballonia.Services.Personalization.Eye;
using Baballonia.ViewModels.SplitViewPane;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Where the guided dot is drawn, in both the headset and the desktop fallback.
/// </summary>
/// <remarks>
/// This exists because of a sign. Gaze degrees are positive upward; the overlay draws at
/// <c>y = centre + TargetY * scale</c> in canvas coordinates, where positive is downward. Getting
/// that backwards would put the "look up" dot below centre, and the recorded fixation would be
/// labelled with the opposite of where the user actually looked — producing a confidently inverted
/// vertical correction rather than an obvious failure.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeGuidedCues))]
public class EyePanelMappingTest
{
    [TestMethod]
    public void HorizontalPanelCoordinatesFollowGazeDirectly()
    {
        Assert.AreEqual(0f, EyeGuidedCues.PanelFromDegrees(0f), 1e-6);
        Assert.IsTrue(EyeGuidedCues.PanelFromDegrees(18f) > 0f, "looking right is right of centre");
        Assert.IsTrue(EyeGuidedCues.PanelFromDegrees(-18f) < 0f, "looking left is left of centre");
    }

    [TestMethod]
    public void LookingUpPutsTheDotAboveCentre()
    {
        // Screen-space: negative is up.
        Assert.IsTrue(EyeGuidedCues.PanelYFromDegrees(6f) < 0f,
            "a 'look up' cue must place the dot above centre, or the label is inverted");
        Assert.IsTrue(EyeGuidedCues.PanelYFromDegrees(-13f) > 0f,
            "a 'look down' cue must place the dot below centre");
        Assert.AreEqual(0f, EyeGuidedCues.PanelYFromDegrees(0f), 1e-6);
    }

    [TestMethod]
    public void PanelCoordinatesStayOnThePanel()
    {
        foreach (var degrees in new[] { -90f, -20f, 0f, 20f, 90f })
        {
            Assert.IsTrue(EyeGuidedCues.PanelFromDegrees(degrees) is >= -1f and <= 1f);
            Assert.IsTrue(EyeGuidedCues.PanelYFromDegrees(degrees) is >= -1f and <= 1f);
        }
    }

    [TestMethod]
    public void TheDesktopDotSpansTheCanvasWithoutLeavingIt()
    {
        // Mirrors the arithmetic in EyePersonalizationViewModel.AdvanceCapture.
        const double radius = 19;
        foreach (var panel in new[] { -1.0, -0.5, 0.0, 0.5, 1.0 })
        {
            var left = (panel + 1) / 2 * EyePersonalizationViewModel.DotCanvasWidth - radius;
            var top = (panel + 1) / 2 * EyePersonalizationViewModel.DotCanvasHeight - radius;

            Assert.IsTrue(left >= -radius && left <= EyePersonalizationViewModel.DotCanvasWidth - radius);
            Assert.IsTrue(top >= -radius && top <= EyePersonalizationViewModel.DotCanvasHeight - radius);
        }

        var centre = (0.0 + 1) / 2 * EyePersonalizationViewModel.DotCanvasWidth - radius;
        Assert.AreEqual(EyePersonalizationViewModel.DotCanvasWidth / 2 - radius, centre, 1e-9);
    }

    [TestMethod]
    public void TheUpwardDotIsAboveTheDownwardOneOnScreen()
    {
        // The end-to-end property, stated the way a person would check it in the headset.
        var up = EyeGuidedCues.PanelYFromDegrees(6f);
        var down = EyeGuidedCues.PanelYFromDegrees(-13f);

        Assert.IsTrue(up < down, $"'look up' ({up}) must draw higher than 'look down' ({down})");
    }
}
