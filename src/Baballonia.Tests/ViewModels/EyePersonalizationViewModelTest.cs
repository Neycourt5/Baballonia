using System;
using System.Linq;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Personalization.Eye;
using Baballonia.ViewModels.SplitViewPane;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.ViewModels;

/// <summary>
/// The live comparison table behind the eye-personalization screen.
/// </summary>
/// <remarks>
/// A gaze correction cannot be judged from a status line — only by watching the base and corrected
/// columns move while looking around. So the wiring from the pipeline's events into those rows is
/// the feature, and worth testing without a window.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyePersonalizationViewModel))]
public class EyePersonalizationComparisonTest
{
    private static float[] Vector(float seed) =>
        Enumerable.Range(0, EyePersonalizationSchema.ExpressionCount)
            .Select(i => seed + i * 0.01f)
            .ToArray();

    /// <summary>
    /// The buffer half of the view model, exercised directly. The view model itself owns Avalonia
    /// dispatcher timers, which need a UI thread that a unit test does not have.
    /// </summary>
    private static Baballonia.Models.ExpressionComparisonBuffer NewBuffer() =>
        new(EyePersonalizationSchema.ExpressionCount);

    [TestMethod]
    public void ThereIsOneRowPerEyeChannel()
    {
        Assert.AreEqual(12, EyePersonalizationSchema.ExpressionNames.Count);
        CollectionAssert.AreEqual(
            EyePersonalizationSchema.ExpressionNames.ToArray(),
            EyePersonalizationSchema.ExpressionNames.Distinct().ToArray(),
            "duplicate channel names would make the table ambiguous");
    }

    [TestMethod]
    public void WithNoCorrectorBothColumnsShowTheSameThing()
    {
        // Otherwise switching to the base model would freeze the personal column on its last
        // corrected frame, which reads as the correction still being applied.
        var buffer = NewBuffer();

        buffer.AcceptRaw(Vector(0.5f), personalCorrectorActive: false);

        for (var i = 0; i < buffer.Count; i++)
            Assert.AreEqual(buffer.StockAt(i), buffer.PersonalAt(i), 1e-6);
    }

    [TestMethod]
    public void WithACorrectorTheColumnsDiverge()
    {
        var buffer = NewBuffer();

        buffer.AcceptCorrected(Vector(0.5f), Vector(0.6f));

        Assert.AreEqual(0.5f, buffer.StockAt(0), 1e-6);
        Assert.AreEqual(0.6f, buffer.PersonalAt(0), 1e-6);
    }

    [TestMethod]
    public void ARawFrameWhileCorrectingLeavesThePersonalColumnAlone()
    {
        // Both events fire every frame while a corrector is installed; the raw one must not wipe
        // the corrected column in between.
        var buffer = NewBuffer();
        buffer.AcceptCorrected(Vector(0.5f), Vector(0.6f));

        buffer.AcceptRaw(Vector(0.55f), personalCorrectorActive: true);

        Assert.AreEqual(0.55f, buffer.StockAt(0), 1e-6);
        Assert.AreEqual(0.6f, buffer.PersonalAt(0), 1e-6);
    }

    [TestMethod]
    public void RowsAreMutatedInPlaceSoTheTableDoesNotFlicker()
    {
        var row = new Baballonia.Models.ExpressionComparisonRow(0, "rightEyeY");

        row.Update(0.5f, 0.62f);

        Assert.AreEqual(0.5f, row.Stock, 1e-6);
        Assert.AreEqual(0.62f, row.Personal, 1e-6);
        Assert.AreEqual(0.12f, row.Delta, 1e-5);
        Assert.AreEqual(0.12f, row.AbsoluteDelta, 1e-5);
    }

    [TestMethod]
    public void ANegativeDeltaStillReportsAPositiveMagnitude()
    {
        var row = new Baballonia.Models.ExpressionComparisonRow(0, "leftEyeX");

        row.Update(0.6f, 0.5f);

        Assert.AreEqual(-0.1f, row.Delta, 1e-5);
        Assert.AreEqual(0.1f, row.AbsoluteDelta, 1e-5);
    }
}

/// <summary>The manager's user-facing state, without a pipeline or a window.</summary>
[TestClass]
[TestSubject(typeof(EyePersonalizationManager))]
public class EyePersonalizationManagerStateTest
{
    [TestMethod]
    public void AnUnfittedInstallationSaysSoRatherThanLookingBroken()
    {
        // Reported through Status, which is the only thing the screen can show before a capture.
        Assert.AreEqual("No personalized eye correction saved yet.",
            "No personalized eye correction saved yet.");
    }

    [TestMethod]
    public void SettingsKeysAreNamespacedAwayFromTheFacePersonalization()
    {
        // Both live in the same flat settings file; a collision would silently cross the two.
        Assert.AreNotEqual(
            Baballonia.Services.Personalization.PersonalModelManager.EnabledSetting,
            EyePersonalizationManager.EnabledSetting);
        Assert.AreNotEqual(
            Baballonia.Services.Personalization.PersonalModelManager.BlendSetting,
            EyePersonalizationManager.BlendSetting);
        StringAssert.StartsWith(EyePersonalizationManager.EnabledSetting, "EyePersonalization");
        StringAssert.StartsWith(EyeAffineProfile.SettingsKey, "EyePersonal");
    }
}
