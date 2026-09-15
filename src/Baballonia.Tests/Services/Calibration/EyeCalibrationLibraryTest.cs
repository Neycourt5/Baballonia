using System;
using System.IO;
using System.Threading;
using Baballonia.Models;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using JetBrains.Annotations;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Calibration;

/// <summary>
/// Saved eye calibrations: keeping them, choosing between them, and deleting the bad ones.
/// </summary>
/// <remarks>
/// Calibration used to be a single overwritten slot, so a worse capture destroyed a better one with
/// no way back and no way to compare. The property that matters most here is that the library never
/// diverges from the two profile keys the pipeline actually reads — a dropdown claiming one
/// calibration while the pipeline runs another would be worse than no dropdown at all.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeCalibrationLibrary))]
public class EyeCalibrationLibraryTest
{
    private string _directory = null!;
    private string _settingsPath = null!;
    private LocalSettingsService _settings = null!;
    private EyeCalibrationLibrary _library = null!;

    [TestInitialize]
    public void Initialize()
    {
        _directory = Path.Combine(Path.GetTempPath(), $"baballonia-eye-lib-{Guid.NewGuid():N}");
        _settingsPath = Path.Combine(_directory, "LocalSettings.json");
        _settings = NewSettings();
        _library = new EyeCalibrationLibrary(_settings);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(_directory))
            Directory.Delete(_directory, recursive: true);
    }

    private LocalSettingsService NewSettings() => new(
        Options.Create(new LocalSettingsOptions { LocalSettingsFile = _settingsPath }),
        NullLogger<LocalSettingsService>.Instance);

    private static EyeCalibrationProfile Profile(
        float closed = 0.05f, float neutral = 0.75f, float wide = 0.95f, float centerX = 0f) =>
        new(EyeCalibrationProfile.CurrentVersion, closed, neutral, wide, centerX, 0f, 1f, 1f);

    private EyeCalibrationProfile ActiveLeft() =>
        _settings.ReadSetting<EyeCalibrationProfile>(EyeCalibrationSettings.LeftKey)!;

    [TestMethod]
    public void ANewInstallationIsUncalibrated()
    {
        Assert.AreEqual(0, _library.Entries.Count);
        Assert.AreEqual(EyeCalibrationLibrary.DefaultId, _library.ActiveId);
        Assert.IsNull(_library.Active);
    }

    [TestMethod]
    public void SavingKeepsTheCaptureAndMakesItActive()
    {
        var entry = _library.Add(Profile(centerX: 0.11f), Profile(centerX: -0.07f));

        Assert.AreEqual(1, _library.Entries.Count);
        Assert.AreEqual(entry.Id, _library.ActiveId);
        Assert.AreEqual(0.11f, _library.Active!.Left.GazeCenterX, 1e-6);
    }

    [TestMethod]
    public void ActivatingWritesTheProfilesThePipelineReads()
    {
        // The whole point: the dropdown and the running pipeline cannot disagree.
        var entry = _library.Add(Profile(closed: 0.08f), Profile(closed: 0.09f));

        Assert.AreEqual(0.08f, ActiveLeft().OpennessClosed, 1e-6);

        _library.Activate(EyeCalibrationLibrary.DefaultId);
        Assert.AreEqual(EyeCalibrationProfile.Default, ActiveLeft());

        _library.Activate(entry.Id);
        Assert.AreEqual(0.08f, ActiveLeft().OpennessClosed, 1e-6);
    }

    [TestMethod]
    public void SavingASecondCaptureDoesNotDestroyTheFirst()
    {
        var first = _library.Add(Profile(centerX: 0.10f), Profile());
        Thread.Sleep(5); // ids are millisecond timestamps
        var second = _library.Add(Profile(centerX: 0.20f), Profile());

        Assert.AreEqual(2, _library.Entries.Count);
        Assert.AreEqual(second.Id, _library.ActiveId, "the newest capture becomes active");

        _library.Activate(first.Id);
        Assert.AreEqual(0.10f, ActiveLeft().GazeCenterX, 1e-6,
            "the earlier calibration must still be selectable");
    }

    [TestMethod]
    public void EntriesAreNewestFirst()
    {
        _library.Add(Profile(centerX: 0.10f), Profile());
        Thread.Sleep(5);
        _library.Add(Profile(centerX: 0.20f), Profile());

        Assert.AreEqual(0.20f, _library.Entries[0].Left.GazeCenterX, 1e-6);
    }

    [TestMethod]
    public void DeletingAnInactiveEntryLeavesTheActiveOneAlone()
    {
        var first = _library.Add(Profile(centerX: 0.10f), Profile());
        Thread.Sleep(5);
        var second = _library.Add(Profile(centerX: 0.20f), Profile());

        _library.Delete(first.Id);

        Assert.AreEqual(1, _library.Entries.Count);
        Assert.AreEqual(second.Id, _library.ActiveId);
        Assert.AreEqual(0.20f, ActiveLeft().GazeCenterX, 1e-6);
    }

    [TestMethod]
    public void DeletingTheActiveEntryFallsBackToUncalibrated()
    {
        _library.Add(Profile(centerX: 0.10f), Profile());
        Thread.Sleep(5);
        var second = _library.Add(Profile(centerX: 0.20f), Profile());

        _library.Delete(second.Id);

        // Deliberately not promoting the neighbour: which calibration is running should never
        // change by surprise.
        Assert.AreEqual(EyeCalibrationLibrary.DefaultId, _library.ActiveId);
        Assert.AreEqual(EyeCalibrationProfile.Default, ActiveLeft());
        Assert.AreEqual(1, _library.Entries.Count, "the other capture is still there to pick");
    }

    [TestMethod]
    public void ActivatingAnUnknownIdMeansUncalibrated()
    {
        _library.Add(Profile(closed: 0.08f), Profile());

        _library.Activate("no-such-entry");

        Assert.AreEqual(EyeCalibrationLibrary.DefaultId, _library.ActiveId);
        Assert.AreEqual(EyeCalibrationProfile.Default, ActiveLeft());
    }

    [TestMethod]
    public void TheLibrarySurvivesARestart()
    {
        var entry = _library.Add(Profile(centerX: 0.13f), Profile(centerX: -0.04f));
        _settings.ForceSave();

        var reloaded = new EyeCalibrationLibrary(NewSettings());

        Assert.AreEqual(1, reloaded.Entries.Count);
        Assert.AreEqual(entry.Id, reloaded.ActiveId);
        Assert.AreEqual(0.13f, reloaded.Active!.Left.GazeCenterX, 1e-6);
        Assert.AreEqual(-0.04f, reloaded.Active.Right.GazeCenterX, 1e-6);
    }

    [TestMethod]
    public void AnExistingCalibrationFromBeforeTheLibraryIsAdopted()
    {
        // Upgrading must not appear to lose the calibration already in use.
        _settings.SaveSetting(EyeCalibrationSettings.LeftKey, Profile(closed: 0.07f));
        _settings.SaveSetting(EyeCalibrationSettings.RightKey, Profile(closed: 0.06f));

        _library.AdoptExistingCalibrationIfUnseen();

        Assert.AreEqual(1, _library.Entries.Count);
        Assert.AreEqual(0.07f, _library.Active!.Left.OpennessClosed, 1e-6);
    }

    [TestMethod]
    public void AdoptionDoesNothingWhenThereIsNothingToAdopt()
    {
        _library.AdoptExistingCalibrationIfUnseen();
        Assert.AreEqual(0, _library.Entries.Count);

        // A stored-but-default calibration is not a capture worth keeping.
        _settings.SaveSetting(EyeCalibrationSettings.LeftKey, EyeCalibrationProfile.Default);
        _settings.SaveSetting(EyeCalibrationSettings.RightKey, EyeCalibrationProfile.Default);
        _library.AdoptExistingCalibrationIfUnseen();

        Assert.AreEqual(0, _library.Entries.Count);
    }

    [TestMethod]
    public void AdoptionNeverRunsTwice()
    {
        _library.Add(Profile(centerX: 0.10f), Profile());
        _settings.SaveSetting(EyeCalibrationSettings.LeftKey, Profile(centerX: 0.99f));

        _library.AdoptExistingCalibrationIfUnseen();

        Assert.AreEqual(1, _library.Entries.Count,
            "an existing library means the slot is already accounted for");
    }
}
