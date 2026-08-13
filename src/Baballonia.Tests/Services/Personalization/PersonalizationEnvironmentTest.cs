using System;
using System.IO;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Covers the detection that drives the Setup checklist and the recording counts.
///
/// These are the answers the one-button workflow depends on: whether training can even start, and
/// what the user should record next. Getting them wrong means either a button that fails when
/// pressed, or advice to record data that is not needed.
/// </summary>
[TestClass]
[TestSubject(typeof(PersonalizationEnvironment))]
public class PersonalizationEnvironmentTest
{
    private string _root = null!;

    [TestInitialize]
    public void Initialize()
    {
        _root = Path.Combine(Path.GetTempPath(), $"babble-env-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_root);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    /// <summary>Creates a session folder shaped like the recorder's output.</summary>
    private string CreateSession(string id, int frames = 3)
    {
        var path = Path.Combine(_root, id);
        Directory.CreateDirectory(Path.Combine(path, "frames"));
        File.WriteAllText(Path.Combine(path, "session.json"), "{}");

        for (var i = 0; i < frames; i++)
            File.WriteAllBytes(Path.Combine(path, "frames", $"{i:D6}.jpg"), [0xFF, 0xD8, 0xFF]);

        return path;
    }

    [TestMethod]
    public void MissingDatasetRoot_ReportsEmptyRatherThanThrowing()
    {
        var status = PersonalizationEnvironment.InspectDataset(
            Path.Combine(_root, "does-not-exist"));

        Assert.AreEqual(0, status.TotalSessions);
        Assert.IsFalse(status.CanTrain);
    }

    [TestMethod]
    public void SessionsAreCountedByType()
    {
        CreateSession("20260813_100000_neutral");
        CreateSession("20260813_110000_neutral");
        CreateSession("20260813_120000_speech", frames: 5);
        CreateSession("20260813_130000_guided");

        var status = PersonalizationEnvironment.InspectDataset(_root);

        Assert.AreEqual(2, status.NeutralSessions);
        Assert.AreEqual(1, status.SpeechSessions);
        Assert.AreEqual(1, status.GuidedSessions);
        Assert.AreEqual(4, status.TotalSessions);
        Assert.AreEqual(14, status.TotalFrames);
    }

    [TestMethod]
    public void FoldersWithoutMetadataAreIgnored()
    {
        CreateSession("20260813_100000_neutral");
        Directory.CreateDirectory(Path.Combine(_root, "20260813_110000_neutral")); // no session.json
        Directory.CreateDirectory(Path.Combine(_root, "random-folder"));

        var status = PersonalizationEnvironment.InspectDataset(_root);

        Assert.AreEqual(1, status.NeutralSessions,
            "A half-written or unrelated folder must not be counted as a recording.");
    }

    [TestMethod]
    public void CanTrain_RequiresANeutralSessionAndASecondSession()
    {
        Assert.IsFalse(PersonalizationEnvironment.InspectDataset(_root).CanTrain);

        CreateSession("20260813_100000_speech");
        Assert.IsFalse(PersonalizationEnvironment.InspectDataset(_root).CanTrain,
            "Speech alone supervises nothing on its own.");

        CreateSession("20260813_110000_neutral");
        Assert.IsTrue(PersonalizationEnvironment.InspectDataset(_root).CanTrain);
    }

    [TestMethod]
    public void Advice_AsksForNeutralFirstThenSpeechThenStops()
    {
        Assert.IsTrue(PersonalizationEnvironment.InspectDataset(_root)
            .NextRecommendation!.Contains("Neutral"));

        CreateSession("20260813_100000_neutral");
        CreateSession("20260813_110000_neutral");
        Assert.IsTrue(PersonalizationEnvironment.InspectDataset(_root)
            .NextRecommendation!.Contains("Speech"));

        CreateSession("20260813_120000_speech");
        CreateSession("20260813_130000_speech");

        var complete = PersonalizationEnvironment.InspectDataset(_root);
        Assert.IsNull(complete.NextRecommendation, "Nothing more should be requested once satisfied.");
        Assert.IsTrue(complete.IsRecommended);
    }

    [TestMethod]
    public void Advice_UsesSingularAndPluralCorrectly()
    {
        CreateSession("20260813_100000_neutral");

        var advice = PersonalizationEnvironment.InspectDataset(_root).NextRecommendation!;

        StringAssert.Contains(advice, "1 more Neutral recording ");
        Assert.IsFalse(advice.Contains("recordings"), $"Should be singular for one: '{advice}'");
    }

    [TestMethod]
    public void TrainingEnvironment_IsNotReadyWhenTheFolderIsMissing()
    {
        Assert.IsFalse(PersonalizationEnvironment.IsTrainingEnvironmentReady(
            Path.Combine(_root, "no-venv-here")));
    }

    [TestMethod]
    public void TrainingEnvironment_IsNotReadyWithoutTorchEvenIfPythonExists()
    {
        var venv = Path.Combine(_root, "venv");
        var scripts = Path.Combine(venv, "Scripts");
        Directory.CreateDirectory(scripts);
        Directory.CreateDirectory(Path.Combine(venv, "Lib", "site-packages"));
        File.WriteAllText(Path.Combine(scripts, "python.exe"), "");

        Assert.IsFalse(PersonalizationEnvironment.IsTrainingEnvironmentReady(venv),
            "A venv without torch cannot train; reporting it ready would produce a failing button.");
    }

    [TestMethod]
    public void TrainingEnvironment_IsReadyWhenTorchIsPresent()
    {
        var venv = Path.Combine(_root, "venv");
        var scripts = Path.Combine(venv, "Scripts");
        Directory.CreateDirectory(scripts);
        Directory.CreateDirectory(Path.Combine(venv, "Lib", "site-packages", "torch"));
        File.WriteAllText(Path.Combine(scripts, "python.exe"), "");

        // Only meaningful on Windows, where VenvPython resolves to Scripts/python.exe.
        if (OperatingSystem.IsWindows())
            Assert.IsTrue(PersonalizationEnvironment.IsTrainingEnvironmentReady(venv));
    }

    [TestMethod]
    public void FindTrainingRoot_LocatesThePackageByWalkingUp()
    {
        var nested = Path.Combine(_root, "src", "App", "bin", "Debug", "net10.0");
        Directory.CreateDirectory(nested);
        Directory.CreateDirectory(Path.Combine(_root, "training", "babble_personal"));

        Assert.AreEqual(
            Path.Combine(_root, "training"),
            PersonalizationEnvironment.FindTrainingRoot(nested));
    }

    [TestMethod]
    public void FindTrainingRoot_ReturnsNullWhenAbsent()
    {
        var nested = Path.Combine(_root, "a", "b", "c");
        Directory.CreateDirectory(nested);

        Assert.IsNull(PersonalizationEnvironment.FindTrainingRoot(nested),
            "Reporting a wrong path would produce a confusing failure later; null is honest.");
    }

    [TestMethod]
    public void RunsAndVenvDirectories_StayOutsideTheRepository()
    {
        var repo = AppContext.BaseDirectory;

        StringAssert.Contains(PersonalizationEnvironment.TrainingRunsDirectory, "ProjectBabble");
        Assert.IsFalse(PersonalizationEnvironment.DefaultVenvDirectory.StartsWith(repo, StringComparison.OrdinalIgnoreCase),
            "The venv must not live inside the repo, which is OneDrive-synced here.");
    }
}
