using System;
using System.IO;
using System.Linq;
using System.Text.Json;
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
    public void TrainingInventory_DistinguishesCumulativeDiscoveryOptimizationAndHoldout()
    {
        CreateInventorySession("20260813_100000_neutral", SessionType.Neutral, 2);
        CreateInventorySession("20260813_110000_neutral", SessionType.Neutral, 3);
        CreateInventorySession("20260813_120000_neutral", SessionType.Neutral, 4);
        CreateInventorySession("20260813_130000_speech", SessionType.Speech, 5);
        CreateInventorySession("20260813_140000_speech", SessionType.Speech, 6);
        CreateInventorySession("20260813_150000_guided", SessionType.Guided, 7,
            guidedDims: [PersonalizationSchema.IndexOf("JawOpen")]);
        CreateInventorySession("20260813_160000_guided", SessionType.Guided, 8,
            guidedDims: [PersonalizationSchema.IndexOf("MouthSmileLeft")]);
        CreateInventorySession("20260813_170000_correction", SessionType.Correction, 9,
            correctedDims: [PersonalizationSchema.IndexOf("JawOpen")]);

        var inventory = PersonalizationEnvironment.InspectTrainingInventory(_root);

        Assert.AreEqual(8, inventory.Discovered.Count);
        Assert.AreEqual(44, inventory.DiscoveredFrames);
        Assert.AreEqual(5, inventory.Training.Count);
        Assert.AreEqual(26, inventory.TrainingFrames);
        Assert.AreEqual(3, inventory.HeldOut.Count);
        Assert.AreEqual(18, inventory.HeldOutFrames);
        Assert.AreEqual("20260813_160000_guided", inventory.LatestGuidedSessionId);
        Assert.IsFalse(inventory.LatestGuidedIsTraining,
            "The newest of two Guided sessions is validation-only under the trainer's default split.");
        CollectionAssert.AreEqual(new[] { "JawOpen" }, inventory.GuidedTrainingExpressions.ToArray(),
            "Coverage must describe optimization, not a rich Guided pass that is only held out.");
        Assert.AreEqual(1, inventory.CorrectionCounts["JawOpen"]);

        var correction = inventory.Training.Single(x => x.Type == SessionType.Correction);
        CollectionAssert.AreEqual(
            new[] { PersonalizationSchema.IndexOf("JawOpen") }, correction.CorrectedDims.ToArray());
    }

    [TestMethod]
    public void TrainingInventory_SkipsStructurallyInvalidJsonLinesWithoutCrashingCleanupUi()
    {
        var session = CreateInventorySession(
            "20260813_100000_guided",
            SessionType.Guided,
            2,
            guidedDims: [PersonalizationSchema.IndexOf("JawOpen")]);
        File.WriteAllLines(Path.Combine(session, "labels.jsonl"),
        [
            // The image for frame zero is real, but cue dimensions are the wrong JSON type.
            "{\"i\":0,\"cue\":{\"phase\":\"hold\",\"dims\":[\"bad\"],\"target\":[1]}}",
            // Syntactically valid JSON that previously threw InvalidOperationException in GetInt32.
            "{\"i\":\"not-an-integer\",\"cue\":{\"phase\":\"hold\",\"dims\":{},\"target\":[]}}",
        ]);

        var inventory = PersonalizationEnvironment.InspectTrainingInventory(_root);

        Assert.AreEqual(1, inventory.Discovered.Count);
        Assert.AreEqual(1, inventory.DiscoveredFrames,
            "The valid frame remains visible while the malformed index is skipped.");
        Assert.AreEqual(0, inventory.GuidedTrainingExpressions.Count,
            "Malformed cue dimensions must not be invented into supervision coverage.");
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

    [TestMethod]
    public void ModelCEmbeddingCoverage_RequiresEveryFrameAndTheCurrentRunnerHash()
    {
        var first = CreateSession("20260813_100000_neutral", frames: 2);
        var second = CreateSession("20260813_110000_speech", frames: 3);

        WriteEmbeddingSidecar(first, frames: 2, modelMd5: "current");
        var partial = PersonalizationEnvironment.InspectEmbeddingDataset(_root, "current");
        Assert.AreEqual(2, partial.Sessions);
        Assert.AreEqual(1, partial.ReadySessions);
        Assert.IsFalse(partial.Ready, "C must stay disabled while even one recording lacks features.");

        WriteEmbeddingSidecar(second, frames: 3, modelMd5: "old");
        Assert.IsFalse(PersonalizationEnvironment.InspectEmbeddingDataset(_root, "current").Ready,
            "Features from a different derived face model must not be mixed into C training.");

        WriteEmbeddingSidecar(second, frames: 3, modelMd5: "current");
        Assert.IsTrue(PersonalizationEnvironment.InspectEmbeddingDataset(_root, "current").Ready);
    }

    private static void WriteEmbeddingSidecar(string session, int frames, string modelMd5)
    {
        File.WriteAllBytes(Path.Combine(session, "embeddings.bin"), new byte[frames * 1280 * 2]);
        File.WriteAllText(Path.Combine(session, "embeddings.json"), JsonSerializer.Serialize(new
        {
            dim = 1280,
            dtype = "float16",
            count = frames,
            model_md5 = modelMd5
        }));
    }

    private string CreateInventorySession(
        string id,
        SessionType type,
        int frames,
        int[]? guidedDims = null,
        int[]? correctedDims = null)
    {
        var path = Path.Combine(_root, id);
        var framesPath = Path.Combine(path, "frames");
        Directory.CreateDirectory(framesPath);
        File.WriteAllText(Path.Combine(path, "session.json"), JsonSerializer.Serialize(new
        {
            SessionId = id,
            SessionType = type.ToString(),
            FrameCount = frames,
        }));

        var target = new float[PersonalizationSchema.ExpressionCount];
        foreach (var dim in guidedDims ?? []) target[dim] = 1f;
        var lines = new string[frames];
        for (var i = 0; i < frames; i++)
        {
            File.WriteAllBytes(Path.Combine(framesPath, $"{i:D6}.jpg"), [0xFF, 0xD8, 0xFF]);
            lines[i] = JsonSerializer.Serialize(new
            {
                i,
                cue = guidedDims == null ? null : new
                {
                    phase = "hold",
                    dims = guidedDims,
                    target,
                }
            });
        }
        File.WriteAllLines(Path.Combine(path, "labels.jsonl"), lines);

        if (correctedDims != null)
        {
            File.WriteAllText(Path.Combine(path, "correction.json"), JsonSerializer.Serialize(new
            {
                CorrectedDims = correctedDims,
                Target = 0f,
            }));
        }

        return path;
    }
}
