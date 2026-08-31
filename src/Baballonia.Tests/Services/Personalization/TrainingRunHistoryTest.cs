using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Baballonia.Contracts;
using Baballonia.Services.Inference;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace Baballonia.Tests.Services.Personalization;

[TestClass]
public sealed class TrainingRunHistoryTest
{
    private string _root = null!;
    private string _runs = null!;
    private string _dataset = null!;

    private static string AssetPath(string name) =>
        Path.Combine(AppContext.BaseDirectory, "Assets", "PersonalModels", name);

    [TestInitialize]
    public void Initialize()
    {
        _root = Path.Combine(Path.GetTempPath(), "BaballoniaRunHistoryTests", Guid.NewGuid().ToString("N"));
        _runs = Path.Combine(_root, "runs");
        _dataset = Path.Combine(_root, "dataset");
        Directory.CreateDirectory(_runs);
        Directory.CreateDirectory(_dataset);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [TestMethod]
    public void Discover_ReturnsNewestFirstWithExactSplitProvenance()
    {
        WriteSession("neutral-old", "Neutral", 100);
        WriteSession("guided-new", "Guided", 250);
        WriteSession("correction-one", "Correction", 40);

        WriteRun(
            "20260814_100000_image_residual_v1",
            "image_residual_v1",
            "imageAdapter.onnx",
            ["neutral-old"],
            ["guided-new"],
            verdict: "unclear",
            improved: 10,
            regressed: 2);
        var newestDirectory = WriteRun(
            "20260814_110000_image_residual_v1",
            "image_residual_v1",
            "imageAdapter.onnx",
            ["guided-new", "correction-one"],
            ["neutral-old"],
            verdict: "better",
            improved: 42,
            regressed: 3);

        var history = TrainingRunHistory.Discover(_runs, _dataset);

        Assert.AreEqual(2, history.Count);
        var newest = history[0];
        Assert.AreEqual("20260814_110000_image_residual_v1", newest.RunId);
        Assert.AreEqual("B", newest.AdapterFamily);
        Assert.AreEqual("image_residual_v1", newest.AdapterType);
        Assert.AreEqual(
            Path.GetFullPath(Path.Combine(newestDirectory, TrainingRunHistory.ExportedModelFileName)),
            newest.ModelPath);
        Assert.IsNotNull(newest.TrainedUtc, "The exported model's exact trained_utc should be retained.");

        CollectionAssert.AreEqual(
            new[] { "guided-new", "correction-one" }, newest.Train.SessionIds.ToArray());
        Assert.AreEqual(2, newest.Train.SessionCount);
        Assert.AreEqual(2, newest.Train.ResolvedSessionCount);
        Assert.AreEqual(290, newest.Train.FrameCount);
        Assert.AreEqual(true, newest.Train.IncludesGuided);
        Assert.AreEqual(true, newest.Train.IncludesCorrection);

        CollectionAssert.AreEqual(new[] { "neutral-old" }, newest.HeldOut.SessionIds.ToArray());
        Assert.AreEqual(1, newest.HeldOut.SessionCount);
        Assert.AreEqual(100, newest.HeldOut.FrameCount);
        Assert.AreEqual(false, newest.HeldOut.IncludesGuided);
        Assert.AreEqual(false, newest.HeldOut.IncludesCorrection);
        Assert.AreEqual("better", newest.Verdict);
        Assert.AreEqual(42, newest.ExpressionsImproved);
        Assert.AreEqual(3, newest.ExpressionsRegressed);
        Assert.AreEqual(0.12, newest.MeanStockMae!.Value, 1e-9);
        Assert.AreEqual(0.04, newest.MeanPersonalMae!.Value, 1e-9);
    }

    [TestMethod]
    public void Discover_IgnoresIncompleteAndMalformedRunsWithoutHidingValidOnes()
    {
        WriteSession("neutral", "Neutral", 25);
        WriteRun(
            "20260814_100000_output_mlp_v1",
            "output_mlp_v1",
            "validAdapter.onnx",
            ["neutral"],
            []);

        var incomplete = Directory.CreateDirectory(
            Path.Combine(_runs, "20260814_110000_image_residual_v1")).FullName;
        File.Copy(AssetPath("imageAdapter.onnx"),
            Path.Combine(incomplete, TrainingRunHistory.ExportedModelFileName));
        File.WriteAllText(Path.Combine(incomplete, TrainingRunHistory.RunManifestFileName), "{}");

        var badManifest = WriteRun(
            "20260814_120000_image_residual_v1",
            "image_residual_v1",
            "imageAdapter.onnx",
            ["neutral"],
            []);
        File.WriteAllText(Path.Combine(badManifest, TrainingRunHistory.RunManifestFileName), "not json");

        var badSummary = WriteRun(
            "20260814_130000_image_residual_v1",
            "image_residual_v1",
            "imageAdapter.onnx",
            ["neutral"],
            []);
        File.WriteAllText(Path.Combine(badSummary, TrainingRunHistory.SummaryFileName), "[]");

        var badOnnx = WriteRun(
            "20260814_140000_image_residual_v1",
            "image_residual_v1",
            "imageAdapter.onnx",
            ["neutral"],
            []);
        File.WriteAllText(Path.Combine(badOnnx, TrainingRunHistory.ExportedModelFileName), "not an onnx model");

        var history = TrainingRunHistory.Discover(_runs, _dataset);

        Assert.AreEqual(1, history.Count);
        Assert.AreEqual("20260814_100000_output_mlp_v1", history[0].RunId);
    }

    [TestMethod]
    public void FrozenInventory_PreservesFactsAndSurfacesDeletedSourceSession()
    {
        WriteSession("guided-source", "Guided", 250);
        WriteRun(
            "20260814_100000_image_residual_v1",
            "image_residual_v1",
            "imageAdapter.onnx",
            ["guided-source"],
            [],
            includeFrozenInventory: true);

        Directory.Delete(Path.Combine(_dataset, "guided-source"), recursive: true);

        var run = TrainingRunHistory.Discover(_runs, _dataset).Single();

        Assert.AreEqual(250, run.Train.FrameCount,
            "Deleting user-owned recordings must not rewrite the corpus used by an old model.");
        Assert.AreEqual(true, run.Train.IncludesGuided);
        Assert.AreEqual(0, run.Train.ResolvedSessionCount);
        CollectionAssert.AreEqual(new[] { "guided-source" }, run.Train.MissingSessionIds.ToArray(),
            "The UI still needs to disclose that the frozen source is no longer on disk.");
    }

    [TestMethod]
    public void LegacyMissingSession_IsUnknownRatherThanInventedAsZeroAndNotGuided()
    {
        WriteRun(
            "20260814_100000_output_mlp_v1",
            "output_mlp_v1",
            "validAdapter.onnx",
            ["deleted-old-session"],
            []);

        var run = TrainingRunHistory.Discover(_runs, _dataset).Single();

        Assert.IsNull(run.Train.FrameCount);
        Assert.IsNull(run.Train.IncludesGuided);
        Assert.IsNull(run.Train.IncludesCorrection);
        CollectionAssert.AreEqual(
            new[] { "deleted-old-session" }, run.Train.MissingSessionIds.ToArray());
    }

    [TestMethod]
    public void ManagerCatalog_KeepsMultipleBRunsAndFiltersIncompatibleHistory()
    {
        WriteSession("neutral", "Neutral", 25);
        foreach (var id in new[]
                 {
                     "20260814_100000_image_residual_v1",
                     "20260814_110000_image_residual_v1",
                     "20260814_120000_image_residual_v1"
                 })
        {
            WriteRun(id, "image_residual_v1", "imageAdapter.onnx", ["neutral"], []);
        }

        // This is a well-formed historical artifact, but its schema is intentionally incompatible
        // with the running build. History remains honest; the selectable runtime catalog must omit it.
        WriteRun(
            "20260814_130000_output_mlp_v1",
            "output_mlp_v1",
            "schemaMismatchAdapter.onnx",
            ["neutral"],
            []);

        using var manager = BuildManager();
        var historical = manager.DiscoverAvailableModels(_runs, _dataset)
            .Where(model => model.Source == PersonalModelArtifactSource.TrainingRun)
            .ToArray();

        Assert.AreEqual(3, historical.Length,
            "Three compatible B exports must remain three independently selectable artifacts.");
        Assert.IsTrue(historical.All(model => model.Kind == "b"));
        Assert.AreEqual(3, historical.Select(model => model.Path)
            .Distinct(StringComparer.OrdinalIgnoreCase).Count());
        CollectionAssert.AreEqual(
            new[]
            {
                "20260814_120000_image_residual_v1",
                "20260814_110000_image_residual_v1",
                "20260814_100000_image_residual_v1"
            },
            historical.Select(model => model.TrainingRun!.RunId).ToArray());
        Assert.IsTrue(historical.All(model => model.Label.Contains(model.TrainingRun!.RunId)));
    }

    private void WriteSession(string id, string type, int frames)
    {
        var directory = Directory.CreateDirectory(Path.Combine(_dataset, id)).FullName;
        File.WriteAllText(Path.Combine(directory, "session.json"), JsonSerializer.Serialize(new
        {
            SessionId = id,
            SessionType = type,
            FrameCount = frames
        }));
    }

    private string WriteRun(
        string id,
        string adapterType,
        string modelAsset,
        string[] trainSessions,
        string[] heldOutSessions,
        string verdict = "better",
        int improved = 5,
        int regressed = 1,
        bool includeFrozenInventory = false)
    {
        var directory = Directory.CreateDirectory(Path.Combine(_runs, id)).FullName;
        var manifest = new Dictionary<string, object>
        {
            ["adapter_type"] = adapterType,
            ["train_sessions"] = trainSessions,
            ["val_sessions"] = heldOutSessions,
        };
        if (includeFrozenInventory)
        {
            object[] Inventory(IEnumerable<string> sessionIds) => sessionIds.Select(sessionId =>
            {
                using var document = JsonDocument.Parse(File.ReadAllText(
                    Path.Combine(_dataset, sessionId, "session.json")));
                var root = document.RootElement;
                return (object)new
                {
                    session_id = sessionId,
                    session_type = root.GetProperty("SessionType").GetString(),
                    frame_count = root.GetProperty("FrameCount").GetInt32(),
                };
            }).ToArray();

            manifest["train_inventory"] = Inventory(trainSessions);
            manifest["val_inventory"] = Inventory(heldOutSessions);
        }

        File.WriteAllText(
            Path.Combine(directory, TrainingRunHistory.RunManifestFileName),
            JsonSerializer.Serialize(manifest));
        File.WriteAllText(
            Path.Combine(directory, TrainingRunHistory.SummaryFileName),
            JsonSerializer.Serialize(new
            {
                summary_version = 2,
                adapter_type = adapterType,
                validated_on_held_out_sessions = heldOutSessions.Length > 0,
                expressions_improved = improved,
                expressions_regressed = regressed,
                mean_stock_mae = 0.12,
                mean_personal_mae = 0.04,
                verdict,
                worst_regressions = Array.Empty<object>()
            }));
        File.Copy(
            AssetPath(modelAsset),
            Path.Combine(directory, TrainingRunHistory.ExportedModelFileName),
            overwrite: true);
        return directory;
    }

    private static PersonalModelManager BuildManager()
    {
        var pipelineManager = (FacePipelineManager)RuntimeHelpers
            .GetUninitializedObject(typeof(FacePipelineManager));
        var pipeline = new FaceProcessingPipeline(new Baballonia.Services.FacePipelineEventBus());
        typeof(FacePipelineManager)
            .GetField("_pipeline", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(pipelineManager, pipeline);

        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<string>(
                PersonalModelManager.PathSetting, It.IsAny<string>(), It.IsAny<bool>()))
            .Returns("");
        settings.Setup(s => s.ReadSetting<bool>(
                PersonalModelManager.EnabledSetting, It.IsAny<bool>(), It.IsAny<bool>()))
            .Returns(false);
        settings.Setup(s => s.ReadSetting<float>(
                PersonalModelManager.BlendSetting, It.IsAny<float>(), It.IsAny<bool>()))
            .Returns(1f);

        return new PersonalModelManager(
            pipelineManager,
            settings.Object,
            NullLogger<PersonalModelManager>.Instance);
    }
}
