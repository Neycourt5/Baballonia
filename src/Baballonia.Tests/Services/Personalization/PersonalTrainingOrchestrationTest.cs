using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services.Inference;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Drives the whole one-button flow - train, export, install, load - against the real Python
/// tooling, using synthetic recordings.
///
/// This is the test that proves the button works. Every part is unit-tested elsewhere, but only
/// running the real chain catches the things that actually break it: a wrong working directory, an
/// argument the trainer rejects, or the installed model being locked by the session still holding
/// it open.
///
/// Skips (rather than fails) when the training environment is absent, so a fresh clone or CI box
/// without PyTorch does not report a false failure.
/// </summary>
[TestClass]
[TestSubject(typeof(PersonalTrainingService))]
public class PersonalTrainingOrchestrationTest
{
    private const int N = PersonalizationSchema.ExpressionCount;
    private static readonly int JawOpen = PersonalizationSchema.IndexOf("JawOpen");

    private readonly List<string> _createdSessions = [];
    private string? _modelBackup;
    private string? _previousBackup;
    private HashSet<string> _runsBefore = [];
    private readonly Dictionary<string, string?> _slotBackups = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string?> _slotPreviousBackups = new(StringComparer.OrdinalIgnoreCase);

    [TestInitialize]
    public void Initialize()
    {
        // Remember pre-existing training runs so cleanup only removes the ones this test caused.
        var runs = PersonalizationEnvironment.TrainingRunsDirectory;
        _runsBefore = Directory.Exists(runs)
            ? Directory.EnumerateDirectories(runs).ToHashSet(StringComparer.OrdinalIgnoreCase)
            : [];

        // Take the user's installed personal model out of harm's way FIRST, and record that it
        // existed. Cleanup uses _modelBackup to decide between "restore what was here" and "delete
        // the file this test installed" - so failing to set it here silently turns cleanup into a
        // deletion of a real, expensively-trained model. That is exactly what used to happen.
        // These tests run against the real data directories, so this must hold even when the test
        // body skips: [TestCleanup] runs after Assert.Inconclusive too.
        _modelBackup = null;
        var installed = PersonalizationPaths.DefaultPersonalModelPath;
        if (File.Exists(installed))
        {
            _modelBackup = Path.Combine(Path.GetTempPath(),
                $"babble-personal-model-backup-{Guid.NewGuid():N}.onnx");
            File.Copy(installed, _modelBackup, overwrite: true);
        }

        _previousBackup = null;
        var previousRollback = Path.ChangeExtension(installed, ".previous.onnx");
        if (File.Exists(previousRollback))
        {
            _previousBackup = Path.Combine(Path.GetTempPath(),
                $"babble-personal-previous-backup-{Guid.NewGuid():N}.onnx");
            File.Copy(previousRollback, _previousBackup, overwrite: true);
        }

        _slotBackups.Clear();
        _slotPreviousBackups.Clear();
        foreach (var kind in new[] { "a", "b", "c" })
        {
            var slot = PersonalizationPaths.PersonalModelPath(kind);
            string? backup = null;
            if (File.Exists(slot))
            {
                backup = Path.Combine(Path.GetTempPath(),
                    $"babble-personal-{kind}-backup-{Guid.NewGuid():N}.onnx");
                File.Copy(slot, backup, overwrite: true);
            }
            _slotBackups[slot] = backup;

            var previous = Path.ChangeExtension(slot, ".previous.onnx");
            string? previousBackup = null;
            if (File.Exists(previous))
            {
                previousBackup = Path.Combine(Path.GetTempPath(),
                    $"babble-personal-{kind}-previous-backup-{Guid.NewGuid():N}.onnx");
                File.Copy(previous, previousBackup, overwrite: true);
            }
            _slotPreviousBackups[slot] = previousBackup;
        }
    }

    /// <summary>
    /// Restores the machine to its prior state. These tests write to the real data directories, so
    /// leaving anything behind would corrupt the user's next real training run.
    /// </summary>
    [TestCleanup]
    public void Cleanup()
    {
        foreach (var directory in _createdSessions.Where(Directory.Exists))
        {
            try { Directory.Delete(directory, recursive: true); } catch { /* best effort */ }
        }

        var model = PersonalizationPaths.DefaultPersonalModelPath;
        try
        {
            if (_modelBackup != null)
            {
                File.Copy(_modelBackup, model, overwrite: true);
                File.Delete(_modelBackup);
            }
            else if (File.Exists(model))
            {
                File.Delete(model);
            }

            // Installing keeps one rollback copy. Restore the user's if there was one; otherwise
            // whatever is there now is this test's residue.
            var previous = Path.ChangeExtension(model, ".previous.onnx");
            if (_previousBackup != null)
            {
                File.Copy(_previousBackup, previous, overwrite: true);
                File.Delete(_previousBackup);
            }
            else if (File.Exists(previous))
            {
                File.Delete(previous);
            }
        }
        catch { /* best effort */ }

        foreach (var (slot, backup) in _slotBackups)
        {
            try
            {
                if (backup != null)
                {
                    File.Copy(backup, slot, overwrite: true);
                    File.Delete(backup);
                }
                else if (File.Exists(slot))
                {
                    File.Delete(slot);
                }
                var previous = Path.ChangeExtension(slot, ".previous.onnx");
                var previousBackup = _slotPreviousBackups.GetValueOrDefault(slot);
                if (previousBackup != null)
                {
                    File.Copy(previousBackup, previous, overwrite: true);
                    File.Delete(previousBackup);
                }
                else if (File.Exists(previous))
                {
                    File.Delete(previous);
                }
            }
            catch { /* best effort */ }
        }

        try
        {
            var runs = PersonalizationEnvironment.TrainingRunsDirectory;
            if (Directory.Exists(runs))
            {
                foreach (var directory in Directory.EnumerateDirectories(runs)
                             .Where(d => !_runsBefore.Contains(d)))
                {
                    Directory.Delete(directory, recursive: true);
                }
            }
        }
        catch { /* best effort */ }
    }

    private static bool EnvironmentReady =>
        PersonalizationEnvironment.FindTrainingRoot() != null &&
        PersonalizationEnvironment.IsTrainingEnvironmentReady(
            PersonalizationEnvironment.DefaultVenvDirectory);

    /// <summary>Writes a session in the recorder's exact on-disk layout.</summary>
    private void CreateSyntheticSession(string suffix, string type, int frames, float jawBias)
    {
        var id = $"zz_test_{suffix}_{Guid.NewGuid():N}"[..24] + $"_{type.ToLowerInvariant()}";
        var path = PersonalizationPaths.SessionDirectory(id);
        Directory.CreateDirectory(Path.Combine(path, "frames"));
        _createdSessions.Add(path);

        var metadata = new
        {
            SchemaVersion = 1,
            SessionId = id,
            SessionType = type,
            StartedUtc = DateTime.UtcNow.ToString("o"),
            ExpressionNames = PersonalizationSchema.ExpressionNames,
            ExpressionSchemaSha256 = PersonalizationSchema.Sha256,
            ImageWidth = 224,
            ImageHeight = 224,
            JpegQuality = 95,
            FrameCount = frames,
            EffectiveFps = 30.0
        };
        File.WriteAllText(Path.Combine(path, "session.json"), JsonSerializer.Serialize(metadata));

        var random = new Random(42);
        var labels = new StringBuilder();

        for (var i = 0; i < frames; i++)
        {
            using var image = new OpenCvSharp.Mat(224, 224, OpenCvSharp.MatType.CV_8UC1,
                new OpenCvSharp.Scalar((i * 7) % 200));
            image.SaveImage(Path.Combine(path, "frames", $"{i:D6}.jpg"));

            // Stock reports jaw activity that the (resting) face is not doing - the defect the
            // adapter is expected to learn away.
            var stock = new float[N];
            for (var d = 0; d < N; d++)
                stock[d] = (float)(random.NextDouble() * 0.01);
            stock[JawOpen] = jawBias;

            labels.AppendLine(JsonSerializer.Serialize(new
            {
                i,
                t = DateTime.UtcNow.Ticks + i * 333_333,
                stock
            }));
        }

        File.WriteAllText(Path.Combine(path, "labels.jsonl"), labels.ToString());
    }

    private static (PersonalModelManager manager, FaceProcessingPipeline pipeline) BuildManager()
    {
        var pipelineManager = (FacePipelineManager)RuntimeHelpers
            .GetUninitializedObject(typeof(FacePipelineManager));

        var pipeline = new FaceProcessingPipeline(new Baballonia.Services.FacePipelineEventBus());
        typeof(FacePipelineManager)
            .GetField("_pipeline", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(pipelineManager, pipeline);

        var enabled = false;
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<bool>(PersonalModelManager.EnabledSetting, It.IsAny<bool>(), It.IsAny<bool>()))
            .Returns(() => enabled);
        settings.Setup(s => s.SaveSetting(PersonalModelManager.EnabledSetting, It.IsAny<bool>(), It.IsAny<bool>()))
            .Callback<string, bool, bool>((_, value, _) => enabled = value);
        settings.Setup(s => s.ReadSetting<string>(PersonalModelManager.PathSetting, It.IsAny<string>(), It.IsAny<bool>()))
            .Returns("");
        settings.Setup(s => s.ReadSetting<float>(PersonalModelManager.BlendSetting, It.IsAny<float>(), It.IsAny<bool>()))
            .Returns(1f);

        var manager = new PersonalModelManager(
            pipelineManager, settings.Object, NullLogger<PersonalModelManager>.Instance);

        return (manager, pipeline);
    }


    /// <summary>
    /// The cleanup in this class used to delete the user's real installed personal model, because
    /// <c>_modelBackup</c> was never populated and cleanup's "no backup" branch is a delete. It ran
    /// on every test in the class, including skipped ones, so simply running the suite destroyed a
    /// trained model that takes minutes of recording and training to reproduce.
    ///
    /// This test stands in for the user's model: it plants a file at the real install path, lets a
    /// full initialize/cleanup cycle run over it, and asserts the file survived byte for byte.
    /// </summary>
    [TestMethod]
    public void Cleanup_RestoresAnAlreadyInstalledModel_RatherThanDeletingIt()
    {
        var installed = PersonalizationPaths.DefaultPersonalModelPath;

        if (File.Exists(installed))
            Assert.Inconclusive("A real personal model is installed; not touching it to run this test.");

        Directory.CreateDirectory(Path.GetDirectoryName(installed)!);

        // Stand-in for a real trained model. Contents are arbitrary but must come back unchanged.
        var sentinel = new byte[] { 0x42, 0x41, 0x42, 0x42, 0x4C, 0x45, 0x00, 0x01, 0x02, 0x03 };
        File.WriteAllBytes(installed, sentinel);

        try
        {
            // A second initialize/cleanup cycle, exactly as another test method in this class
            // would trigger. Initialize must notice the file; Cleanup must put it back.
            Initialize();
            Cleanup();

            Assert.IsTrue(File.Exists(installed),
                "Cleanup deleted an already-installed personal model. Retraining one is expensive; " +
                "the test suite must never destroy user data.");
            CollectionAssert.AreEqual(sentinel, File.ReadAllBytes(installed),
                "The restored model does not match what was installed before the test.");
        }
        finally
        {
            if (File.Exists(installed)) File.Delete(installed);
            var previous = Path.ChangeExtension(installed, ".previous.onnx");
            if (File.Exists(previous)) File.Delete(previous);
        }
    }

    [TestMethod]
    public async Task TrainWithoutRecordings_FailsWithAdviceRatherThanAnError()
    {
        var (manager, _) = BuildManager();
        using (manager)
        {
            var service = new PersonalTrainingService(manager, NullLogger<PersonalTrainingService>.Instance);

            // Point the dataset check at reality; if the user happens to have recordings this test
            // is not meaningful, so only assert the no-data path.
            if (PersonalizationEnvironment.InspectDataset().CanTrain)
                Assert.Inconclusive("Real recordings exist; skipping the empty-dataset path.");

            var result = await service.TrainAsync("a");

            Assert.IsFalse(result.Success);
            Assert.IsTrue(
                result.Message.Contains("recordings", StringComparison.OrdinalIgnoreCase) ||
                result.Message.Contains("training environment", StringComparison.OrdinalIgnoreCase),
                $"Expected actionable advice, got: {result.Message}");
            Assert.IsFalse(result.Message.Contains("exit code"), "Raw process errors must not reach the user.");
        }
    }

    [TestMethod]
    public async Task FullPipeline_TrainsExportsInstallsAndLoads()
    {
        if (!EnvironmentReady)
            Assert.Inconclusive("Python training environment not set up; skipping the live pipeline test.");

        if (PersonalizationEnvironment.InspectDataset().TotalSessions > 0)
            Assert.Inconclusive("Real recordings are present; not mixing synthetic data into them.");

        // Preserve any existing installed model.
        var modelPath = PersonalizationPaths.DefaultPersonalModelPath;
        if (File.Exists(modelPath))
        {
            _modelBackup = modelPath + ".testbackup";
            File.Copy(modelPath, _modelBackup, overwrite: true);
        }

        CreateSyntheticSession("a", "Neutral", 40, jawBias: 0.30f);
        CreateSyntheticSession("b", "Neutral", 40, jawBias: 0.30f);

        var (manager, pipeline) = BuildManager();
        using (manager)
        {
            var service = new PersonalTrainingService(manager, NullLogger<PersonalTrainingService>.Instance);

            var stages = new List<TrainingStage>();
            var progress = new Progress<TrainingProgress>(p => stages.Add(p.Stage));

            using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(5));
            var result = await service.TrainAsync("a", progress, cancellation.Token);

            Assert.IsTrue(result.Success,
                $"Pipeline failed: {result.Message}\n--- details ---\n{service.DetailLog}");

            Assert.AreEqual(PersonalizationPaths.PersonalModelPath("a"), manager.ModelPath);
            Assert.IsTrue(File.Exists(manager.ModelPath), "The trained model should have been installed in A's slot.");
            Assert.IsNotNull(pipeline.Corrector, "The installed model should be live on the pipeline.");
            Assert.IsTrue(manager.IsActive);

            Assert.IsNotNull(result.Summary, "The results screen needs the measured summary.");
            Assert.AreEqual("output_mlp_v1", result.Summary.AdapterType);

            // Two neutral sessions means one is held out, so the verdict is allowed to be a claim.
            Assert.IsTrue(result.Summary.ValidatedOnHeldOutSessions);
            Assert.IsTrue(
                result.Summary.NeutralPersonalFalseActivation <= result.Summary.NeutralStockFalseActivation,
                "The adapter should not make resting-face activation worse.");

            CollectionAssert.Contains(stages, TrainingStage.Training, "Progress should reach Training.");
            CollectionAssert.Contains(stages, TrainingStage.Installing, "Progress should reach Installing.");
        }
    }

    [TestMethod]
    public async Task InstallingOverAnActiveModel_Succeeds()
    {
        // A loaded InferenceSession holds the file open on Windows, so retraining while a model is
        // active would fail with a sharing violation unless it is unloaded first.
        var source = Path.Combine(AppContext.BaseDirectory, "Assets", "PersonalModels", "validAdapter.onnx");
        if (!File.Exists(source))
            Assert.Inconclusive("Adapter fixture missing.");

        var modelPath = PersonalizationPaths.DefaultPersonalModelPath;
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

        if (File.Exists(modelPath))
        {
            _modelBackup = modelPath + ".testbackup";
            File.Copy(modelPath, _modelBackup, overwrite: true);
        }

        File.Copy(source, modelPath, overwrite: true);

        var (manager, pipeline) = BuildManager();
        using (manager)
        {
            manager.SetEnabled(true);
            var load = await manager.ReloadAsync();
            Assert.IsTrue(load.Success, load.Message);
            Assert.IsNotNull(pipeline.Corrector);

            var service = new PersonalTrainingService(manager, NullLogger<PersonalTrainingService>.Instance);
            var install = typeof(PersonalTrainingService)
                .GetMethod("InstallModelToPath", BindingFlags.NonPublic | BindingFlags.Instance)!;

            var error = install.Invoke(service, [source, modelPath]) as string;

            Assert.IsNull(error, $"Overwriting a loaded model should succeed, got: {error}");
            Assert.IsTrue(File.Exists(Path.ChangeExtension(modelPath, ".previous.onnx")),
                "The replaced model should be kept as a backup.");
        }
    }
}
