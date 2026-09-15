using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Inference.Models;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.C2;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace Baballonia.Tests.Services.Personalization;

[TestClass]
public class C2ModelManagerTest
{
    private string _root = null!;
    private C2CandidateCorrector? _installed;
    private string SelectionPath => Path.Combine(_root, "selection.json");

    [TestInitialize]
    public void Initialize() => _root = Path.Combine(Path.GetTempPath(), "babble-c2-keep-" + Guid.NewGuid().ToString("N"));

    private static C2PreparedCandidate Prepared(bool matches = true) => new(new C2CandidateCorrector(
        Path.Combine(AppContext.BaseDirectory, "Assets", "PersonalModels", "c2Adapter.onnx"), "fixture-context"), () => matches);

    private C2ModelManager Manager(Func<string, Task<C2PreparedCandidate>>? prepare = null) => new(
        SelectionPath, (directory, _) => (prepare ?? (_ => Task.FromResult(Prepared())))(directory),
        (candidate, _) => { var previous = _installed; _installed = candidate; return previous; },
        NullLogger<C2ModelManager>.Instance);

    [TestCleanup]
    public void Cleanup()
    {
        _installed?.Dispose();
        if (Directory.Exists(_root)) Directory.Delete(_root, true);
    }

    [TestMethod]
    public async Task KeptCandidateSurvivesPageExitAndRestoresExactSelectionAfterRestart()
    {
        var selected = Path.Combine(_root, "Candidates", "chosen-not-latest");
        using (var manager = Manager())
        {
            await manager.ActivateAsync(selected, keep: true);
            manager.EndTrial(); // The page calls this on navigation/disposal.
            Assert.IsTrue(manager.IsKept);
            Assert.IsNotNull(_installed);
            Assert.AreEqual(selected, JsonSerializer.Deserialize<C2Selection>(File.ReadAllText(SelectionPath))!.CandidateDirectory);
        }
        Assert.IsNull(_installed);
        string? restored = null;
        using var reopened = Manager(directory => { restored = directory; return Task.FromResult(Prepared()); });
        await reopened.StartAsync(CancellationToken.None);
        Assert.AreEqual(selected, restored);
        Assert.IsTrue(reopened.IsKept);
        Assert.IsFalse(reopened.IsTrial);
    }

    [TestMethod]
    public async Task TrialCanBeKeptAndReturnToCRemainsOffAfterRestart()
    {
        var selected = Path.Combine(_root, "Candidates", "one");
        using (var manager = Manager())
        {
            await manager.ActivateAsync(selected, keep: false);
            Assert.IsTrue(manager.IsTrial);
            Assert.IsFalse(File.Exists(SelectionPath));
            await manager.ActivateAsync(selected, keep: true);
            manager.ReturnToModelC();
            Assert.IsFalse(manager.IsActive);
            Assert.IsNull(_installed);
        }
        using var reopened = Manager(_ => throw new AssertFailedException("Return to C must not reload C2."));
        await reopened.StartAsync(CancellationToken.None);
        Assert.IsFalse(reopened.IsActive);
        Assert.IsNull(reopened.RestoreFailure);
    }

    [TestMethod]
    public async Task TemporaryTrialEndsOnPageExitWithoutSaving()
    {
        using var manager = Manager();
        await manager.ActivateAsync(Path.Combine(_root, "trial"), keep: false);
        manager.EndTrial();
        Assert.IsFalse(manager.IsActive);
        Assert.IsNull(_installed);
        Assert.IsFalse(File.Exists(SelectionPath));
    }

    [TestMethod]
    public async Task KeepAndStartupUseEverydayAudioPolicyWhileTrialUsesComparisonPolicy()
    {
        var modes = new List<bool>();
        C2ModelManager Create() => new(SelectionPath,
            (_, keep) => { modes.Add(keep); return Task.FromResult(Prepared()); },
            (candidate, _) => { var previous = _installed; _installed = candidate; return previous; },
            NullLogger<C2ModelManager>.Instance);
        using (var manager = Create())
        {
            await manager.ActivateAsync(Path.Combine(_root, "chosen"), keep: false);
            await manager.ActivateAsync(Path.Combine(_root, "chosen"), keep: true);
        }
        using var reopened = Create();
        await reopened.StartAsync(CancellationToken.None);
        CollectionAssert.AreEqual(new[] { false, true, true }, modes.ToArray());
    }

    [TestMethod]
    public async Task LeavingPageDuringLoadCannotActivateOrPersistLater()
    {
        var ready = new TaskCompletionSource<C2PreparedCandidate>();
        using var manager = Manager(_ => ready.Task);
        var activation = manager.ActivateAsync(Path.Combine(_root, "pending"), keep: true);
        manager.EndTrial();
        ready.SetResult(Prepared());
        await Assert.ThrowsExactlyAsync<OperationCanceledException>(async () => await activation);
        Assert.IsNull(_installed);
        Assert.IsFalse(File.Exists(SelectionPath));
    }

    [TestMethod]
    public async Task InvalidOrChangedCandidateKeepsPreviousSelectionAndRuntime()
    {
        var selected = Path.Combine(_root, "good");
        using var manager = Manager(directory => directory.EndsWith("bad")
            ? throw new InvalidDataException("Candidate changed") : Task.FromResult(Prepared()));
        await manager.ActivateAsync(selected, keep: true);
        var previous = _installed;
        await Assert.ThrowsExactlyAsync<InvalidDataException>(() => manager.ActivateAsync(Path.Combine(_root, "bad"), keep: true));
        Assert.AreSame(previous, _installed);
        Assert.AreEqual(selected, manager.CandidateDirectory);
        Assert.AreEqual(selected, JsonSerializer.Deserialize<C2Selection>(File.ReadAllText(SelectionPath))!.CandidateDirectory);
    }

    [TestMethod]
    public async Task FailedStartupUsesBaseWithoutDeletingSavedCandidateChoice()
    {
        C2Contract.WriteAtomic(SelectionPath, new C2Selection(Path.Combine(_root, "missing")));
        var before = File.ReadAllText(SelectionPath);
        using var manager = Manager(_ => throw new FileNotFoundException("Candidate missing"));
        await manager.StartAsync(CancellationToken.None);
        Assert.IsFalse(manager.IsActive);
        Assert.IsNull(_installed);
        StringAssert.Contains(manager.RestoreFailure!, "Candidate missing");
        Assert.AreEqual(before, File.ReadAllText(SelectionPath));
    }

    [TestMethod]
    public async Task ChangedLiveContextCannotBeKept()
    {
        using var manager = Manager(_ => Task.FromResult(Prepared(matches: false)));
        await Assert.ThrowsExactlyAsync<InvalidOperationException>(() => manager.ActivateAsync(Path.Combine(_root, "changed"), keep: true));
        Assert.IsNull(_installed);
        Assert.IsFalse(File.Exists(SelectionPath));
    }

    [TestMethod]
    public async Task UnwritableSelectionDoesNotReplaceRunningModel()
    {
        using var manager = Manager();
        await manager.ActivateAsync(Path.Combine(_root, "trial"), keep: false);
        var previous = _installed;
        Directory.CreateDirectory(SelectionPath); // A directory cannot be atomically replaced by the selection file.
        var failure = await Assert.ThrowsAsync<Exception>(() => manager.ActivateAsync(Path.Combine(_root, "keep"), keep: true));
        Assert.IsTrue(failure is IOException or UnauthorizedAccessException);
        Assert.AreSame(previous, _installed);
        Assert.IsTrue(manager.IsTrial);
    }

    [TestMethod]
    public void CandidateRequiresCompletedReportMatchingContextAndModelBytes()
    {
        Directory.CreateDirectory(_root);
        var modelPath = Path.Combine(_root, "candidate.onnx");
        File.WriteAllText(modelPath, "original");
        Assert.ThrowsExactly<FileNotFoundException>(() => C2RuntimeContext.ValidateCandidate(_root, "context"));
        C2Contract.WriteAtomic(Path.Combine(_root, "summary.json"), new
        {
            context_sha256 = "context", candidate_sha256 = C2Contract.Hash(modelPath)
        });
        Assert.AreEqual(modelPath, C2RuntimeContext.ValidateCandidate(_root, "context"));
        Assert.ThrowsExactly<InvalidOperationException>(() => C2RuntimeContext.ValidateCandidate(_root, "different"));
        File.AppendAllText(modelPath, "changed");
        Assert.ThrowsExactly<InvalidDataException>(() => C2RuntimeContext.ValidateCandidate(_root, "context"));
    }

    [TestMethod]
    public void ApplicationRelocationKeepsCompatibilityButModelAndSettingChangesDoNot()
    {
        var output = new C2OutputContext(Enumerable.Range(0, 45).Select(_ => new[] {0f,1f,0f,1f}).ToArray(), 1, true, .5f, 3);
        var recorded = new C2ReferenceContract("old/stock", "stock-hash", "old/features", "features-hash",
            "C.onnx", "C-hash", 1, output, "camera");
        C2RuntimeContext.ValidateContext(recorded, recorded with { StockPath = "new/stock", FeaturePath = "new/features" });
        foreach (var changed in new[]
        {
            recorded with { StockSha256 = "changed" }, recorded with { FeatureSha256 = "changed" },
            recorded with { ReferenceSha256 = "changed" }, recorded with { ReferenceBlend = .5f },
            recorded with { CameraSettings = "changed" }, recorded with { Output = output with { Beta = 4 } }
        }) Assert.ThrowsExactly<InvalidOperationException>(() => C2RuntimeContext.ValidateContext(recorded, changed));
    }

    [TestMethod]
    public void KeptCandidateAllowsUncorrectedCalibrationButComparisonAndCorrectedRangesStayExact()
    {
        var output = new C2OutputContext(Enumerable.Range(0, 45).Select(_ => new[] {0f,1f,0f,1f}).ToArray(), 1, true, .5f, .5f);
        output.Ranges[33][0] = .4f;
        var recorded = new C2ReferenceContract("stock", "stock-hash", "features", "features-hash",
            "C.onnx", "C-hash", 1, output, "camera");
        C2Contract.WriteAtomic(Path.Combine(_root, "summary.json"), new { support = new[] { "JawOpen", "MouthSmileLeft", "MouthSmileRight" } });
        var support = C2RuntimeContext.ReadSupportedChannels(_root)!;
        var changedRanges = output.Ranges.Select(r => r.ToArray()).ToArray();
        changedRanges[33][0] = .3f; // Synthetic change to an uncorrected channel.
        var current = recorded with { Output = output with { Ranges = changedRanges } };
        C2RuntimeContext.ValidateContext(recorded, current, support.Contains);
        var comparisonError = Assert.ThrowsExactly<InvalidOperationException>(() => C2RuntimeContext.ValidateContext(recorded, current));
        StringAssert.Contains(comparisonError.Message, "TongueOut");
        foreach (var channel in support)
        {
            changedRanges[channel][0] = .1f;
            var error = Assert.ThrowsExactly<InvalidOperationException>(() => C2RuntimeContext.ValidateContext(recorded, current, support.Contains));
            StringAssert.Contains(error.Message, PersonalizationSchema.ExpressionNames[channel]);
            changedRanges[channel][0] = 0;
        }
        foreach (var incompatible in new[]
        {
            current with { ReferenceSha256 = "different" }, current with { ReferenceBlend = .5f },
            current with { CameraSettings = "different" }, current with { Output = current.Output with { Beta = 2 } }
        }) Assert.ThrowsExactly<InvalidOperationException>(() => C2RuntimeContext.ValidateContext(recorded, incompatible, support.Contains));
        Assert.AreEqual(.4f, recorded.Output.Ranges[33][0], "Compatibility must not rewrite the training contract.");
    }

    [TestMethod]
    public void MissingSupportStaysStrictAndInvalidSupportIsRejected()
    {
        var path = Path.Combine(_root, "summary.json");
        C2Contract.WriteAtomic(path, new { status = "comparison_needed" });
        Assert.IsNull(C2RuntimeContext.ReadSupportedChannels(_root));
        foreach (var support in new[] { Array.Empty<string>(), new[] { "Unknown" }, new[] { "JawOpen", "JawOpen" } })
        {
            C2Contract.WriteAtomic(path, new { support });
            Assert.ThrowsExactly<InvalidDataException>(() => C2RuntimeContext.ReadSupportedChannels(_root));
        }
        C2Contract.WriteAtomic(path, new { support = "JawOpen" });
        Assert.ThrowsExactly<InvalidDataException>(() => C2RuntimeContext.ReadSupportedChannels(_root));
    }

    [TestMethod]
    public void RestoredContextAllowsCameraStartupThenRejectsChangedTransform()
    {
        var pipeline = new FaceProcessingPipeline(new FacePipelineEventBus(), new PipelineMetrics())
        {
            ImageTransformer = new ImageTransformer()
        };
        var face = (FacePipelineManager)RuntimeHelpers.GetUninitializedObject(typeof(FacePipelineManager));
        typeof(FacePipelineManager).GetField("_pipeline", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(face, pipeline);
        var settings = new Mock<ILocalSettingsService>();
        var transform = new CameraSettings(Camera.Face, new RegionOfInterest(12, 34, 180, 170), .5f);
        settings.Setup(s => s.ReadSetting<CameraSettings>("FaceCamera", null, false)).Returns(transform);
        settings.Setup(s => s.ReadSetting<string>("LastOpenedFaceCamera", null, false)).Returns("camera");
        settings.Setup(s => s.ReadSetting<float>("AppSettings_OneEuroMinFreqCutoff", default, false)).Returns(.5f);
        settings.Setup(s => s.ReadSetting<float>(PersonalModelManager.BlendSetting, 1f, false)).Returns(1f);
        var model = new PersonalModelManager(face, settings.Object, NullLogger<PersonalModelManager>.Instance);
        var corrector = new Mock<IPersonalCorrector>();
        typeof(PersonalModelManager).GetField("_corrector", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, corrector.Object);
        typeof(PersonalModelManager).GetProperty(nameof(PersonalModelManager.ActiveModelPath))!.SetValue(model, "C.onnx");
        var calibration = new CalibrationService(settings.Object);
        var output = C2OutputContext.Capture(calibration, settings.Object);
        var contract = new C2ReferenceContract("stock", "hash", "feature", "hash", "C.onnx", "hash", model.Blend,
            output, JsonSerializer.Serialize(new { Address = "camera", Backend = (string?)null, Transform = transform }));
        var context = new C2RuntimeContext(model, calibration, settings.Object, face);
        var matches = context.ContextGuard(contract);
        Assert.IsTrue(matches(), "The inactive default transformer must not reject restoration before camera startup.");
        face.SetTransformation(transform with { });
        face.SetVideoSource(new Mock<IVideoSource>().Object);
        Assert.IsTrue(matches(), "An equal transform reconstructed on camera connect must keep C2 active.");
        settings.Setup(s => s.ReadSetting<bool>("AudioAssist_Enabled", false, false)).Returns(true);
        Assert.IsFalse(matches(), "Capture and comparison still require audio off.");
        StringAssert.Contains(context.ContextFailure(contract)()!, "Audio Assist is on");
        var everydayFailure = context.ContextFailure(contract, allowAudioAssist: true);
        Assert.IsNull(everydayFailure(), "A kept visual C2 must remain compatible with downstream Audio Assist.");
        var supportedFailure = context.ContextFailure(contract, allowAudioAssist: true, correctsChannel: i => i is 4 or 19 or 20);
        calibration.SetExpression("/tongueOutLower", .3f);
        Assert.IsNull(supportedFailure(), "Uncorrected TongueOut calibration must not stop a kept jaw/smile candidate.");
        StringAssert.Contains(everydayFailure()!, "TongueOut", "Unknown support must still use the strict policy.");
        calibration.SetExpression("/jawOpenLower", .1f);
        StringAssert.Contains(supportedFailure()!, "JawOpen");
        calibration.SetExpression("/jawOpenLower", 0);
        face.SetTransformation(transform with { RotationRadians = 1f });
        Assert.IsFalse(matches(), "Changing the live crop/rotation must cause fallback.");
        StringAssert.Contains(everydayFailure()!, "Face camera crop");
    }

    [TestMethod]
    public void ExistingTrainedCandidateRoundTripsAndLoadsWithoutRetraining()
    {
        var directory = Environment.GetEnvironmentVariable("BABALLONIA_C2_TEST_CANDIDATE");
        if (string.IsNullOrWhiteSpace(directory)) Assert.Inconclusive("Set BABALLONIA_C2_TEST_CANDIDATE for read-only compatibility validation.");
        var contract = JsonSerializer.Deserialize<C2ReferenceContract>(File.ReadAllText(Path.Combine(directory, "contract.json")))!;
        var bytes = JsonSerializer.SerializeToUtf8Bytes(contract, PersonalizationPaths.IndentedJson);
        var hash = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(bytes));
        var path = C2RuntimeContext.ValidateCandidate(directory, hash);
        using var candidate = new C2CandidateCorrector(path, hash, contract.EmbeddingDim);
        Assert.IsNull(candidate.Failure);
        var currentStock = Path.Combine(AppContext.BaseDirectory, "faceModel.onnx");
        C2RuntimeContext.ValidateContext(contract, contract with { StockPath = currentStock, StockSha256 = C2Contract.Hash(currentStock) });
        var support = C2RuntimeContext.ReadSupportedChannels(directory)!;
        Assert.IsNotNull(support);
        var adjustedRanges = contract.Output.Ranges.Select(r => r.ToArray()).ToArray();
        adjustedRanges[33][0] = contract.Output.Ranges[33][0] + .01f;
        C2RuntimeContext.ValidateContext(contract, contract with { Output = contract.Output with { Ranges = adjustedRanges } }, support.Contains);
    }
}
