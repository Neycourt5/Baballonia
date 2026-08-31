using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using Baballonia.Services.events;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

[TestClass]
public class GrimacePreviewServiceTest
{
    private static readonly int Jaw = PersonalizationSchema.IndexOf("JawOpen");
    private static readonly int LowerLeft = PersonalizationSchema.IndexOf("MouthLowerDownLeft");
    private static readonly int LowerRight = PersonalizationSchema.IndexOf("MouthLowerDownRight");
    private static readonly int StretchLeft = PersonalizationSchema.IndexOf("MouthStretchLeft");
    private static readonly int StretchRight = PersonalizationSchema.IndexOf("MouthStretchRight");
    private static readonly int[] ExpectedDims =
        [Jaw, LowerLeft, LowerRight, StretchLeft, StretchRight];

    [TestMethod]
    public void CandidateCatalogueContainsOnlyThreeSchemaValidSymmetricHypotheses()
    {
        Assert.AreEqual(3, GrimaceCandidateCatalog.All.Count,
            "the experiment must stay small enough to compare in-headset");

        foreach (var candidate in GrimaceCandidateCatalog.All)
        {
            CollectionAssert.AreEquivalent(ExpectedDims, candidate.Dims.ToArray(), candidate.Id);
            var target = candidate.CreateTarget();

            Assert.AreEqual(PersonalizationSchema.ExpressionCount, target.Length);
            Assert.IsTrue(target.All(value => value is >= 0f and <= 1f), candidate.Id);
            Assert.AreEqual(target[LowerLeft], target[LowerRight], 1e-6, candidate.Id);
            Assert.AreEqual(target[StretchLeft], target[StretchRight], 1e-6, candidate.Id);

            for (var dim = 0; dim < target.Length; dim++)
            {
                if (ExpectedDims.Contains(dim)) continue;
                Assert.AreEqual(0f, target[dim], $"{candidate.Id} moves unrelated dim {dim}");
            }
        }

        Assert.IsFalse(GuidedCues.All.Any(cue => cue.Id == "Grimace"),
            "experimental candidates must not enter the ordinary trainable cue catalogue");
        Assert.IsFalse(GuidedCues.CombinationPass.Any(cue => cue.Id == "Grimace"));
    }

    [TestMethod]
    public void UnconfirmedCandidateNeverSurfacesATrainableCue()
    {
        var fixture = CreateFixture();
        using var service = fixture.Service;

        Assert.IsNull(service.Confirmation);
        Assert.IsNull(service.ConfirmedCandidate);
        Assert.IsNull(service.ConfirmedCue);
        Assert.IsNull(service.ConfirmedRoutineChoice);

        var attempted = service.ConfirmCurrentCandidate();
        Assert.IsFalse(attempted.Success);
        Assert.IsNull(service.ConfirmedCue);
    }

    [TestMethod]
    public void PreviewCommandsAvatarButNeverPublishesARecorderCue()
    {
        var cueState = new CueStateSource();
        var fixture = CreateFixture();
        using var service = fixture.Service;

        var started = service.BeginPreview();

        Assert.IsTrue(started.Success);
        Assert.IsTrue(service.IsPreviewing);
        Assert.IsNull(cueState.CurrentCue(),
            "preview has no CueStateSource dependency and must never stamp supervision");

        var target = fixture.Override.SampleTarget();
        Assert.IsNotNull(target);
        CollectionAssert.AreEqual(service.CurrentCandidate!.CreateTarget(), target);
        StringAssert.Contains(fixture.Presenter.Frames[^1].Title, "PREVIEW ONLY");
        StringAssert.Contains(fixture.Presenter.Frames[^1].Instruction, "NOT TRAINING DATA");

        service.CancelPreview();

        Assert.IsFalse(service.IsPreviewing);
        Assert.IsFalse(fixture.Presenter.IsPresenting);
        Assert.IsTrue(fixture.Override.SampleTarget()!.All(value => value == 0f),
            "cancel must immediately neutralize before releasing live tracking");
    }

    [TestMethod]
    public async Task RecorderAndPreviewMutuallyExcludeEachOther()
    {
        var gate = new TrainingCaptureGate();
        var bus = new FacePipelineEventBus();
        using var recorder = new DatasetRecorderService(
            bus,
            NullLogger<DatasetRecorderService>.Instance,
            cueState: null,
            captureGate: gate);
        var fixture = CreateFixture(gate: gate);
        using var preview = fixture.Service;

        string? sessionId = null;
        try
        {
            sessionId = recorder.StartSession(SessionType.Neutral);

            var whileRecording = preview.BeginPreview();
            Assert.IsFalse(whileRecording.Success);
            StringAssert.Contains(whileRecording.Message, "training recording");
            Assert.IsFalse(fixture.Override.IsActive);

            await recorder.StopSessionAsync();
            Assert.IsTrue(preview.BeginPreview().Success);

            var blocked = Assert.ThrowsExactly<InvalidOperationException>(() =>
                recorder.StartSession(SessionType.Guided));
            StringAssert.Contains(blocked.Message, "Grimace candidate preview");
        }
        finally
        {
            preview.CancelPreview();
            if (recorder.IsRecording)
                await recorder.StopSessionAsync();
            if (sessionId != null)
            {
                var directory = PersonalizationPaths.SessionDirectory(sessionId);
                if (Directory.Exists(directory))
                    Directory.Delete(directory, recursive: true);
            }
        }
    }

    [TestMethod]
    public void VrActionsCycleCandidatesAndCancelSafely()
    {
        var fixture = CreateFixture();
        using var service = fixture.Service;
        Assert.IsTrue(service.BeginPreview().Success);
        Assert.AreEqual(0, service.CurrentCandidateIndex);

        fixture.Presenter.NextAction = VrCalibrationAction.Skip;
        Assert.IsTrue(service.Tick());
        Assert.AreEqual(1, service.CurrentCandidateIndex);

        fixture.Presenter.NextAction = VrCalibrationAction.Retry;
        Assert.IsTrue(service.Tick());
        Assert.AreEqual(0, service.CurrentCandidateIndex);

        fixture.Presenter.NextAction = VrCalibrationAction.Cancel;
        Assert.IsFalse(service.Tick());
        Assert.IsFalse(service.IsPreviewing);
        Assert.IsTrue(fixture.Override.SampleTarget()!.All(value => value == 0f));
    }

    [TestMethod]
    public void ExplicitConfirmationPersistsExactCandidateAndOnlyThenBuildsRoutine()
    {
        var settings = new MemorySettings();
        var fixture = CreateFixture(settings: settings);
        using var service = fixture.Service;
        Assert.IsTrue(service.BeginPreview().Success);

        var chosen = GrimaceCandidateCatalog.All[1];
        Assert.IsTrue(service.ShowCandidate(chosen.Id).Success);
        var confirmed = service.ConfirmCurrentCandidate();

        Assert.IsTrue(confirmed.Success);
        Assert.IsFalse(service.IsPreviewing);
        Assert.AreEqual(chosen.Id, service.Confirmation!.CandidateId);
        Assert.AreEqual(chosen.Id, service.ConfirmedCandidate!.Id);

        var cue = service.ConfirmedCue;
        Assert.IsNotNull(cue);
        Assert.IsTrue(cue.IsBinary,
            "only the exact empirically accepted vector is valid; no untested half-level is inferred");
        CollectionAssert.AreEqual(chosen.CreateTarget(), cue.TargetAt(1f));

        var routine = service.ConfirmedRoutineChoice;
        Assert.IsNotNull(routine);
        var holds = routine.Build().Where(step => step.Phase == "hold").ToArray();
        Assert.IsTrue(holds.Length > 0);
        foreach (var hold in holds)
            CollectionAssert.AreEqual(chosen.CreateTarget(), hold.To);

        // Confirmation survives a new service instance because it is a persisted decision, not an
        // in-memory selection made by opening the preview page.
        var second = CreateFixture(settings: settings);
        using var restored = second.Service;
        Assert.AreEqual(chosen.Id, restored.ConfirmedCandidate!.Id);
        Assert.IsNotNull(restored.ConfirmedCue);
    }

    [TestMethod]
    public void StaleOrTamperedConfirmationCannotBecomeSupervision()
    {
        var settings = new MemorySettings();
        var candidate = GrimaceCandidateCatalog.All[0];
        settings.SaveSetting(
            GrimacePreviewService.ConfirmationSetting,
            new GrimaceCandidateConfirmation(
                candidate.Id,
                GrimaceCandidateCatalog.Version,
                PersonalizationSchema.Sha256,
                "not-the-confirmed-vector-fingerprint",
                DateTime.UtcNow.ToString("o")));

        var fixture = CreateFixture(settings: settings);
        using var service = fixture.Service;

        Assert.IsNull(service.ConfirmedCandidate);
        Assert.IsNull(service.ConfirmedCue);
        Assert.IsNull(service.ConfirmedRoutineChoice);
    }

    [TestMethod]
    public void StalledDeadmanCannotConfirmACandidate()
    {
        var clock = Stopwatch.GetTimestamp();
        var expressionOverride = new ExpressionOverrideService(() => clock, Stopwatch.Frequency);
        var fixture = CreateFixture(expressionOverride: expressionOverride);
        using var service = fixture.Service;
        Assert.IsTrue(service.BeginPreview().Success);

        clock += (long)((ExpressionOverrideService.KeepAliveTimeout.TotalSeconds + 0.1) *
                        Stopwatch.Frequency);

        Assert.IsFalse(expressionOverride.IsCommandHealthy);
        var result = service.ConfirmCurrentCandidate();
        Assert.IsFalse(result.Success);
        Assert.IsNull(service.Confirmation);
        Assert.IsNull(service.ConfirmedCue);
    }

    [TestMethod]
    public void UnhealthyHeadsetCannotConfirmAndTickStopsPreviewSafely()
    {
        var fixture = CreateFixture();
        using var service = fixture.Service;
        Assert.IsTrue(service.BeginPreview().Success);
        fixture.Presenter.Healthy = false;

        var confirmation = service.ConfirmCurrentCandidate();
        Assert.IsFalse(confirmation.Success);
        Assert.IsNull(service.Confirmation);

        Assert.IsFalse(service.Tick());
        Assert.IsFalse(service.IsPreviewing);
        Assert.IsFalse(fixture.Presenter.IsPresenting);
        Assert.IsTrue(fixture.Override.SampleTarget()!.All(value => value == 0f));
        StringAssert.Contains(service.Status, "stopped safely");
    }

    [TestMethod]
    public void ConfirmationIsNotReportedUntilThePersistedDecisionReadsBack()
    {
        var settings = new MemorySettings { DropWrites = true };
        var fixture = CreateFixture(settings: settings);
        using var service = fixture.Service;
        Assert.IsTrue(service.BeginPreview().Success);

        var result = service.ConfirmCurrentCandidate();

        Assert.IsFalse(result.Success);
        Assert.IsTrue(service.IsPreviewing,
            "a failed save keeps the preview alive so the user can retry");
        Assert.IsNull(service.Confirmation);
        Assert.IsNull(service.ConfirmedCue);
    }

    [TestMethod]
    public void DisposeNeutralizesAndReleasesCaptureGate()
    {
        var fixture = CreateFixture();
        Assert.IsTrue(fixture.Service.BeginPreview().Success);

        fixture.Service.Dispose();

        Assert.IsTrue(fixture.Override.SampleTarget()!.All(value => value == 0f));
        Assert.IsTrue(fixture.Gate.TryEnter("a later recording", out var lease, out _));
        lease!.Dispose();
    }

    private static Fixture CreateFixture(
        TrainingCaptureGate? gate = null,
        MemorySettings? settings = null,
        ExpressionOverrideService? expressionOverride = null)
    {
        gate ??= new TrainingCaptureGate();
        settings ??= new MemorySettings();
        expressionOverride ??= new ExpressionOverrideService();
        var presenter = new FakePresenter();
        var service = new GrimacePreviewService(
            expressionOverride,
            gate,
            settings,
            presenter,
            NullLogger<GrimacePreviewService>.Instance);
        return new Fixture(service, expressionOverride, presenter, gate);
    }

    private sealed record Fixture(
        GrimacePreviewService Service,
        ExpressionOverrideService Override,
        FakePresenter Presenter,
        TrainingCaptureGate Gate);

    private sealed class FakePresenter : IVrCalibrationPresenter
    {
        public bool IsAvailable => true;
        public bool IsPresenting { get; private set; }
        public bool Healthy { get; set; } = true;
        public bool IsHealthy => IsPresenting && Healthy;
        public string Status => IsPresenting ? "presenting" : "ready";
        public List<VrCalibrationFrame> Frames { get; } = [];
        public VrCalibrationAction NextAction { get; set; }

        public VrPresenterStartResult Begin(string sessionTitle)
        {
            IsPresenting = true;
            return VrPresenterStartResult.Success();
        }

        public void Present(VrCalibrationFrame frame) => Frames.Add(frame);

        public VrCalibrationAction ConsumeAction()
        {
            var action = NextAction;
            NextAction = VrCalibrationAction.None;
            return action;
        }

        public void End(string? completionMessage = null) => IsPresenting = false;
        public void Dispose() => IsPresenting = false;
    }

    private sealed class MemorySettings : ILocalSettingsService
    {
        private readonly Dictionary<string, object?> _values = [];
        public bool DropWrites { get; init; }

        public T ReadSetting<T>(string key, T? defaultValue = default, bool forceLocal = false) =>
            _values.TryGetValue(key, out var value) && value is T typed ? typed : defaultValue!;

        public void SaveSetting<T>(string key, T value, bool forceLocal = false)
        {
            if (!DropWrites)
                _values[key] = value;
        }

        public void Save(object target) { }
        public void Load(object target) { }
        public void ForceSave() { }
    }
}
