using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services.Calibration;
using Baballonia.Services.events;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The label side of guided calibration, and the preflight that stops a session producing data that
/// is confidently wrong.
///
/// <see cref="CueStateSource"/> exists so that the value sent to the avatar and the value written
/// into the dataset are the same number. Both interpolate the same immutable snapshot, so the tests
/// that matter are the ones showing the two agree - including mid-transition, where a stale or
/// separately-computed target would diverge without anything failing.
///
/// The OSC-prefix preflight is the other. With a prefix configured the VRCFT module never matches
/// the commanded addresses, so the avatar does not move, the user has nothing to imitate, and the
/// recording fills with neutral faces labelled as expressions. Nothing about that fails at the time.
/// </summary>
[TestClass]
public class CueStateSourceTest
{
    private static readonly int Jaw = PersonalizationSchema.IndexOf("JawOpen");
    private long _clock;

    private static float[] Vector(float jaw)
    {
        var values = new float[PersonalizationSchema.ExpressionCount];
        values[Jaw] = jaw;
        return values;
    }

    private CuePhase Phase(string id, string phase, float from, float to, double seconds) =>
        new(id, phase, [Jaw], Vector(from), Vector(to), seconds, _clock, to, 0);

    private void Advance(double seconds) => _clock += (long)(seconds * Stopwatch.Frequency);

    [TestInitialize]
    public void Initialize() => _clock = 5_000_000;

    [TestMethod]
    public void NoPhaseMeansNoCueStamp()
    {
        var source = new CueStateSource(() => _clock, Stopwatch.Frequency);

        Assert.IsNull(source.CurrentCue(), "frames outside calibration must not carry a cue");
    }

    [TestMethod]
    public void HoldReportsTheCommandedValue()
    {
        var source = new CueStateSource(() => _clock, Stopwatch.Frequency);
        source.SetPhase(Phase("JawOpen50", "hold", 0.5f, 0.5f, 3.0));

        Advance(1.5);
        var cue = source.CurrentCue();

        Assert.IsNotNull(cue);
        Assert.AreEqual("JawOpen50", cue.Id);
        Assert.AreEqual("hold", cue.Phase);
        Assert.AreEqual(0.5f, cue.Target[Jaw], 1e-6);
        CollectionAssert.AreEqual(new[] { Jaw }, cue.Dims.ToList());
    }

    [TestMethod]
    public void TransitionInterpolates()
    {
        var source = new CueStateSource(() => _clock, Stopwatch.Frequency);
        source.SetPhase(Phase("JawOpen100", "transition", 0f, 1f, 1.0));

        Advance(0.5);

        Assert.AreEqual(0.5f, source.CurrentCue()!.Target[Jaw], 0.02);
    }

    [TestMethod]
    public void TargetIsClampedToTheUnitRange()
    {
        var source = new CueStateSource(() => _clock, Stopwatch.Frequency);
        source.SetPhase(Phase("X", "hold", 0f, 1f, 1.0));

        Advance(10.0);  // far past the end

        Assert.AreEqual(1.0f, source.CurrentCue()!.Target[Jaw], 1e-6);
    }

    [TestMethod]
    public void UncuedExpressionsStayAtZero()
    {
        var source = new CueStateSource(() => _clock, Stopwatch.Frequency);
        source.SetPhase(Phase("JawOpen100", "hold", 1f, 1f, 2.0));

        var target = source.CurrentCue()!.Target;

        for (var d = 0; d < PersonalizationSchema.ExpressionCount; d++)
        {
            if (d == Jaw) continue;
            Assert.AreEqual(0f, target[d], $"expression {d} was commanded but not cued");
        }
    }

    [TestMethod]
    public void ClearStopsStampingImmediately()
    {
        var source = new CueStateSource(() => _clock, Stopwatch.Frequency);
        source.SetPhase(Phase("X", "hold", 1f, 1f, 5.0));

        source.Clear();

        Assert.IsNull(source.CurrentCue());
    }

    [TestMethod]
    public void AvatarDeadmanLapseStopsCueStampingAtTheSameBoundary()
    {
        var overrideService = new ExpressionOverrideService(() => _clock, 1);
        var source = new CueStateSource(
            () => _clock, 1, () => overrideService.IsCommandHealthy);
        var phase = new CuePhase(
            "JawOpen100", "hold", [Jaw], Vector(1), Vector(1), 10, _clock, 1, 0);

        overrideService.Activate();
        overrideService.PushPhase(phase);
        source.SetPhase(phase);
        Assert.IsNotNull(source.CurrentCue());

        _clock += 2; // longer than the shared one-second sender/cue deadman

        Assert.IsFalse(overrideService.IsCommandHealthy);
        Assert.IsNull(source.CurrentCue(),
            "a label must not outlive the avatar command that visually teaches it");
    }

    [TestMethod]
    public void SourceLabelDistinguishesAvatarFromBar()
    {
        var source = new CueStateSource(() => _clock, Stopwatch.Frequency) { Source = "bar" };
        source.SetPhase(Phase("X", "hold", 1f, 1f, 1.0));

        Assert.AreEqual("bar", source.CurrentCue()!.Source);
    }

    [TestMethod]
    public void RecordedTargetMatchesWhatTheAvatarWasSent()
    {
        // Two components, one snapshot, one clock. If they ever diverged, the dataset would describe
        // an expression the avatar never showed - and nothing would report an error.
        var overrideService = new ExpressionOverrideService(() => _clock, Stopwatch.Frequency);
        var source = new CueStateSource(() => _clock, Stopwatch.Frequency);

        var phase = Phase("JawOpen100", "transition", 0f, 1f, 2.0);
        overrideService.Activate();
        overrideService.PushPhase(phase);
        source.SetPhase(phase);

        foreach (var step in new[] { 0.25, 0.5, 0.75 })
        {
            Advance(step);

            // What GuidedCalibrationService.Tick does every UI tick. Without it the override's
            // deadman lapses after a second and hands control back to live tracking - which is the
            // correct behaviour, and exactly why the real loop keeps checking in.
            overrideService.KeepAlive();

            var sent = overrideService.SampleTarget();
            var recorded = source.CurrentCue()!.Target;

            Assert.IsNotNull(sent);
            for (var d = 0; d < PersonalizationSchema.ExpressionCount; d++)
                Assert.AreEqual(sent[d], recorded[d], 1e-6,
                    $"sent and recorded targets disagree at expression {d}");
        }
    }

    [TestMethod]
    public void MalformedPhaseIsRejected()
    {
        var source = new CueStateSource(() => _clock, Stopwatch.Frequency);
        var bad = new CuePhase("X", "hold", [Jaw], new float[3], new float[3], 1.0, _clock);

        Assert.ThrowsExactly<ArgumentException>(() => source.SetPhase(bad));
    }
}

/// <summary>The preflight and lifecycle of a guided session.</summary>
[TestClass]
public class GuidedCalibrationServiceTest
{
    private static GuidedCalibrationService Build(string oscPrefix, DatasetRecorderService recorder)
    {
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<string>(
                GuidedCalibrationService.OscPrefixSetting, It.IsAny<string>(), It.IsAny<bool>()))
            .Returns(oscPrefix);

        return new GuidedCalibrationService(
            new ExpressionOverrideService(),
            new CueStateSource(),
            recorder,
            settings.Object,
            NullLogger<GuidedCalibrationService>.Instance,
            cameraFrameReady: () => true);
    }

    private static DatasetRecorderService Recorder() =>
        new(new Baballonia.Services.FacePipelineEventBus(),
            NullLogger<DatasetRecorderService>.Instance);

    [TestMethod]
    public void PreflightPassesWithNoOscPrefix()
    {
        using var recorder = Recorder();
        var service = Build("", recorder);

        Assert.IsTrue(service.Preflight().Started);
    }

    [TestMethod]
    public void PreflightRequiresARealRecentFaceInferenceFrame()
    {
        var bus = new Baballonia.Services.FacePipelineEventBus();
        using var recorder = new DatasetRecorderService(
            bus, NullLogger<DatasetRecorderService>.Instance);
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<string>(
                GuidedCalibrationService.OscPrefixSetting, It.IsAny<string>(), It.IsAny<bool>()))
            .Returns("");
        var service = new GuidedCalibrationService(
            new ExpressionOverrideService(), new CueStateSource(), recorder,
            settings.Object, NullLogger<GuidedCalibrationService>.Instance);

        var before = service.Preflight();
        Assert.IsFalse(before.Started);
        StringAssert.Contains(before.Message, "face-camera inference frame");

        using var image = new Mat(8, 8, MatType.CV_8UC1, Scalar.All(127));
        bus.Publish(new FacePipelineEvents.NewRawExpressionsEvent(
            image, new float[PersonalizationSchema.ExpressionCount], DateTime.UtcNow.Ticks));

        Assert.IsTrue(service.Preflight().Started,
            "the same usable raw-expression frame consumed by recording should satisfy readiness");
    }

    [TestMethod]
    public void PreflightRefusesWhenAnOscPrefixIsConfigured()
    {
        // The failure this prevents is silent: the avatar would sit still while the recorder happily
        // labelled a neutral face as a series of expressions.
        using var recorder = Recorder();
        var service = Build("/avatar", recorder);

        var result = service.Preflight();

        Assert.IsFalse(result.Started);
        StringAssert.Contains(result.Message, "prefix");
        StringAssert.Contains(result.Message, "avatar would not move");
    }

    [TestMethod]
    public void PreflightRefusesWhileAlreadyRecording()
    {
        using var recorder = Recorder();
        var service = Build("", recorder);
        var sessionId = recorder.StartSession(SessionType.Neutral);

        try
        {
            Assert.IsFalse(service.Preflight().Started);
        }
        finally
        {
            recorder.StopSessionAsync().GetAwaiter().GetResult();
            var directory = PersonalizationPaths.SessionDirectory(sessionId);
            if (System.IO.Directory.Exists(directory))
                System.IO.Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public void StartingIsRefusedWhenPreflightFailsAndNothingIsRecorded()
    {
        using var recorder = Recorder();
        var service = Build("/prefix", recorder);

        var result = service.Start(new GuidedCaptureRoutine(
            GuidedCaptureRoutine.BuildJawOpenRoutine(repetitions: 1)));

        Assert.IsFalse(result.Started);
        Assert.IsFalse(recorder.IsRecording, "a refused session must not leave a recording running");
        Assert.IsFalse(service.IsRunning);
    }

    [TestMethod]
    public void TickWithoutAStartDoesNothing()
    {
        using var recorder = Recorder();
        var service = Build("", recorder);

        Assert.IsFalse(service.Tick());
    }

    [TestMethod]
    public async Task StoppingWithoutStartingIsSafe()
    {
        using var recorder = Recorder();
        var service = Build("", recorder);

        Assert.IsNull(await service.StopAsync());
    }

    [TestMethod]
    public void DisposeReleasesTheAvatar()
    {
        // Navigating away mid-session must not leave the avatar frozen mid-expression.
        var overrideService = new ExpressionOverrideService();
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<string>(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<bool>()))
            .Returns("");

        using var recorder = Recorder();
        var service = new GuidedCalibrationService(overrideService, new CueStateSource(), recorder,
            settings.Object, NullLogger<GuidedCalibrationService>.Instance);

        overrideService.Activate();
        service.Dispose();

        // Deactivate serves a brief neutral flush, then releases control back to live tracking.
        var target = overrideService.SampleTarget();
        Assert.IsTrue(target is null || target.All(v => v == 0f),
            "after disposal the avatar must be neutral or released, never held mid-expression");
    }

    [TestMethod]
    public async Task VrCancelInvalidatesThePartialAttemptBeforeFinalizing()
    {
        long clock = 1_000;
        var presenter = new ActionPresenter();
        var overrideService = new ExpressionOverrideService(() => clock, 1_000);
        var cueState = new CueStateSource(() => clock, 1_000, () => overrideService.IsCommandHealthy);
        using var recorder = new DatasetRecorderService(
            new Baballonia.Services.FacePipelineEventBus(),
            NullLogger<DatasetRecorderService>.Instance,
            cueState);
        var service = BuildAdvanced(overrideService, cueState, recorder, presenter);
        var routine = ShortRoutine(() => clock);
        string? directory = null;

        try
        {
            Assert.IsTrue(service.Start(routine).Started);
            directory = PersonalizationPaths.SessionDirectory(recorder.CurrentSessionId!);
            ReachFirstHold(service, routine, ref clock);

            presenter.NextAction = VrCalibrationAction.Cancel;
            Assert.IsFalse(service.Tick());
            await service.StopAsync();

            CollectionAssert.Contains(QualityReasons(directory), "cancelled-in-vr");
            Assert.IsFalse(recorder.IsRecording);
        }
        finally
        {
            service.Dispose();
            DeleteSession(directory);
        }
    }

    [TestMethod]
    public async Task DeadmanLapseAbortsAndInvalidatesTheInterruptedAttempt()
    {
        long clock = 1_000;
        var presenter = new ActionPresenter();
        var overrideService = new ExpressionOverrideService(() => clock, 1_000);
        var cueState = new CueStateSource(() => clock, 1_000, () => overrideService.IsCommandHealthy);
        using var recorder = new DatasetRecorderService(
            new Baballonia.Services.FacePipelineEventBus(),
            NullLogger<DatasetRecorderService>.Instance,
            cueState);
        var service = BuildAdvanced(overrideService, cueState, recorder, presenter);
        var routine = ShortRoutine(() => clock);
        string? directory = null;

        try
        {
            Assert.IsTrue(service.Start(routine).Started);
            directory = PersonalizationPaths.SessionDirectory(recorder.CurrentSessionId!);
            ReachFirstHold(service, routine, ref clock);

            clock += 1_500;
            Assert.IsFalse(service.Tick(), "a lapsed visual command must abort the guided session");
            await service.StopAsync();

            CollectionAssert.Contains(QualityReasons(directory), "override-deadman-lapsed");
            Assert.IsFalse(recorder.IsRecording);
        }
        finally
        {
            service.Dispose();
            DeleteSession(directory);
        }
    }

    [TestMethod]
    public void DisposeFinalizesTheRecorderAndPersistsRetryQuality()
    {
        long clock = 1_000;
        var presenter = new ActionPresenter();
        var overrideService = new ExpressionOverrideService(() => clock, 1_000);
        var cueState = new CueStateSource(() => clock, 1_000, () => overrideService.IsCommandHealthy);
        using var recorder = new DatasetRecorderService(
            new Baballonia.Services.FacePipelineEventBus(),
            NullLogger<DatasetRecorderService>.Instance,
            cueState);
        var service = BuildAdvanced(overrideService, cueState, recorder, presenter);
        var routine = ShortRoutine(() => clock);
        string? directory = null;

        try
        {
            Assert.IsTrue(service.Start(routine).Started);
            directory = PersonalizationPaths.SessionDirectory(recorder.CurrentSessionId!);
            ReachFirstHold(service, routine, ref clock);

            presenter.NextAction = VrCalibrationAction.Retry;
            Assert.IsTrue(service.Tick());
            service.Dispose();

            Assert.IsFalse(recorder.IsRecording,
                "disposing guided calibration must finalize, not leave the recorder running");
            CollectionAssert.Contains(QualityReasons(directory), "retried-in-vr");
        }
        finally
        {
            service.Dispose();
            DeleteSession(directory);
        }
    }

    [TestMethod]
    public async Task PresenterUpdateFailureClearsCueAndInvalidatesAttemptImmediately()
    {
        long clock = 1_000;
        var presenter = new ActionPresenter();
        var overrideService = new ExpressionOverrideService(() => clock, 1_000);
        var cueState = new CueStateSource(() => clock, 1_000, () => overrideService.IsCommandHealthy);
        using var recorder = new DatasetRecorderService(
            new Baballonia.Services.FacePipelineEventBus(),
            NullLogger<DatasetRecorderService>.Instance,
            cueState);
        var service = BuildAdvanced(overrideService, cueState, recorder, presenter);
        var routine = ShortRoutine(() => clock);
        string? directory = null;

        try
        {
            Assert.IsTrue(service.Start(routine).Started);
            directory = PersonalizationPaths.SessionDirectory(recorder.CurrentSessionId!);
            ReachFirstHold(service, routine, ref clock);
            Assert.IsNotNull(cueState.CurrentCue());

            presenter.FailNextPresent = true;
            Assert.IsFalse(service.Tick());
            Assert.IsNull(cueState.CurrentCue(),
                "a frame arriving before StopAsync must not retain a stale guided target");
            Assert.IsFalse(overrideService.IsCommandHealthy);
            var safeTarget = overrideService.SampleTarget();
            Assert.IsTrue(safeTarget is null || safeTarget.All(value => value == 0f),
                "presenter failure may use the normal neutral flush but must not hold an expression");

            await service.StopAsync();
            CollectionAssert.Contains(QualityReasons(directory), "vr-presenter-failed");
        }
        finally
        {
            service.Dispose();
            DeleteSession(directory);
        }
    }

    private static GuidedCalibrationService BuildAdvanced(
        ExpressionOverrideService overrideService,
        CueStateSource cueState,
        DatasetRecorderService recorder,
        IVrCalibrationPresenter presenter)
    {
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<string>(
                GuidedCalibrationService.OscPrefixSetting, It.IsAny<string>(), It.IsAny<bool>()))
            .Returns("");
        return new GuidedCalibrationService(
            overrideService, cueState, recorder, settings.Object,
            NullLogger<GuidedCalibrationService>.Instance, presenter,
            cameraFrameReady: () => true);
    }

    private static GuidedCaptureRoutine ShortRoutine(Func<long> clock) => new(
        GuidedCaptureRoutine.BuildJawOpenRoutine(
            repetitions: 1, levels: [1f], holdSeconds: 1,
            restSeconds: 0.1, transitionSeconds: 0.1, leadInSeconds: 0.1),
        clock, 1_000);

    private static void ReachFirstHold(
        GuidedCalibrationService service,
        GuidedCaptureRoutine routine,
        ref long clock)
    {
        // Walks to the first hold rather than counting steps, so inserting or reordering the
        // lead-in/prep/transition preamble does not silently point these tests at the wrong phase.
        for (var step = 0; step < 32 && routine.Current?.Phase != "hold"; step++)
        {
            Assert.IsTrue(service.Tick(), "the session ended before reaching a hold");
            clock += 200;
        }

        Assert.AreEqual("hold", routine.Current!.Phase, "never reached a hold");
    }

    private static string[] QualityReasons(string directory)
    {
        var path = Path.Combine(directory, "guided_quality_overrides.json");
        Assert.IsTrue(File.Exists(path), $"expected quality overrides at {path}");
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.GetProperty("attempts").EnumerateArray()
            .Select(item => item.GetProperty("reason").GetString() ?? "")
            .ToArray();
    }

    private static void DeleteSession(string? directory)
    {
        if (!string.IsNullOrWhiteSpace(directory) && Directory.Exists(directory))
            Directory.Delete(directory, recursive: true);
    }

    private sealed class ActionPresenter : IVrCalibrationPresenter
    {
        public bool IsAvailable => true;
        public bool IsPresenting { get; private set; }
        public bool IsHealthy { get; private set; } = true;
        public string Status => IsPresenting ? "presenting" : "stopped";
        public VrCalibrationAction NextAction { get; set; }
        public bool FailNextPresent { get; set; }

        public VrPresenterStartResult Begin(string sessionTitle)
        {
            if (IsPresenting)
                return VrPresenterStartResult.Failure("already active");
            IsPresenting = true;
            IsHealthy = true;
            return VrPresenterStartResult.Success();
        }

        public void Present(VrCalibrationFrame frame)
        {
            if (!FailNextPresent) return;
            FailNextPresent = false;
            IsHealthy = false;
            IsPresenting = false;
        }

        public VrCalibrationAction ConsumeAction()
        {
            var action = NextAction;
            NextAction = VrCalibrationAction.None;
            return action;
        }

        public void End(string? completionMessage = null)
        {
            IsPresenting = false;
            IsHealthy = false;
        }
        public void Dispose() => End();
    }
}
