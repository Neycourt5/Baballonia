using Baballonia.Contracts;
using Baballonia.Desktop.Calibration;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using Baballonia.Services.events;
using Baballonia.Services.EyeV2;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.Tests.Services.EyeV2;

[TestClass]
public sealed class EyeV2VrPresentationTest
{
    private string _directory = null!;
    private string _path = null!;

    [TestInitialize]
    public void Initialize()
    {
        _directory = Path.Combine(Path.GetTempPath(), "BaballoniaEyeVrTests", Guid.NewGuid().ToString("N"));
        _path = Path.Combine(_directory, EyeV2CalibrationStore.FileName);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    [TestMethod]
    public async Task CancelFromHeadset_PreservesPreviouslySavedCalibration()
    {
        var old = EyeV2CalibrationAndMapperTest.IdentityCalibration() with
        {
            Capture = new EyeV2CaptureMetadata { CreatedUtc = new DateTime(2026, 8, 1, 12, 0, 0, DateTimeKind.Utc) },
        };
        var store = Store();
        await store.SaveAsync(old);
        var manager = Manager(store);
        var presenter = new RecordingPresenter { NextAction = VrCalibrationAction.Cancel };
        var bus = new EyePipelineEventBus();
        using var service = new EyeV2CalibrationService(
            bus, store, manager, Mock.Of<ILogger<EyeV2CalibrationService>>(), presenter,
            (_, _) => Task.CompletedTask);

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(() => service.CalibrateAsync());

        Assert.IsTrue(store.TryLoad(out var preserved, out var error), error);
        Assert.AreEqual(old.Capture.CreatedUtc, preserved!.Capture.CreatedUtc);
        Assert.IsTrue(presenter.Frames.Count > 0);
        Assert.AreEqual(VrCalibrationPhase.Preparing, presenter.Frames[0].Phase);
        Assert.IsNull(presenter.CompletionMessage);
        Assert.IsFalse(presenter.IsPresenting);
    }

    [TestMethod]
    public async Task FullCalibration_PresentsBehaviorAndFiveTargetsInHeadset()
    {
        var store = Store();
        var manager = Manager(store);
        var presenter = new RecordingPresenter();
        var bus = new EyePipelineEventBus();
        long ticks = DateTime.UnixEpoch.Ticks;
        var blinkSample = 0;

        Task Delay(int _, CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            ticks += 33 * TimeSpan.TicksPerMillisecond;
            var frame = presenter.LastFrame;
            if (frame != null)
            {
                var raw = RawFor(frame, ref blinkSample);
                using var image = new Mat();
                bus.Publish(new EyePipelineEvents.NewRawExpressionsEvent(image, raw, ticks));
            }
            return Task.CompletedTask;
        }

        using var service = new EyeV2CalibrationService(
            bus, store, manager, Mock.Of<ILogger<EyeV2CalibrationService>>(), presenter, Delay);

        var result = await service.CalibrateAsync();

        Assert.IsTrue(result.IsValid());
        Assert.AreEqual(EyeTrackingMode.ExperimentalV2, manager.Mode);
        Assert.AreEqual("Eye V2-A mapping was saved and activated.", presenter.CompletionMessage);
        Assert.AreEqual(1d, presenter.Frames.Max(frame => frame.OverallProgress), 0.0001,
            "the progress bar must reach exactly full: the denominator is the sum of the steps, so " +
            "this catches a step whose real duration drifted away from its declared one");
        Assert.IsTrue(presenter.Frames.Any(frame => frame.Title == "BLINK SLOWLY THREE TIMES"));
        Assert.IsTrue(presenter.Frames.Any(frame => frame.Title == "SQUINT AND HOLD"));
        Assert.IsTrue(presenter.Frames.Any(frame => frame.Title == "OPEN YOUR EYES WIDE"));

        var targets = presenter.Frames
            .Where(frame => frame.Phase == VrCalibrationPhase.Target)
            .Select(frame => (frame.TargetX, frame.TargetY))
            .Distinct()
            .ToArray();
        CollectionAssert.AreEquivalent(
            new (float?, float?)[] { (0, 0), (-0.7f, 0), (0.7f, 0), (0, -0.7f), (0, 0.7f) },
            targets);

        // For every sampling state, its preparing state was uploaded first.
        foreach (var title in presenter.Frames
                     .Where(frame => frame.Phase is VrCalibrationPhase.Sampling or
                         VrCalibrationPhase.Hold or VrCalibrationPhase.Target)
                     .Select(frame => frame.Title)
                     .Distinct())
        {
            var firstPrep = presenter.Frames.FindIndex(frame =>
                frame.Title == title && frame.Phase == VrCalibrationPhase.Preparing);
            var firstSample = presenter.Frames.FindIndex(frame =>
                frame.Title == title && frame.Phase != VrCalibrationPhase.Preparing);
            Assert.IsTrue(firstPrep >= 0 && firstPrep < firstSample, title);
        }
    }

    [TestMethod]
    public void GazeTargetsSitAtComfortableVisualAngles()
    {
        // Straight ahead is straight ahead.
        var (cx, cy, cz) = OpenVrCalibrationPresenter.GazeTargetPosition(0f, 0f);
        Assert.AreEqual(0f, cx, 1e-5);
        Assert.AreEqual(0f, cy, 1e-5);
        Assert.IsTrue(cz < 0, "the target must be in front of the viewer");

        var (lx, _, _) = OpenVrCalibrationPresenter.GazeTargetPosition(-0.7f, 0f);
        var (rx, _, _) = OpenVrCalibrationPresenter.GazeTargetPosition(0.7f, 0f);
        Assert.IsTrue(lx < 0 && rx > 0, "left and right anchors must be on opposite sides");
        Assert.AreEqual(-lx, rx, 1e-5, "the anchors must be symmetric");

        var (_, uy, _) = OpenVrCalibrationPresenter.GazeTargetPosition(0f, -0.7f);
        var (_, dy, _) = OpenVrCalibrationPresenter.GazeTargetPosition(0f, 0.7f);
        Assert.IsTrue(uy > 0, "negative Y is up");
        Assert.IsTrue(dy < 0);

        // Comfortably inside eye-in-head range. A target far enough out to need head movement
        // measures the neck rather than the eye, and the whole point of the anchors is the eye.
        foreach (var (x, y) in new[] { (-0.7f, 0f), (0.7f, 0f), (0f, -0.7f), (0f, 0.7f) })
        {
            var (px, py, pz) = OpenVrCalibrationPresenter.GazeTargetPosition(x, y);
            var degrees = float.RadiansToDegrees(
                MathF.Atan2(MathF.Sqrt(px * px + py * py), -pz));
            Assert.IsTrue(degrees is > 4 and < 15,
                $"({x},{y}) sits at {degrees:F1} degrees, outside the comfortable band");
        }
    }

    [TestMethod]
    public async Task GazeSamplingIgnoresTheEyeStillTravellingToTheTarget()
    {
        // The eye needs time to saccade and settle. If sampling began the instant the target
        // appeared, every anchor would be dragged toward wherever the user had been looking, and
        // the whole gaze map would inherit that bias. Here the "in flight" frames carry a wildly
        // wrong gaze; a correct implementation must not let them reach the fit at all.
        var contaminatedFrames = 0;

        var clean = await RunCalibrationAsync(null);
        var withFlight = await RunCalibrationAsync(ticksOnTarget =>
        {
            // 800 ms of acquisition at a 250 ms cadence covers the first three updates.
            if (ticksOnTarget >= 3) return false;
            contaminatedFrames++;
            return true;
        });

        Assert.IsTrue(contaminatedFrames > 0, "the test never produced any in-flight frames");

        // Byte-for-byte identical: not "close enough", because any leakage at all would move the
        // medians the anchors are built from.
        foreach (var (a, b) in new[] { (clean.Left, withFlight.Left), (clean.Right, withFlight.Right) })
        {
            Assert.AreEqual(a.Gaze.XX, b.Gaze.XX, 1e-9, "in-flight gaze reached the fit");
            Assert.AreEqual(a.Gaze.YY, b.Gaze.YY, 1e-9, "in-flight gaze reached the fit");
            Assert.AreEqual(a.Gaze.OffsetX, b.Gaze.OffsetX, 1e-9, "in-flight gaze reached the fit");
            Assert.AreEqual(a.Gaze.OffsetY, b.Gaze.OffsetY, 1e-9, "in-flight gaze reached the fit");
        }
    }

    /// <summary>
    /// Runs a full calibration against synthetic eye data. <paramref name="corruptWhileAcquiring"/>
    /// is asked, for each update spent on a gaze target, whether that frame should carry a wildly
    /// wrong gaze - letting a test express "the eye had not arrived yet".
    /// </summary>
    private async Task<EyeV2Calibration> RunCalibrationAsync(Func<int, bool>? corruptWhileAcquiring)
    {
        var store = Store();
        var manager = Manager(store);
        var presenter = new RecordingPresenter();
        var bus = new EyePipelineEventBus();
        long ticks = DateTime.UnixEpoch.Ticks;
        var blinkSample = 0;
        (float? X, float? Y) previousTarget = (null, null);
        var ticksOnThisTarget = 0;

        Task Delay(int _, CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            ticks += 33 * TimeSpan.TicksPerMillisecond;
            var frame = presenter.LastFrame;
            if (frame != null)
            {
                var raw = RawFor(frame, ref blinkSample);

                if (frame.Phase == VrCalibrationPhase.Target)
                {
                    var isNewTarget = previousTarget.X != frame.TargetX ||
                                      previousTarget.Y != frame.TargetY;
                    ticksOnThisTarget = isNewTarget ? 0 : ticksOnThisTarget + 1;
                    previousTarget = (frame.TargetX, frame.TargetY);

                    if (corruptWhileAcquiring?.Invoke(ticksOnThisTarget) == true)
                        for (var i = 0; i < 6; i++) raw[i] = 5f;
                }

                using var image = new Mat();
                bus.Publish(new EyePipelineEvents.NewRawExpressionsEvent(image, raw, ticks));
            }
            return Task.CompletedTask;
        }

        using var service = new EyeV2CalibrationService(
            bus, store, manager, Mock.Of<ILogger<EyeV2CalibrationService>>(), presenter, Delay);

        return await service.CalibrateAsync();
    }

    [TestMethod]
    public async Task PresenterRenderFailureAbortsBeforeAnyCalibrationCanBeCommitted()
    {
        var old = EyeV2CalibrationAndMapperTest.IdentityCalibration() with
        {
            Capture = new EyeV2CaptureMetadata
            {
                CreatedUtc = new DateTime(2026, 8, 2, 12, 0, 0, DateTimeKind.Utc),
            },
        };
        var store = Store();
        await store.SaveAsync(old);
        var manager = Manager(store);
        var presenter = new RecordingPresenter { FailOnPresentNumber = 1 };
        var bus = new EyePipelineEventBus();
        var delayCalls = 0;
        using var service = new EyeV2CalibrationService(
            bus, store, manager, Mock.Of<ILogger<EyeV2CalibrationService>>(), presenter,
            (_, _) => { delayCalls++; return Task.CompletedTask; });

        await Assert.ThrowsExactlyAsync<InvalidOperationException>(() => service.CalibrateAsync());

        Assert.AreEqual(0, delayCalls,
            "sampling must not advance after the first headset frame failed to render");
        Assert.IsTrue(store.TryLoad(out var preserved, out var error), error);
        Assert.AreEqual(old.Capture.CreatedUtc, preserved!.Capture.CreatedUtc);
        Assert.AreEqual(EyeTrackingMode.DefaultBaballonia, manager.Mode);
        Assert.IsNull(presenter.CompletionMessage);
    }

    [TestMethod]
    public async Task CalibrationActivationFailureRollsBackFileAndPreviouslyActiveMapper()
    {
        var old = EyeV2CalibrationAndMapperTest.IdentityCalibration() with
        {
            Capture = new EyeV2CaptureMetadata
            {
                CreatedUtc = new DateTime(2026, 8, 3, 12, 0, 0, DateTimeKind.Utc),
            },
        };
        var store = Store();
        await store.SaveAsync(old);
        var originalBytes = await File.ReadAllBytesAsync(_path);
        var settings = Settings();
        IEyeStateMapper? installed = null;
        var failNextMapperInstall = false;
        var manager = new EyeV2Manager(null!, settings.Object, store,
            Mock.Of<ILogger<EyeV2Manager>>(), mapper =>
            {
                if (failNextMapperInstall && mapper != null)
                {
                    failNextMapperInstall = false;
                    throw new InvalidOperationException("mapper install failed");
                }
                installed = mapper;
            });
        Assert.IsTrue(manager.TrySetMode(EyeTrackingMode.ExperimentalV2));
        var priorMapper = installed;
        var priorCalibration = manager.Calibration;
        failNextMapperInstall = true;

        var presenter = new RecordingPresenter();
        var bus = new EyePipelineEventBus();
        using var service = new EyeV2CalibrationService(
            bus, store, manager, Mock.Of<ILogger<EyeV2CalibrationService>>(), presenter,
            PublishingDelay(bus, presenter));

        await Assert.ThrowsExactlyAsync<InvalidOperationException>(() => service.CalibrateAsync());

        Assert.AreEqual(EyeTrackingMode.ExperimentalV2, manager.Mode);
        Assert.AreSame(priorMapper, installed);
        Assert.AreSame(priorCalibration, manager.Calibration);
        Assert.IsTrue(store.TryLoad(out var preserved, out var error), error);
        Assert.AreEqual(old.Capture.CreatedUtc, preserved!.Capture.CreatedUtc,
            "the newly fitted file must be rolled back byte-for-byte on activation failure");
        CollectionAssert.AreEqual(originalBytes, await File.ReadAllBytesAsync(_path));
    }

    [TestMethod]
    public async Task RecenterSettingsFailureRollsBackFileAndPreviouslyActiveMapper()
    {
        var old = EyeV2CalibrationAndMapperTest.IdentityCalibration() with
        {
            Capture = new EyeV2CaptureMetadata
            {
                CreatedUtc = new DateTime(2026, 8, 4, 12, 0, 0, DateTimeKind.Utc),
            },
        };
        var store = Store();
        await store.SaveAsync(old);
        var originalBytes = await File.ReadAllBytesAsync(_path);
        var failSettingsSave = false;
        var settings = Settings(() => failSettingsSave);
        IEyeStateMapper? installed = null;
        var manager = new EyeV2Manager(null!, settings.Object, store,
            Mock.Of<ILogger<EyeV2Manager>>(), mapper => installed = mapper);
        Assert.IsTrue(manager.TrySetMode(EyeTrackingMode.ExperimentalV2));
        var priorMapper = installed;
        var priorCalibration = manager.Calibration;
        failSettingsSave = true;

        var presenter = new RecordingPresenter();
        var bus = new EyePipelineEventBus();
        using var service = new EyeV2CalibrationService(
            bus, store, manager, Mock.Of<ILogger<EyeV2CalibrationService>>(), presenter,
            PublishingDelay(bus, presenter));

        await Assert.ThrowsExactlyAsync<InvalidOperationException>(() => service.RecenterAsync());

        Assert.AreEqual(EyeTrackingMode.ExperimentalV2, manager.Mode);
        Assert.AreSame(priorMapper, installed);
        Assert.AreSame(priorCalibration, manager.Calibration);
        Assert.IsTrue(store.TryLoad(out var preserved, out var error), error);
        Assert.AreEqual(old.Capture.CreatedUtc, preserved!.Capture.CreatedUtc);
        CollectionAssert.AreEqual(originalBytes, await File.ReadAllBytesAsync(_path));
    }

    [TestMethod]
    public async Task RecenterSaveFailureLeavesFileAndActiveModeUntouched()
    {
        var old = EyeV2CalibrationAndMapperTest.IdentityCalibration() with
        {
            Capture = new EyeV2CaptureMetadata
            {
                CreatedUtc = new DateTime(2026, 8, 5, 12, 0, 0, DateTimeKind.Utc),
            },
        };
        var store = Store();
        await store.SaveAsync(old);
        var originalBytes = await File.ReadAllBytesAsync(_path);
        var settings = Settings();
        IEyeStateMapper? installed = null;
        var manager = new EyeV2Manager(null!, settings.Object, store,
            Mock.Of<ILogger<EyeV2Manager>>(), mapper => installed = mapper);
        Assert.IsTrue(manager.TrySetMode(EyeTrackingMode.ExperimentalV2));
        var priorMapper = installed;
        var priorCalibration = manager.Calibration;

        // SaveAsync uses this exact temporary path; a directory there deterministically makes the
        // replacement fail after the prior calibration snapshot has been captured.
        Directory.CreateDirectory(_path + ".tmp");
        var presenter = new RecordingPresenter();
        var bus = new EyePipelineEventBus();
        using var service = new EyeV2CalibrationService(
            bus, store, manager, Mock.Of<ILogger<EyeV2CalibrationService>>(), presenter,
            PublishingDelay(bus, presenter));

        Exception? failure = null;
        try
        {
            await service.RecenterAsync();
        }
        catch (Exception ex)
        {
            failure = ex;
        }
        Assert.IsNotNull(failure);
        Assert.AreEqual(EyeTrackingMode.ExperimentalV2, manager.Mode);
        Assert.AreSame(priorMapper, installed);
        Assert.AreSame(priorCalibration, manager.Calibration);
        Assert.IsTrue(store.TryLoad(out var preserved, out var error), error);
        Assert.AreEqual(old.Capture.CreatedUtc, preserved!.Capture.CreatedUtc);
        CollectionAssert.AreEqual(originalBytes, await File.ReadAllBytesAsync(_path));
    }

    [TestMethod]
    public void ControllerButtons_AcceptBothOpenVrYOriginsAndRejectTheRestOfThePanel()
    {
        var frame = new VrCalibrationFrame(
            "test", "test", VrCalibrationPhase.Hold, 0,
            AllowRetry: true, AllowSkip: true, AllowCancel: true);

        Assert.AreEqual(VrCalibrationAction.Retry,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 100, 700));
        Assert.AreEqual(VrCalibrationAction.Skip,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 500, 68));
        Assert.AreEqual(VrCalibrationAction.Cancel,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 800, 700));
        Assert.AreEqual(VrCalibrationAction.None,
            OpenVrCalibrationPresenter.HitTestControllerButton(frame, 800, 400));
        Assert.AreEqual(VrCalibrationAction.None,
            OpenVrCalibrationPresenter.HitTestControllerButton(
                frame with { AllowCancel = false }, 800, 700));
    }

    [TestMethod]
    public void GuidedRetryAndSkip_AreAttemptScoped()
    {
        long clock = 1;
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine(
            repetitions: 2, levels: [1f], holdSeconds: 1, restSeconds: 1,
            transitionSeconds: 1, leadInSeconds: 1, prepSeconds: 1);
        var routine = new GuidedCaptureRoutine(steps, () => clock, 1);

        AdvanceToHold(routine, ref clock);
        var original = routine.CurrentAttempt!.Value;

        var replay = routine.RetryCurrentAttempt();
        Assert.IsNotNull(replay);
        Assert.AreEqual("prep", routine.Current!.Phase,
            "Retry replays the whole attempt, countdown included, so the user is not thrown " +
            "straight back into an expression they just asked to redo.");
        Assert.AreEqual(original.Attempt + 1, routine.CurrentAttempt!.Value.Attempt);

        AdvanceToHold(routine, ref clock);
        var skipped = routine.CurrentAttempt!.Value;
        var next = routine.SkipCurrentAttempt();
        Assert.IsNotNull(next);
        Assert.AreNotEqual(skipped.Repetition, routine.CurrentAttempt!.Value.Repetition,
            "Skip must advance only past the current repetition, not end the whole routine.");
    }

    /// <summary>
    /// Walks the routine to its next hold. Written as a search rather than a step count so that
    /// changing the preamble - the lead-in, the countdown, the ramp - cannot quietly leave these
    /// assertions pointing at the wrong phase.
    /// </summary>
    private static void AdvanceToHold(GuidedCaptureRoutine routine, ref long clock)
    {
        for (var step = 0; step < 32 && routine.Current?.Phase != "hold"; step++)
        {
            clock += 2;
            routine.Tick();
        }

        Assert.AreEqual("hold", routine.Current!.Phase, "never reached a hold");
    }

    private EyeV2CalibrationStore Store() =>
        new(Mock.Of<ILogger<EyeV2CalibrationStore>>(), _path);

    private static Mock<ILocalSettingsService> Settings(Func<bool>? failSave = null)
    {
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(service => service.ReadSetting(
                It.IsAny<string>(), It.IsAny<EyeTrackingMode>(), It.IsAny<bool>()))
            .Returns(EyeTrackingMode.DefaultBaballonia);
        settings.Setup(service => service.SaveSetting(
                It.IsAny<string>(), It.IsAny<EyeTrackingMode>(), It.IsAny<bool>()))
            .Callback((string _, EyeTrackingMode _, bool _) =>
            {
                if (failSave?.Invoke() == true)
                    throw new InvalidOperationException("settings save failed");
            });
        return settings;
    }

    private static EyeV2Manager Manager(EyeV2CalibrationStore store)
    {
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(service => service.ReadSetting(
                It.IsAny<string>(), It.IsAny<EyeTrackingMode>(), It.IsAny<bool>()))
            .Returns(EyeTrackingMode.DefaultBaballonia);
        return new EyeV2Manager(null!, settings.Object, store,
            Mock.Of<ILogger<EyeV2Manager>>(), _ => { });
    }

    private static Func<int, CancellationToken, Task> PublishingDelay(
        EyePipelineEventBus bus,
        RecordingPresenter presenter)
    {
        long ticks = DateTime.UnixEpoch.Ticks;
        var blinkSample = 0;
        return (_, token) =>
        {
            token.ThrowIfCancellationRequested();
            ticks += 33 * TimeSpan.TicksPerMillisecond;
            if (presenter.LastFrame is { } frame)
            {
                using var image = new Mat();
                bus.Publish(new EyePipelineEvents.NewRawExpressionsEvent(
                    image, RawFor(frame, ref blinkSample), ticks));
            }
            return Task.CompletedTask;
        };
    }

    private static float[] RawFor(VrCalibrationFrame frame, ref int blinkSample)
    {
        var leftX = 0f;
        var leftY = 0f;
        var rightX = 0f;
        var rightY = 0f;
        var leftOpen = 0.75f;
        var rightOpen = 0.70f;

        if (frame.Title == "BLINK SLOWLY THREE TIMES")
        {
            var closed = blinkSample++ % 8 is >= 2 and <= 4;
            if (closed) { leftOpen = 0.08f; rightOpen = 0.12f; }
        }
        else if (frame.Title == "SQUINT AND HOLD")
        {
            leftOpen = 0.45f;
            rightOpen = 0.42f;
        }
        else if (frame.Title == "OPEN YOUR EYES WIDE")
        {
            leftOpen = 0.95f;
            rightOpen = 0.91f;
        }
        else if (frame.TargetX is { } x && frame.TargetY is { } y)
        {
            leftX = 0.15f + x / 1.4f + 0.10f * y;
            leftY = -0.10f + y / 1.2f - 0.05f * x;
            rightX = -0.08f + x / 1.1f - 0.07f * y;
            rightY = 0.12f + y / 1.35f + 0.04f * x;
            if (y < 0) { leftOpen += 0.03f; rightOpen += 0.03f; }
        }

        return EyeV2CalibrationAndMapperTest.Raw(
            leftX, leftY, leftOpen, rightX, rightY, rightOpen);
    }

    private sealed class RecordingPresenter : IVrCalibrationPresenter
    {
        public bool IsAvailable => true;
        public bool IsPresenting { get; private set; }
        public bool IsHealthy { get; private set; } = true;
        public string Status => IsPresenting ? "recording" : "stopped";
        public List<VrCalibrationFrame> Frames { get; } = [];
        public VrCalibrationFrame? LastFrame { get; private set; }
        public VrCalibrationAction NextAction { get; set; }
        public string? CompletionMessage { get; private set; }
        public int FailOnPresentNumber { get; set; }

        public VrPresenterStartResult Begin(string sessionTitle)
        {
            IsPresenting = true;
            IsHealthy = true;
            return VrPresenterStartResult.Success();
        }

        public void Present(VrCalibrationFrame frame)
        {
            LastFrame = frame;
            Frames.Add(frame);
            if (FailOnPresentNumber > 0 && Frames.Count == FailOnPresentNumber)
            {
                IsHealthy = false;
                IsPresenting = false;
            }
        }

        public VrCalibrationAction ConsumeAction()
        {
            var action = NextAction;
            NextAction = VrCalibrationAction.None;
            return action;
        }

        public void End(string? completionMessage = null)
        {
            CompletionMessage = completionMessage;
            IsPresenting = false;
            IsHealthy = false;
        }

        public void Dispose() { }
    }
}
