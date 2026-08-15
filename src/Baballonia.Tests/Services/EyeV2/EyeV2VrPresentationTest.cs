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
            "the displayed routine total must match its actual 60 seconds");
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
            transitionSeconds: 1, leadInSeconds: 1);
        var routine = new GuidedCaptureRoutine(steps, () => clock, 1);

        // Reach the first hold.
        routine.Tick();
        clock += 2;
        routine.Tick();
        clock += 2;
        routine.Tick();
        Assert.AreEqual("hold", routine.Current!.Phase);
        var original = routine.CurrentAttempt!.Value;

        var replay = routine.RetryCurrentAttempt();
        Assert.IsNotNull(replay);
        Assert.AreEqual("transition", routine.Current!.Phase);
        Assert.AreEqual(original.Attempt + 1, routine.CurrentAttempt!.Value.Attempt);

        clock += 2;
        routine.Tick();
        Assert.AreEqual("hold", routine.Current!.Phase);
        var skipped = routine.CurrentAttempt!.Value;
        var next = routine.SkipCurrentAttempt();
        Assert.IsNotNull(next);
        Assert.AreNotEqual(skipped.Repetition, routine.CurrentAttempt!.Value.Repetition,
            "Skip must advance only past the current repetition, not end the whole routine.");
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
