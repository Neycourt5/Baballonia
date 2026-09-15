using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Services;
using Baballonia.Services.Inference;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services;

[TestClass]
public class CameraWatchdogServiceTest
{
    [TestMethod]
    public void BackoffIsOneTwoFourEightThenHoldsAtTenSeconds()
    {
        var delays = Enumerable.Range(1, 8)
            .Select(CameraWatchdogService.DelayFor)
            .Select(delay => delay.TotalSeconds)
            .ToArray();

        CollectionAssert.AreEqual(new[] { 1d, 2d, 4d, 8d, 10d, 10d, 10d, 10d }, delays);
        Assert.AreEqual(CameraWatchdogService.Backoff[0], CameraWatchdogService.DelayFor(0));
        Assert.AreEqual(CameraWatchdogService.Backoff[0], CameraWatchdogService.DelayFor(-1));
    }

    [TestMethod]
    public void TimingConstantsPreserveStartGraceAndFastObservation()
    {
        Assert.IsTrue(CameraWatchdogService.StartGrace > CameraWatchdogService.StallTimeout);
        Assert.IsTrue(CameraWatchdogService.TickInterval < CameraWatchdogService.StallTimeout / 4);
        Assert.IsTrue(CameraWatchdogService.PresencePollInterval < CameraWatchdogService.StallTimeout);
        Assert.IsTrue(CameraWatchdogService.FirstFrameTimeout <
                      SingleCameraSourceFactory.DefaultFirstFrameTimeout);
        Assert.IsTrue(CameraWatchdogService.FirstFrameTimeout <= CameraWatchdogService.Backoff[^1]);
    }

    [TestMethod]
    public async Task ReconnectingSlotGetsAnImmediateAttemptAndCountsRecovery()
    {
        var slot = FakeSlot.Reconnecting(recover: true);
        var watchdog = Create(slot);

        await watchdog.TickAsync();

        Assert.AreEqual(1, slot.RecoverCalls);
        Assert.AreEqual(CameraWatchdogService.FirstFrameTimeout, slot.LastTimeout);
        Assert.AreEqual(1, watchdog.ReconnectCount);
        Assert.AreEqual(CameraState.Running, slot.State);
    }

    [TestMethod]
    public async Task FailedAttemptIsBackedOffInsteadOfRepeatedEveryTick()
    {
        var slot = FakeSlot.Reconnecting(recover: false);
        var watchdog = Create(slot);

        await watchdog.TickAsync();
        await watchdog.TickAsync();

        Assert.AreEqual(1, slot.RecoverCalls);
        Assert.AreEqual(0, watchdog.ReconnectCount);
    }

    [TestMethod]
    public async Task StalledRunningSlotRecoversButStartupGraceDoesNotTrip()
    {
        var stalled = FakeSlot.Running(
            installed: DateTime.UtcNow - CameraWatchdogService.StartGrace - TimeSpan.FromSeconds(1),
            age: CameraWatchdogService.StallTimeout + TimeSpan.FromSeconds(1));
        var startingNormally = FakeSlot.Running(
            installed: DateTime.UtcNow,
            age: CameraWatchdogService.StallTimeout + TimeSpan.FromSeconds(1));
        var watchdog = Create(stalled, startingNormally);

        await watchdog.TickAsync();

        Assert.AreEqual(1, stalled.RecoverCalls);
        Assert.AreEqual(0, startingNormally.RecoverCalls);
    }

    [TestMethod]
    public async Task RemovedDeviceRecoversEvenWhenBackendStillReportsFreshFrames()
    {
        var removed = FakeSlot.Running(
            installed: DateTime.UtcNow - CameraWatchdogService.StartGrace - TimeSpan.FromSeconds(1),
            age: TimeSpan.Zero);
        removed.IsPresent = false;
        var watchdog = Create(removed);

        await watchdog.TickAsync();

        Assert.AreEqual(1, removed.RecoverCalls);
        Assert.AreEqual(CameraState.Reconnecting, removed.State);
    }

    [TestMethod]
    public async Task StoppedSlotIsNeverResurrected()
    {
        var slot = FakeSlot.Reconnecting(recover: true);
        slot.Target = null;
        slot.State = CameraState.Stopped;
        var watchdog = Create(slot);

        await watchdog.TickAsync();

        Assert.AreEqual(0, slot.RecoverCalls);
    }

    [TestMethod]
    public async Task CollapsedTopologyCannotRecoverTheSamePhysicalSlotTwice()
    {
        var shared = FakeSlot.Reconnecting(recover: true);
        var host = new FakeHost { CurrentSlots = [shared, shared] };
        var watchdog = Create(host);

        await watchdog.TickAsync();

        Assert.AreEqual(1, shared.RecoverCalls);
    }

    [TestMethod]
    public async Task TopologyRemovalDropsOldBackoffState()
    {
        var old = FakeSlot.Reconnecting(recover: false);
        var replacement = FakeSlot.Reconnecting(recover: false);
        var host = new FakeHost { CurrentSlots = [old] };
        var watchdog = Create(host);

        await watchdog.TickAsync();
        host.CurrentSlots = [replacement];
        await watchdog.TickAsync();
        host.CurrentSlots = [old];
        await watchdog.TickAsync();

        Assert.AreEqual(2, old.RecoverCalls,
            "re-adding a topology slot should not retain its obsolete retry delay");
    }

    private static CameraWatchdogService Create(params FakeSlot[] slots) =>
        Create(new FakeHost { CurrentSlots = slots });

    private static CameraWatchdogService Create(FakeHost host) =>
        new([host], NullLogger<CameraWatchdogService>.Instance);

    private sealed class FakeHost : ICameraSlotHost
    {
        public IReadOnlyList<IRecoverableCameraSlot> CurrentSlots { get; set; } = [];
        public IReadOnlyList<IRecoverableCameraSlot> Slots => CurrentSlots;
    }

    private sealed class FakeSlot : IRecoverableCameraSlot
    {
        private readonly bool _recover;

        private FakeSlot(bool recover) => _recover = recover;

        public string Name { get; init; } = "Test";
        public CameraTarget? Target { get; set; } = new("camera", "backend");
        public CameraState State { get; set; }
        public event Action<CameraState>? StateChanged;
        public DateTime? SourceInstalledAtUtc { get; set; }
        public TimeSpan? TimeSinceLastFrame { get; set; }
        public bool IsPresent { get; set; } = true;
        public int RecoverCalls { get; private set; }
        public TimeSpan LastTimeout { get; private set; }

        public static FakeSlot Reconnecting(bool recover) => new(recover)
        {
            State = CameraState.Reconnecting,
        };

        public static FakeSlot Running(DateTime installed, TimeSpan age) => new(recover: true)
        {
            State = CameraState.Running,
            SourceInstalledAtUtc = installed,
            TimeSinceLastFrame = age,
        };

        public bool IsTargetPresent(string address) => IsPresent;

        public Task<bool> RecoverAsync(TimeSpan firstFrameTimeout, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            RecoverCalls++;
            LastTimeout = firstFrameTimeout;
            if (_recover && IsPresent)
            {
                State = CameraState.Running;
                SourceInstalledAtUtc = DateTime.UtcNow;
                TimeSinceLastFrame = TimeSpan.Zero;
                StateChanged?.Invoke(State);
            }
            else
            {
                State = CameraState.Reconnecting;
                SourceInstalledAtUtc = null;
            }
            return Task.FromResult(_recover);
        }
    }
}
