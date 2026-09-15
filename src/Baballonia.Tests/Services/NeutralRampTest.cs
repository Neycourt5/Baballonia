using System;
using System.Collections.Generic;
using Baballonia.Services;
using Baballonia.Services.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services;

[TestClass]
public class NeutralRampTest
{
    private DateTime _now;

    [TestInitialize]
    public void Initialize() => _now = new DateTime(2026, 8, 29, 0, 0, 0, DateTimeKind.Utc);

    [TestMethod]
    public void FaceStartsAtLastSentAndArrivesAtExactZeroOnce()
    {
        var ramp = new NeutralExpressionRamp(_ => 0f, utcNow: () => _now);
        var expressive = Map(("/jawOpen", 0.8f), ("/mouthSmileLeft", 0.4f));
        ramp.Observe(expressive);

        var start = ramp.Sample(reconnecting: true)!;
        Assert.AreEqual(0.8f, start["/jawOpen"], 1e-6f);
        Assert.AreEqual(0.4f, start["/mouthSmileLeft"], 1e-6f);

        _now += TimeSpan.FromMilliseconds(250);
        var midpoint = ramp.Sample(reconnecting: true)!;
        Assert.AreEqual(0.4f, midpoint["/jawOpen"], 1e-6f);
        Assert.AreEqual(0.2f, midpoint["/mouthSmileLeft"], 1e-6f);

        _now += TimeSpan.FromMilliseconds(250);
        var neutral = ramp.Sample(reconnecting: true)!;
        Assert.AreEqual(0f, neutral["/jawOpen"]);
        Assert.AreEqual(0f, neutral["/mouthSmileLeft"]);
        Assert.IsNull(ramp.Sample(reconnecting: true),
            "the endpoint is emitted once, then downstream holds neutral");
    }

    [TestMethod]
    public void EyeNeutralOpensLidsAndZerosEveryOtherKnownChannel()
    {
        var from = Map(
            ("/leftEyeX", -0.7f),
            ("/leftEyeLid", 0.1f),
            ("/rightEyeLid", 0.3f),
            ("/rightEyeSquint", 0.9f));

        var neutral = NeutralExpressionRamp.Interpolate(
            from,
            key => key is "/leftEyeLid" or "/rightEyeLid" ? 1f : 0f,
            progress: 1d);

        Assert.AreEqual(0f, neutral["/leftEyeX"]);
        Assert.AreEqual(1f, neutral["/leftEyeLid"]);
        Assert.AreEqual(1f, neutral["/rightEyeLid"]);
        Assert.AreEqual(0f, neutral["/rightEyeSquint"]);
    }

    [TestMethod]
    public void RampSnapshotsMutableRunnerOutputAndDoesNothingForAUserStop()
    {
        var ramp = new NeutralExpressionRamp(_ => 0f, utcNow: () => _now);
        var reused = Map(("/jawOpen", 0.75f));
        ramp.Observe(reused);
        reused["/jawOpen"] = 0.1f;

        Assert.IsNull(ramp.Sample(reconnecting: false));
        Assert.AreEqual(0.75f, ramp.Sample(reconnecting: true)!["/jawOpen"], 1e-6f);
    }

    [TestMethod]
    public void NoPriorOutputMeansThereIsNothingToRelax()
    {
        var ramp = new NeutralExpressionRamp(_ => 0f, utcNow: () => _now);

        Assert.IsNull(ramp.Sample(reconnecting: true));
        Assert.IsNull(ramp.Sample(reconnecting: true));
    }

    [TestMethod]
    public void EyeOutageRequiresEveryTargetedSlotToBeReconnecting()
    {
        var left = new FakeSlot { State = CameraState.Reconnecting };
        var right = new FakeSlot { State = CameraState.Running };

        Assert.IsFalse(EyePipelineManager.AllTargetedSlotsReconnecting([left, right]),
            "one live eye must prevent total-outage neutral decay");

        right.State = CameraState.Reconnecting;
        Assert.IsTrue(EyePipelineManager.AllTargetedSlotsReconnecting([left, right]));

        right.Target = null;
        Assert.IsTrue(EyePipelineManager.AllTargetedSlotsReconnecting([left, right]),
            "untargeted slots do not represent a missing requested camera");
    }

    private static OrderedFloatMap Map(params (string Key, float Value)[] values)
    {
        var map = new OrderedFloatMap(Array.ConvertAll(values, value => value.Key));
        foreach (var value in values)
            map[value.Key] = value.Value;
        return map;
    }

    private sealed class FakeSlot : IRecoverableCameraSlot
    {
        public string Name => "eye";
        public CameraTarget? Target { get; set; } = new("camera", "backend");
        public CameraState State { get; set; }
        public event Action<CameraState>? StateChanged
        {
            add { }
            remove { }
        }
        public DateTime? SourceInstalledAtUtc => null;
        public TimeSpan? TimeSinceLastFrame => null;
        public bool IsTargetPresent(string address) => true;
        public System.Threading.Tasks.Task<bool> RecoverAsync(
            TimeSpan firstFrameTimeout,
            System.Threading.CancellationToken cancellationToken) =>
            System.Threading.Tasks.Task.FromResult(false);
    }
}
