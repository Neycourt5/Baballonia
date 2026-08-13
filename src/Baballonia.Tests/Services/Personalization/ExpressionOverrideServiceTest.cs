using System;
using System.Linq;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Exercises the calibration override state machine on a fake clock, so the safety timeouts are
/// tested deterministically instead of by sleeping.
///
/// The behavior that matters most here is the failure modes: a stuck override would leave the user's
/// avatar frozen mid-expression in a live social VR session.
/// </summary>
[TestClass]
[TestSubject(typeof(ExpressionOverrideService))]
public class ExpressionOverrideServiceTest
{
    private const int N = PersonalizationSchema.ExpressionCount;
    private const long TicksPerSecond = 1000; // 1 tick == 1 ms, keeps the arithmetic readable

    private long _now;
    private ExpressionOverrideService _service = null!;

    [TestInitialize]
    public void Initialize()
    {
        _now = 10_000;
        _service = new ExpressionOverrideService(() => _now, TicksPerSecond);
    }

    private void AdvanceMs(long ms) => _now += ms;

    private static float[] Vector(float value, params int[] dims)
    {
        var v = new float[N];
        foreach (var d in dims) v[d] = value;
        return v;
    }

    private CuePhase Hold(float level, int dim, double seconds = 2d) =>
        new("Cue", "hold", [dim], Vector(level, dim), Vector(level, dim), seconds, _now, level);

    private CuePhase Ramp(float from, float to, int dim, double seconds) =>
        new("Cue", "transition", [dim], Vector(from, dim), Vector(to, dim), seconds, _now);

    [TestMethod]
    public void Inactive_ReturnsNull_SoLiveTrackingIsUntouched()
    {
        Assert.IsNull(_service.SampleTarget());
        Assert.IsFalse(_service.IsActive);
    }

    [TestMethod]
    public void Activate_BeforeAnyPhase_ServesNeutral()
    {
        _service.Activate();

        var target = _service.SampleTarget();

        Assert.IsNotNull(target);
        Assert.AreEqual(N, target.Length);
        Assert.IsTrue(target.All(v => v == 0f), "Avatar should settle to rest before the first cue.");
    }

    [TestMethod]
    public void Hold_ServesConstantCommandedLevel()
    {
        _service.Activate();
        _service.PushPhase(Hold(0.75f, dim: 4));

        AdvanceMs(500);
        var mid = _service.SampleTarget()!;
        AdvanceMs(500);
        var later = _service.SampleTarget()!;

        Assert.AreEqual(0.75f, mid[4], 1e-6);
        Assert.AreEqual(0.75f, later[4], 1e-6);
        Assert.AreEqual(0f, mid[19], 1e-6, "Uncued dimensions stay at rest.");
    }

    [TestMethod]
    public void Transition_InterpolatesLinearlyAndClampsAtEnds()
    {
        _service.Activate();
        _service.PushPhase(Ramp(0f, 1f, dim: 4, seconds: 1d));

        Assert.AreEqual(0f, _service.SampleTarget()![4], 1e-6);

        AdvanceMs(250);
        Assert.AreEqual(0.25f, _service.SampleTarget()![4], 1e-3);

        AdvanceMs(250);
        Assert.AreEqual(0.5f, _service.SampleTarget()![4], 1e-3);

        AdvanceMs(500);
        Assert.AreEqual(1f, _service.SampleTarget()![4], 1e-3);
    }

    [TestMethod]
    public void KeepAliveExpiry_ReleasesToLiveTracking()
    {
        _service.Activate();
        _service.PushPhase(Hold(1f, dim: 4, seconds: 60d));

        AdvanceMs(900);
        _service.KeepAlive();
        AdvanceMs(900);
        Assert.IsNotNull(_service.SampleTarget(), "Still within the keep-alive window.");

        // Cue engine dies here: no further check-ins.
        AdvanceMs(1_100);

        Assert.IsNull(_service.SampleTarget(),
            "A hung cue engine must hand control back to live tracking rather than freeze the avatar.");
        Assert.IsFalse(_service.IsActive);
    }

    [TestMethod]
    public void PhaseOverrun_FallsBackToNeutralNotTheLastTarget()
    {
        _service.Activate();
        _service.PushPhase(Hold(1f, dim: 4, seconds: 1d));

        // Engine stalls but keeps checking in, so the deadman never fires. Advance past the phase
        // duration (1s) plus the overrun grace (2s).
        for (var i = 0; i < 8; i++)
        {
            AdvanceMs(500);
            _service.KeepAlive();
        }

        var target = _service.SampleTarget();

        Assert.IsNotNull(target, "Keep-alive is healthy, so the override stays engaged.");
        Assert.IsTrue(target.All(v => v == 0f),
            "Past the overrun grace the avatar must relax rather than hold a stale expression.");
    }

    [TestMethod]
    public void Deactivate_FlushesNeutralThenReleases()
    {
        _service.Activate();
        _service.PushPhase(Hold(1f, dim: 4));
        Assert.AreEqual(1f, _service.SampleTarget()![4], 1e-6);

        _service.Deactivate();

        var flushed = _service.SampleTarget();
        Assert.IsNotNull(flushed, "Deactivate must first drive the avatar to neutral...");
        Assert.IsTrue(flushed.All(v => v == 0f));

        AdvanceMs((long)ExpressionOverrideService.NeutralFlushDuration.TotalMilliseconds + 1);

        Assert.IsNull(_service.SampleTarget(), "...then hand control back to live tracking.");
        Assert.IsFalse(_service.IsActive);
    }

    [TestMethod]
    public void Deactivate_WhenInactive_IsHarmless()
    {
        _service.Deactivate();
        Assert.IsNull(_service.SampleTarget());
    }

    [TestMethod]
    public void Reactivate_AfterDeactivate_Works()
    {
        _service.Activate();
        _service.Deactivate();
        AdvanceMs(500);
        Assert.IsNull(_service.SampleTarget());

        _service.Activate();
        _service.PushPhase(Hold(0.5f, dim: 11));

        Assert.AreEqual(0.5f, _service.SampleTarget()![11], 1e-6);
    }

    [TestMethod]
    public void PushPhase_RejectsWrongLengthVectors()
    {
        _service.Activate();
        var bad = new CuePhase("Bad", "hold", [0], new float[3], new float[N], 1d, _now);

        Assert.ThrowsExactly<ArgumentException>(() => _service.PushPhase(bad));
    }

    [TestMethod]
    public void SampleTarget_ReturnsIndependentArrays()
    {
        _service.Activate();
        _service.PushPhase(Hold(0.5f, dim: 4));

        var first = _service.SampleTarget()!;
        first[4] = 999f;

        Assert.AreEqual(0.5f, _service.SampleTarget()![4], 1e-6,
            "Callers must not be able to corrupt subsequent samples.");
    }

    [TestMethod]
    public void CommandedValuesAreClampedToUnitRange()
    {
        _service.Activate();
        var over = new CuePhase("Over", "hold", [4], Vector(5f, 4), Vector(5f, 4), 1d, _now);
        _service.PushPhase(over);

        Assert.AreEqual(1f, _service.SampleTarget()![4], 1e-6);
    }
}
