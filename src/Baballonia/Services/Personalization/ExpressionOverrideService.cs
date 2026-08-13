using System;
using System.Diagnostics;
using System.Threading;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Drives the VRChat avatar to independently generated target expressions during guided
/// calibration, by temporarily replacing the face values Baballonia sends downstream.
///
/// Targets travel Baballonia's normal path (OSC -> VRCFT module -> UnifiedExpressions -> avatar),
/// so no avatar-specific parameter encoding is duplicated here. Crucially the targets originate
/// from the cue engine, never from inference, which is what keeps the resulting training data
/// non-circular: the avatar shows a commanded expression, the user imitates it, and the commanded
/// vector becomes the label while the camera records the user's real face.
///
/// Threading: the cue engine (UI thread) pushes immutable <see cref="CuePhase"/> snapshots;
/// <see cref="SampleTarget"/> is pulled from the sender loop (background thread) and interpolates
/// at call time. Sampling on the transmitting clock keeps avatar motion smooth regardless of UI
/// tick jitter, and means sent values and recorded labels derive from one source.
///
/// Safety: an override that gets stuck would leave the avatar frozen mid-expression. Three
/// independent layers prevent that - a keep-alive deadman, a per-phase overrun cap, and a neutral
/// flush on deactivate. See <see cref="SampleTarget"/>.
/// </summary>
public sealed class ExpressionOverrideService : IExpressionOverrideSource
{
    /// <summary>Stop overriding if the cue engine stops checking in (hung UI, torn-down page).</summary>
    public static readonly TimeSpan KeepAliveTimeout = TimeSpan.FromSeconds(1);

    /// <summary>Fall back to neutral if a phase runs far past its declared duration.</summary>
    public static readonly TimeSpan PhaseOverrunGrace = TimeSpan.FromSeconds(2);

    /// <summary>Hold neutral briefly on deactivate so the avatar relaxes instead of freezing.</summary>
    public static readonly TimeSpan NeutralFlushDuration = TimeSpan.FromMilliseconds(200);

    private readonly Func<long> _now;
    private readonly double _ticksPerSecond;

    private volatile CuePhase? _phase;
    private volatile bool _active;

    private long _lastKeepAlive;
    private long _flushUntil;

    public ExpressionOverrideService() : this(Stopwatch.GetTimestamp, Stopwatch.Frequency)
    {
    }

    /// <summary>Test seam: inject a deterministic clock so the safety timers can be exercised.</summary>
    public ExpressionOverrideService(Func<long> timestampProvider, double ticksPerSecond)
    {
        _now = timestampProvider;
        _ticksPerSecond = ticksPerSecond;
    }

    /// <summary>True while calibration is commanding the avatar.</summary>
    public bool IsActive => _active;

    /// <summary>
    /// Begins overriding. Until the first <see cref="PushPhase"/> the target is neutral, so the
    /// avatar settles to a rest face rather than jumping.
    /// </summary>
    public void Activate()
    {
        Interlocked.Exchange(ref _flushUntil, 0);
        Interlocked.Exchange(ref _lastKeepAlive, _now());
        _phase = null;
        _active = true;
    }

    /// <summary>Publishes the phase the avatar should now be animating. Cheap; no allocation beyond the record.</summary>
    public void PushPhase(CuePhase phase)
    {
        ArgumentNullException.ThrowIfNull(phase);
        phase.Validate();

        Interlocked.Exchange(ref _lastKeepAlive, _now());
        _phase = phase;
    }

    /// <summary>
    /// Deadman check-in. The cue engine must call this regularly (every UI tick) for the override to
    /// keep applying; if it stops, live tracking resumes automatically.
    /// </summary>
    public void KeepAlive() => Interlocked.Exchange(ref _lastKeepAlive, _now());

    /// <summary>
    /// Ends the override. Serves neutral for <see cref="NeutralFlushDuration"/> first so the avatar
    /// visibly relaxes, then hands control back to live tracking. Safe to call repeatedly, and safe
    /// to call from an exception handler or page teardown.
    /// </summary>
    public void Deactivate()
    {
        if (!_active)
            return;

        _phase = null;
        Interlocked.Exchange(ref _flushUntil, _now() + (long)(NeutralFlushDuration.TotalSeconds * _ticksPerSecond));
    }

    /// <inheritdoc />
    public float[]? SampleTarget()
    {
        if (!_active)
            return null;

        var now = _now();

        // Layer 3: neutral flush after Deactivate, then release control.
        var flushUntil = Interlocked.Read(ref _flushUntil);
        if (flushUntil != 0)
        {
            if (now < flushUntil)
                return Neutral();

            _active = false;
            Interlocked.Exchange(ref _flushUntil, 0);
            return null;
        }

        // Layer 1: deadman. Releasing to live tracking is the safer failure here - a frozen avatar
        // is worse than tracking resuming mid-session, and the session is already broken.
        if (SecondsSince(Interlocked.Read(ref _lastKeepAlive), now) > KeepAliveTimeout.TotalSeconds)
        {
            _active = false;
            _phase = null;
            return null;
        }

        var phase = _phase;
        if (phase == null)
            return Neutral();

        var elapsed = SecondsSince(phase.StartTimestamp, now);

        // Layer 2: the engine stalled without dying. Park on neutral rather than the last target.
        if (elapsed > phase.DurationSeconds + PhaseOverrunGrace.TotalSeconds)
            return Neutral();

        return Interpolate(phase, elapsed);
    }

    private double SecondsSince(long start, long now) => (now - start) / _ticksPerSecond;

    private static float[] Neutral() => new float[PersonalizationSchema.ExpressionCount];

    private static float[] Interpolate(CuePhase phase, double elapsedSeconds)
    {
        var t = phase.DurationSeconds <= 0
            ? 1d
            : Math.Clamp(elapsedSeconds / phase.DurationSeconds, 0d, 1d);

        var result = new float[PersonalizationSchema.ExpressionCount];
        for (var i = 0; i < result.Length; i++)
        {
            var from = phase.FromVector[i];
            var value = from + (phase.ToVector[i] - from) * (float)t;
            result[i] = Math.Clamp(value, 0f, 1f);
        }

        return result;
    }
}
