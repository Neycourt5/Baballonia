using System;
using System.Diagnostics;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Supplies the recorder with the commanded target at the moment each frame was captured.
///
/// This is the other half of the guided-calibration loop. <see cref="ExpressionOverrideService"/>
/// sends the commanded vector to the avatar; this reports the same vector to the dataset recorder,
/// so what the user was looking at and what the trainer is told are the same number by construction
/// rather than by two code paths agreeing.
///
/// Both read the same immutable <see cref="CuePhase"/> snapshot and interpolate it at call time, on
/// their own clocks. There is deliberately no shared mutable "current target" that one could update
/// and the other read stale - the snapshot is swapped whole, and interpolation is a pure function of
/// it and the time.
/// </summary>
public sealed class CueStateSource : IReadOnlyCueStateSource
{
    private readonly Func<long> _now;
    private readonly double _ticksPerSecond;

    private volatile CuePhase? _phase;

    /// <summary>"avatar" when the VRChat avatar is the visual teacher; "bar" for the on-screen fallback.</summary>
    public string Source { get; set; } = "avatar";

    public CueStateSource() : this(Stopwatch.GetTimestamp, Stopwatch.Frequency)
    {
    }

    /// <summary>Test seam: a deterministic clock, matching ExpressionOverrideService's.</summary>
    public CueStateSource(Func<long> clock, double ticksPerSecond)
    {
        _now = clock;
        _ticksPerSecond = ticksPerSecond;
    }

    /// <summary>Publishes the phase now being commanded. Null means "not calibrating".</summary>
    public void SetPhase(CuePhase? phase)
    {
        phase?.Validate();
        _phase = phase;
    }

    /// <summary>Stops stamping cues onto frames. Recording itself is unaffected.</summary>
    public void Clear() => _phase = null;

    /// <inheritdoc />
    public FrameLabel.CueLabel? CurrentCue()
    {
        var phase = _phase;
        if (phase is null)
            return null;

        return new FrameLabel.CueLabel
        {
            Id = phase.CueId,
            Phase = phase.PhaseName,
            Dims = phase.Dims,
            Target = Interpolate(phase),
            Level = phase.Level,
            Repetition = phase.RepetitionIndex,
            Source = Source
        };
    }

    /// <summary>
    /// The commanded vector at this instant, linearly across the phase.
    /// </summary>
    /// <remarks>
    /// Deliberately the same arithmetic as <see cref="ExpressionOverrideService"/>. During a hold the
    /// endpoints are equal so this is a constant, which is the case that carries the label weight -
    /// transitions are masked out by the labeller precisely because a moving target does not describe
    /// a moving face well.
    /// </remarks>
    private float[] Interpolate(CuePhase phase)
    {
        var count = PersonalizationSchema.ExpressionCount;
        var result = new float[count];

        var progress = phase.DurationSeconds <= 0
            ? 1.0
            : Math.Clamp((_now() - phase.StartTimestamp) / _ticksPerSecond / phase.DurationSeconds, 0, 1);

        for (var i = 0; i < count; i++)
        {
            var value = phase.FromVector[i] + (phase.ToVector[i] - phase.FromVector[i]) * (float)progress;
            result[i] = Math.Clamp(value, 0f, 1f);
        }

        return result;
    }
}
