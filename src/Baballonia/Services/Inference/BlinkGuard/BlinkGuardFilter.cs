using System;

namespace Baballonia.Services.Inference.BlinkGuard;

/// <summary>
/// BlinkGuard: suppresses the gaze snap that follows a blink, and nothing else.
/// </summary>
/// <remarks>
/// <para>Owns one <see cref="BlinkGuardEye"/> per eye, because the two eyes genuinely do blink
/// independently - a wink must not drag the open eye into a hold - and keeps the counters and the
/// optional capture ring.</para>
///
/// <para><strong>Where this must and must not run.</strong> It is applied inside
/// <c>EyeOutputPostProcessor</c>, which the eye pipeline invokes <em>after</em> snapshotting
/// <c>RawEyeResult</c> for the native/DFR path. That ordering is what keeps foveated rendering on an
/// unmodified, minimum-latency stream while the social/VRCFT gaze gets the treatment. The exclusion
/// is therefore structural rather than a convention someone can forget.</para>
///
/// <para><strong>On confidence.</strong> The vendor VRCFT modules publish a confidence value in
/// shared memory; this application does not consume those - it runs its own model over the eye
/// cameras, including the Beyond's, and produces no confidence channel. Every default here works
/// from validity and temporal consistency alone. <see cref="GazeSample.Confidence"/> and
/// <see cref="BlinkGuardSettings.UseConfidence"/> exist so a source that does supply one can be
/// wired in without changing a signature.</para>
/// </remarks>
public sealed class BlinkGuardFilter
{
    private readonly BlinkGuardEye _left;
    private readonly BlinkGuardEye _right;
    private readonly object _gate = new();

    private BlinkGuardSettings _settings;
    private double _clock;

    public BlinkGuardFilter(BlinkGuardSettings? settings = null, BlinkGuardDiagnostics? diagnostics = null)
    {
        _settings = (settings ?? BlinkGuardSettings.Balanced).Sanitized();
        Diagnostics = diagnostics ?? new BlinkGuardDiagnostics();
        _left = new BlinkGuardEye(_settings);
        _right = new BlinkGuardEye(_settings);
    }

    public BlinkGuardDiagnostics Diagnostics { get; }

    public BlinkGuardSettings Settings
    {
        get { lock (_gate) return _settings; }
    }

    public BlinkGuardState LeftState => _left.State;
    public BlinkGuardState RightState => _right.State;
    public double LeftReacquiringSeconds => _left.ReacquiringSeconds;
    public double RightReacquiringSeconds => _right.ReacquiringSeconds;

    /// <summary>Whether either eye is currently not passing gaze straight through.</summary>
    public bool IsIntervening =>
        _left.State is not BlinkGuardState.Normal || _right.State is not BlinkGuardState.Normal;

    public void UpdateSettings(BlinkGuardSettings settings)
    {
        var sanitized = settings.Sanitized();
        lock (_gate)
        {
            _settings = sanitized;
            _left.UpdateSettings(sanitized);
            _right.UpdateSettings(sanitized);
        }
    }

    public void Reset()
    {
        lock (_gate)
        {
            _left.Reset();
            _right.Reset();
            _clock = 0;
        }
    }

    /// <summary>
    /// Filters one frame. Returns the gaze that should be published for social tracking.
    /// </summary>
    /// <param name="left">Left eye's raw gaze and eyelid state.</param>
    /// <param name="right">Right eye's raw gaze and eyelid state.</param>
    /// <param name="dt">Seconds since the previous frame. Real elapsed time, never an assumed rate.</param>
    public (GazeResult Left, GazeResult Right) Process(in GazeSample left, in GazeSample right, double dt)
    {
        BlinkGuardSettings settings;
        lock (_gate) settings = _settings;

        if (!settings.Enabled)
        {
            return (new GazeResult(left.X, left.Y, BlinkGuardState.Normal),
                    new GazeResult(right.X, right.Y, BlinkGuardState.Normal));
        }

        if (!double.IsFinite(dt) || dt < 0)
            dt = 0;

        BlinkGuardStep leftStep, rightStep;
        lock (_gate)
        {
            _clock += dt;
            leftStep = _left.Process(left, dt);
            rightStep = _right.Process(right, dt);
        }

        Account(leftStep);
        Account(rightStep);

        Diagnostics.Add(new BlinkGuardRecord(
            _clock,
            left.X, left.Y, leftStep.X, leftStep.Y,
            left.Closedness, left.Valid, left.Confidence,
            leftStep.State, leftStep.Rejection, leftStep.CandidateCount,
            leftStep.DegreesFromPreviousRaw, leftStep.DegreesFromPreviousOutput,
            right.X, right.Y, rightStep.X, rightStep.Y,
            right.Closedness, right.Valid, right.Confidence,
            rightStep.State, rightStep.Rejection, rightStep.CandidateCount,
            rightStep.DegreesFromPreviousRaw, rightStep.DegreesFromPreviousOutput));

        return (new GazeResult(leftStep.X, leftStep.Y, leftStep.State),
                new GazeResult(rightStep.X, rightStep.Y, rightStep.State));
    }

    private void Account(in BlinkGuardStep step)
    {
        var counters = Diagnostics.Counters;

        if (step.BlinkStarted)
            counters.BlinksDetected++;

        if (step.Rejection is not BlinkGuardRejection.None and not BlinkGuardRejection.StillClosed)
            counters.SamplesRejected++;

        // Counted on the size of the suppression rather than on the rejection reason, because the
        // worst case is a sample that is perfectly valid and simply wrong - it is never "rejected",
        // it just loses to the held position. DegreesFromPreviousOutput is the gap between what the
        // tracker reported this frame and what was actually published, so it is exactly the flick
        // the avatar did not make.
        //
        // "Prevented" is reserved for gaps big enough to have been visible. A sample a few degrees
        // out is jitter nobody would have noticed; counting those would make the number look
        // impressive and mean nothing.
        if (step.State is BlinkGuardState.BlinkHold or BlinkGuardState.Reacquiring &&
            step.DegreesFromPreviousOutput >= LargeGlitchDegrees)
        {
            counters.GlitchesPrevented++;
        }

        if (step.ReacquisitionSucceeded || step.ReacquisitionTimedOut)
        {
            counters.ReacquisitionCount++;
            counters.ReacquisitionTotalSeconds += step.ReacquisitionSeconds;
            counters.ReacquisitionMaxSeconds =
                Math.Max(counters.ReacquisitionMaxSeconds, step.ReacquisitionSeconds);
        }

        if (step.ReacquisitionTimedOut)
            counters.TimeoutCount++;
    }

    /// <summary>A suppressed jump has to be this big before it counts as a glitch prevented.</summary>
    public const float LargeGlitchDegrees = 10f;
}

/// <summary>The published gaze for one eye, plus the state that produced it.</summary>
public readonly record struct GazeResult(float X, float Y, BlinkGuardState State);
