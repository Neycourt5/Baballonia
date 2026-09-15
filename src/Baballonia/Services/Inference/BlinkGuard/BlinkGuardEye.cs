using System;

namespace Baballonia.Services.Inference.BlinkGuard;

public enum BlinkGuardState
{
    /// <summary>Tracking is trusted and passed through untouched.</summary>
    Normal,

    /// <summary>The lid is down. Output is held; incoming gaze is not believed.</summary>
    BlinkHold,

    /// <summary>The lid is up but the gaze has not yet earned trust.</summary>
    Reacquiring,

    /// <summary>A trustworthy gaze was found and is being eased into.</summary>
    Transition,
}

/// <summary>Why a post-blink sample was not counted toward reacquisition.</summary>
public enum BlinkGuardRejection
{
    None,
    NotFinite,
    Invalid,
    LowConfidence,
    Disagreed,
    StillClosed,
}

/// <summary>What one eye did on one frame, for diagnostics and the UI.</summary>
public readonly record struct BlinkGuardStep(
    float X,
    float Y,
    BlinkGuardState State,
    BlinkGuardRejection Rejection,
    int CandidateCount,
    float DegreesFromPreviousRaw,
    float DegreesFromPreviousOutput,
    bool BlinkStarted,
    bool BlinkEnded,
    bool ReacquisitionSucceeded,
    bool ReacquisitionTimedOut,
    float ReacquisitionSeconds);

/// <summary>
/// The per-eye blink/reacquisition state machine.
/// </summary>
/// <remarks>
/// <para>The failure this exists for: the lid crosses the pupil, the tracker's gaze estimate becomes
/// meaningless for a few frames, and on reopening the very first estimate is accepted verbatim - so
/// the avatar's eye snaps somewhere absurd and corrects a few frames later. Small jitter is
/// forgivable; a 30° flick is not, because it reads as the avatar looking at something.</para>
///
/// <para>The design constraint that shapes everything: it must <em>not</em> be a smoother. Ordinary
/// saccades stay untouched and instant, because in <see cref="BlinkGuardState.Normal"/> the input is
/// passed through unchanged. Every intervention is confined to the window between the lid closing
/// and gaze earning trust again.</para>
///
/// <para>The subtle requirement is that people move their eyes <em>during</em> blinks. So this must
/// never drag the gaze back to where it was before the blink; it only refuses to believe the first
/// sample. Once several samples agree with each other - however far that is from the pre-blink
/// position - that is the new truth and it is accepted.</para>
///
/// <para>Deterministic and clock-free: every decision is driven by the caller's timestamps, so the
/// whole thing is testable by feeding synthetic samples.</para>
/// </remarks>
public sealed class BlinkGuardEye
{
    private readonly GazeHistory _recent;
    private readonly GazeHistory _candidates;

    private BlinkGuardSettings _settings;

    private float _outputX, _outputY;
    private float _heldX, _heldY;
    private float _transitionFromX, _transitionFromY;
    private float _previousRawX, _previousRawY;
    private bool _hasPrevious;

    private double _reacquiringSeconds;
    private double _transitionSeconds;
    private int _agreeingRun;

    // One-frame veto state for the optional normal-mode guard.
    private bool _spikeHeld;
    private float _spikeCandidateX, _spikeCandidateY;

    public BlinkGuardEye(BlinkGuardSettings settings)
    {
        _settings = settings.Sanitized();
        _recent = new GazeHistory(Math.Max(8, _settings.StableWindowSamples));
        _candidates = new GazeHistory(16);
    }

    public BlinkGuardState State { get; private set; } = BlinkGuardState.Normal;

    /// <summary>Seconds spent in the current reacquisition, for the UI's live timer.</summary>
    public double ReacquiringSeconds => _reacquiringSeconds;

    /// <summary>
    /// Coefficients only. Deliberately does not reset the state machine: the settings screen writes
    /// on every keystroke, and rebuilding mid-blink would strand the eye in a held position.
    /// </summary>
    public void UpdateSettings(BlinkGuardSettings settings) => _settings = settings.Sanitized();

    public void Reset()
    {
        State = BlinkGuardState.Normal;
        _recent.Clear();
        _candidates.Clear();
        _hasPrevious = false;
        _agreeingRun = 0;
        _reacquiringSeconds = 0;
        _transitionSeconds = 0;
        _spikeHeld = false;
    }

    /// <summary>Advances one frame and returns the gaze that should be published.</summary>
    /// <param name="sample">This frame's raw gaze and eyelid state.</param>
    /// <param name="dt">Seconds since the previous call. Non-finite or negative is treated as 0.</param>
    public BlinkGuardStep Process(in GazeSample sample, double dt)
    {
        if (!double.IsFinite(dt) || dt < 0)
            dt = 0;

        // A broken sample can never become the output, in any state. The finite guard upstream owns
        // substituting something sane; here it simply must not poison the history or the hold.
        if (!sample.IsFinite)
        {
            return Step(BlinkGuardRejection.NotFinite, degreesFromRaw: 0f);
        }

        var degreesFromRaw = _hasPrevious ? sample.DegreesTo(_previousRawX, _previousRawY) : 0f;
        _previousRawX = sample.X;
        _previousRawY = sample.Y;
        _hasPrevious = true;

        var closed = sample.Closedness >= _settings.ClosedThreshold;
        var open = sample.Closedness <= _settings.ReopenThreshold;

        // Closing wins from any state. A second blink during a reacquisition is common - people
        // flutter - and must restart the cycle rather than race it.
        if (closed && State is not BlinkGuardState.BlinkHold)
        {
            EnterBlinkHold();
            return Step(BlinkGuardRejection.StillClosed, degreesFromRaw, blinkStarted: true);
        }

        switch (State)
        {
            case BlinkGuardState.Normal:
                return Normal(sample, degreesFromRaw);

            case BlinkGuardState.BlinkHold:
                if (!open)
                    return Step(BlinkGuardRejection.StillClosed, degreesFromRaw);

                State = BlinkGuardState.Reacquiring;
                _reacquiringSeconds = 0;
                _candidates.Clear();
                _agreeingRun = 0;
                return Step(BlinkGuardRejection.None, degreesFromRaw, blinkEnded: true);

            case BlinkGuardState.Reacquiring:
                return Reacquiring(sample, dt, degreesFromRaw);

            case BlinkGuardState.Transition:
                return Transition(sample, dt, degreesFromRaw);

            default:
                return Normal(sample, degreesFromRaw);
        }
    }

    private BlinkGuardStep Normal(in GazeSample sample, float degreesFromRaw)
    {
        _recent.Add(sample);

        var x = sample.X;
        var y = sample.Y;

        // Optional, off by default. One frame of veto costs a real saccade a single frame and kills
        // an impossible one-frame flick outright.
        if (_settings.NormalSpikeGuardEnabled && _recent.Count > 1)
        {
            var jump = sample.DegreesTo(_outputX, _outputY);
            if (!_spikeHeld && jump > _settings.NormalSpikeGuardDegrees)
            {
                _spikeHeld = true;
                _spikeCandidateX = sample.X;
                _spikeCandidateY = sample.Y;
                return Step(BlinkGuardRejection.Disagreed, degreesFromRaw);
            }

            if (_spikeHeld)
            {
                _spikeHeld = false;
                // Confirmed by a second frame near the first: it was a real movement after all.
                if (sample.DegreesTo(_spikeCandidateX, _spikeCandidateY) >
                    _settings.StabilityToleranceDegrees)
                {
                    return Step(BlinkGuardRejection.Disagreed, degreesFromRaw);
                }
            }
        }

        _outputX = x;
        _outputY = y;
        return Step(BlinkGuardRejection.None, degreesFromRaw);
    }

    private void EnterBlinkHold()
    {
        State = BlinkGuardState.BlinkHold;

        // The position to hold is the median of the frames just before the lid arrived, not the last
        // one - by the time closedness crosses the threshold the lid is already over the pupil and
        // the newest samples are the least trustworthy of the lot.
        if (_recent.TryMedian(_settings.StableWindowSamples, out var mx, out var my))
        {
            _heldX = mx;
            _heldY = my;
        }
        else
        {
            _heldX = _outputX;
            _heldY = _outputY;
        }

        _outputX = _heldX;
        _outputY = _heldY;
        _candidates.Clear();
        _agreeingRun = 0;
        _spikeHeld = false;
    }

    private BlinkGuardStep Reacquiring(in GazeSample sample, double dt, float degreesFromRaw)
    {
        _reacquiringSeconds += dt;

        // Output stays where it was held. The eyelid is barely off the pupil; whatever the tracker
        // reports now has not earned the avatar's eye yet.
        _outputX = _heldX;
        _outputY = _heldY;

        var rejection = Evaluate(sample);
        if (rejection is BlinkGuardRejection.None)
        {
            // Agreement is with the run so far, not with the pre-blink position - which is what lets
            // a genuinely new gaze direction be accepted. A sample that breaks the run still seeds
            // the next one, so a sustained move to a new position wins after RequiredStableSamples;
            // it is only reported as a disagreement so the diagnostics say why trust was reset.
            if (_candidates.Count > 0 &&
                sample.DegreesTo(_candidates[0]) > _settings.StabilityToleranceDegrees)
            {
                _agreeingRun = 0;
                _candidates.Clear();
                rejection = BlinkGuardRejection.Disagreed;
            }

            _candidates.Add(sample);
            _agreeingRun++;
        }
        else
        {
            _agreeingRun = 0;
            _candidates.Clear();
        }

        var settled = _agreeingRun >= _settings.RequiredStableSamples &&
                      _candidates.DispersionDegrees(_settings.RequiredStableSamples) <=
                      _settings.StabilityToleranceDegrees;

        if (settled && _reacquiringSeconds >= _settings.MinimumHoldSeconds)
        {
            _candidates.TryMedian(_settings.RequiredStableSamples, out var tx, out var ty);
            return Accept(tx, ty, degreesFromRaw, timedOut: false);
        }

        if (_reacquiringSeconds >= _settings.ReacquisitionTimeoutSeconds)
        {
            // Never freeze indefinitely. Take the best thing available and move on.
            if (_candidates.Count > 0)
            {
                _candidates.TryMedian(_candidates.Count, out var tx, out var ty);
                return Accept(tx, ty, degreesFromRaw, timedOut: true);
            }

            if (sample.Valid)
                return Accept(sample.X, sample.Y, degreesFromRaw, timedOut: true);

            // Nothing valid at all: resume normal rather than hold forever, and let the ordinary
            // path deal with whatever the tracker is doing.
            State = BlinkGuardState.Normal;
            return Step(rejection, degreesFromRaw, reacquisitionTimedOut: true);
        }

        return Step(rejection, degreesFromRaw);
    }

    private BlinkGuardRejection Evaluate(in GazeSample sample)
    {
        if (!sample.Valid)
            return BlinkGuardRejection.Invalid;

        if (_settings.UseConfidence && sample.Confidence < _settings.ConfidenceThreshold)
            return BlinkGuardRejection.LowConfidence;

        return BlinkGuardRejection.None;
    }

    private BlinkGuardStep Accept(float targetX, float targetY, float degreesFromRaw, bool timedOut)
    {
        var reacquired = (float)_reacquiringSeconds;
        _reacquiringSeconds = 0;

        var distance = new GazeSample(targetX, targetY, 0f, true, 1f).DegreesTo(_heldX, _heldY);

        // Already where we were holding, or no blend configured: resume immediately rather than
        // spend 80ms easing toward a position the eye is effectively at.
        if (distance <= _settings.ImmediateResumeDegrees || _settings.TransitionSeconds <= 0f)
        {
            State = BlinkGuardState.Normal;
            _outputX = targetX;
            _outputY = targetY;
            _recent.Clear();
            _recent.Add(new GazeSample(targetX, targetY, 0f, true, 1f));
            return Step(BlinkGuardRejection.None, degreesFromRaw,
                reacquisitionSucceeded: !timedOut, reacquisitionTimedOut: timedOut,
                reacquisitionSeconds: reacquired);
        }

        State = BlinkGuardState.Transition;
        _transitionSeconds = 0;
        _transitionFromX = _heldX;
        _transitionFromY = _heldY;
        _outputX = _heldX;
        _outputY = _heldY;
        return Step(BlinkGuardRejection.None, degreesFromRaw,
            reacquisitionSucceeded: !timedOut, reacquisitionTimedOut: timedOut,
            reacquisitionSeconds: reacquired);
    }

    private BlinkGuardStep Transition(in GazeSample sample, double dt, float degreesFromRaw)
    {
        _transitionSeconds += dt;
        _recent.Add(sample);

        var t = _settings.TransitionSeconds <= 0f
            ? 1f
            : (float)Math.Clamp(_transitionSeconds / _settings.TransitionSeconds, 0d, 1d);

        // Smoothstep rather than a linear ramp: the ease-out is what makes the arrival invisible.
        var eased = t * t * (3f - 2f * t);

        // Blended toward the LIVE sample, not a frozen target. Easing to a stale position and then
        // jumping to wherever the eye had moved on to would just relocate the snap.
        _outputX = _transitionFromX + (sample.X - _transitionFromX) * eased;
        _outputY = _transitionFromY + (sample.Y - _transitionFromY) * eased;

        if (t >= 1f)
        {
            State = BlinkGuardState.Normal;
            _outputX = sample.X;
            _outputY = sample.Y;
        }

        return Step(BlinkGuardRejection.None, degreesFromRaw);
    }

    private BlinkGuardStep Step(
        BlinkGuardRejection rejection,
        float degreesFromRaw,
        bool blinkStarted = false,
        bool blinkEnded = false,
        bool reacquisitionSucceeded = false,
        bool reacquisitionTimedOut = false,
        float reacquisitionSeconds = 0f)
    {
        var fromOutput = _hasPrevious
            ? new GazeSample(_previousRawX, _previousRawY, 0f, true, 1f).DegreesTo(_outputX, _outputY)
            : 0f;

        return new BlinkGuardStep(
            _outputX, _outputY, State, rejection, _agreeingRun,
            degreesFromRaw, fromOutput,
            blinkStarted, blinkEnded,
            reacquisitionSucceeded, reacquisitionTimedOut, reacquisitionSeconds);
    }
}
