using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Baballonia.Services.Personalization;

/// <summary>
/// One step of a guided routine: what to command, for how long, and how much to trust the result.
/// </summary>
/// <param name="CueId">
/// Stamped into every frame's label. Distinct per level (JawOpen50, JawOpen100) rather than shared,
/// so two holds at different intensities can never be mistaken for one continuous segment.
/// </param>
/// <param name="Phase">prep, transition, hold or rest - this is what decides the label weight.</param>
/// <param name="From">Commanded vector at the start of the step.</param>
/// <param name="To">Commanded vector at the end. Equal to <paramref name="From"/> for a hold.</param>
public sealed record CueStep(
    string CueId,
    string Phase,
    IReadOnlyList<int> Dims,
    float[] From,
    float[] To,
    double DurationSeconds,
    float Level,
    int Repetition,
    string Instruction,
    string DisplayName = "",
    int RepetitionCount = 0);

/// <summary>Stable identity for one independently repeatable/validatable guided attempt.</summary>
public readonly record struct GuidedAttemptIdentity(string CueId, int Repetition, int Attempt);

/// <summary>
/// Drives the avatar through a known sequence of expressions so the user can copy it.
///
/// The problem this solves is that nothing in the corpus currently says what a *correct* non-zero
/// expression looks like. Neutral sessions say "all zeros"; speech pseudo-labels just echo the stock
/// model's own opinion. So the adapter can learn to quieten a resting face and nothing else - which
/// is exactly the ceiling P1 hit.
///
/// Asking someone to produce "JawOpen = 0.5" on demand does not work; nobody knows what half a jaw
/// feels like. Showing them an avatar already doing it does, because imitation is a thing humans are
/// good at. The commanded vector remains the label, so the supervision is non-circular by
/// construction: it originates in this routine, never in inference.
///
/// <para><b>Why holds rather than ramps.</b> Both the avatar's own animator smoothing and the user's
/// reaction time make the commanded value a poor description of the face while it is moving. During
/// a settled hold it is a good one. So holds carry full weight, transitions carry none, and the
/// first half-second of every hold is trimmed away by the labeller.</para>
///
/// This class is a pure state machine over a step list: it owns no timer and no threads. The caller
/// ticks it, which keeps it testable without waiting in real time.
/// </summary>
public sealed class GuidedCaptureRoutine
{
    /// <summary>
    /// Levels for the first routine. 0.5 and 1.0 only, deliberately: finer gradations are worth
    /// having but worthless if the user cannot reproduce them, and whether they can is exactly what
    /// the first real session measures. Expand once the monotonicity numbers say it is safe.
    /// </summary>
    public static readonly float[] DefaultLevels = [0.5f, 1.0f];

    private readonly IReadOnlyList<CueStep> _steps;
    private readonly Func<long> _now;
    private readonly double _ticksPerSecond;

    private int _index = -1;
    private long _stepStarted;
    private bool _finished;
    private readonly Dictionary<(string CueId, int Repetition), int> _attempts = new();

    public GuidedCaptureRoutine(IReadOnlyList<CueStep> steps, Func<long>? clock = null,
                                double? ticksPerSecond = null)
    {
        if (steps.Count == 0)
            throw new ArgumentException("A routine needs at least one step.", nameof(steps));

        _steps = steps;
        _now = clock ?? Stopwatch.GetTimestamp;
        _ticksPerSecond = ticksPerSecond ?? Stopwatch.Frequency;
    }

    /// <summary>Step currently being commanded, or null before the first tick and after the end.</summary>
    public CueStep? Current => _index >= 0 && _index < _steps.Count && !_finished ? _steps[_index] : null;

    public int StepIndex => _index;
    public int StepCount => _steps.Count;
    public bool IsFinished => _finished;
    public double CurrentStepProgress => Current == null
        ? (_finished ? 1 : 0)
        : Math.Clamp(ElapsedInStep / Math.Max(Current.DurationSeconds, 1e-6), 0, 1);
    public double CurrentStepSecondsRemaining => Current == null
        ? 0
        : Math.Max(0, Current.DurationSeconds - ElapsedInStep);

    public GuidedAttemptIdentity? CurrentAttempt => Current is not { } step
        ? null
        : new GuidedAttemptIdentity(step.CueId, step.Repetition,
            _attempts.GetValueOrDefault((step.CueId, step.Repetition)));

    /// <summary>Total length of the routine, for telling the user what they are signing up for.</summary>
    public double TotalSeconds => _steps.Sum(s => s.DurationSeconds);

    /// <summary>What the user should be doing right now.</summary>
    public string Instruction => Current?.Instruction ?? (_finished ? "Done - thank you." : "Get ready...");

    /// <summary>Rough progress for a bar, by elapsed time rather than step count.</summary>
    public double Progress
    {
        get
        {
            if (_finished) return 1.0;
            if (_index < 0) return 0.0;

            var elapsedSteps = _steps.Take(_index).Sum(s => s.DurationSeconds);
            return Math.Clamp((elapsedSteps + ElapsedInStep) / Math.Max(TotalSeconds, 1e-6), 0, 1);
        }
    }

    private double ElapsedInStep => (_now() - _stepStarted) / _ticksPerSecond;

    /// <summary>
    /// Advances the machine. Returns the phase to command when it changed, otherwise null.
    /// </summary>
    /// <remarks>
    /// Only returns a phase on a *change*, so the caller pushes one snapshot per step rather than
    /// one per tick. The override service interpolates within the phase itself, so there is nothing
    /// to update in between.
    /// </remarks>
    public CuePhase? Tick()
    {
        if (_finished)
            return null;

        if (_index < 0)
        {
            _index = 0;
            _stepStarted = _now();
            return ToPhase(_steps[0], _stepStarted);
        }

        if (ElapsedInStep < _steps[_index].DurationSeconds)
            return null;

        _index++;
        if (_index >= _steps.Count)
        {
            _finished = true;
            return null;
        }

        _stepStarted = _now();
        return ToPhase(_steps[_index], _stepStarted);
    }

    /// <summary>Replays only the current cue/level/repetition and gives the replay a new attempt id.</summary>
    public CuePhase? RetryCurrentAttempt()
    {
        if (Current is not { Phase: "hold" } current) return null;
        var key = (current.CueId, current.Repetition);
        _attempts[key] = _attempts.GetValueOrDefault(key) + 1;

        var first = _index;
        while (first > 0 && SameAttempt(_steps[first - 1], current)) first--;
        _index = first;
        _stepStarted = _now();
        _finished = false;
        return ToPhase(_steps[_index], _stepStarted);
    }

    /// <summary>Skips the remainder of only the current cue/level/repetition.</summary>
    public CuePhase? SkipCurrentAttempt()
    {
        if (Current is not { Phase: "hold" } current) return null;
        var next = _index + 1;
        while (next < _steps.Count && SameAttempt(_steps[next], current)) next++;
        if (next >= _steps.Count)
        {
            _index = _steps.Count;
            _finished = true;
            return null;
        }

        _index = next;
        _stepStarted = _now();
        return ToPhase(_steps[_index], _stepStarted);
    }

    private static bool SameAttempt(CueStep candidate, CueStep current) =>
        candidate.CueId == current.CueId && candidate.Repetition == current.Repetition;

    private CuePhase ToPhase(CueStep step, long startedAt) => new(
        CueId: step.CueId,
        PhaseName: step.Phase,
        Dims: step.Dims,
        FromVector: step.From,
        ToVector: step.To,
        DurationSeconds: step.DurationSeconds,
        StartTimestamp: startedAt,
        Level: step.Level,
        RepetitionIndex: step.Repetition,
        AttemptIndex: _attempts.GetValueOrDefault((step.CueId, step.Repetition)));

    // =============================================================================================
    // Routine construction
    // =============================================================================================

    /// <summary>
    /// How much of the front of each hold the trainer discards, because the face is still arriving.
    /// Mirrors <c>labels.HOLD_SETTLE_TRIM_SECONDS</c> in the Python trainer; the two must be changed
    /// together. Used here only to tell the user when the trusted part of a hold begins - nothing in
    /// C# does the trimming.
    /// </summary>
    public const double HoldSettleTrimSeconds = 0.75;

    /// <summary>Timing shared by every routine, so passes stay comparable to each other.</summary>
    /// <param name="HoldSeconds">
    /// The trusted window. The labeller trims its settle-in period off the front of this, so the
    /// usable part is shorter than the number here.
    /// </param>
    /// <param name="PrepSeconds">
    /// Counted down before each attempt with the expression named on the headset. Without it the
    /// only warning was the transition itself, which asked the user to notice a cue, understand it
    /// and produce it inside three quarters of a second - and a cue produced late lands in the hold
    /// window as a face that is still moving.
    /// </param>
    public sealed record CueTiming(
        double HoldSeconds = 3.0,
        double RestSeconds = 2.0,
        double TransitionSeconds = 1.0,
        double LeadInSeconds = 5.0,
        double PrepSeconds = 3.0)
    {
        public static readonly CueTiming Default = new();
    }

    /// <summary>
    /// Builds the step list for one or more cues, back to back.
    /// </summary>
    /// <remarks>
    /// Every cue contributes the same shape - lead-in, then per level per repetition
    /// transition/hold/transition/rest - so a multi-expression pass is exactly a concatenation and
    /// nothing downstream has to know whether the session covered one expression or eight.
    /// </remarks>
    public static IReadOnlyList<CueStep> BuildRoutine(
        IReadOnlyList<GuidedCue> cues,
        int repetitions = 3,
        CueTiming? timing = null)
    {
        if (cues.Count == 0)
            throw new ArgumentException("A routine needs at least one cue.", nameof(cues));
        if (repetitions < 1)
            throw new ArgumentOutOfRangeException(nameof(repetitions));

        timing ??= CueTiming.Default;
        var steps = new List<CueStep>();

        foreach (var cue in cues)
        {
            var dims = cue.Dims;
            var neutral = new float[PersonalizationSchema.ExpressionCount];

            // Lead-in: the avatar sits at neutral while the user finds it and settles. Labelled as
            // a rest so the settled part still supplies known-closed frames rather than being
            // wasted. In a multi-cue pass it doubles as "get ready for the next expression".
            steps.Add(new CueStep($"{cue.Id}LeadIn", "rest", dims, neutral, neutral,
                timing.LeadInSeconds, 0f, 0, cue.LeadInInstruction, cue.DisplayName, repetitions));

            for (var rep = 0; rep < repetitions; rep++)
            {
                // EffectiveLevels, not Levels: the raw property is nullable and means "use the
                // shared default ladder", which is the case for most of the catalogue.
                foreach (var level in cue.EffectiveLevels)
                {
                    // Distinct id per level. The labeller measures its settle-in trim from where
                    // segment identity last changed, so two holds sharing an id would let the
                    // second inherit the first's start time and skip its trim.
                    var id = $"{cue.Id}{(int)Math.Round(level * 100)}";
                    var target = cue.TargetAt(level);

                    // Prep: the avatar stays neutral while the headset names the coming expression
                    // and counts down. Deliberately carries no supervision - the labeller ignores
                    // this phase entirely - because the user is being told what to do, not doing it.
                    steps.Add(new CueStep(id, "prep", dims, neutral, neutral,
                        timing.PrepSeconds, level, rep, cue.PrepInstruction,
                        cue.DisplayName, repetitions));

                    steps.Add(new CueStep(id, "transition", dims, neutral, target,
                        timing.TransitionSeconds, level, rep, cue.ApproachInstruction(level),
                        cue.DisplayName, repetitions));

                    steps.Add(new CueStep(id, "hold", dims, target, target,
                        timing.HoldSeconds, level, rep, cue.HoldInstruction(level),
                        cue.DisplayName, repetitions));

                    steps.Add(new CueStep(id, "transition", dims, target, neutral,
                        timing.TransitionSeconds, level, rep, "And relax...",
                        cue.DisplayName, repetitions));

                    steps.Add(new CueStep(id, "rest", dims, neutral, neutral,
                        timing.RestSeconds, level, rep, cue.RestInstruction,
                        cue.DisplayName, repetitions));
                }
            }
        }

        return steps;
    }

    /// <summary>
    /// The JawOpen routine: the one the user's actual complaint calls for.
    /// </summary>
    /// <remarks>
    /// Produces both halves of the discrimination the model needs. The holds are known-open frames;
    /// the rests and the lead-in are known-closed frames captured in the same session, under the same
    /// lighting, minutes apart. Neutral recordings alone can never supply the first half, which is
    /// why "make the jaw quieter" was the only thing the adapter could previously learn.
    /// </remarks>
    public static IReadOnlyList<CueStep> BuildJawOpenRoutine(
        int repetitions = 3,
        IReadOnlyList<float>? levels = null,
        double holdSeconds = 3.0,
        double restSeconds = 2.0,
        double transitionSeconds = 1.0,
        double leadInSeconds = 5.0,
        double prepSeconds = 3.0)
    {
        var cue = levels is null ? GuidedCues.JawOpen : GuidedCues.JawOpen with { Levels = levels };

        return BuildRoutine([cue], repetitions,
            new CueTiming(holdSeconds, restSeconds, transitionSeconds, leadInSeconds, prepSeconds));
    }
}
