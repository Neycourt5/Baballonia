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
    string Instruction);

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

    private static CuePhase ToPhase(CueStep step, long startedAt) => new(
        CueId: step.CueId,
        PhaseName: step.Phase,
        Dims: step.Dims,
        FromVector: step.From,
        ToVector: step.To,
        DurationSeconds: step.DurationSeconds,
        StartTimestamp: startedAt,
        Level: step.Level,
        RepetitionIndex: step.Repetition);

    // =============================================================================================
    // Routine construction
    // =============================================================================================

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
        double transitionSeconds = 0.75,
        double leadInSeconds = 5.0)
    {
        levels ??= DefaultLevels;

        var jaw = PersonalizationSchema.IndexOf("JawOpen");
        int[] dims = [jaw];
        var neutral = new float[PersonalizationSchema.ExpressionCount];

        float[] At(float level)
        {
            var vector = new float[PersonalizationSchema.ExpressionCount];
            vector[jaw] = level;
            return vector;
        }

        var steps = new List<CueStep>
        {
            // Lead-in: the avatar sits at neutral while the user finds it and settles. Labelled as a
            // rest so the settled part still supplies known-closed frames rather than being wasted.
            new("JawOpenLeadIn", "rest", dims, neutral, neutral, leadInSeconds, 0f, 0,
                "Look at your avatar and relax your face."),
        };

        for (var rep = 0; rep < repetitions; rep++)
        {
            foreach (var level in levels)
            {
                var id = $"JawOpen{(int)Math.Round(level * 100)}";
                var target = At(level);
                var percent = (int)Math.Round(level * 100);

                steps.Add(new CueStep(id, "transition", dims, neutral, target, transitionSeconds,
                    level, rep, $"Open your jaw to match the avatar ({percent}%)..."));

                steps.Add(new CueStep(id, "hold", dims, target, target, holdSeconds,
                    level, rep, $"Hold it - match the avatar's mouth ({percent}%)."));

                steps.Add(new CueStep(id, "transition", dims, target, neutral, transitionSeconds,
                    level, rep, "And relax..."));

                steps.Add(new CueStep(id, "rest", dims, neutral, neutral, restSeconds,
                    level, rep, "Rest. Let your mouth close naturally."));
            }
        }

        return steps;
    }
}
