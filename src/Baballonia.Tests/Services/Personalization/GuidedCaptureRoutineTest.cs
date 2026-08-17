using System;
using System.Diagnostics;
using System.Linq;
using Baballonia.Services.Personalization;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The cue engine, driven by an injected clock so a minute-long routine runs instantly.
///
/// The property worth the most here is that a hold at 50% and a hold at 100% are different segments
/// as far as the labeller is concerned. They share a repetition and would share a cue id if the ids
/// were per-expression, and the labeller measures its settle-in trim from the point the segment
/// identity last changed - so if they looked identical, the second hold would inherit the first
/// one's start time and never be trimmed. That bug already existed once, on the phase boundary; the
/// distinct ids here are the belt to the labeller's braces.
/// </summary>
[TestClass]
public class GuidedCaptureRoutineTest
{
    private long _clock;

    /// <summary>An injected clock in stopwatch ticks, so time can be advanced deliberately.</summary>
    private GuidedCaptureRoutine Build(params CueStep[] steps) =>
        new(steps, () => _clock, Stopwatch.Frequency);

    private void Advance(double seconds) => _clock += (long)(seconds * Stopwatch.Frequency);

    private static float[] Vector(float jaw)
    {
        var values = new float[PersonalizationSchema.ExpressionCount];
        values[PersonalizationSchema.IndexOf("JawOpen")] = jaw;
        return values;
    }

    private static CueStep Step(string id, string phase, double seconds, float level = 0f) =>
        new(id, phase, [PersonalizationSchema.IndexOf("JawOpen")],
            Vector(0), Vector(level), seconds, level, 0, "do the thing");

    [TestInitialize]
    public void Initialize() => _clock = 1_000_000;

    [TestMethod]
    public void FirstTickCommandsTheFirstStep()
    {
        var routine = Build(Step("A", "hold", 1.0), Step("B", "rest", 1.0));

        var phase = routine.Tick();

        Assert.IsNotNull(phase);
        Assert.AreEqual("A", phase.CueId);
        Assert.AreEqual(0, routine.StepIndex);
    }

    [TestMethod]
    public void NoNewPhaseUntilTheStepElapses()
    {
        var routine = Build(Step("A", "hold", 1.0), Step("B", "rest", 1.0));
        routine.Tick();

        Advance(0.5);

        Assert.IsNull(routine.Tick(), "the override interpolates within a phase; nothing to re-push");
        Assert.AreEqual(0, routine.StepIndex);
    }

    [TestMethod]
    public void StepAdvancesWhenItsDurationPasses()
    {
        var routine = Build(Step("A", "hold", 1.0), Step("B", "rest", 1.0));
        routine.Tick();

        Advance(1.01);
        var phase = routine.Tick();

        Assert.IsNotNull(phase);
        Assert.AreEqual("B", phase.CueId);
        Assert.AreEqual("rest", phase.PhaseName);
    }

    [TestMethod]
    public void RoutineFinishesAfterTheLastStep()
    {
        var routine = Build(Step("A", "hold", 1.0));
        routine.Tick();

        Advance(1.01);

        Assert.IsNull(routine.Tick());
        Assert.IsTrue(routine.IsFinished);
        Assert.IsNull(routine.Current);
        Assert.AreEqual(1.0, routine.Progress, 1e-6);
    }

    [TestMethod]
    public void FinishedRoutineStaysFinished()
    {
        var routine = Build(Step("A", "hold", 1.0));
        routine.Tick();
        Advance(2.0);
        routine.Tick();

        Assert.IsNull(routine.Tick());
        Assert.IsTrue(routine.IsFinished);
    }

    [TestMethod]
    public void ProgressAdvancesMonotonically()
    {
        var routine = Build(Step("A", "hold", 2.0), Step("B", "rest", 2.0));
        routine.Tick();

        var readings = new System.Collections.Generic.List<double> { routine.Progress };
        for (var i = 0; i < 8; i++)
        {
            Advance(0.5);
            routine.Tick();
            readings.Add(routine.Progress);
        }

        for (var i = 1; i < readings.Count; i++)
            Assert.IsTrue(readings[i] >= readings[i - 1] - 1e-9, "progress went backwards");

        Assert.AreEqual(1.0, readings[^1], 1e-6);
    }

    // =============================================================================================
    // The JawOpen routine
    // =============================================================================================

    [TestMethod]
    public void JawRoutineCommandsBothOpenAndClosedStates()
    {
        // Both halves in one session under identical conditions is the entire point: neutral
        // recordings can supply known-closed frames, but nothing else supplies known-open ones.
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine(repetitions: 2);
        var jaw = PersonalizationSchema.IndexOf("JawOpen");

        var holds = steps.Where(s => s.Phase == "hold").ToList();
        var rests = steps.Where(s => s.Phase == "rest").ToList();

        Assert.IsTrue(holds.Count > 0 && rests.Count > 0);
        Assert.IsTrue(holds.All(s => s.To[jaw] > 0), "holds must command an open jaw");
        Assert.IsTrue(rests.All(s => s.To[jaw] == 0), "rests must command a closed jaw");
    }

    [TestMethod]
    public void EachLevelGetsItsOwnCueId()
    {
        // If 50% and 100% shared an id, the labeller would treat consecutive holds as one segment
        // and skip the settle-in trim on the second.
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine(repetitions: 1);

        var byLevel = steps.Where(s => s.Phase == "hold")
            .GroupBy(s => s.Level)
            .ToDictionary(g => g.Key, g => g.Select(s => s.CueId).Distinct().ToList());

        Assert.AreEqual(2, byLevel.Count, "default routine holds at two levels");
        foreach (var (_, ids) in byLevel)
            Assert.AreEqual(1, ids.Count, "one id per level");

        var allIds = byLevel.Values.SelectMany(v => v).Distinct().ToList();
        Assert.AreEqual(2, allIds.Count, "the two levels must not share an id");
    }

    [TestMethod]
    public void RepetitionsAreNumbered()
    {
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine(repetitions: 3);

        var reps = steps.Where(s => s.Phase == "hold").Select(s => s.Repetition).Distinct().OrderBy(r => r);

        CollectionAssert.AreEqual(new[] { 0, 1, 2 }, reps.ToList());
    }

    [TestMethod]
    public void TransitionsSeparateHoldsFromRests()
    {
        // Whatever the face is doing mid-movement, the commanded value does not describe it. The
        // labeller masks these out; the routine has to actually emit them.
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine(repetitions: 1);

        for (var i = 0; i < steps.Count - 1; i++)
        {
            if (steps[i].Phase == "hold")
                Assert.AreEqual("transition", steps[i + 1].Phase,
                    "a hold must be followed by a transition, never directly by a rest");
        }
    }

    [TestMethod]
    public void RoutineStartsWithARestSoTheUserCanSettle()
    {
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine();

        Assert.AreEqual("rest", steps[0].Phase);
        Assert.IsTrue(steps[0].DurationSeconds >= 3,
            "the lead-in has to be long enough to find the avatar and relax");
    }

    [TestMethod]
    public void RoutineOnlyEverCommandsTheJaw()
    {
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine(repetitions: 2);
        var jaw = PersonalizationSchema.IndexOf("JawOpen");

        foreach (var step in steps)
        {
            CollectionAssert.AreEqual(new[] { jaw }, step.Dims.ToList());
            for (var d = 0; d < PersonalizationSchema.ExpressionCount; d++)
            {
                if (d == jaw) continue;
                Assert.AreEqual(0f, step.To[d], "a JawOpen routine must not move anything else");
                Assert.AreEqual(0f, step.From[d]);
            }
        }
    }

    [TestMethod]
    public void RoutineLengthIsReasonableForAHumanToSitThrough()
    {
        var routine = new GuidedCaptureRoutine(GuidedCaptureRoutine.BuildJawOpenRoutine());

        Assert.IsTrue(routine.TotalSeconds is > 30 and < 120,
            $"routine is {routine.TotalSeconds:F0}s; long enough to be useful, short enough to finish");
    }

    [TestMethod]
    public void EveryStepCarriesAnInstruction()
    {
        foreach (var step in GuidedCaptureRoutine.BuildJawOpenRoutine())
            Assert.IsFalse(string.IsNullOrWhiteSpace(step.Instruction), $"{step.CueId}/{step.Phase}");
    }

    [TestMethod]
    public void WholeRoutineRunsToCompletion()
    {
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine(repetitions: 2);
        var routine = Build(steps.ToArray());

        var pushed = 0;
        for (var i = 0; i < 10_000 && !routine.IsFinished; i++)
        {
            if (routine.Tick() is not null) pushed++;
            Advance(0.05);
        }

        Assert.IsTrue(routine.IsFinished, "routine never terminated");
        Assert.AreEqual(steps.Count, pushed, "every step should have been commanded exactly once");
    }

    [TestMethod]
    public void EmptyRoutineIsRejected()
    {
        Assert.ThrowsExactly<ArgumentException>(() => new GuidedCaptureRoutine([]));
    }

    [TestMethod]
    public void CustomLevelsAreHonoured()
    {
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine(repetitions: 1, levels: [0.25f, 0.75f]);
        var levels = steps.Where(s => s.Phase == "hold").Select(s => s.Level).Distinct().OrderBy(l => l);

        CollectionAssert.AreEqual(new[] { 0.25f, 0.75f }, levels.ToList());
    }

    // =============================================================================================
    // The cue catalogue
    // =============================================================================================

    /// <summary>
    /// Every cue must produce both halves of the discrimination: frames where the expression is
    /// commanded on, and frames where it is commanded off, in the same session under the same
    /// lighting. Neutral recordings supply only the second kind, which is why "be quieter" was
    /// previously the only thing the corpus could teach.
    /// </summary>
    [TestMethod]
    public void EveryCueCommandsBothOnAndOffStates()
    {
        foreach (var cue in GuidedCues.All)
        {
            var steps = GuidedCaptureRoutine.BuildRoutine([cue], repetitions: 1);

            var holds = steps.Where(s => s.Phase == "hold").ToList();
            var rests = steps.Where(s => s.Phase == "rest").ToList();

            Assert.IsTrue(holds.Count > 0, $"{cue.Id} has no holds");
            Assert.IsTrue(rests.Count > 0, $"{cue.Id} has no rests");

            foreach (var dim in cue.Dims)
            {
                Assert.IsTrue(holds.Any(h => h.To[dim] > 0), $"{cue.Id} never commands {dim} on");
                Assert.IsTrue(rests.All(r => r.To[dim] == 0), $"{cue.Id} rests do not release {dim}");
            }
        }
    }

    [TestMethod]
    public void NoCueMovesAnExpressionItDidNotDeclare()
    {
        // The labeller trusts Dims to decide what a cue is claiming. A vector that moves an
        // undeclared expression would be supervised as "stay at rest" while visibly not at rest.
        foreach (var cue in GuidedCues.All)
        {
            foreach (var step in GuidedCaptureRoutine.BuildRoutine([cue], repetitions: 1))
            {
                for (var dim = 0; dim < PersonalizationSchema.ExpressionCount; dim++)
                {
                    if (cue.Dims.Contains(dim)) continue;

                    Assert.AreEqual(0f, step.To[dim],
                        $"{cue.Id}/{step.Phase} moves undeclared expression {dim}");
                    Assert.AreEqual(0f, step.From[dim], $"{cue.Id}/{step.Phase} (from)");
                }
            }
        }
    }

    [TestMethod]
    public void EveryCueGivesEachLevelItsOwnId()
    {
        // Shared ids would let a second hold inherit the first's start time and skip the
        // settle-in trim - the exact aliasing bug M1 fixed on the phase boundary.
        foreach (var cue in GuidedCues.All)
        {
            var byLevel = GuidedCaptureRoutine.BuildRoutine([cue], repetitions: 1)
                .Where(s => s.Phase == "hold")
                .GroupBy(s => s.Level)
                .ToDictionary(g => g.Key, g => g.Select(s => s.CueId).Distinct().ToList());

            Assert.AreEqual(cue.EffectiveLevels.Count, byLevel.Count, $"{cue.Id} level count");

            var allIds = byLevel.Values.SelectMany(v => v).Distinct().ToList();
            Assert.AreEqual(cue.EffectiveLevels.Count, allIds.Count,
                $"{cue.Id} reuses a cue id across levels");
        }
    }

    [TestMethod]
    public void CueIdsAreUniqueAcrossTheCatalogue()
    {
        // Two cues sharing an id would merge into one segment in the labeller's eyes.
        var ids = GuidedCues.All.Select(c => c.Id).ToList();

        CollectionAssert.AllItemsAreUnique(ids);
    }

    [TestMethod]
    public void TongueIsCommandedAsBinary()
    {
        // A tongue is out or it is not; a commanded 50 % would be a label nobody can reproduce.
        Assert.IsTrue(GuidedCues.TongueOut.IsBinary);
        Assert.AreEqual(1, GuidedCues.TongueOut.EffectiveLevels.Count);
        Assert.AreEqual(1.0f, GuidedCues.TongueOut.EffectiveLevels[0]);
    }

    [TestMethod]
    public void CombinationCuesCommandTheJawGentlyRatherThanFully()
    {
        // Both expressions at maximum is usually not a pose a person can hold, and an unholdable
        // cue produces confident wrong labels.
        var jaw = PersonalizationSchema.IndexOf("JawOpen");

        foreach (var cue in GuidedCues.CombinationPass)
        {
            Assert.IsTrue(cue.Dims.Contains(jaw), $"{cue.Id} should involve the jaw");

            var full = cue.TargetAt(1.0f);
            var primary = cue.Dims.First(d => d != jaw);

            Assert.IsTrue(full[jaw] < full[primary],
                $"{cue.Id} commands the jaw as hard as the primary expression");
            Assert.IsTrue(full[jaw] > 0, $"{cue.Id} does not open the jaw at all");
        }
    }

    [TestMethod]
    public void CombinationCuesDriveEveryExpressionTheyName()
    {
        foreach (var cue in GuidedCues.CombinationPass)
        {
            var target = cue.TargetAt(1.0f);
            foreach (var dim in cue.Dims)
                Assert.IsTrue(target[dim] > 0, $"{cue.Id} declares {dim} but never moves it");
        }
    }

    [TestMethod]
    public void TargetsStayInTheUnitRange()
    {
        foreach (var cue in GuidedCues.All)
            foreach (var level in cue.EffectiveLevels)
                Assert.IsTrue(cue.TargetAt(level).All(v => v is >= 0f and <= 1f), cue.Id);
    }

    [TestMethod]
    public void EveryCueCarriesInstructionsForEveryPhase()
    {
        foreach (var cue in GuidedCues.All)
            foreach (var step in GuidedCaptureRoutine.BuildRoutine([cue], repetitions: 1))
                Assert.IsFalse(string.IsNullOrWhiteSpace(step.Instruction),
                    $"{cue.Id}/{step.Phase} has no instruction");
    }

    [TestMethod]
    public void AMultiCueRoutineIsTheConcatenationOfItsParts()
    {
        // Nothing downstream should be able to tell a multi-expression pass from several
        // single-expression ones stitched together.
        var separate = GuidedCues.CorePass
            .SelectMany(c => GuidedCaptureRoutine.BuildRoutine([c], repetitions: 2))
            .ToList();
        var combined = GuidedCaptureRoutine.BuildRoutine(GuidedCues.CorePass, repetitions: 2);

        Assert.AreEqual(separate.Count, combined.Count);
        for (var i = 0; i < separate.Count; i++)
        {
            Assert.AreEqual(separate[i].CueId, combined[i].CueId, $"step {i}");
            Assert.AreEqual(separate[i].Phase, combined[i].Phase, $"step {i}");
        }
    }

    [TestMethod]
    public void TheFullPassRunsToCompletionInAReasonableTime()
    {
        var choice = GuidedRoutineChoice.All.First(c => c.Id == "core");
        var steps = choice.Build();
        var routine = Build(steps.ToArray());

        var pushed = 0;
        for (var i = 0; i < 200_000 && !routine.IsFinished; i++)
        {
            if (routine.Tick() is not null) pushed++;
            Advance(0.05);
        }

        Assert.IsTrue(routine.IsFinished);
        Assert.AreEqual(steps.Count, pushed, "every step should be commanded exactly once");
        Assert.IsTrue(routine.TotalSeconds is > 120 and < 420,
            $"full pass is {routine.TotalSeconds:F0}s - long enough to be useful, short enough to finish");
    }

    [TestMethod]
    public void RoutineChoicesAreWellFormed()
    {
        Assert.IsTrue(GuidedRoutineChoice.All.Count >= 4);
        CollectionAssert.AllItemsAreUnique(GuidedRoutineChoice.All.Select(c => c.Id).ToList());

        foreach (var choice in GuidedRoutineChoice.All)
        {
            Assert.IsTrue(choice.Cues.Count > 0, $"{choice.Id} covers nothing");
            Assert.IsFalse(string.IsNullOrWhiteSpace(choice.DisplayName), choice.Id);
            Assert.IsFalse(string.IsNullOrWhiteSpace(choice.Description), choice.Id);
            Assert.IsTrue(choice.EstimatedSeconds > 0, choice.Id);
            Assert.IsTrue(choice.Build().Count > 0, choice.Id);
        }
    }

    [TestMethod]
    public void TheFirstOfferedRoutineIsTheFullPass()
    {
        // It is the default selection, so it has to be the one most users should run.
        Assert.AreEqual("core", GuidedRoutineChoice.All[0].Id);
    }

    [TestMethod]
    public void EstimatedTimeMatchesTheBuiltRoutine()
    {
        foreach (var choice in GuidedRoutineChoice.All)
        {
            var actual = choice.Build().Sum(s => s.DurationSeconds);

            Assert.AreEqual(choice.EstimatedSeconds, actual, 0.01,
                $"{choice.Id}: the picker would misreport how long this takes");
        }
    }

    [TestMethod]
    public void CuesCanBeLookedUpById()
    {
        Assert.AreEqual(GuidedCues.Smile.Id, GuidedCues.ById("Smile")!.Id);
        Assert.AreEqual(GuidedCues.Smile.Id, GuidedCues.ById("smile")!.Id);
        Assert.IsNull(GuidedCues.ById("NotACue"));
    }

    // =============================================================================================
    // Pacing
    //
    // The user gets a countdown with the expression named before every attempt. Previously the only
    // warning was the ramp itself, which asked them to notice a cue, understand it and produce it
    // inside three quarters of a second - and an expression that arrives late lands in the hold as a
    // face that is still moving, which is a wrong label at full weight.
    // =============================================================================================

    /// <summary>
    /// Walks the routine to its next hold. A search rather than a step count, so changing the
    /// preamble cannot quietly leave these assertions pointing at the wrong phase.
    /// </summary>
    private static void AdvanceToHold(GuidedCaptureRoutine routine, ref long clock)
    {
        for (var step = 0; step < 32 && routine.Current?.Phase != "hold"; step++)
        {
            clock += 2;
            routine.Tick();
        }

        Assert.AreEqual("hold", routine.Current!.Phase, "never reached a hold");
    }

    [TestMethod]
    public void GuidedRetryAndSkip_AreAttemptScoped()
    {
        long clock = 1;
        var steps = GuidedCaptureRoutine.BuildJawOpenRoutine(
            repetitions: 2, levels: [1f], holdSeconds: 1, restSeconds: 1,
            transitionSeconds: 1, leadInSeconds: 1, prepSeconds: 1);
        var routine = new GuidedCaptureRoutine(steps, () => clock, 1);

        AdvanceToHold(routine, ref clock);
        var original = routine.CurrentAttempt!.Value;

        var replay = routine.RetryCurrentAttempt();
        Assert.IsNotNull(replay);
        Assert.AreEqual("prep", routine.Current!.Phase,
            "Retry replays the whole attempt, countdown included, so the user is not thrown " +
            "straight back into an expression they just asked to redo.");
        Assert.AreEqual(original.Attempt + 1, routine.CurrentAttempt!.Value.Attempt);

        AdvanceToHold(routine, ref clock);
        var skipped = routine.CurrentAttempt!.Value;
        var next = routine.SkipCurrentAttempt();
        Assert.IsNotNull(next);
        Assert.AreNotEqual(skipped.Repetition, routine.CurrentAttempt!.Value.Repetition,
            "Skip must advance only past the current repetition, not end the whole routine.");
    }

    [TestMethod]
    public void EveryAttemptOpensWithACountdownBeforeTheAvatarMoves()
    {
        var steps = GuidedCaptureRoutine.BuildRoutine([GuidedCues.Smile], repetitions: 2);

        var attempts = steps
            .Where(step => !step.CueId.EndsWith("LeadIn", StringComparison.Ordinal))
            .GroupBy(step => (step.CueId, step.Repetition));

        foreach (var attempt in attempts)
        {
            var phases = attempt.Select(step => step.Phase).ToArray();
            CollectionAssert.AreEqual(
                new[] { "prep", "transition", "hold", "transition", "rest" }, phases,
                $"{attempt.Key} did not follow prep -> ramp -> hold -> ramp -> rest");
        }
    }

    [TestMethod]
    public void ThePrepStepCommandsNothing()
    {
        // Data integrity, not cosmetics: the countdown is when the user is being *told* what to do.
        // Commanding the expression here would move the avatar before they were ready and stamp the
        // frames with a target they are not yet making.
        var prep = GuidedCaptureRoutine.BuildRoutine([GuidedCues.JawOpen], repetitions: 1)
            .Where(step => step.Phase == "prep")
            .ToArray();

        Assert.IsTrue(prep.Length > 0);
        foreach (var step in prep)
        {
            Assert.IsTrue(step.From.All(v => v == 0f), "prep must start neutral");
            Assert.IsTrue(step.To.All(v => v == 0f), "prep must stay neutral");
        }
    }

    [TestMethod]
    public void PrepNamesTheExpressionInWords()
    {
        // An avatar's smile can be subtle enough to miss entirely, and a cue the user never noticed
        // still gets recorded as though they performed it.
        var prep = GuidedCaptureRoutine.BuildRoutine([GuidedCues.Smile], repetitions: 1)
            .First(step => step.Phase == "prep");

        StringAssert.Contains(prep.Instruction.ToLowerInvariant(), "smile");
    }

    [TestMethod]
    public void RampIsLongEnoughToBeFollowed()
    {
        Assert.IsTrue(GuidedCaptureRoutine.CueTiming.Default.TransitionSeconds >= 1.0,
            "a ramp shorter than a second outruns ordinary reaction time");
        Assert.IsTrue(GuidedCaptureRoutine.CueTiming.Default.PrepSeconds >= 2.0,
            "the countdown has to be long enough to read and act on");
    }

    [TestMethod]
    public void TrustedHoldWindowSurvivesTheSettleTrim()
    {
        // The trainer discards the front of every hold while the face is still arriving. If the trim
        // ever grew past the hold, guided capture would silently record nothing usable.
        var timing = GuidedCaptureRoutine.CueTiming.Default;

        Assert.IsTrue(GuidedCaptureRoutine.HoldSettleTrimSeconds < timing.HoldSeconds,
            "the settle trim must not consume the whole hold");
        Assert.IsTrue(timing.HoldSeconds - GuidedCaptureRoutine.HoldSettleTrimSeconds >= 2.0,
            "at least two seconds of each hold must remain trustworthy");
    }
}
