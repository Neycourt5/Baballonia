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
}
