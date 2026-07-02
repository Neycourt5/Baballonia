using BabblePersonalizer.Core.Calibration;
using BabblePersonalizer.Core.Inventory;
using BabblePersonalizer.Core.Models;
using System.Diagnostics;

namespace BabblePersonalizer.Tests;

[TestClass]
public sealed class GuidedCalibrationPlanTests
{
    [TestMethod]
    public void EveryGuidedParameterHasThreeTrainingAndOneHeldOutRepetition()
    {
        var parameters = LegacyBaballoniaFaceCatalog.CreateDefinitions();
        var plan = GuidedCalibrationPlan.Create(parameters);
        foreach (var parameter in parameters.Where(x => x.CalibrationStrategy is not
                     (CalibrationStrategy.ManualReviewOnly or CalibrationStrategy.UnsupportedPassthrough)))
        {
            var ramps = plan.Segments.Where(x => x.CanonicalTarget == parameter.CanonicalName && x.StepType.EndsWith("Ramp"));
            CollectionAssert.AreEquivalent(new[] { 1, 2, 3 }, ramps.Where(x => !x.IsValidation).Select(x => x.Repetition).Distinct().ToArray());
            CollectionAssert.AreEquivalent(new[] { 4 }, ramps.Where(x => x.IsValidation).Select(x => x.Repetition).Distinct().ToArray());
        }
        Assert.IsFalse(plan.Segments.Any(x => x.CanonicalTarget == "TongueSquish"));
        Assert.IsTrue(plan.Segments.Any(x => x.StepType == "NeutralHold"));
        Assert.IsTrue(plan.Segments.Any(x => x.StepType == "Speech"));
        Assert.IsTrue(plan.Segments.Any(x => x.StepType == "ValidationNeutral"));
    }

    [TestMethod]
    public void EngineSupportsCancellationSkipAndCompletionProgress()
    {
        var plan = new GuidedCalibrationPlan
        {
            Segments = new[]
            {
                new CalibrationSegment("A", "A", "A", 1, 0, "Hold", "TrainingRamp", TimeSpan.FromMilliseconds(10)),
                new CalibrationSegment("B", "B", "B", 1, 1, "Hold", "TrainingRamp", TimeSpan.FromMilliseconds(10))
            }
        };
        var engine = new GuidedCalibrationEngine(plan); var start = Stopwatch.GetTimestamp(); engine.Start(start);
        engine.SkipCurrentParameter(); Assert.AreEqual("B", engine.Current?.CanonicalTarget);
        engine.Update(start + Stopwatch.Frequency); Assert.IsTrue(engine.IsComplete); Assert.AreEqual(1d, engine.OverallProgress);
        engine.Start(start); engine.Cancel(); Assert.IsFalse(engine.IsRunning);
    }
}
