using BabblePersonalizer.Core.Calibration;

namespace BabblePersonalizer.Tests;

[TestClass]
public sealed class RobustStatisticsTests
{
    [TestMethod]
    public void MedianPercentileAndMadAreDeterministic()
    {
        var values = new[] { 1f, 9f, 3f, 5f, 7f };
        Assert.AreEqual(5f, RobustStatistics.Median(values), .0001f);
        Assert.AreEqual(3f, RobustStatistics.Percentile(values, .25f), .0001f);
        Assert.AreEqual(2f, RobustStatistics.MedianAbsoluteDeviation(values), .0001f);
    }

    [TestMethod]
    public void OutlierAndSingleFrameSpikeAreRejected()
    {
        var values = Enumerable.Repeat(.1f, 10).Concat(new[] { 1f }).Concat(Enumerable.Repeat(.1f, 10)).ToArray();
        var result = RobustStatistics.RejectOutliersAndSpikes(values);
        Assert.AreEqual(20, result.Accepted.Count);
        Assert.AreEqual(1, result.Reasons.Values.Sum());
    }

    [TestMethod]
    public void IsotonicCurveIsMonotonicAndInterpolates()
    {
        var curve = RobustStatistics.IsotonicCurve(new[]
        {
            new ResponseCurvePoint(0, 0), new ResponseCurvePoint(.25f, .4f),
            new ResponseCurvePoint(.5f, .35f), new ResponseCurvePoint(1, 1)
        });
        Assert.IsTrue(curve.Zip(curve.Skip(1), (a, b) => b.Output >= a.Output).All(x => x));
        var value = RobustStatistics.Interpolate(curve, .75f);
        Assert.IsTrue(value >= curve.First(x => x.Input >= .5f).Output && value <= 1);
    }

    [TestMethod]
    public void StabilityPenalizesNoisySignal()
    {
        var stable = RobustStatistics.Stability(new[] { .5f, .5f, .501f, .499f, .5f });
        var noisy = RobustStatistics.Stability(new[] { .1f, .8f, .2f, .9f, .4f });
        Assert.IsTrue(stable > noisy);
    }
}
