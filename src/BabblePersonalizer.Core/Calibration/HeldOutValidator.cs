using BabblePersonalizer.Core.Storage;

namespace BabblePersonalizer.Core.Calibration;

public static class HeldOutValidator
{
    public static HeldOutValidationReport Validate(
        PersonalCalibrationProfile profile, IReadOnlyList<CalibrationSample> samples)
    {
        var validation = samples.Where(x => x.IsValidation && x.CanonicalTarget.Length > 0 &&
                                            x.StepType.EndsWith("Ramp", StringComparison.Ordinal))
            .GroupBy(x => x.CanonicalTarget)
            .SelectMany(group => group.Where(x => x.Attempt == group.Max(y => y.Attempt))).ToArray();
        var correction = new PersonalCorrectionEngine(profile);
        var metrics = new List<ParameterValidationMetric>();
        foreach (var parameter in profile.Parameters)
        {
            var relevant = validation.Where(x => x.CanonicalTarget == parameter.CanonicalName).ToArray();
            if (relevant.Length == 0) continue;
            var stockPoints = relevant.Select(x => (x.RequestedIntensity,
                x.RawOutput[parameter.OriginalOutputIndex])).ToArray();
            var correctedValues = relevant.Select(x => correction.Apply(x.RawOutput)[parameter.OriginalOutputIndex]).ToArray();
            var personalizedPoints = relevant.Select((x, i) => (x.RequestedIntensity, correctedValues[i])).ToArray();
            var stockError = stockPoints.Average(x => Math.Abs(x.RequestedIntensity - x.Item2));
            var personalError = personalizedPoints.Average(x => Math.Abs(x.RequestedIntensity - x.Item2));
            var stockMono = RobustStatistics.SpearmanLikeMonotonicity(stockPoints);
            var personalMono = RobustStatistics.SpearmanLikeMonotonicity(personalizedPoints);
            var stockSpikes = SpikeRate(stockPoints.Select(x => x.Item2));
            var personalSpikes = SpikeRate(personalizedPoints.Select(x => x.Item2));
            metrics.Add(new(parameter.CanonicalName, relevant.Length, stockError, personalError,
                stockMono, personalMono, stockSpikes, personalSpikes,
                personalError < stockError && personalMono >= stockMono && personalSpikes <= stockSpikes));
        }
        var stockScore = metrics.Count == 0 ? 0 : metrics.Average(x => (1 - x.StockTargetError) * .6f + x.StockMonotonicity * .4f);
        var personalScore = metrics.Count == 0 ? 0 : metrics.Average(x => (1 - x.PersonalizedTargetError) * .6f + x.PersonalizedMonotonicity * .4f);
        var improved = metrics.Count >= 3 && personalScore > stockScore + .01f &&
                       metrics.Count(x => x.Improved) > metrics.Count / 2;
        return new HeldOutValidationReport
        {
            CreatedUtc = DateTimeOffset.UtcNow, Parameters = metrics, StockScore = stockScore,
            PersonalizedScore = personalScore, ImprovementSupported = improved,
            Summary = metrics.Count == 0 ? "No held-out samples were available."
                : improved ? "Held-out metrics support the personalized correction."
                : "Held-out metrics do not yet support claiming an improvement; review or repeat weak poses."
        };
    }

    private static float SpikeRate(IEnumerable<float> source)
    {
        var values = source.ToArray(); if (values.Length < 3) return 0;
        var spikes = 0;
        for (var i = 1; i + 1 < values.Length; i++)
            if (Math.Abs(values[i] - values[i - 1]) > .25f &&
                Math.Abs(values[i] - values[i + 1]) > .25f) spikes++;
        return spikes / (float)(values.Length - 2);
    }
}
