namespace BabblePersonalizer.Core.Calibration;

public static class RobustStatistics
{
    public static float Median(IEnumerable<float> source) => Percentile(source, .5f);

    public static float Percentile(IEnumerable<float> source, float percentile)
    {
        var values = source.Where(float.IsFinite).OrderBy(x => x).ToArray();
        if (values.Length == 0) return 0;
        percentile = Math.Clamp(percentile, 0, 1);
        var position = percentile * (values.Length - 1);
        var lower = (int)Math.Floor(position); var upper = (int)Math.Ceiling(position);
        if (lower == upper) return values[lower];
        return values[lower] + (values[upper] - values[lower]) * (position - lower);
    }

    public static float MedianAbsoluteDeviation(IEnumerable<float> source)
    {
        var values = source.Where(float.IsFinite).ToArray();
        if (values.Length == 0) return 0;
        var median = Median(values);
        return Median(values.Select(x => Math.Abs(x - median)));
    }

    public static IReadOnlyList<float> WindowMedians(IReadOnlyList<float> source, int windowSize = 7)
    {
        if (source.Count == 0) return Array.Empty<float>();
        windowSize = Math.Clamp(windowSize, 1, source.Count);
        var result = new List<float>((source.Count + windowSize - 1) / windowSize);
        for (var i = 0; i < source.Count; i += windowSize)
            result.Add(Median(source.Skip(i).Take(Math.Min(windowSize, source.Count - i))));
        return result;
    }

    public static RejectionResult RejectOutliersAndSpikes(
        IReadOnlyList<float> source, float madMultiplier = 6, float spikeMultiplier = 8)
    {
        if (source.Count < 5)
            return new(source.Where(float.IsFinite).ToArray(), new Dictionary<string, int>());
        var finite = source.Where(float.IsFinite).ToArray();
        var median = Median(finite); var mad = Math.Max(MedianAbsoluteDeviation(finite), .0001f);
        var accepted = new List<float>(finite.Length);
        var reasons = new Dictionary<string, int>(StringComparer.Ordinal);
        void Reject(string reason) => reasons[reason] = reasons.GetValueOrDefault(reason) + 1;
        for (var i = 0; i < finite.Length; i++)
        {
            if (Math.Abs(finite[i] - median) > madMultiplier * mad) { Reject("RobustOutlier"); continue; }
            if (i > 0 && i + 1 < finite.Length &&
                Math.Abs(finite[i] - finite[i - 1]) > spikeMultiplier * mad &&
                Math.Abs(finite[i] - finite[i + 1]) > spikeMultiplier * mad)
            { Reject("SingleFrameSpike"); continue; }
            accepted.Add(finite[i]);
        }
        return new(accepted, reasons);
    }

    public static float Stability(IEnumerable<float> source)
    {
        var values = source.Where(float.IsFinite).ToArray();
        if (values.Length < 3) return 0;
        var robustRange = Math.Max(Percentile(values, .95f) - Percentile(values, .05f), .01f);
        var sigma = MedianAbsoluteDeviation(values) * 1.4826f;
        return Math.Clamp(1 - sigma / robustRange, 0, 1);
    }

    public static float SpearmanLikeMonotonicity(IReadOnlyList<(float Target, float Observed)> points)
    {
        if (points.Count < 3) return 0;
        var ordered = points.OrderBy(x => x.Target).ToArray();
        var consistent = 0; var comparable = 0;
        for (var i = 1; i < ordered.Length; i++)
        {
            if (Math.Abs(ordered[i].Target - ordered[i - 1].Target) < .001f) continue;
            comparable++;
            if (ordered[i].Observed >= ordered[i - 1].Observed) consistent++;
        }
        return comparable == 0 ? 0 : consistent / (float)comparable;
    }

    public static List<ResponseCurvePoint> IsotonicCurve(IEnumerable<ResponseCurvePoint> source)
    {
        var points = source.Where(x => float.IsFinite(x.Input) && float.IsFinite(x.Output))
            .OrderBy(x => x.Input).ToArray();
        if (points.Length == 0) return new() { new(0, 0), new(1, 1) };
        var blocks = new List<Block>();
        foreach (var point in points)
        {
            blocks.Add(new Block(point.Input, point.Input, point.Output, 1));
            while (blocks.Count > 1 && blocks[^2].Value > blocks[^1].Value)
            {
                var right = blocks[^1]; blocks.RemoveAt(blocks.Count - 1);
                var left = blocks[^1]; blocks.RemoveAt(blocks.Count - 1);
                var weight = left.Weight + right.Weight;
                blocks.Add(new Block(left.Start, right.End,
                    (left.Value * left.Weight + right.Value * right.Weight) / weight, weight));
            }
        }
        var result = new List<ResponseCurvePoint>();
        foreach (var block in blocks)
            result.AddRange(points.Where(x => x.Input >= block.Start && x.Input <= block.End)
                .Select(x => new ResponseCurvePoint(x.Input, Math.Clamp(block.Value, 0, 1))));
        return result.GroupBy(x => x.Input).Select(x => new ResponseCurvePoint(x.Key, Median(x.Select(y => y.Output))))
            .OrderBy(x => x.Input).ToList();
    }

    public static float Interpolate(IReadOnlyList<ResponseCurvePoint> curve, float input)
    {
        if (curve.Count == 0) return input;
        if (input <= curve[0].Input) return curve[0].Output;
        for (var i = 1; i < curve.Count; i++)
        {
            if (input > curve[i].Input) continue;
            var width = curve[i].Input - curve[i - 1].Input;
            if (width <= float.Epsilon) return curve[i].Output;
            var t = (input - curve[i - 1].Input) / width;
            return curve[i - 1].Output + (curve[i].Output - curve[i - 1].Output) * t;
        }
        return curve[^1].Output;
    }

    private sealed record Block(float Start, float End, float Value, int Weight);
}

public sealed record RejectionResult(IReadOnlyList<float> Accepted, IReadOnlyDictionary<string, int> Reasons);
public sealed record ResponseCurvePoint(float Input, float Output);
