using System;

namespace Baballonia.Services.Inference.BlinkGuard;

/// <summary>One eye, one frame, as BlinkGuard sees it.</summary>
/// <param name="X">Horizontal gaze, [-1,1], spanning ±<see cref="BlinkGuardSettings.GazeRangeDegrees"/>.</param>
/// <param name="Y">Vertical gaze, same units.</param>
/// <param name="Closedness">0 fully open, 1 fully shut. The eyelid signal the state machine keys on.</param>
/// <param name="Valid">Whether the producer believes this sample at all.</param>
/// <param name="Confidence">
/// 0..1 where a source supplies one. This pipeline does not, so it is 1 and
/// <see cref="BlinkGuardSettings.UseConfidence"/> is off; the field exists so a source that does
/// have one needs no signature change.
/// </param>
public readonly record struct GazeSample(
    float X,
    float Y,
    float Closedness,
    bool Valid,
    float Confidence)
{
    public bool IsFinite => float.IsFinite(X) && float.IsFinite(Y) && float.IsFinite(Closedness);

    /// <summary>Angular distance to another gaze direction, in degrees.</summary>
    /// <remarks>
    /// A small-angle planar distance rather than a true spherical one. Across ±45° the error is a
    /// few percent, which is far below the tolerances anything here compares against, and it costs
    /// one square root instead of trigonometry in a per-frame path.
    /// </remarks>
    public float DegreesTo(float x, float y)
    {
        var dx = (X - x) * BlinkGuardSettings.GazeRangeDegrees;
        var dy = (Y - y) * BlinkGuardSettings.GazeRangeDegrees;
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    public float DegreesTo(in GazeSample other) => DegreesTo(other.X, other.Y);
}

/// <summary>
/// A small fixed-capacity ring of recent samples. Allocation-free after construction.
/// </summary>
/// <remarks>
/// Used for two things only: the median that defines the pre-blink stable gaze, and the dispersion
/// that decides whether post-blink samples agree. Both are read on a handful of entries, so an
/// insertion-free ring with an on-demand copy beats anything cleverer at this size.
/// </remarks>
public sealed class GazeHistory(int capacity)
{
    private readonly GazeSample[] _items = new GazeSample[Math.Max(1, capacity)];
    private readonly float[] _scratch = new float[Math.Max(1, capacity)];
    private int _next;

    public int Count { get; private set; }

    public int Capacity => _items.Length;

    public void Clear()
    {
        _next = 0;
        Count = 0;
    }

    public void Add(in GazeSample sample)
    {
        _items[_next] = sample;
        _next = (_next + 1) % _items.Length;
        if (Count < _items.Length)
            Count++;
    }

    /// <summary>Newest first: 0 is the most recent sample.</summary>
    public GazeSample this[int indexFromNewest]
    {
        get
        {
            if (indexFromNewest < 0 || indexFromNewest >= Count)
                throw new ArgumentOutOfRangeException(nameof(indexFromNewest));

            var index = _next - 1 - indexFromNewest;
            if (index < 0) index += _items.Length;
            return _items[index];
        }
    }

    /// <summary>
    /// Median of the most recent <paramref name="window"/> samples, per axis.
    /// </summary>
    /// <remarks>
    /// Median rather than mean: a single bad frame as the lid comes down must not drag the position
    /// the entire blink will be held at. Axes are taken independently, which is not a true spatial
    /// median but is what makes it a cheap per-axis rank selection.
    /// </remarks>
    public bool TryMedian(int window, out float x, out float y)
    {
        x = y = 0f;
        var n = Math.Min(window, Count);
        if (n <= 0)
            return false;

        x = MedianOf(n, axisX: true);
        y = MedianOf(n, axisX: false);
        return true;
    }

    private float MedianOf(int n, bool axisX)
    {
        for (var i = 0; i < n; i++)
            _scratch[i] = axisX ? this[i].X : this[i].Y;

        Array.Sort(_scratch, 0, n);
        var middle = n / 2;
        return n % 2 == 1 ? _scratch[middle] : (_scratch[middle - 1] + _scratch[middle]) / 2f;
    }

    /// <summary>
    /// The widest angular gap between any two of the most recent <paramref name="window"/> samples.
    /// </summary>
    /// <remarks>
    /// Pairwise rather than distance-to-mean, because a lone outlier barely moves a mean of three
    /// but is exactly what this has to catch. The window is single digits, so the quadratic scan is
    /// a handful of operations.
    /// </remarks>
    public float DispersionDegrees(int window)
    {
        var n = Math.Min(window, Count);
        if (n < 2)
            return 0f;

        var worst = 0f;
        for (var i = 0; i < n; i++)
        {
            var a = this[i];
            for (var j = i + 1; j < n; j++)
                worst = MathF.Max(worst, a.DegreesTo(this[j]));
        }

        return worst;
    }
}
