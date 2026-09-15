using System;
using System.Collections.Generic;

namespace Baballonia.Services.Inference.Filters;

public class OneEuroFilter : IFilter
{
    private readonly float _initialMinCutoff;
    private readonly float _initialBeta;
    private readonly (string A, string B)[] _pairedKeys;
    private bool _isInitialized;

    /// <summary>
    /// For each channel, the index of the channel whose speed also raises this one's cutoff, or -1.
    /// </summary>
    private int[] _speedPartner;

    private string[] _keys;
    private float[] _minCutoff;
    private float[] _beta;
    private float[] _dCutoff;
    private float[] _xPrev;
    private float[] _dxPrev;

    private float[] _dx;
    private float[] _dxHat;
    private float[] _cutoff;
    private float[] _xHat;

    // Reusable output map
    private OrderedFloatMap _output;
    // The input map we sized our state to; detects a model hot-swap.
    private OrderedFloatMap _source;
    private DateTime _tPrev;
    private readonly Func<DateTime> _clock;

    /// <param name="pairedKeys">
    /// Channel pairs that should share one speed estimate when choosing their cutoff.
    /// </param>
    /// <remarks>
    /// <para>Why pairing is needed at all. One Euro widens its cutoff with the channel's <em>own</em>
    /// speed, so a channel that reports a smaller movement is also smoothed harder. For two eyes
    /// looking at the same point that is a bug in disguise: the eye whose camera reports a
    /// compressed range - typically the one looking outward, where the iris is near the corner - is
    /// penalised twice, once for reporting less movement and again by being filtered more for it.
    /// The two eyes then arrive at different times.</para>
    ///
    /// <para>Pairing takes the larger of the two speeds, so a saccade one eye reports clearly opens
    /// the cutoff for both. This can only ever <em>raise</em> a cutoff, never lower one, so no
    /// channel is made laggier than it was; the cost is that the quieter eye passes a little more
    /// noise during a movement, which is exactly when noise is least visible.</para>
    /// </remarks>
    public OneEuroFilter(
        float minCutoff = 1.0f, float beta = 0.0f, (string A, string B)[]? pairedKeys = null,
        Func<DateTime>? clock = null)
    {
        _initialMinCutoff = minCutoff;
        _initialBeta = beta;
        _pairedKeys = pairedKeys ?? [];
        _clock = clock ?? (() => DateTime.UtcNow);
        _isInitialized = false;
    }

    /// <summary>The two eyes look at one point, so their gaze channels share a speed estimate.</summary>
    public static (string A, string B)[] EyeGazePairs { get; } =
        [("/leftEyeX", "/rightEyeX"), ("/leftEyeY", "/rightEyeY")];

    public OrderedFloatMap Filter(OrderedFloatMap x)
    {
        // (Re)init on first frame or after a model swap. State is sized to the first frame;
        // a shrunk output overruns it, a grown one gets truncated/mis-keyed.
        if (!_isInitialized || ShapeChanged(x))
        {
            Initialize(x);
            return _output; // Return the initial state on the first frame
        }

        DateTime now = _clock();
        float elapsedTime = (float)(now - _tPrev).TotalSeconds;

        ReadOnlySpan<float> xSpan = x.ValuesSpan;
        Span<float> outSpan = _output.ValuesSpan;
        int length = _xPrev.Length;

        if (elapsedTime <= 0.0f)
        {
            ReadOnlySpan<float> src = xSpan.Slice(0, length);
            src.CopyTo(outSpan);
            src.CopyTo(_xPrev);
            return _output;
        }

        // Speeds first, for every channel: a paired channel's cutoff depends on its partner's
        // speed, which is not known until the partner has been through this.
        for (int i = 0; i < length; i++)
        {
            _dx[i] = (xSpan[i] - _xPrev[i]) / elapsedTime;

            float r_d = MathF.Tau * _dCutoff[i] * elapsedTime;
            float a_d = r_d / (r_d + 1.0f);

            _dxHat[i] = a_d * _dx[i] + (1.0f - a_d) * _dxPrev[i];
        }

        for (int i = 0; i < length; i++)
        {
            float val = xSpan[i];

            var speed = MathF.Abs(_dxHat[i]);
            int partner = _speedPartner[i];
            if (partner >= 0)
                speed = MathF.Max(speed, MathF.Abs(_dxHat[partner]));

            _cutoff[i] = _minCutoff[i] + _beta[i] * speed;

            float r = MathF.Tau * _cutoff[i] * elapsedTime;
            float a = r / (r + 1.0f);

            _xHat[i] = a * val + (1.0f - a) * _xPrev[i];

            _xPrev[i] = _xHat[i];
            _dxPrev[i] = _dxHat[i];

            outSpan[i] = _xHat[i];
        }

        _tPrev = now;
        return _output;
    }

    // True when x's output set differs from ours. The runner reuses one map per model, so a
    // reference change flags a swap; length/keys confirm it so an equivalent map never resets us.
    private bool ShapeChanged(OrderedFloatMap x)
    {
        if (ReferenceEquals(x, _source))
            return false;

        if (x.Count != _keys.Length)
            return true;

        int i = 0;
        foreach (string key in x.Keys)
        {
            if (!string.Equals(key, _keys[i], StringComparison.Ordinal))
                return true;
            i++;
        }

        // Same shape, new instance — adopt it for the fast path.
        _source = x;
        return false;
    }

    private void Initialize(OrderedFloatMap x0)
    {
        int length = x0.Count;

        _keys = new string[length];
        _minCutoff = new float[length];
        _beta = new float[length];
        _dCutoff = new float[length];
        _xPrev = new float[length];
        _dxPrev = new float[length];

        _dx = new float[length];
        _dxHat = new float[length];
        _cutoff = new float[length];
        _xHat = new float[length];

        int i = 0;
        foreach (KeyValuePair<string, float> kvp in x0)
        {
            _keys[i] = kvp.Key;
            _xPrev[i] = kvp.Value;
            _minCutoff[i] = _initialMinCutoff;
            _beta[i] = _initialBeta;
            _dCutoff[i] = 1.0f;
            _dxPrev[i] = 0.0f;
            i++;
        }

        _speedPartner = new int[length];
        Array.Fill(_speedPartner, -1);
        foreach (var (a, b) in _pairedKeys)
        {
            int ia = Array.IndexOf(_keys, a);
            int ib = Array.IndexOf(_keys, b);
            if (ia < 0 || ib < 0)
                continue;

            _speedPartner[ia] = ib;
            _speedPartner[ib] = ia;
        }

        _output = new OrderedFloatMap(_keys);
        _source = x0;

        _xPrev.CopyTo(_output.ValuesSpan);

        _tPrev = _clock();
        _isInitialized = true;
    }
}
