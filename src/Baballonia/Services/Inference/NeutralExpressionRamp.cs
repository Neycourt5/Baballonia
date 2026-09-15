using System;
using System.Linq;

namespace Baballonia.Services.Inference;

/// <summary>
/// Holds the last emitted keyed expression and produces a bounded relaxation toward a key-specific
/// neutral value while its camera is reconnecting. One neutral sample is emitted at the endpoint;
/// subsequent samples are silent so downstream can hold neutral without needless traffic.
/// </summary>
public sealed class NeutralExpressionRamp(
    Func<string, float> neutralForKey,
    TimeSpan? duration = null,
    Func<DateTime>? utcNow = null)
{
    public static readonly TimeSpan DefaultDuration = TimeSpan.FromMilliseconds(500);

    private readonly TimeSpan _duration = duration ?? DefaultDuration;
    private readonly Func<DateTime> _utcNow = utcNow ?? (() => DateTime.UtcNow);
    private OrderedFloatMap? _lastEmitted;
    private OrderedFloatMap? _rampFrom;
    private DateTime? _startedUtc;
    private bool _settled;

    /// <summary>Snapshots a real pipeline result; runner/filter maps are mutable and reused.</summary>
    public void Observe(OrderedFloatMap expression)
    {
        _lastEmitted = Clone(expression);
        ResetOutage();
    }

    public OrderedFloatMap? Sample(bool reconnecting)
    {
        if (!reconnecting)
        {
            ResetOutage();
            return null;
        }

        if (_settled)
            return null;

        var now = _utcNow();
        if (_startedUtc is null)
        {
            if (_lastEmitted is null)
            {
                _settled = true;
                return null;
            }

            _rampFrom = Clone(_lastEmitted);
            _startedUtc = now;
        }

        var elapsed = now - _startedUtc.Value;
        var denominator = Math.Max(_duration.TotalMilliseconds, double.Epsilon);
        var t = Math.Clamp(elapsed.TotalMilliseconds / denominator, 0d, 1d);
        var result = Interpolate(_rampFrom!, neutralForKey, t);

        if (t >= 1d)
            _settled = true;

        return result;
    }

    public static OrderedFloatMap Interpolate(
        OrderedFloatMap from,
        Func<string, float> neutralForKey,
        double progress)
    {
        var keys = from.Keys.ToArray();
        var result = new OrderedFloatMap(keys);
        var t = Math.Clamp(progress, 0d, 1d);

        foreach (var key in keys)
        {
            var start = from[key];
            var neutral = neutralForKey(key);
            result[key] = (float)(start + (neutral - start) * t);
        }

        return result;
    }

    private void ResetOutage()
    {
        _rampFrom = null;
        _startedUtc = null;
        _settled = false;
    }

    private static OrderedFloatMap Clone(OrderedFloatMap source)
    {
        var result = new OrderedFloatMap(source.Keys.ToArray());
        foreach (var pair in source)
            result[pair.Key] = pair.Value;
        return result;
    }
}
