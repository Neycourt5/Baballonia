using System;

namespace Baballonia.Services.Inference;

/// <summary>
/// Shapes one eyelid's final value: closing passes through untouched, reopening is rate-limited.
/// </summary>
/// <remarks>
/// <para>The asymmetry is the whole design. A blink's <em>onset</em> is the part people notice, and
/// any delay there reads immediately as laggy tracking - so the falling edge is passed through with
/// zero added latency. The <em>reopen</em> is where the model is least certain: the lid crosses the
/// whole openness range in a few frames while the eyelash and iris are still moving, and small
/// disagreements between consecutive frames show up as a visible flutter just after the eye opens.
/// Limiting only that edge removes the flutter without touching how a blink starts.</para>
///
/// <para><see cref="RisePerSecond"/> of 10 means a full closed-to-open recovery takes about 100 ms,
/// which is close to how fast a real lid reopens anyway - so this is shaping the value toward
/// plausibility rather than slowing it down.</para>
///
/// <para>One instance per eye; nothing is shared, so winks and asymmetric blinks are preserved.</para>
/// </remarks>
public sealed class BlinkReopenLimiter
{
    /// <summary>Maximum increase in openness per second. 10/s ≈ 100 ms for a full reopen.</summary>
    public const float RisePerSecond = 10.0f;

    private float _value;
    private bool _primed;

    public void Reset()
    {
        _value = 0f;
        _primed = false;
    }

    /// <summary>
    /// Returns the shaped lid value for this frame.
    /// </summary>
    /// <param name="target">Calibrated openness, 1 = open.</param>
    /// <param name="dt">Seconds since the previous frame.</param>
    public float Apply(float target, float dt)
    {
        if (!float.IsFinite(target))
            return _primed ? _value : 1f;

        // The first frame, and the first frame after a reset, must not ramp up from zero: the eye
        // would appear to blink open every time tracking starts or a camera reconnects.
        if (!_primed)
        {
            _primed = true;
            _value = target;
            return _value;
        }

        if (!float.IsFinite(dt) || dt <= 0f)
            return _value;

        // Closing: straight through. Any delay here is felt as latency.
        if (target <= _value)
        {
            _value = target;
            return _value;
        }

        _value = Math.Min(target, _value + RisePerSecond * dt);
        return _value;
    }
}
