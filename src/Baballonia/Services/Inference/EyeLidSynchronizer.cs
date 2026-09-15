using System;

namespace Baballonia.Services.Inference;

/// <summary>Which eye the other one is pulled toward.</summary>
/// <remarks>
/// <see cref="Average"/> treats the two cameras as equally trustworthy, which is right when nothing
/// is known about them. When one camera genuinely reads better — a better angle, better lighting, a
/// lash that keeps catching on the other side — naming it is strictly more useful than averaging,
/// because averaging drags the good eye halfway toward the bad one.
/// </remarks>
public enum EyeSyncSource
{
    Average,
    Left,
    Right,

    /// <summary>
    /// Whichever eye reports the expression more strongly wins, so an expression only one camera
    /// picks up still reaches both eyes at full strength.
    /// </summary>
    /// <remarks>
    /// <para>The right answer when the two cameras disagree by <em>missing</em> things rather than
    /// by being biased. Averaging a squint one camera reads at 0.9 and the other at 0.2 produces a
    /// half-hearted 0.55 on both eyes - the expression is weakened rather than synchronized, which
    /// is the opposite of what someone asking for a stronger squint wants.</para>
    ///
    /// <para>Applies to squint and widen only. On the lid channel "stronger" would have to mean
    /// "more closed", and that makes a single camera mis-reading one eye close both, while a droopy
    /// resting eye drags the other one down with it. Lids fall back to
    /// <see cref="Average"/>.</para>
    /// </remarks>
    Stronger,
}

/// <summary>
/// Pulls the two eyelids toward each other, and gets out of the way for a deliberate wink.
/// </summary>
/// <remarks>
/// <para>Real eyelids are yoked: both are driven by the same nerve, so they blink together and rest
/// at nearly the same openness. The model estimates each eye separately from its own camera, so its
/// two lid values disagree by whatever the two images happen to disagree about — different lighting,
/// a lash, a slightly different angle. That disagreement is not information; on an avatar it reads
/// as one eye lagging, or as a permanent half-squint on one side.</para>
///
/// <para>Coupling them fixes that but would also make winking impossible, so the coupling yields to
/// a wink. Releasing is <em>immediate</em>, because a wink that takes an eighth of a second to start
/// reads as broken tracking, while re-coupling waits for <see cref="RecoupleDwellSeconds"/>, because
/// re-engaging the instant a wink ends would chatter.</para>
///
/// <para>One instance per pipeline — this is inherently a two-eye operation.</para>
/// </remarks>
public sealed class EyeLidSynchronizer
{
    /// <summary>A lid at or below this openness counts as closed for wink detection.</summary>
    public const float WinkClosedOpenness = 0.25f;

    /// <summary>A lid at or above this openness counts as open for wink detection.</summary>
    public const float WinkOpenOpenness = 0.60f;

    /// <summary>How long the wink shape must be absent before coupling re-engages.</summary>
    public const float RecoupleDwellSeconds = 0.120f;

    private readonly float _amount;
    private readonly EyeSyncSource _source;
    private readonly bool _detectWinks;
    private bool _released;
    private float _steadySeconds;

    /// <param name="amount">
    /// 0 = fully independent (the lids are returned untouched), 1 = both lids take the source value
    /// exactly. Intermediate values pull each lid that fraction of the way toward it.
    /// </param>
    /// <param name="source">Which eye is authoritative. See <see cref="EyeSyncSource"/>.</param>
    /// <param name="detectWinks">
    /// When false the coupling never releases. For someone who does not wink, the escape hatch is
    /// pure downside: every false positive is a moment of the asymmetry they turned coupling on to
    /// remove, bought for a gesture they never make.
    /// </param>
    public EyeLidSynchronizer(
        float amount, EyeSyncSource source = EyeSyncSource.Average, bool detectWinks = true)
    {
        _amount = Math.Clamp(amount, 0f, 1f);
        _source = source;
        _detectWinks = detectWinks;
    }

    /// <summary>True while a wink has released the coupling. Diagnostics only.</summary>
    public bool IsReleased => _released;

    /// <summary>How strongly the lids are pulled together.</summary>
    public float Amount => _amount;

    /// <summary>Which eye the other follows.</summary>
    public EyeSyncSource Source => _source;

    /// <summary>Whether a wink can release the coupling at all.</summary>
    public bool DetectsWinks => _detectWinks;

    public void Reset()
    {
        _released = false;
        _steadySeconds = 0f;
    }

    /// <summary>
    /// Whether these two lids look like a deliberate wink: one genuinely closed while the other
    /// stays genuinely open.
    /// </summary>
    /// <remarks>
    /// <para>Deliberately <em>not</em> "the lids differ a lot". That test cannot work, because a
    /// large disagreement between the two cameras is the exact symptom this class exists to remove —
    /// keying the escape hatch on it means the coupling switches itself off precisely for the people
    /// who need it, and a resting asymmetry latches the release permanently. Absolute positions do
    /// not have that problem: two lids that disagree while both sit in the open range are two
    /// cameras arguing, not a wink.</para>
    ///
    /// <para>When a <see cref="EyeSyncSource"/> names one eye as authoritative, only that eye
    /// closing counts as a wink. If you have said the left camera reads correctly, then the right
    /// one reading closed while the left reads open is the right camera being wrong — and a wink is
    /// something you do on purpose, so it will show up on the eye that tracks you properly.</para>
    /// </remarks>
    public bool IsWinkShape(float left, float right)
    {
        if (!_detectWinks)
            return false;

        var leftWink = left <= WinkClosedOpenness && right >= WinkOpenOpenness;
        var rightWink = right <= WinkClosedOpenness && left >= WinkOpenOpenness;

        return _source switch
        {
            EyeSyncSource.Left => leftWink,
            EyeSyncSource.Right => rightWink,
            _ => leftWink || rightWink,
        };
    }

    /// <summary>Returns the synchronized pair. Openness values, 1 = open.</summary>
    public (float Left, float Right) Apply(float left, float right, float dt)
    {
        if (_amount <= 0f)
            return (left, right);

        // Never invent a value from a broken one; the finite guard upstream owns that decision.
        if (!float.IsFinite(left) || !float.IsFinite(right))
            return (left, right);

        if (!float.IsFinite(dt) || dt < 0f)
            dt = 0f;

        if (IsWinkShape(left, right))
        {
            // Immediate: any delay here is felt as a wink that will not start.
            _released = true;
            _steadySeconds = 0f;
        }
        else if (_released)
        {
            _steadySeconds += dt;
            if (_steadySeconds >= RecoupleDwellSeconds)
            {
                _released = false;
                _steadySeconds = 0f;
            }
        }

        // Lids never use Stronger: see the remarks on EyeSyncSource.Stronger.
        var source = _source == EyeSyncSource.Stronger ? EyeSyncSource.Average : _source;
        return _released ? (left, right) : Pull(left, right, _amount, source);
    }

    /// <summary>
    /// Pulls a left/right pair toward the chosen source. Shared with the expression channels, which
    /// need the same arithmetic without the wink state machine.
    /// </summary>
    /// <remarks>
    /// With a named source that eye is returned untouched and only the other one moves, so naming
    /// your better camera costs the good eye nothing.
    /// </remarks>
    public static (float Left, float Right) Pull(
        float left, float right, float amount, EyeSyncSource source)
    {
        if (amount <= 0f || !float.IsFinite(left) || !float.IsFinite(right))
            return (left, right);

        var target = source switch
        {
            EyeSyncSource.Left => left,
            EyeSyncSource.Right => right,
            EyeSyncSource.Stronger => Math.Max(left, right),
            _ => (left + right) * 0.5f,
        };

        return (left + (target - left) * amount, right + (target - right) * amount);
    }
}
