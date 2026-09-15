using System;

namespace Baballonia.Services.Inference;

/// <summary>
/// Infers widen and squint for one eye from its calibrated openness, for models that do not emit
/// those channels themselves.
/// </summary>
/// <remarks>
/// <para><strong>This is a fallback, not a feature.</strong> The user's tuned eye model emits real
/// per-eye Widen/Squint/Brow, and those are always preferred - a value the network actually
/// predicted carries information that openness alone does not. This exists so the stock
/// six-output model, and older tuned models without metadata, are not simply missing two
/// expressions.</para>
///
/// <para>Openness alone is an ambiguous signal: a half-closed eye is a squint, a blink in progress,
/// or a blink recovering, and the difference matters a great deal to how an avatar reads. Three
/// mechanisms keep that ambiguity from becoming visible flicker:</para>
///
/// <list type="bullet">
/// <item><description><em>Hysteresis</em> - entering a state takes a stronger signal than staying
/// in it, so openness hovering on a threshold cannot oscillate.</description></item>
/// <item><description><em>Dwell</em> - a state change must be wanted continuously for
/// <see cref="DwellSeconds"/> before it is committed, so a single noisy frame changes
/// nothing.</description></item>
/// <item><description><em>Slew</em> - the output moves at a bounded rate, so even a committed
/// change arrives as a movement rather than a step.</description></item>
/// </list>
///
/// <para>Squint is suppressed while a blink is in progress. Without that, every blink passes
/// through the squint band on the way down and again on the way up, and the avatar squints twice
/// per blink - the single most objectionable artefact of deriving these at all.</para>
///
/// <para>One instance per eye. Nothing here is shared between the two, which is what preserves
/// winks and natural left/right asymmetry.</para>
/// </remarks>
public sealed class DerivedEyeShapeEstimator
{
    /// <summary>Below this calibrated openness the eye is treated as blinking, not squinting.</summary>
    public const float BlinkThreshold = 0.15f;

    /// <summary>Openness required to start widening.</summary>
    public const float WidenEnter = 0.85f;

    /// <summary>Openness required to keep widening once started.</summary>
    public const float WidenExit = 0.80f;

    /// <summary>Openness required to start squinting.</summary>
    public const float SquintEnter = 0.50f;

    /// <summary>Openness required to keep squinting once started.</summary>
    public const float SquintExit = 0.55f;

    /// <summary>How long a state change must be wanted continuously before it is committed.</summary>
    public const float DwellSeconds = 0.100f;

    /// <summary>Maximum output change per second, in expression units.</summary>
    public const float SlewPerSecond = 4.0f;

    public enum Shape
    {
        Neutral,
        Widen,
        Squint,
    }

    private Shape _committed = Shape.Neutral;
    private Shape _candidate = Shape.Neutral;
    private float _candidateHeldSeconds;
    private float _widen;
    private float _squint;

    /// <summary>The committed state. Exposed for diagnostics and tests, not for control flow.</summary>
    public Shape State => _committed;

    public float Widen => _widen;

    public float Squint => _squint;

    public void Reset()
    {
        _committed = Shape.Neutral;
        _candidate = Shape.Neutral;
        _candidateHeldSeconds = 0f;
        _widen = 0f;
        _squint = 0f;
    }

    /// <summary>
    /// Advances one frame.
    /// </summary>
    /// <param name="calibratedOpenness">
    /// Openness in output space (1 = open), after per-eye calibration. Using the calibrated value
    /// rather than the raw one is what makes the fixed thresholds meaningful across different
    /// people and camera placements.
    /// </param>
    /// <param name="dt">Seconds since the previous frame. Non-finite or negative is treated as 0.</param>
    public void Update(float calibratedOpenness, float dt)
    {
        if (!float.IsFinite(calibratedOpenness))
            calibratedOpenness = EyeCalibrationRest;
        calibratedOpenness = Math.Clamp(calibratedOpenness, 0f, 1f);

        if (!float.IsFinite(dt) || dt < 0f)
            dt = 0f;

        var blinking = calibratedOpenness < BlinkThreshold;
        var desired = DesiredShape(calibratedOpenness, blinking);

        if (desired == _committed)
        {
            // Already where we want to be; drop any half-accumulated intent to change.
            _candidate = _committed;
            _candidateHeldSeconds = 0f;
        }
        else if (desired == _candidate)
        {
            _candidateHeldSeconds += dt;
            if (_candidateHeldSeconds >= DwellSeconds)
            {
                _committed = desired;
                _candidateHeldSeconds = 0f;
            }
        }
        else
        {
            _candidate = desired;
            _candidateHeldSeconds = 0f;
        }

        var (widenTarget, squintTarget) = TargetsFor(_committed, calibratedOpenness);

        var step = SlewPerSecond * dt;
        _widen = Approach(_widen, widenTarget, step);
        _squint = Approach(_squint, squintTarget, step);
    }

    private Shape DesiredShape(float openness, bool blinking) => _committed switch
    {
        // Leaving a state needs less than entering it did: that gap is the hysteresis.
        Shape.Widen => openness > WidenExit ? Shape.Widen : Shape.Neutral,
        Shape.Squint => !blinking && openness < SquintExit ? Shape.Squint : Shape.Neutral,
        _ when openness > WidenEnter => Shape.Widen,
        // A blink passes straight through the squint band twice. Suppressing squint here is what
        // stops every blink from reading as a deliberate squint.
        _ when !blinking && openness < SquintEnter => Shape.Squint,
        _ => Shape.Neutral,
    };

    private static (float Widen, float Squint) TargetsFor(Shape shape, float openness) => shape switch
    {
        Shape.Widen => (Math.Clamp((openness - WidenExit) / (1f - WidenExit), 0f, 1f), 0f),
        // Saturates at the blink threshold rather than at fully closed, so a hard squint reads as a
        // hard squint instead of asymptotically approaching one somewhere inside a blink.
        Shape.Squint => (0f, Math.Clamp((SquintExit - openness) / (SquintExit - BlinkThreshold), 0f, 1f)),
        _ => (0f, 0f),
    };

    private static float Approach(float current, float target, float maxStep)
    {
        var delta = target - current;
        if (Math.Abs(delta) <= maxStep)
            return target;
        return current + Math.Sign(delta) * maxStep;
    }

    /// <summary>
    /// Where calibration puts a resting eye. Used only as the safe substitute for a non-finite
    /// input, so corruption reads as "at rest" rather than as a sudden wide-eyed stare.
    /// </summary>
    private const float EyeCalibrationRest = 0.75f;
}
