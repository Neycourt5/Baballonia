using System;

namespace Baballonia.Services.Inference.BlinkGuard;

/// <summary>How hard BlinkGuard works, for people who do not want to read the advanced settings.</summary>
public enum BlinkGuardPreset
{
    Off,
    Subtle,
    Balanced,
    Strong,
    Custom,
}

/// <summary>
/// Every tunable BlinkGuard has, in one record so none of them become a magic constant.
/// </summary>
/// <remarks>
/// <para>All durations are in seconds and all angles in degrees, because that is what the settings
/// screen and the diagnostics talk in. Gaze itself arrives as [-1,1] per axis spanning
/// <see cref="GazeRangeDegrees"/>, and the conversion happens once, here.</para>
///
/// <para>The defaults are the Balanced preset. They are deliberately short: the whole point is to
/// suppress the post-blink snap without anyone noticing a delay, so every window is measured in
/// tens of milliseconds rather than the tenths of a second a general smoother would use.</para>
/// </remarks>
public sealed record BlinkGuardSettings
{
    /// <summary>Half-range of the gaze axes. ±1 on an axis is ±45°, matching the eye model.</summary>
    public const float GazeRangeDegrees = 45f;

    public bool Enabled { get; init; } = true;

    /// <summary>
    /// Closedness at or above which the eye counts as shut.
    /// </summary>
    /// <remarks>
    /// Separate from <see cref="ReopenThreshold"/> on purpose. One threshold for both directions
    /// chatters when the lid sits near it, which would flip the state machine several times per
    /// blink and defeat the whole feature.
    /// </remarks>
    public float ClosedThreshold { get; init; } = 0.60f;

    /// <summary>Closedness at or below which the eye counts as open again.</summary>
    public float ReopenThreshold { get; init; } = 0.30f;

    /// <summary>
    /// The shortest time the held gaze is kept after the lid reopens, before any new sample can be
    /// accepted. Covers the frames where the lid is off the pupil but the image is still smeared.
    /// </summary>
    public float MinimumHoldSeconds { get; init; } = 0.030f;

    /// <summary>How many consecutive mutually-agreeing samples establish a trustworthy gaze.</summary>
    public int RequiredStableSamples { get; init; } = 3;

    /// <summary>How far apart those samples may be and still count as agreeing.</summary>
    public float StabilityToleranceDegrees { get; init; } = 5f;

    /// <summary>
    /// After this long, stop waiting for agreement and take the best candidate available.
    /// </summary>
    /// <remarks>
    /// The avatar's gaze must never be frozen indefinitely. A timeout is a worse outcome than a
    /// clean reacquisition but a far better one than an eye that has stopped moving.
    /// </remarks>
    public float ReacquisitionTimeoutSeconds { get; init; } = 0.175f;

    /// <summary>How long the eased blend from the held gaze to the reacquired gaze takes.</summary>
    public float TransitionSeconds { get; init; } = 0.080f;

    /// <summary>
    /// If the reacquired gaze is within this of where it was held, skip the blend entirely.
    /// </summary>
    /// <remarks>
    /// The common case: the user blinked and is still looking at the same thing. Blending toward a
    /// position we are already at would add latency for no visible benefit.
    /// </remarks>
    public float ImmediateResumeDegrees { get; init; } = 2f;

    /// <summary>How many samples the pre-blink stable position is taken from.</summary>
    /// <remarks>
    /// A median over a handful of frames, not a mean: one bad frame just before the lid closes must
    /// not become the position the whole blink is held at. Kept tiny so normal tracking gains no
    /// perceptible latency - this window is only ever *read*, never output, during Normal.
    /// </remarks>
    public int StableWindowSamples { get; init; } = 5;

    /// <summary>
    /// Whether to require a confidence value before trusting a post-blink sample.
    /// </summary>
    /// <remarks>
    /// Off, and it is not an oversight. This pipeline runs its own model over the eye cameras and
    /// produces no confidence channel - unlike the vendor VRCFT modules, which expose one in shared
    /// memory. The plumbing is here so a source that does have one can switch it on, but every
    /// default is built to work from temporal consistency and validity alone.
    /// </remarks>
    public bool UseConfidence { get; init; }

    /// <summary>Confidence below which a sample is rejected, when <see cref="UseConfidence"/> is on.</summary>
    public float ConfidenceThreshold { get; init; } = 0.5f;

    /// <summary>
    /// A very conservative single-frame spike veto outside blinks. Off by default.
    /// </summary>
    /// <remarks>
    /// Deliberately separate from BlinkGuard proper: the post-blink snap is the actual complaint,
    /// and a guard that runs during ordinary tracking risks clipping real saccades, which is a
    /// worse and much more noticeable failure.
    /// </remarks>
    public bool NormalSpikeGuardEnabled { get; init; }

    /// <summary>A single-frame jump beyond this is held for one frame pending confirmation.</summary>
    public float NormalSpikeGuardDegrees { get; init; } = 25f;

    public BlinkGuardPreset Preset { get; init; } = BlinkGuardPreset.Balanced;

    public static BlinkGuardSettings Balanced { get; } = new();

    public static BlinkGuardSettings ForPreset(BlinkGuardPreset preset) => preset switch
    {
        BlinkGuardPreset.Off => new BlinkGuardSettings { Enabled = false, Preset = preset },

        // Intervenes only on the clearest failures, and gives up quickly.
        BlinkGuardPreset.Subtle => new BlinkGuardSettings
        {
            Preset = preset,
            MinimumHoldSeconds = 0.020f,
            RequiredStableSamples = 2,
            StabilityToleranceDegrees = 7f,
            ReacquisitionTimeoutSeconds = 0.120f,
            TransitionSeconds = 0.060f,
        },

        BlinkGuardPreset.Balanced => new BlinkGuardSettings { Preset = preset },

        // Waits for more agreement before believing the eye. Costs a little more time to reacquire.
        BlinkGuardPreset.Strong => new BlinkGuardSettings
        {
            Preset = preset,
            MinimumHoldSeconds = 0.045f,
            RequiredStableSamples = 4,
            StabilityToleranceDegrees = 4f,
            ReacquisitionTimeoutSeconds = 0.200f,
            TransitionSeconds = 0.100f,
        },

        _ => new BlinkGuardSettings { Preset = BlinkGuardPreset.Custom },
    };

    /// <summary>
    /// Clamps every field into a range the algorithm is known to behave in.
    /// </summary>
    /// <remarks>
    /// Applied at load, not only in the UI, so a hand-edited settings file cannot put the state
    /// machine somewhere it was never tested - an inverted hysteresis pair especially, which would
    /// latch the eye shut.
    /// </remarks>
    public BlinkGuardSettings Sanitized()
    {
        var closed = Clamp(ClosedThreshold, 0.10f, 0.95f, 0.60f);
        var reopen = Clamp(ReopenThreshold, 0.05f, 0.90f, 0.30f);

        // Hysteresis, not a coin toss: reopening must be strictly easier than closing.
        if (reopen >= closed)
            reopen = Math.Max(0.05f, closed - 0.10f);

        return this with
        {
            ClosedThreshold = closed,
            ReopenThreshold = reopen,
            MinimumHoldSeconds = Clamp(MinimumHoldSeconds, 0f, 0.200f, 0.030f),
            RequiredStableSamples = Math.Clamp(RequiredStableSamples, 1, 15),
            StabilityToleranceDegrees = Clamp(StabilityToleranceDegrees, 0.5f, 30f, 5f),
            ReacquisitionTimeoutSeconds = Clamp(ReacquisitionTimeoutSeconds, 0.030f, 1f, 0.175f),
            TransitionSeconds = Clamp(TransitionSeconds, 0f, 0.500f, 0.080f),
            ImmediateResumeDegrees = Clamp(ImmediateResumeDegrees, 0f, 20f, 2f),
            StableWindowSamples = Math.Clamp(StableWindowSamples, 1, 32),
            ConfidenceThreshold = Clamp(ConfidenceThreshold, 0f, 1f, 0.5f),
            NormalSpikeGuardDegrees = Clamp(NormalSpikeGuardDegrees, 5f, 90f, 25f),
        };
    }

    private static float Clamp(float value, float min, float max, float fallback) =>
        float.IsFinite(value) ? Math.Clamp(value, min, max) : fallback;
}
