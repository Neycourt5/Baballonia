using System;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Personalization.Audio;

/// <summary>
/// An optional last stage that may adjust expression values before they are smoothed and sent.
/// </summary>
/// <remarks>
/// Same shape as <see cref="IExpressionCorrector"/> and the same rule: null means the stage does not
/// exist, and the pipeline is byte-for-byte what it was without it.
/// </remarks>
public interface IExpressionEnhancer
{
    /// <summary>
    /// Returns adjusted values, or the input unchanged. Must never throw, and must return a new
    /// array rather than mutating the caller's.
    /// </summary>
    float[] Enhance(float[] expressions);
}

/// <summary>
/// Makes speech look more animated by amplifying mouth movement the face tracker already sees.
///
/// The problem it addresses is that accurate tracking can still read as *subtle*: the mouth is doing
/// the right thing, just not very much of it, and on an avatar across a room that reads as flat. The
/// fix is not to invent motion but to scale up motion that is genuinely there, in proportion to how
/// energetically the person is speaking.
///
/// <para><b>The rule that matters.</b> This multiplies, it never adds. If the tracker says
/// <c>MouthSmileLeft = 0</c>, then <c>0 x anything</c> is still 0, no matter how loudly the user
/// shouts. Audio can make a real smile bigger; it cannot manufacture one. That is the whole
/// difference between this and the viseme-driven mouth animation it is meant to avoid becoming - the
/// camera remains the only thing that decides *what* the face is doing.</para>
///
/// Only jaw and mouth expressions are touched. Cheeks, nose and tongue are left alone because
/// loudness says nothing about them.
/// </summary>
public sealed class ProsodyEnhancer : IExpressionEnhancer
{
    /// <summary>Largest multiplier at full strength and full speech energy.</summary>
    public const float MaxGain = 1.6f;

    /// <summary>
    /// Below this, an expression is treated as "not really happening" and left alone.
    /// </summary>
    /// <remarks>
    /// Without it, amplification would scale up the tracker's own noise floor while the face is at
    /// rest - turning a steady 0.02 into 0.03 across the mouth, which is exactly the resting-jitter
    /// complaint this project spent a milestone fixing.
    /// </remarks>
    public const float MovementFloor = 0.05f;

    /// <summary>Features older than this are ignored; a stalled capture must not freeze the gain.</summary>
    public static readonly TimeSpan MaxFeatureAge = TimeSpan.FromMilliseconds(250);

    private static readonly int[] AffectedDims = BuildAffectedDims();

    private readonly IAudioFeatureSource _source;
    private readonly Func<long> _now;

    private float _smoothedGain = 1f;

    /// <summary>0..1. Scales the whole effect; 0 makes this an exact passthrough.</summary>
    public float Strength { get; set; } = 0.5f;

    /// <summary>
    /// Audio is sampled this far in the past, to line up with the camera.
    /// </summary>
    /// <remarks>
    /// Microphone and camera do not arrive with the same latency, and the visible failure is a mouth
    /// that swells slightly before or after the sound. Configurable because the right value depends
    /// on the capture stack and the headset, and it is easier to nudge than to derive.
    /// </remarks>
    public TimeSpan SyncOffset { get; set; } = TimeSpan.FromMilliseconds(50);

    /// <summary>Gain applied on the most recent call, for the diagnostics readout.</summary>
    public float LastGain => _smoothedGain;

    public ProsodyEnhancer(IAudioFeatureSource source, Func<long>? clock = null)
    {
        _source = source;
        _now = clock ?? (() => DateTime.UtcNow.Ticks);
    }

    private static int[] BuildAffectedDims()
    {
        // Jaw and mouth only. Named rather than indexed so a schema reorder cannot silently point
        // this at the tongue.
        var wanted = new List<int>();
        for (var i = 0; i < PersonalizationSchema.ExpressionCount; i++)
        {
            var name = PersonalizationSchema.ExpressionNames[i];
            if (name.StartsWith("Jaw", StringComparison.Ordinal) ||
                name.StartsWith("Mouth", StringComparison.Ordinal))
            {
                wanted.Add(i);
            }
        }

        return wanted.ToArray();
    }

    /// <inheritdoc />
    public float[] Enhance(float[] expressions)
    {
        var output = (float[])expressions.Clone();

        try
        {
            var gain = ComputeGain();

            // Exactly 1 means there is nothing to do, and skipping keeps the "assist on but silent"
            // case bit-identical to the visual path rather than merely close to it.
            if (Math.Abs(gain - 1f) < 1e-6f)
                return output;

            foreach (var dim in AffectedDims)
            {
                var value = output[dim];
                if (value <= MovementFloor)
                    continue;

                // Only the part above the floor is scaled, so a value sitting just over the
                // threshold is not jerked upward as it crosses it.
                var boosted = MovementFloor + (value - MovementFloor) * gain;
                output[dim] = Math.Clamp(boosted, 0f, 1f);
            }

            return output;
        }
        catch
        {
            // An enhancement is never worth breaking tracking over.
            return output;
        }
    }

    private float ComputeGain()
    {
        var strength = Math.Clamp(Strength, 0f, 1f);
        if (strength <= 0f)
        {
            _smoothedGain = 1f;
            return 1f;
        }

        var features = _source.Latest;
        var now = _now();

        // Stale or silent both mean "no reason to boost". Decaying rather than snapping avoids a
        // visible step when the user stops talking or capture hiccups.
        var target = 1f;
        if (features.TimestampTicks > 0)
        {
            var age = features.AgeAt(now - SyncOffset.Ticks);
            if (age <= MaxFeatureAge && features.IsVoiced)
            {
                // Emphasis counts for a little extra: a stressed syllable is what makes speech look
                // animated rather than merely loud.
                var drive = Math.Clamp(features.SpeechEnergy + features.Onset * 0.5f, 0f, 1f);
                target = 1f + (MaxGain - 1f) * drive * strength;
            }
        }

        _smoothedGain += (target - _smoothedGain) * 0.25f;

        if (Math.Abs(_smoothedGain - 1f) < 1e-4f)
            _smoothedGain = 1f;

        return Math.Clamp(_smoothedGain, 1f, MaxGain);
    }
}
