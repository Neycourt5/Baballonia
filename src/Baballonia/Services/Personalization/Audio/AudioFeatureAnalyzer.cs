using System;

namespace Baballonia.Services.Personalization.Audio;

/// <summary>
/// Turns windows of microphone samples into <see cref="AudioFeatures"/>.
///
/// Deliberately old-fashioned signal processing - RMS, a minimum-follower noise floor,
/// autocorrelation pitch - rather than anything learned. Three reasons: it costs microseconds, it
/// behaves predictably on audio nobody tested it against, and it cannot do anything beyond what it
/// says on the tin. A model would be a second thing to train, ship, and reason about, for a feature
/// whose entire job is answering "is this person talking, and how energetically?"
///
/// The class is stateful across windows (noise floor, smoothing, onset history) but touches no
/// hardware, so it can be driven from a test at any rate with any waveform.
/// </summary>
public sealed class AudioFeatureAnalyzer
{
    /// <summary>Analysis window. Long enough for a pitch period at 60 Hz, short enough to feel live.</summary>
    public const int DefaultSampleRate = 16000;

    /// <summary>Human speech fundamentals sit inside this band; searching wider mostly finds noise.</summary>
    private const float MinPitchHz = 60f;
    private const float MaxPitchHz = 400f;

    /// <summary>Voice has to clear the noise floor by this much to count. ~6 dB.</summary>
    private const float VoicedRatio = 2.0f;

    /// <summary>Below this the signal is too quiet to say anything about, whatever the ratio says.</summary>
    private const float AbsoluteSilence = 1e-4f;

    /// <summary>
    /// RMS above the floor that counts as "fully animated speech". Normal conversation sits well
    /// below this, so the enhancer's strongest boost is reserved for genuinely raised voices.
    /// </summary>
    private const float FullEnergyRms = 0.20f;

    // Smoothing. Fast attack so emphasis is not missed, slow release so the mouth does not snap
    // shut between words - the same asymmetry a compressor uses, for the same reason.
    private const float EnergyAttack = 0.35f;
    private const float EnergyRelease = 0.06f;

    // The noise floor rises slowly (so a sustained sound cannot become "silence") and falls quickly
    // (so it recovers when a fan switches off).
    private const float FloorRise = 0.0005f;
    private const float FloorFall = 0.05f;

    private readonly int _sampleRate;

    /// <summary>
    /// Rolling history that pitch estimation runs over, separate from the caller's window.
    /// </summary>
    /// <remarks>
    /// Energy wants a short window so the mouth reacts promptly; autocorrelation wants a long one,
    /// because measuring a 60 Hz fundamental needs at least a couple of its 16.7 ms periods and a
    /// typical 20 ms hop does not contain them. Keeping ~64 ms of history lets each get what it
    /// needs - the standard overlapping-window arrangement - instead of compromising on one length
    /// that serves neither.
    /// </remarks>
    private readonly float[] _history;
    private int _historyWritten;
    private int _historyFilled;

    private float _noiseFloor = 0.01f;
    private float _energy;
    private float _slowEnergy;
    private bool _initialized;

    public AudioFeatureAnalyzer(int sampleRate = DefaultSampleRate)
    {
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));

        _sampleRate = sampleRate;

        // Four periods of the lowest pitch we look for: enough support for a stable estimate
        // without making the correlation loop expensive.
        var samplesPerLowestPeriod = (int)(sampleRate / MinPitchHz);
        _history = new float[samplesPerLowestPeriod * 4];
    }

    /// <summary>Current tracked background level, exposed for diagnostics.</summary>
    public float NoiseFloor => _noiseFloor;

    /// <summary>Forgets everything learned. Used when capture restarts on a different device.</summary>
    public void Reset()
    {
        _noiseFloor = 0.01f;
        _energy = 0f;
        _slowEnergy = 0f;
        _initialized = false;
        _historyWritten = 0;
        _historyFilled = 0;
        Array.Clear(_history);
    }

    /// <summary>Appends a window to the pitch history, oldest samples falling off the front.</summary>
    private void AppendHistory(ReadOnlySpan<float> samples)
    {
        // A window longer than the history only contributes its tail; anything earlier would be
        // overwritten within this same call anyway.
        var start = Math.Max(0, samples.Length - _history.Length);

        for (var i = start; i < samples.Length; i++)
        {
            _history[_historyWritten] = samples[i];
            _historyWritten = (_historyWritten + 1) % _history.Length;
            if (_historyFilled < _history.Length)
                _historyFilled++;
        }
    }

    /// <summary>Copies the history into chronological order for correlation.</summary>
    private float[] HistoryInOrder()
    {
        var ordered = new float[_historyFilled];
        var start = (_historyWritten - _historyFilled + _history.Length) % _history.Length;

        for (var i = 0; i < _historyFilled; i++)
            ordered[i] = _history[(start + i) % _history.Length];

        return ordered;
    }

    /// <summary>
    /// Analyses one window of mono samples in [-1, 1].
    /// </summary>
    /// <param name="samples">The window. Typically 10-30 ms worth.</param>
    /// <param name="timestampTicks">UTC ticks at the end of the window.</param>
    public AudioFeatures Analyze(ReadOnlySpan<float> samples, long timestampTicks)
    {
        if (samples.Length == 0)
            return AudioFeatures.Silent(timestampTicks);

        var rms = ComputeRms(samples);

        // Seed the floor from the first window so a loud start does not spend seconds decaying.
        if (!_initialized)
        {
            _noiseFloor = Math.Max(rms, AbsoluteSilence);
            _initialized = true;
        }

        UpdateNoiseFloor(rms);

        // Energy comes from the caller's window so it stays responsive; pitch runs over the longer
        // history, which is the only way a low fundamental has enough periods to be measurable.
        AppendHistory(samples);
        var (pitch, clarity) = EstimatePitch(HistoryInOrder());

        // Both tests must pass. Loudness alone calls a door slam speech; periodicity alone calls a
        // steady hum speech. Together they are wrong far less often, which matters because a false
        // "voiced" reading makes the avatar's mouth move when the user is silent.
        var aboveFloor = rms > Math.Max(_noiseFloor * VoicedRatio, AbsoluteSilence);
        var isVoiced = aboveFloor && clarity > 0.3f;

        var target = isVoiced ? NormalizeEnergy(rms) : 0f;

        // Asymmetric smoothing: rise quickly, fall slowly.
        var coefficient = target > _energy ? EnergyAttack : EnergyRelease;
        _energy += (target - _energy) * coefficient;
        _energy = Math.Clamp(_energy, 0f, 1f);

        // Onset compares the fast envelope against a slower one, so a syllable landing harder than
        // the recent average registers as emphasis rather than merely as volume.
        _slowEnergy += (_energy - _slowEnergy) * 0.02f;
        var onset = Math.Clamp(_energy - _slowEnergy, 0f, 1f);

        return new AudioFeatures(
            Rms: rms,
            NoiseFloor: _noiseFloor,
            IsVoiced: isVoiced,
            SpeechEnergy: _energy,
            PitchHz: isVoiced ? pitch : 0f,
            PitchClarity: clarity,
            Onset: onset,
            TimestampTicks: timestampTicks);
    }

    private static float ComputeRms(ReadOnlySpan<float> samples)
    {
        double sum = 0;
        foreach (var sample in samples)
            sum += (double)sample * sample;

        return (float)Math.Sqrt(sum / samples.Length);
    }

    private void UpdateNoiseFloor(float rms)
    {
        if (rms < _noiseFloor)
            _noiseFloor += (rms - _noiseFloor) * FloorFall;
        else
            _noiseFloor += (rms - _noiseFloor) * FloorRise;

        _noiseFloor = Math.Max(_noiseFloor, AbsoluteSilence);
    }

    /// <summary>Maps RMS above the noise floor onto 0..1, with a square root so quiet speech still registers.</summary>
    private float NormalizeEnergy(float rms)
    {
        var excess = Math.Max(0f, rms - _noiseFloor);
        var normalized = excess / Math.Max(FullEnergyRms, 1e-6f);
        return Math.Clamp(MathF.Sqrt(Math.Clamp(normalized, 0f, 1f)), 0f, 1f);
    }

    /// <summary>
    /// Autocorrelation pitch estimate, returning (hz, clarity).
    /// </summary>
    /// <remarks>
    /// Normalised so clarity is comparable across volumes - it measures how periodic the window is,
    /// not how loud. That is what lets it act as the second half of the voiced test: speech is
    /// strongly periodic, room noise is not, regardless of level.
    ///
    /// Cost is O(window x lag range). At 16 kHz with a 20 ms window that is roughly 70k multiply-adds
    /// per window, 50 times a second - a few hundred microseconds, on a thread that is not the
    /// inference tick.
    /// </remarks>
    private (float Hz, float Clarity) EstimatePitch(ReadOnlySpan<float> samples)
    {
        var minLag = (int)(_sampleRate / MaxPitchHz);
        var maxLag = (int)(_sampleRate / MinPitchHz);

        // Two full periods of the lowest pitch searched. Below that the correlation has too little
        // overlap to mean anything, so reporting nothing is more honest than reporting noise.
        if (samples.Length < maxLag * 2 || minLag >= maxLag)
            return (0f, 0f);

        double energy = 0;
        for (var i = 0; i < samples.Length; i++)
            energy += (double)samples[i] * samples[i];

        if (energy <= 1e-12)
            return (0f, 0f);

        var bestLag = 0;
        double bestCorrelation = 0;

        for (var lag = minLag; lag <= maxLag; lag++)
        {
            double correlation = 0;
            double lagEnergy = 0;
            var count = samples.Length - lag;

            for (var i = 0; i < count; i++)
            {
                correlation += (double)samples[i] * samples[i + lag];
                lagEnergy += (double)samples[i + lag] * samples[i + lag];
            }

            if (lagEnergy <= 1e-12)
                continue;

            // Normalised cross-correlation: independent of amplitude, so it measures periodicity.
            var normalized = correlation / Math.Sqrt(energy * lagEnergy);
            if (normalized > bestCorrelation)
            {
                bestCorrelation = normalized;
                bestLag = lag;
            }
        }

        if (bestLag == 0)
            return (0f, 0f);

        return ((float)_sampleRate / bestLag, (float)Math.Clamp(bestCorrelation, 0, 1));
    }
}
