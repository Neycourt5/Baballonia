using System;

namespace Baballonia.Services.Personalization.Audio;

/// <summary>
/// A short window of microphone audio, described by a handful of numbers.
///
/// Everything here is non-semantic on purpose: loudness, periodicity, pitch. Nothing in this project
/// transcribes speech, infers meaning, or recognises words - the enhancer only needs to know *how
/// energetically* someone is talking, not what they are saying. That keeps the whole feature local,
/// cheap, and impossible to misuse for anything it was not intended for.
///
/// No audio is retained. A window is analysed into this record and discarded.
/// </summary>
/// <param name="Rms">Root-mean-square amplitude of the window, linear 0..1.</param>
/// <param name="NoiseFloor">Tracked background level, the reference "silence" is measured against.</param>
/// <param name="IsVoiced">Whether the window looks like voice rather than room noise.</param>
/// <param name="SpeechEnergy">
/// Smoothed 0..1 measure of how energetically the user is speaking. This is the one value the
/// enhancer actually consumes; the rest are diagnostics and inputs to it.
/// </param>
/// <param name="PitchHz">Estimated fundamental, or 0 when unvoiced or unclear.</param>
/// <param name="PitchClarity">Confidence in <paramref name="PitchHz"/>, 0..1.</param>
/// <param name="Onset">
/// How sharply energy rose against its recent average, 0..1. Emphasis - a stressed syllable - shows
/// up here, which is what makes speech look animated rather than merely loud.
/// </param>
/// <param name="TimestampTicks">UTC ticks at the end of the analysed window, for sync.</param>
public readonly record struct AudioFeatures(
    float Rms,
    float NoiseFloor,
    bool IsVoiced,
    float SpeechEnergy,
    float PitchHz,
    float PitchClarity,
    float Onset,
    long TimestampTicks)
{
    /// <summary>The "nothing is happening" reading, used when audio is off or unavailable.</summary>
    public static AudioFeatures Silent(long ticks = 0) =>
        new(0f, 0f, false, 0f, 0f, 0f, 0f, ticks);

    /// <summary>Age of this reading, for deciding whether it can still be trusted.</summary>
    public TimeSpan AgeAt(long nowTicks) => TimeSpan.FromTicks(Math.Max(0, nowTicks - TimestampTicks));
}

/// <summary>
/// Somewhere audio features come from.
/// </summary>
/// <remarks>
/// An interface rather than a concrete microphone because capture is the platform-specific,
/// permission-dependent, hardware-dependent part, and none of the logic that consumes it should
/// have to care. It also means the analyser and the enhancer are testable against synthetic audio
/// with no device involved at all.
/// </remarks>
public interface IAudioFeatureSource : IDisposable
{
    /// <summary>Whether audio is currently being captured and analysed.</summary>
    bool IsRunning { get; }

    /// <summary>Why capture is not running, when it should be. Empty when healthy.</summary>
    string StatusMessage { get; }

    /// <summary>
    /// Most recent analysed window, or <see cref="AudioFeatures.Silent"/> when there is none.
    /// </summary>
    /// <remarks>
    /// Must be safe to call from the inference tick and must never block: capture and analysis run
    /// on their own thread, and this only reads the latest published result.
    /// </remarks>
    AudioFeatures Latest { get; }

    /// <summary>Starts capture. Returns false when unavailable; never throws.</summary>
    bool Start();

    /// <summary>Stops capture. Safe to call when not running.</summary>
    void Stop();
}

/// <summary>
/// The source used when audio assist is off, or when no capture backend exists on this platform.
/// </summary>
/// <remarks>
/// Always reports silence, which the enhancer turns into a gain of exactly 1. That is what makes
/// "audio off" and "no microphone" produce byte-identical output to the visual-only path rather
/// than merely similar output.
/// </remarks>
public sealed class NullAudioFeatureSource : IAudioFeatureSource
{
    public bool IsRunning => false;
    public string StatusMessage => "Audio capture is not available on this platform.";
    public AudioFeatures Latest => AudioFeatures.Silent();

    public bool Start() => false;
    public void Stop() { }
    public void Dispose() { }
}
