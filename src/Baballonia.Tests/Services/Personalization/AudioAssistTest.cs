using System;
using System.Diagnostics;
using System.Linq;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.Audio;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Signal analysis, driven by generated waveforms so no microphone is involved.
///
/// The two failure modes worth testing are opposite. Missing real speech makes the feature useless;
/// firing on room noise makes the avatar's mouth move while the user is silent, which is worse than
/// useless. So the tests come in pairs: a voiced tone must register, and noise at a comparable level
/// must not.
/// </summary>
[TestClass]
public class AudioFeatureAnalyzerTest
{
    private const int Rate = AudioFeatureAnalyzer.DefaultSampleRate;
    private const int Window = Rate / 50;  // 20 ms

    private static float[] Tone(float hz, float amplitude, int samples = Window, int phase = 0)
    {
        var buffer = new float[samples];
        for (var i = 0; i < samples; i++)
            buffer[i] = amplitude * MathF.Sin(2 * MathF.PI * hz * (i + phase) / Rate);
        return buffer;
    }

    /// <summary>A vowel-ish waveform: a fundamental plus a couple of harmonics.</summary>
    private static float[] Voice(float hz, float amplitude, int samples = Window, int phase = 0)
    {
        var buffer = new float[samples];
        for (var i = 0; i < samples; i++)
        {
            var t = 2 * MathF.PI * (i + phase) / Rate;
            buffer[i] = amplitude * (
                MathF.Sin(hz * t) +
                0.5f * MathF.Sin(2 * hz * t) +
                0.25f * MathF.Sin(3 * hz * t)) / 1.75f;
        }
        return buffer;
    }

    private static float[] Noise(float amplitude, int seed = 0, int samples = Window)
    {
        var random = new Random(seed);
        var buffer = new float[samples];
        for (var i = 0; i < samples; i++)
            buffer[i] = (float)(random.NextDouble() * 2 - 1) * amplitude;
        return buffer;
    }

    private static float[] Silence(int samples = Window) => new float[samples];

    /// <summary>Feeds many windows so the noise floor and smoothing settle, as they would live.</summary>
    private static AudioFeatures Settle(AudioFeatureAnalyzer analyzer, Func<int, float[]> generator,
                                        int windows)
    {
        var features = AudioFeatures.Silent();
        var ticks = DateTime.UtcNow.Ticks;

        for (var i = 0; i < windows; i++)
        {
            ticks += TimeSpan.TicksPerMillisecond * 20;
            features = analyzer.Analyze(generator(i), ticks);
        }

        return features;
    }

    [TestMethod]
    public void SilenceIsNotVoicedAndCarriesNoEnergy()
    {
        var analyzer = new AudioFeatureAnalyzer();

        var features = Settle(analyzer, _ => Silence(), 50);

        Assert.IsFalse(features.IsVoiced);
        Assert.AreEqual(0f, features.SpeechEnergy, 1e-3);
    }

    [TestMethod]
    public void SustainedVoiceIsDetected()
    {
        var analyzer = new AudioFeatureAnalyzer();

        // A moment of quiet first, so the noise floor is realistic rather than seeded from speech.
        Settle(analyzer, _ => Noise(0.002f), 40);
        var features = Settle(analyzer, i => Voice(140f, 0.25f, phase: i * Window), 40);

        Assert.IsTrue(features.IsVoiced, "a sustained voiced tone should register as speech");
        Assert.IsTrue(features.SpeechEnergy > 0.3f, $"energy was {features.SpeechEnergy:F3}");
    }

    [TestMethod]
    public void BroadbandNoiseIsNotMistakenForSpeech()
    {
        // The important negative. Loudness alone would call this speech; requiring periodicity too
        // is what keeps a fan or a keyboard from animating the avatar's mouth.
        var analyzer = new AudioFeatureAnalyzer();

        Settle(analyzer, _ => Noise(0.002f), 40);
        var features = Settle(analyzer, i => Noise(0.25f, seed: i + 1), 40);

        Assert.IsFalse(features.IsVoiced,
            $"noise was treated as voice (clarity {features.PitchClarity:F2})");
    }

    [TestMethod]
    public void PitchIsRecoveredFromAToneWithinACoupleOfPercent()
    {
        var analyzer = new AudioFeatureAnalyzer();

        Settle(analyzer, _ => Noise(0.002f), 20);
        var features = Settle(analyzer, i => Voice(150f, 0.3f, phase: i * Window), 20);

        Assert.IsTrue(Math.Abs(features.PitchHz - 150f) < 8f,
            $"estimated {features.PitchHz:F1} Hz for a 150 Hz source");
        Assert.IsTrue(features.PitchClarity > 0.8f);
    }

    [TestMethod]
    public void LouderSpeechProducesMoreEnergy()
    {
        var quiet = new AudioFeatureAnalyzer();
        Settle(quiet, _ => Noise(0.002f), 40);
        var quietFeatures = Settle(quiet, i => Voice(140f, 0.06f, phase: i * Window), 40);

        var loud = new AudioFeatureAnalyzer();
        Settle(loud, _ => Noise(0.002f), 40);
        var loudFeatures = Settle(loud, i => Voice(140f, 0.30f, phase: i * Window), 40);

        Assert.IsTrue(loudFeatures.SpeechEnergy > quietFeatures.SpeechEnergy,
            $"quiet {quietFeatures.SpeechEnergy:F3} vs loud {loudFeatures.SpeechEnergy:F3}");
    }

    [TestMethod]
    public void EnergyDecaysAfterSpeechStops()
    {
        var analyzer = new AudioFeatureAnalyzer();
        Settle(analyzer, _ => Noise(0.002f), 30);
        var speaking = Settle(analyzer, i => Voice(140f, 0.3f, phase: i * Window), 40);

        var afterwards = Settle(analyzer, _ => Silence(), 60);

        Assert.IsTrue(speaking.SpeechEnergy > 0.3f);
        Assert.IsTrue(afterwards.SpeechEnergy < 0.05f, "energy should fall away once speech stops");
    }

    [TestMethod]
    public void TheNoiseFloorAdaptsToASteadyBackground()
    {
        // A noisy room must not read as permanent speech, which is what a fixed threshold would do.
        var analyzer = new AudioFeatureAnalyzer();

        var features = Settle(analyzer, i => Noise(0.05f, seed: i), 200);

        Assert.IsTrue(analyzer.NoiseFloor > 0.005f, "the floor should have risen to meet the noise");
        Assert.IsFalse(features.IsVoiced);
    }

    [TestMethod]
    public void AnalysisIsCheapEnoughToRunOnEveryWindow()
    {
        // Budget is one 20 ms window every 20 ms, on a background thread. Anything near that would
        // be a problem; this asserts it is nowhere close.
        var analyzer = new AudioFeatureAnalyzer();
        var samples = Voice(140f, 0.3f);

        for (var i = 0; i < 20; i++)
            analyzer.Analyze(samples, DateTime.UtcNow.Ticks);

        const int iterations = 500;
        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < iterations; i++)
            analyzer.Analyze(samples, DateTime.UtcNow.Ticks);
        stopwatch.Stop();

        var perWindowMs = stopwatch.Elapsed.TotalMilliseconds / iterations;
        Assert.IsTrue(perWindowMs < 5.0,
            $"{perWindowMs:F3} ms per 20 ms window - too close to real time");
    }

    [TestMethod]
    public void EmptyInputIsSafe()
    {
        var analyzer = new AudioFeatureAnalyzer();

        var features = analyzer.Analyze(ReadOnlySpan<float>.Empty, 123);

        Assert.IsFalse(features.IsVoiced);
        Assert.AreEqual(123, features.TimestampTicks);
    }
}

/// <summary>
/// The enhancer, and specifically the promise that it cannot invent an expression.
/// </summary>
[TestClass]
public class ProsodyEnhancerTest
{
    private static readonly int Jaw = PersonalizationSchema.IndexOf("JawOpen");
    private static readonly int SmileLeft = PersonalizationSchema.IndexOf("MouthSmileLeft");
    private static readonly int TongueOut = PersonalizationSchema.IndexOf("TongueOut");
    private static readonly int CheekPuff = PersonalizationSchema.IndexOf("CheekPuffLeft");

    /// <summary>A source that reports whatever the test wants, with no device behind it.</summary>
    private sealed class FakeSource : IAudioFeatureSource
    {
        public AudioFeatures Latest { get; set; } = AudioFeatures.Silent();
        public bool IsRunning => true;
        public string StatusMessage => "";
        public bool Start() => true;
        public void Stop() { }
        public void Dispose() { }
    }

    private static AudioFeatures Speaking(float energy, long ticks, float onset = 0f) =>
        new(Rms: 0.2f, NoiseFloor: 0.01f, IsVoiced: true, SpeechEnergy: energy,
            PitchHz: 140f, PitchClarity: 0.9f, Onset: onset, TimestampTicks: ticks);

    private static float[] Expressions(params (int Dim, float Value)[] values)
    {
        var result = new float[PersonalizationSchema.ExpressionCount];
        foreach (var (dim, value) in values)
            result[dim] = value;
        return result;
    }

    /// <summary>Runs enough calls for the enhancer's own smoothing to settle.</summary>
    private static float[] Settle(ProsodyEnhancer enhancer, float[] input, int calls = 40)
    {
        var output = input;
        for (var i = 0; i < calls; i++)
            output = enhancer.Enhance(input);
        return output;
    }

    [TestMethod]
    public void LoudSpeechCannotInventAnExpression()
    {
        // The single most important property. The tracker says the user is not smiling; no amount
        // of shouting may put a smile on the avatar. Multiplication guarantees it - the alternative
        // is the canned viseme behaviour this feature exists to avoid becoming.
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = Speaking(1.0f, now) };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };

        var input = Expressions((Jaw, 0.6f), (SmileLeft, 0f));
        var output = Settle(enhancer, input);

        Assert.AreEqual(0f, output[SmileLeft], 1e-6, "audio must never create movement from nothing");
        Assert.IsTrue(output[Jaw] > input[Jaw], "existing movement should be amplified");
    }

    [TestMethod]
    public void SilenceLeavesExpressionsExactlyAlone()
    {
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = AudioFeatures.Silent(now) };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };

        var input = Expressions((Jaw, 0.6f), (SmileLeft, 0.4f));
        var output = Settle(enhancer, input);

        CollectionAssert.AreEqual(input, output, "with no speech the output must be untouched");
    }

    [TestMethod]
    public void ZeroStrengthIsAnExactPassthrough()
    {
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = Speaking(1.0f, now) };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 0f };

        var input = Expressions((Jaw, 0.6f), (SmileLeft, 0.4f));

        CollectionAssert.AreEqual(input, Settle(enhancer, input));
    }

    [TestMethod]
    public void OnlyJawAndMouthAreAffected()
    {
        // Loudness says nothing about cheeks or the tongue, so it must not move them.
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = Speaking(1.0f, now) };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };

        var input = Expressions((Jaw, 0.6f), (TongueOut, 0.6f), (CheekPuff, 0.6f));
        var output = Settle(enhancer, input);

        Assert.IsTrue(output[Jaw] > input[Jaw]);
        Assert.AreEqual(input[TongueOut], output[TongueOut], 1e-6);
        Assert.AreEqual(input[CheekPuff], output[CheekPuff], 1e-6);
    }

    [TestMethod]
    public void OutputStaysInTheUnitRange()
    {
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = Speaking(1.0f, now, onset: 1.0f) };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };

        var output = Settle(enhancer, Expressions((Jaw, 0.98f), (SmileLeft, 0.95f)));

        Assert.IsTrue(output.All(v => v is >= 0f and <= 1f), "values escaped [0,1]");
    }

    [TestMethod]
    public void RestingNoiseIsNotAmplified()
    {
        // Scaling everything would inflate the tracker's own resting jitter, undoing the exact
        // problem an earlier milestone was spent fixing.
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = Speaking(1.0f, now) };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };

        var input = Expressions((Jaw, 0.02f), (SmileLeft, 0.03f));
        var output = Settle(enhancer, input);

        CollectionAssert.AreEqual(input, output, "values below the movement floor must be left alone");
    }

    [TestMethod]
    public void StaleFeaturesStopTheBoost()
    {
        // A stalled capture thread must not freeze the gain at whatever it last was.
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = Speaking(1.0f, now) };
        var clock = now;
        var enhancer = new ProsodyEnhancer(source, () => clock) { Strength = 1.0f };

        var input = Expressions((Jaw, 0.6f));
        Settle(enhancer, input);

        // Jump the clock forward; the features are now far older than the freshness window.
        clock = now + TimeSpan.TicksPerSecond * 5;
        var output = Settle(enhancer, input);

        CollectionAssert.AreEqual(input, output, "stale audio must decay back to no enhancement");
    }

    [TestMethod]
    public void UnvoicedAudioDoesNotBoost()
    {
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource
        {
            Latest = new AudioFeatures(0.3f, 0.01f, IsVoiced: false, SpeechEnergy: 0.9f,
                                       0f, 0.1f, 0f, now)
        };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };

        var input = Expressions((Jaw, 0.6f));

        CollectionAssert.AreEqual(input, Settle(enhancer, input));
    }

    [TestMethod]
    public void StrongerSpeechGivesAStrongerBoost()
    {
        var now = DateTime.UtcNow.Ticks;
        var input = Expressions((Jaw, 0.5f));

        float Boosted(float energy)
        {
            var source = new FakeSource { Latest = Speaking(energy, now) };
            var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };
            return Settle(enhancer, input)[Jaw];
        }

        Assert.IsTrue(Boosted(0.9f) > Boosted(0.3f));
        Assert.IsTrue(Boosted(0.3f) > input[Jaw]);
    }

    [TestMethod]
    public void GainIsBounded()
    {
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = Speaking(1.0f, now, onset: 1.0f) };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };

        Settle(enhancer, Expressions((Jaw, 0.5f)));

        Assert.IsTrue(enhancer.LastGain <= ProsodyEnhancer.MaxGain + 1e-6);
        Assert.IsTrue(enhancer.LastGain >= 1f);
    }

    [TestMethod]
    public void TheCallersArrayIsNeverMutated()
    {
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = Speaking(1.0f, now) };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };

        var input = Expressions((Jaw, 0.6f));
        var copy = (float[])input.Clone();

        var output = enhancer.Enhance(input);

        CollectionAssert.AreEqual(copy, input, "the input array was modified in place");
        Assert.AreNotSame(input, output);
    }

    [TestMethod]
    public void AThrowingSourceCannotBreakTracking()
    {
        var enhancer = new ProsodyEnhancer(new ThrowingSource());
        var input = Expressions((Jaw, 0.6f));

        CollectionAssert.AreEqual(input, enhancer.Enhance(input));
    }

    private sealed class ThrowingSource : IAudioFeatureSource
    {
        public bool IsRunning => true;
        public string StatusMessage => "";
        public AudioFeatures Latest => throw new InvalidOperationException("device exploded");
        public bool Start() => true;
        public void Stop() { }
        public void Dispose() { }
    }

    [TestMethod]
    public void EnhancementCostIsNegligibleOnTheTick()
    {
        var now = DateTime.UtcNow.Ticks;
        var source = new FakeSource { Latest = Speaking(0.8f, now) };
        var enhancer = new ProsodyEnhancer(source, () => now) { Strength = 1.0f };
        var input = Expressions((Jaw, 0.6f), (SmileLeft, 0.4f));

        for (var i = 0; i < 100; i++)
            enhancer.Enhance(input);

        const int iterations = 10_000;
        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < iterations; i++)
            enhancer.Enhance(input);
        stopwatch.Stop();

        var perCallMs = stopwatch.Elapsed.TotalMilliseconds / iterations;
        Assert.IsTrue(perCallMs < 0.05,
            $"{perCallMs:F4} ms per call against a 10 ms tick budget");
    }
}

/// <summary>The null source, which is what "audio off" and "no microphone" both resolve to.</summary>
[TestClass]
public class NullAudioFeatureSourceTest
{
    [TestMethod]
    public void ReportsSilenceAndRefusesToStart()
    {
        using var source = new NullAudioFeatureSource();

        Assert.IsFalse(source.Start());
        Assert.IsFalse(source.IsRunning);
        Assert.IsFalse(source.Latest.IsVoiced);
        Assert.AreEqual(0f, source.Latest.SpeechEnergy);
    }

    [TestMethod]
    public void ProducesAnExactPassthroughThroughTheEnhancer()
    {
        // The property that makes "no audio backend on this platform" safe rather than degraded.
        using var source = new NullAudioFeatureSource();
        var enhancer = new ProsodyEnhancer(source) { Strength = 1.0f };

        var input = new float[PersonalizationSchema.ExpressionCount];
        input[PersonalizationSchema.IndexOf("JawOpen")] = 0.7f;

        for (var i = 0; i < 20; i++)
            CollectionAssert.AreEqual(input, enhancer.Enhance(input));
    }
}
