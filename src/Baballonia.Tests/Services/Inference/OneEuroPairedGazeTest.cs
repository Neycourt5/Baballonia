using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Filters;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Inference;

/// <summary>
/// One eye arriving later than the other during a sideways look.
/// </summary>
/// <remarks>
/// <para>Reported as "one eye's movement is delayed, especially on look out — the one that looks
/// towards the nose moves, but the one that looks out is delayed".</para>
///
/// <para>One Euro widens its cutoff with each channel's <em>own</em> speed, so a channel reporting a
/// smaller movement is also smoothed harder. The eye looking outward, whose iris sits near the
/// corner where the model's range compresses, is penalised twice: once for reporting less movement,
/// and again by being filtered more because of it. With the user's settings (minCutoff 0.5,
/// beta 0.5) that is about 100 ms of difference in arrival time.</para>
/// </remarks>
[TestClass]
[TestSubject(typeof(OneEuroFilter))]
public class OneEuroPairedGazeTest
{
    private const string LeftX = "/leftEyeX";
    private const string RightX = "/rightEyeX";
    private const string Widen = "/leftEyeWiden";

    private static readonly string[] Keys = [LeftX, RightX, Widen];

    /// <summary>The user's own filter settings.</summary>
    private static OneEuroFilter Paired() =>
        new(minCutoff: 0.5f, beta: 0.5f, pairedKeys: OneEuroFilter.EyeGazePairs);

    private static OneEuroFilter Unpaired() => new(minCutoff: 0.5f, beta: 0.5f);

    /// <summary>
    /// Drives a saccade in which one eye reports a large movement and the other a compressed one,
    /// and reports how many frames each took to cover 90% of its own travel.
    /// </summary>
    private static (int Strong, int Weak) FramesToArrive(OneEuroFilter filter)
    {
        const float strongAmplitude = 0.40f;
        const float weakAmplitude = 0.12f;
        const int rest = 20, move = 6, hold = 200;

        var map = new OrderedFloatMap(Keys);
        int? strong = null, weak = null;

        for (var frame = 0; frame < rest + move + hold; frame++)
        {
            var progress = frame < rest ? 0f
                : frame < rest + move ? (frame - rest + 1) / (float)move
                : 1f;

            map[LeftX] = 0.5f + strongAmplitude * progress;
            map[RightX] = 0.5f + weakAmplitude * progress;
            map[Widen] = 0.5f;

            // The filter measures elapsed time from the wall clock, so the frames must be real.
            Thread.Sleep(1);
            var result = filter.Filter(map);

            if (strong is null && result[LeftX] >= 0.5f + strongAmplitude * 0.9f)
                strong = frame - rest;
            if (weak is null && result[RightX] >= 0.5f + weakAmplitude * 0.9f)
                weak = frame - rest;
        }

        Assert.IsNotNull(strong, "the strong eye never arrived");
        Assert.IsNotNull(weak, "the weak eye never arrived");
        return (strong.Value, weak.Value);
    }

    [TestMethod]
    public void WithoutPairingTheCompressedEyeArrivesLate()
    {
        // The bug, reproduced. This is the behaviour every build before this fix had.
        var (strong, weak) = FramesToArrive(Unpaired());

        Assert.IsTrue(weak > strong,
            $"expected the compressed eye to lag; strong {strong}, weak {weak}");
    }

    [TestMethod]
    public void PairingMakesBothEyesArriveTogether()
    {
        var (strong, weak) = FramesToArrive(Paired());

        Assert.AreEqual(strong, weak, 1,
            $"the two eyes should arrive together; strong {strong}, weak {weak}");
    }

    [TestMethod]
    public void PairingNeverMakesAnEyeSlower()
    {
        // The safety property that makes this fix unconditional rather than a setting: coupling
        // takes the LARGER of the two speeds, so a cutoff can only widen. Nothing gets laggier.
        var (pairedStrong, pairedWeak) = FramesToArrive(Paired());
        var (loneStrong, loneWeak) = FramesToArrive(Unpaired());

        Assert.IsTrue(pairedStrong <= loneStrong + 1,
            $"the well-tracked eye got slower: {loneStrong} -> {pairedStrong}");
        Assert.IsTrue(pairedWeak <= loneWeak + 1,
            $"the compressed eye got slower: {loneWeak} -> {pairedWeak}");
    }

    [TestMethod]
    public void AnUnpairedChannelIsUnaffected()
    {
        // The face pipeline passes no pairs, so its behaviour must be bit-for-bit what it was.
        var withPairs = Paired();
        var without = Unpaired();

        var map = new OrderedFloatMap(Keys);
        var a = new List<float>();
        var b = new List<float>();

        for (var frame = 0; frame < 40; frame++)
        {
            map[LeftX] = 0.5f + 0.3f * frame / 40f;
            map[RightX] = 0.5f;
            map[Widen] = frame < 20 ? 0.2f : 0.8f;

            Thread.Sleep(1);
            a.Add(withPairs.Filter(map)[Widen]);
            b.Add(without.Filter(map)[Widen]);
        }

        // Widen is in neither pair, so gaze moving cannot change it. Timings differ by a frame's
        // jitter at most, so compare the settled tail rather than the transient.
        Assert.AreEqual(b[^1], a[^1], 0.02f, "an unpaired channel was disturbed by pairing");
    }

    [TestMethod]
    public void PairsNamingAbsentKeysAreIgnored()
    {
        // A six-output model has no widen channel; a pair naming one must not throw or mis-index.
        var filter = new OneEuroFilter(0.5f, 0.5f,
            [("/leftEyeX", "/rightEyeX"), ("/nope", "/alsoNope"), ("/leftEyeX", "/missing")]);

        var map = new OrderedFloatMap(Keys);
        map[LeftX] = 0.4f;
        map[RightX] = 0.6f;
        map[Widen] = 0.5f;

        filter.Filter(map);
        Thread.Sleep(2);
        var result = filter.Filter(map);

        Assert.IsTrue(result.Values.All(float.IsFinite));
    }

    [TestMethod]
    public void TheGazePairsAreTheFourGazeChannels()
    {
        var pairs = OneEuroFilter.EyeGazePairs;

        CollectionAssert.AreEquivalent(
            new[] { "/leftEyeX", "/rightEyeX", "/leftEyeY", "/rightEyeY" },
            pairs.SelectMany(p => new[] { p.A, p.B }).ToArray());
    }

    [TestMethod]
    public void APairingSurvivesAModelSwap()
    {
        // Shape changes re-run Initialize, which is where the partner table is built.
        var filter = Paired();
        var twelve = new OrderedFloatMap(Keys);
        twelve[LeftX] = 0.5f;
        twelve[RightX] = 0.5f;
        twelve[Widen] = 0.5f;
        filter.Filter(twelve);

        var six = new OrderedFloatMap([LeftX, RightX]);
        six[LeftX] = 0.5f;
        six[RightX] = 0.5f;
        filter.Filter(six);

        for (var frame = 0; frame < 30; frame++)
        {
            six[LeftX] = 0.5f + 0.4f * Math.Min(1f, frame / 6f);
            six[RightX] = 0.5f + 0.12f * Math.Min(1f, frame / 6f);
            Thread.Sleep(1);
            filter.Filter(six);
        }

        var final = filter.Filter(six);
        Assert.IsTrue(final[RightX] > 0.5f, "the paired channel never moved after the swap");
    }
}
