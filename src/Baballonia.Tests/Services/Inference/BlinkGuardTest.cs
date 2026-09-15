using System;
using System.Collections.Generic;
using System.Linq;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.BlinkGuard;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Inference;

/// <summary>
/// BlinkGuard: the post-blink gaze snap, and everything it must not break while removing it.
/// </summary>
/// <remarks>
/// <para>The failure being fixed: the eyelid crosses the pupil, the tracker's gaze estimate goes
/// meaningless for a few frames, and on reopening the first estimate is accepted verbatim — so the
/// avatar's eye flicks somewhere absurd and corrects a few frames later. That reads as the avatar
/// looking at something, which is far worse than jitter.</para>
///
/// <para>Most of these tests are about restraint rather than correction. A filter that removed the
/// snap by smoothing everything would pass a naive "no snap" test and ruin ordinary tracking, so
/// saccade speed, wink independence and gaze-changed-during-blink are all pinned explicitly.</para>
/// </remarks>
[TestClass]
[TestSubject(typeof(BlinkGuardFilter))]
public class BlinkGuardTest
{
    private const double Frame = 1d / 90d;
    private const float Deg = BlinkGuardSettings.GazeRangeDegrees;

    private static GazeSample Open(float x, float y, bool valid = true, float confidence = 1f) =>
        new(x, y, 0f, valid, confidence);

    private static GazeSample Closed(float x, float y) => new(x, y, 1f, true, 1f);

    /// <summary>Degrees between two gaze directions, the unit every threshold is stated in.</summary>
    private static float Degrees(float x1, float y1, float x2, float y2) =>
        MathF.Sqrt(MathF.Pow((x1 - x2) * Deg, 2) + MathF.Pow((y1 - y2) * Deg, 2));

    /// <summary>Drives both eyes with the same sample, which is the ordinary bilateral case.</summary>
    private static GazeResult Drive(BlinkGuardFilter f, in GazeSample s, int frames = 1)
    {
        var result = default(GazeResult);
        for (var i = 0; i < frames; i++)
            result = f.Process(s, s, Frame).Left;
        return result;
    }

    private static BlinkGuardFilter Filter(BlinkGuardSettings? settings = null) =>
        new(settings ?? BlinkGuardSettings.Balanced);

    // ---------------------------------------------------------------- 1. normal smooth gaze

    [TestMethod]
    public void NormalGazeIsPassedThroughUntouched()
    {
        var f = Filter();
        for (var i = 0; i < 60; i++)
        {
            var x = i / 200f;
            var result = Drive(f, Open(x, 0.1f));
            Assert.AreEqual(x, result.X, 1e-6, "normal tracking must not be filtered at all");
            Assert.AreEqual(BlinkGuardState.Normal, result.State);
        }
    }

    // ---------------------------------------------------------------- 2. legitimate fast saccade

    [TestMethod]
    public void ALegitimateSaccadeIsNotSlowedDown()
    {
        // The single most important negative result: BlinkGuard is not a smoother.
        var f = Filter();
        Drive(f, Open(0f, 0f), 20);

        var jumped = Drive(f, Open(0.6f, -0.4f));

        Assert.AreEqual(0.6f, jumped.X, 1e-6, "a saccade outside a blink must arrive instantly");
        Assert.AreEqual(-0.4f, jumped.Y, 1e-6);
    }

    // ---------------------------------------------------------------- 3. normal bilateral blink

    [TestMethod]
    public void ABlinkHoldsGazeInsteadOfFollowingTheLid()
    {
        var f = Filter();
        Drive(f, Open(0.3f, 0.2f), 20);

        // Lid down, and the tracker starts reporting nonsense as it crosses the pupil.
        var held = Drive(f, Closed(-0.9f, 0.8f), 5);

        Assert.AreEqual(BlinkGuardState.BlinkHold, held.State);
        Assert.IsTrue(Degrees(held.X, held.Y, 0.3f, 0.2f) < 2f,
            $"gaze should have held near the pre-blink position, got ({held.X:F2},{held.Y:F2})");
    }

    [TestMethod]
    public void ABlinkDoesNotForceGazeToCentre()
    {
        // An easy wrong implementation: recentre on blink. It looks tidy and is completely wrong.
        var f = Filter();
        Drive(f, Open(0.5f, 0.5f), 20);
        var held = Drive(f, Closed(0.5f, 0.5f), 5);

        Assert.IsTrue(Degrees(held.X, held.Y, 0f, 0f) > 20f, "gaze must not be pulled to centre");
    }

    // ---------------------------------------------------------------- 4. single-eye wink

    [TestMethod]
    public void AWinkLeavesTheOpenEyeAlone()
    {
        var f = Filter();
        for (var i = 0; i < 20; i++)
            f.Process(Open(0.2f, 0f), Open(0.2f, 0f), Frame);

        // Left shut and reporting rubbish; right open and tracking a genuine movement.
        var (left, right) = f.Process(Closed(-0.8f, 0.7f), Open(0.45f, 0f), Frame);

        Assert.AreEqual(BlinkGuardState.BlinkHold, left.State);
        Assert.AreEqual(BlinkGuardState.Normal, right.State);
        Assert.AreEqual(0.45f, right.X, 1e-6, "the open eye must keep tracking during a wink");
    }

    // ---------------------------------------------------------------- 5. one-frame post-blink outlier

    [TestMethod]
    public void AOneFrameThirtyDegreeOutlierNeverReachesTheAvatar()
    {
        var f = Filter();
        Drive(f, Open(0f, 0f), 20);
        Drive(f, Closed(0f, 0f), 5);

        var outputs = new List<GazeResult>();

        // Reopened. Frame B is a wild reacquisition error between two good ones.
        outputs.Add(Drive(f, Open(0.02f, 0f)));
        outputs.Add(Drive(f, Open(0.7f, 0.6f)));   // ~40 degrees away
        outputs.Add(Drive(f, Open(0.01f, 0f)));
        outputs.Add(Drive(f, Open(0.02f, 0f)));
        outputs.Add(Drive(f, Open(0.0f, 0f)));

        foreach (var o in outputs)
        {
            Assert.IsTrue(Degrees(o.X, o.Y, 0f, 0f) < 8f,
                $"the outlier leaked to the output: ({o.X:F2},{o.Y:F2})");
        }
    }

    // ---------------------------------------------------------------- 6. several bad samples

    [TestMethod]
    public void SeveralDisagreeingSamplesAreAllRefused()
    {
        var f = Filter();
        Drive(f, Open(0f, 0f), 20);
        Drive(f, Closed(0f, 0f), 5);

        // Four mutually contradictory readings: nothing here deserves to be believed.
        foreach (var (x, y) in new[] { (0.8f, 0.1f), (-0.7f, 0.4f), (0.3f, -0.9f), (-0.2f, 0.8f) })
        {
            var o = Drive(f, Open(x, y));
            Assert.IsTrue(Degrees(o.X, o.Y, 0f, 0f) < 8f,
                $"a disagreeing sample reached the output: ({o.X:F2},{o.Y:F2})");
        }
    }

    // ---------------------------------------------------------------- 7. gaze legitimately changed

    [TestMethod]
    public void AGazeThatGenuinelyChangedDuringTheBlinkIsAccepted()
    {
        // The requirement that stops this being a "snap back" filter. Looking left, blink, and the
        // user is now looking right - that is real, and must win.
        var f = Filter();
        Drive(f, Open(-0.6f, 0f), 20);
        Drive(f, Closed(-0.6f, 0f), 6);

        GazeResult last = default;
        for (var i = 0; i < 40; i++)
            last = Drive(f, Open(0.6f, 0f));

        Assert.AreEqual(0.6f, last.X, 0.02f,
            "several coherent samples at the new position must be accepted, not dragged back");
        Assert.AreEqual(BlinkGuardState.Normal, last.State);
    }

    [TestMethod]
    public void TheMoveToANewGazeIsEasedRatherThanSnapped()
    {
        var f = Filter();
        Drive(f, Open(-0.6f, 0f), 20);
        Drive(f, Closed(-0.6f, 0f), 6);

        var path = new List<float>();
        for (var i = 0; i < 30; i++)
            path.Add(Drive(f, Open(0.6f, 0f)).X);

        // It must arrive...
        Assert.AreEqual(0.6f, path[^1], 0.02f);

        // ...but not in one frame, and never overshooting either end.
        var biggestStep = 0f;
        for (var i = 1; i < path.Count; i++)
            biggestStep = MathF.Max(biggestStep, MathF.Abs(path[i] - path[i - 1]));

        Assert.IsTrue(biggestStep < 1.2f * 0.6f, "the transition should be eased, not a single jump");
        Assert.IsTrue(path.All(v => v >= -0.61f && v <= 0.61f), "the ease must not overshoot");
    }

    [TestMethod]
    public void ReturningToTheSamePlaceResumesWithoutABlend()
    {
        // The common case: blinked, still looking at the same thing. Spending 80 ms easing toward a
        // position we are already at would be latency for nothing.
        var f = Filter();
        Drive(f, Open(0.25f, 0.1f), 20);
        Drive(f, Closed(0.25f, 0.1f), 5);

        GazeResult last = default;
        for (var i = 0; i < 8; i++)
            last = Drive(f, Open(0.25f, 0.1f));

        Assert.AreEqual(BlinkGuardState.Normal, last.State);
        Assert.AreEqual(0.25f, last.X, 1e-3);
    }

    // ---------------------------------------------------------------- 8. low confidence

    [TestMethod]
    public void LowConfidenceSamplesAreRefusedWhenGatingIsOn()
    {
        var f = Filter(BlinkGuardSettings.Balanced with
        {
            UseConfidence = true,
            ConfidenceThreshold = 0.5f,
        });

        Drive(f, Open(0f, 0f), 20);
        Drive(f, Closed(0f, 0f), 5);

        for (var i = 0; i < 5; i++)
        {
            var o = Drive(f, Open(0.7f, 0.5f, confidence: 0.1f));
            Assert.IsTrue(Degrees(o.X, o.Y, 0f, 0f) < 8f, "a low-confidence sample was accepted");
        }
    }

    [TestMethod]
    public void ConfidenceGatingIsOffByDefaultBecauseThisPipelineHasNone()
    {
        // This application runs its own model on the eye cameras and produces no confidence
        // channel. If the default ever flips on, every post-blink sample would be rejected and the
        // eye would sit on the timeout path forever.
        Assert.IsFalse(BlinkGuardSettings.Balanced.UseConfidence);
    }

    // ---------------------------------------------------------------- 9. invalid samples

    [TestMethod]
    public void InvalidSamplesDuringReopeningAreRefused()
    {
        var f = Filter();
        Drive(f, Open(0.1f, 0.1f), 20);
        Drive(f, Closed(0.1f, 0.1f), 5);

        for (var i = 0; i < 4; i++)
        {
            var o = Drive(f, Open(0.9f, -0.9f, valid: false));
            Assert.IsTrue(Degrees(o.X, o.Y, 0.1f, 0.1f) < 8f, "an invalid sample was accepted");
        }
    }

    [TestMethod]
    public void NonFiniteSamplesNeverReachTheOutput()
    {
        var f = Filter();
        Drive(f, Open(0.2f, 0.2f), 20);

        var o = Drive(f, Open(float.NaN, 0.2f));

        Assert.IsTrue(float.IsFinite(o.X) && float.IsFinite(o.Y));
    }

    // ---------------------------------------------------------------- 10. timeout

    [TestMethod]
    public void ReacquisitionTimesOutRatherThanFreezingForever()
    {
        // Never leave the avatar's eyes frozen. A timeout is worse than a clean reacquisition and
        // far better than an eye that has stopped moving.
        var f = Filter();
        Drive(f, Open(0f, 0f), 20);
        Drive(f, Closed(0f, 0f), 5);

        // Endless mutually-contradictory samples: agreement can never be reached.
        var rng = new Random(7);
        for (var i = 0; i < 120; i++)
            Drive(f, Open((float)(rng.NextDouble() - 0.5), (float)(rng.NextDouble() - 0.5)));

        Assert.IsTrue(f.Diagnostics.Counters.TimeoutCount > 0, "the timeout never fired");
        Assert.AreNotEqual(BlinkGuardState.BlinkHold, f.LeftState, "the eye is still frozen");
    }

    // ---------------------------------------------------------------- 11. rapid repeated blinking

    [TestMethod]
    public void RapidBlinkingDoesNotWedgeTheStateMachine()
    {
        var f = Filter();
        Drive(f, Open(0.2f, 0f), 20);

        for (var blink = 0; blink < 8; blink++)
        {
            Drive(f, Closed(0.2f, 0f), 4);
            Drive(f, Open(0.2f, 0f), 6);
        }

        // Settle, then confirm it is tracking again rather than stuck holding.
        GazeResult last = default;
        for (var i = 0; i < 30; i++)
            last = Drive(f, Open(0.2f, 0f));

        Assert.AreEqual(BlinkGuardState.Normal, last.State);
        Assert.AreEqual(0.2f, last.X, 1e-2);
        Assert.IsTrue(f.Diagnostics.Counters.BlinksDetected >= 8);
    }

    [TestMethod]
    public void ABlinkDuringReacquisitionRestartsTheCycle()
    {
        var f = Filter();
        Drive(f, Open(0f, 0f), 20);
        Drive(f, Closed(0f, 0f), 5);
        Drive(f, Open(0.05f, 0f));          // reopening, one candidate so far

        var again = Drive(f, Closed(0f, 0f));

        Assert.AreEqual(BlinkGuardState.BlinkHold, again.State,
            "a second blink must restart the cycle rather than race the first");
    }

    // ---------------------------------------------------------------- 12. non-blink tracking loss

    [TestMethod]
    public void TrackingLossWithoutABlinkDoesNotEnterBlinkHold()
    {
        // Invalid samples with the eye wide open are a different failure, and BlinkGuard should not
        // claim it — the eye is not shut, so there is no lid to hide a held position behind.
        var f = Filter();
        Drive(f, Open(0.3f, 0f), 20);

        var o = Drive(f, Open(0.3f, 0f, valid: false), 5);

        Assert.AreNotEqual(BlinkGuardState.BlinkHold, o.State);
        Assert.AreEqual(0, f.Diagnostics.Counters.BlinksDetected);
    }

    // ---------------------------------------------------------------- 13. irregular frame timing

    [TestMethod]
    public void IrregularFrameTimingIsHandledFromRealDeltas()
    {
        // The pipeline runs on a timer that degrades under load, so a fixed rate cannot be assumed.
        var f = Filter();
        var rng = new Random(11);

        for (var i = 0; i < 20; i++)
            f.Process(Open(0f, 0f), Open(0f, 0f), 0.004 + rng.NextDouble() * 0.03);

        f.Process(Closed(0f, 0f), Closed(0f, 0f), 0.05);
        f.Process(Closed(0f, 0f), Closed(0f, 0f), 0.01);

        GazeResult last = default;
        for (var i = 0; i < 30; i++)
            last = f.Process(Open(0.4f, 0f), Open(0.4f, 0f), 0.004 + rng.NextDouble() * 0.03).Left;

        Assert.AreEqual(BlinkGuardState.Normal, last.State);
        Assert.AreEqual(0.4f, last.X, 0.02f);
    }

    [TestMethod]
    public void AZeroDeltaFrameCannotStallTheTimeout()
    {
        // Duplicate timestamps happen. They must not advance time, but must not hang it either.
        var f = Filter();
        Drive(f, Open(0f, 0f), 20);
        f.Process(Closed(0f, 0f), Closed(0f, 0f), 0);
        f.Process(Open(0f, 0f), Open(0f, 0f), 0);

        Assert.IsTrue(f.Diagnostics.Counters.BlinksDetected > 0);
    }

    // ---------------------------------------------------------------- disabled / settings

    [TestMethod]
    public void DisabledIsExactlyPassThrough()
    {
        var f = Filter(BlinkGuardSettings.Balanced with { Enabled = false });

        var (left, right) = f.Process(Closed(-0.9f, 0.9f), Closed(0.9f, -0.9f), Frame);

        Assert.AreEqual(-0.9f, left.X, 0f);
        Assert.AreEqual(0.9f, right.X, 0f);
    }

    [TestMethod]
    public void InvertedHysteresisIsRepairedRatherThanLatchingTheEyeShut()
    {
        // A hand-edited settings file with reopen above closed would mean the eye could enter the
        // hold and never satisfy the exit condition.
        var settings = (BlinkGuardSettings.Balanced with
        {
            ClosedThreshold = 0.4f,
            ReopenThreshold = 0.9f,
        }).Sanitized();

        Assert.IsTrue(settings.ReopenThreshold < settings.ClosedThreshold);
    }

    [TestMethod]
    public void PresetsAreOrderedFromLeastToMostCautious()
    {
        var subtle = BlinkGuardSettings.ForPreset(BlinkGuardPreset.Subtle);
        var balanced = BlinkGuardSettings.ForPreset(BlinkGuardPreset.Balanced);
        var strong = BlinkGuardSettings.ForPreset(BlinkGuardPreset.Strong);

        Assert.IsFalse(BlinkGuardSettings.ForPreset(BlinkGuardPreset.Off).Enabled);
        Assert.IsTrue(subtle.RequiredStableSamples <= balanced.RequiredStableSamples);
        Assert.IsTrue(balanced.RequiredStableSamples <= strong.RequiredStableSamples);
        Assert.IsTrue(subtle.MinimumHoldSeconds <= strong.MinimumHoldSeconds);
        Assert.AreEqual(BlinkGuardPreset.Balanced, BlinkGuardSettings.Balanced.Preset);
    }

    [TestMethod]
    public void TheNormalSpikeGuardIsOffByDefault()
    {
        // It risks clipping real saccades, which is a more noticeable failure than the one it fixes.
        Assert.IsFalse(BlinkGuardSettings.Balanced.NormalSpikeGuardEnabled);
    }

    // ---------------------------------------------------------------- diagnostics

    [TestMethod]
    public void CountersOnlyCallLargeSuppressionsGlitchesPrevented()
    {
        var f = Filter();
        Drive(f, Open(0f, 0f), 20);
        Drive(f, Closed(0f, 0f), 5);
        Drive(f, Open(0.8f, 0.7f));          // far enough out to count
        Drive(f, Open(0f, 0f), 10);

        Assert.IsTrue(f.Diagnostics.Counters.GlitchesPrevented > 0);
    }

    [TestMethod]
    public void CaptureIsBoundedAndOffUntilStarted()
    {
        var f = Filter();

        Drive(f, Open(0f, 0f), 50);
        Assert.AreEqual(0, f.Diagnostics.RecordCount, "nothing should be recorded before a capture");

        f.Diagnostics.StartCapture();
        Drive(f, Open(0f, 0f), 30);
        Assert.AreEqual(30, f.Diagnostics.RecordCount);

        f.Diagnostics.StopCapture();
        Drive(f, Open(0f, 0f), 10);
        Assert.AreEqual(30, f.Diagnostics.RecordCount, "recording continued after stop");

        var csv = f.Diagnostics.ToCsv();
        StringAssert.StartsWith(csv, "t,l_raw_x");
        Assert.AreEqual(31, csv.TrimEnd('\r', '\n').Split('\n').Length, "header plus one row each");
    }

    [TestMethod]
    public void TheCaptureRingOverwritesRatherThanGrowing()
    {
        var f = new BlinkGuardFilter(BlinkGuardSettings.Balanced, new BlinkGuardDiagnostics(64));
        f.Diagnostics.StartCapture();

        Drive(f, Open(0f, 0f), 500);

        Assert.AreEqual(64, f.Diagnostics.RecordCount, "the ring must be bounded");
    }
}

/// <summary>
/// BlinkGuard where it actually runs: inside the post-processor, on the social path only.
/// </summary>
/// <remarks>
/// The hard requirement is that foveated rendering never sees this. The eye pipeline snapshots
/// <c>RawEyeResult</c> for the native/DFR path <em>before</em> calling the post-processor, so putting
/// BlinkGuard in the post-processor makes that a property of the pipeline's shape. These tests pin
/// the half that lives in the post-processor; the ordering itself is pinned by the pipeline tests.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeOutputPostProcessor))]
public class BlinkGuardIntegrationTest
{
    private const float Frame = 1f / 90f;

    private static OrderedFloatMap TunedModelMap() => new([
        "/rightEyeY", "/rightEyeX", "/rightEyeLid", "/rightEyeWiden", "/rightEyeSquint", "/rightEyeBrow",
        "/leftEyeY", "/leftEyeX", "/leftEyeLid", "/leftEyeWiden", "/leftEyeSquint", "/leftEyeBrow",
    ]);

    /// <summary>Raw lid is closedness; geometry has already inverted it to openness by this point.</summary>
    private static void Set(OrderedFloatMap map, float x, float y, float openness)
    {
        map["/leftEyeX"] = x;
        map["/rightEyeX"] = x;
        map["/leftEyeY"] = y;
        map["/rightEyeY"] = y;
        map["/leftEyeLid"] = openness;
        map["/rightEyeLid"] = openness;
    }

    [TestMethod]
    public void WithoutABlinkGuardThePostProcessorIsUnchanged()
    {
        // Off by default, and "off" has to mean the previous behaviour exactly.
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();
        Set(map, 0.4f, -0.2f, 0.9f);

        var result = post.Apply(map, Frame);

        Assert.AreEqual(0.4f, result["/leftEyeX"], 1e-6);
        Assert.AreEqual(-0.2f, result["/leftEyeY"], 1e-6);
    }

    [TestMethod]
    public void ABlinkHoldsGazeThroughTheWholePostProcessor()
    {
        var post = new EyeOutputPostProcessor
        {
            BlinkGuard = new BlinkGuardFilter(BlinkGuardSettings.Balanced),
        };

        var map = TunedModelMap();
        for (var i = 0; i < 20; i++)
        {
            Set(map, 0.35f, 0.15f, 0.95f);
            post.Apply(map, Frame);
        }

        // Lid shut, and the tracker reports a wild direction from behind it.
        OrderedFloatMap result = map;
        for (var i = 0; i < 4; i++)
        {
            Set(map, -0.9f, 0.85f, 0.02f);
            result = post.Apply(map, Frame);
        }

        Assert.AreEqual(0.35f, result["/leftEyeX"], 0.05f,
            "the wild sample behind a closed lid reached the social gaze output");
    }

    [TestMethod]
    public void TheEyelidItselfKeepsAnimatingDuringAHold()
    {
        // BlinkGuard holds gaze, never the blink. If it froze openness the avatar would stop
        // blinking, which is a far more obvious defect than the one being fixed.
        var post = new EyeOutputPostProcessor
        {
            BlinkGuard = new BlinkGuardFilter(BlinkGuardSettings.Balanced),
        };

        var map = TunedModelMap();
        for (var i = 0; i < 20; i++)
        {
            Set(map, 0.2f, 0f, 0.95f);
            post.Apply(map, Frame);
        }

        Set(map, 0.2f, 0f, 0.02f);
        var closed = post.Apply(map, Frame);

        Assert.IsTrue(closed["/leftEyeLid"] < 0.2f, "the lid must still read as closed");
    }

    [TestMethod]
    public void BlinkGuardIsAppliedBeforeGazeSyncSoAHoldIsNotSpreadByCoupling()
    {
        // Coupling a held eye to a snapping one would spread the snap to both, which is worse than
        // not having BlinkGuard at all.
        var post = new EyeOutputPostProcessor(gazeSyncAmount: 1f)
        {
            BlinkGuard = new BlinkGuardFilter(BlinkGuardSettings.Balanced),
        };

        var map = TunedModelMap();
        for (var i = 0; i < 20; i++)
        {
            Set(map, 0.3f, 0f, 0.95f);
            post.Apply(map, Frame);
        }

        OrderedFloatMap result = map;
        for (var i = 0; i < 4; i++)
        {
            map["/leftEyeLid"] = 0.02f;
            map["/rightEyeLid"] = 0.02f;
            map["/leftEyeX"] = -0.9f;
            map["/rightEyeX"] = 0.9f;
            result = post.Apply(map, Frame);
        }

        Assert.IsTrue(MathF.Abs(result["/leftEyeX"] - 0.3f) < 0.1f,
            $"a held eye was dragged by coupling, got {result["/leftEyeX"]:F2}");
    }

    [TestMethod]
    public void GazeStaysInRangeThroughAHold()
    {
        var post = new EyeOutputPostProcessor
        {
            BlinkGuard = new BlinkGuardFilter(BlinkGuardSettings.Balanced),
        };

        var map = TunedModelMap();
        for (var i = 0; i < 30; i++)
        {
            Set(map, 0.9f, -0.9f, i < 15 ? 0.95f : 0.01f);
            var result = post.Apply(map, Frame);

            foreach (var key in new[] { "/leftEyeX", "/leftEyeY", "/rightEyeX", "/rightEyeY" })
                Assert.IsTrue(result[key] is >= -1f and <= 1f, $"{key} left [-1,1]");
        }
    }
}
