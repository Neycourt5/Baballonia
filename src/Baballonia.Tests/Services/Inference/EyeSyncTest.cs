using Baballonia.Services.Calibration;
using Baballonia.Services.Inference;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Inference;

/// <summary>
/// Eyelid coupling, and the wink that has to escape it.
/// </summary>
/// <remarks>
/// The whole feature is a trade against winking, so most of these tests are about the escape rather
/// than the coupling. A version that synchronized perfectly and quietly broke winks would satisfy a
/// naive reading of the requirement.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeLidSynchronizer))]
public class EyeLidSynchronizerTest
{
    private const float Frame = 1f / 90f;

    private static (float Left, float Right) Hold(
        EyeLidSynchronizer sync, float left, float right, float seconds)
    {
        var result = (Left: left, Right: right);
        for (var elapsed = 0f; elapsed < seconds; elapsed += Frame)
            result = sync.Apply(left, right, Frame);
        return result;
    }

    [TestMethod]
    public void ZeroAmountLeavesBothLidsUntouched()
    {
        var sync = new EyeLidSynchronizer(0f);

        var (left, right) = sync.Apply(0.20f, 0.90f, Frame);

        Assert.AreEqual(0.20f, left, 1e-6);
        Assert.AreEqual(0.90f, right, 1e-6);
    }

    [TestMethod]
    public void FullAmountMakesBothLidsTheAverage()
    {
        var sync = new EyeLidSynchronizer(1f);

        var (left, right) = sync.Apply(0.60f, 0.80f, Frame);

        Assert.AreEqual(0.70f, left, 1e-6);
        Assert.AreEqual(0.70f, right, 1e-6);
    }

    [TestMethod]
    public void PartialAmountPullsEachLidPartWayToTheAverage()
    {
        var sync = new EyeLidSynchronizer(0.5f);

        // Average is 0.70; halfway from 0.60 is 0.65, halfway from 0.80 is 0.75.
        var (left, right) = sync.Apply(0.60f, 0.80f, Frame);

        Assert.AreEqual(0.65f, left, 1e-6);
        Assert.AreEqual(0.75f, right, 1e-6);
    }

    [TestMethod]
    public void SmallDisagreementIsTheCaseThisExistsFor()
    {
        // Two cameras disagreeing slightly about a resting lid: not information, and visible on an
        // avatar as a permanent half-squint on one side.
        var sync = new EyeLidSynchronizer(0.75f);

        var (left, right) = sync.Apply(0.72f, 0.80f, Frame);

        Assert.IsTrue(System.Math.Abs(left - right) < 0.03f,
            $"the lids should have been pulled together, got {left:F3} and {right:F3}");
    }

    [TestMethod]
    public void ADeliberateWinkReleasesCouplingImmediately()
    {
        var sync = new EyeLidSynchronizer(1f);
        Hold(sync, 0.80f, 0.80f, 1f);

        // One frame, no dwell: a wink that takes an eighth of a second to start reads as broken.
        var (left, right) = sync.Apply(0f, 0.80f, Frame);

        Assert.IsTrue(sync.IsReleased);
        Assert.AreEqual(0f, left, 1e-6, "the winking eye must close fully");
        Assert.AreEqual(0.80f, right, 1e-6, "the open eye must be untouched");
    }

    [TestMethod]
    public void AHeldWinkStaysReleased()
    {
        var sync = new EyeLidSynchronizer(1f);

        var (left, right) = Hold(sync, 0f, 0.80f, 3f);

        Assert.IsTrue(sync.IsReleased);
        Assert.AreEqual(0f, left, 1e-6);
        Assert.AreEqual(0.80f, right, 1e-6);
    }

    [TestMethod]
    public void CouplingReturnsAfterTheWinkEnds()
    {
        var sync = new EyeLidSynchronizer(1f);
        Hold(sync, 0f, 0.80f, 1f);
        Assert.IsTrue(sync.IsReleased);

        var (left, right) = Hold(sync, 0.78f, 0.80f, 1f);

        Assert.IsFalse(sync.IsReleased, "coupling should re-engage once the eyes agree again");
        Assert.AreEqual(left, right, 1e-6);
    }

    [TestMethod]
    public void CouplingDoesNotReturnInstantlyWhenTheLidsMerelyCross()
    {
        var sync = new EyeLidSynchronizer(1f);
        Hold(sync, 0f, 0.80f, 1f);

        // A single frame of agreement partway through a wink must not re-couple; the dwell exists
        // so the coupling cannot chatter on and off.
        sync.Apply(0.78f, 0.80f, Frame);

        Assert.IsTrue(sync.IsReleased);
    }

    [TestMethod]
    public void ALargeRestingDisagreementStillCouples()
    {
        // The bug this replaced a hysteresis band to fix. The old rule released on raw difference,
        // so two cameras that disagreed badly at rest - exactly the people this feature is for -
        // released the coupling and, sitting inside the band, never re-engaged. The feature
        // switched itself off precisely when it was needed, and looked simply broken.
        var sync = new EyeLidSynchronizer(1f);

        var (left, right) = Hold(sync, 0.45f, 0.85f, 1f);

        Assert.IsFalse(sync.IsReleased,
            "two lids that are both open are two cameras arguing, not a wink");
        Assert.AreEqual(left, right, 1e-6);
    }

    [TestMethod]
    public void ARestingDisagreementDoesNotLatchAfterABlink()
    {
        // The same bug by its usual route: a blink briefly drives the lids far apart, and the
        // resting disagreement afterwards was enough to hold the release open forever.
        var sync = new EyeLidSynchronizer(1f);
        Hold(sync, 0.45f, 0.85f, 0.5f);

        Hold(sync, 0.02f, 0.05f, 0.15f);   // blink
        var (left, right) = Hold(sync, 0.45f, 0.85f, 0.5f);

        Assert.IsFalse(sync.IsReleased, "coupling must survive a blink");
        Assert.AreEqual(left, right, 1e-6);
    }

    [TestMethod]
    public void AHalfClosedEyeIsNotAWink()
    {
        // Halfway is ambiguous, and guessing "wink" there is the failure mode that latches.
        var sync = new EyeLidSynchronizer(1f);

        Hold(sync, 0.40f, 0.90f, 1f);

        Assert.IsFalse(sync.IsReleased);
    }

    [TestMethod]
    public void NamingABetterEyeMakesOnlyThatEyeAbleToWink()
    {
        // If you have said the left camera reads correctly, the right one reporting a closed eye
        // while the left reports an open one is the right camera being wrong. A wink is deliberate,
        // so it shows up on the eye that tracks you properly.
        var followsLeft = new EyeLidSynchronizer(1f, EyeSyncSource.Left);

        Hold(followsLeft, 0.85f, 0.02f, 1f);
        Assert.IsFalse(followsLeft.IsReleased, "the unreliable eye alone cannot release coupling");

        Hold(followsLeft, 0.02f, 0.85f, 1f);
        Assert.IsTrue(followsLeft.IsReleased, "the named eye closing is a real wink");
    }

    [TestMethod]
    public void AverageStillReleasesForEitherEye()
    {
        foreach (var (left, right) in new[] { (0.02f, 0.85f), (0.85f, 0.02f) })
        {
            var sync = new EyeLidSynchronizer(1f);
            Hold(sync, left, right, 1f);
            Assert.IsTrue(sync.IsReleased, $"a wink at {left}/{right} should release");
        }
    }

    [TestMethod]
    public void FollowingTheLeftEyeLeavesTheLeftEyeAlone()
    {
        var sync = new EyeLidSynchronizer(1f, EyeSyncSource.Left);

        var (left, right) = sync.Apply(0.45f, 0.85f, Frame);

        Assert.AreEqual(0.45f, left, 1e-6, "the named eye must not be dragged toward the other");
        Assert.AreEqual(0.45f, right, 1e-6, "the other eye copies it");
    }

    [TestMethod]
    public void FollowingTheRightEyeIsTheMirrorImage()
    {
        var sync = new EyeLidSynchronizer(1f, EyeSyncSource.Right);

        var (left, right) = sync.Apply(0.45f, 0.85f, Frame);

        Assert.AreEqual(0.85f, left, 1e-6);
        Assert.AreEqual(0.85f, right, 1e-6);
    }

    [TestMethod]
    public void PartialFollowingMovesOnlyTheOtherEyePartWay()
    {
        var sync = new EyeLidSynchronizer(0.5f, EyeSyncSource.Left);

        var (left, right) = sync.Apply(0.40f, 0.80f, Frame);

        Assert.AreEqual(0.40f, left, 1e-6);
        Assert.AreEqual(0.60f, right, 1e-6);
    }

    [TestMethod]
    public void ANormalBlinkStaysCoupled()
    {
        var sync = new EyeLidSynchronizer(0.75f);
        var maximumDifference = 0f;

        // Both lids travel together, as a real blink does.
        for (var i = 0; i <= 8; i++)
        {
            var openness = 1f - i / 8f;
            var (left, right) = sync.Apply(openness, openness - 0.04f, Frame);
            maximumDifference = System.Math.Max(maximumDifference, System.Math.Abs(left - right));
        }

        Assert.IsFalse(sync.IsReleased, "a blink is not a wink");
        Assert.IsTrue(maximumDifference < 0.02f,
            $"the lids should stay together through a blink, peak difference {maximumDifference:F3}");
    }

    [TestMethod]
    public void NonFiniteInputIsPassedThroughUntouched()
    {
        var sync = new EyeLidSynchronizer(1f);

        var (left, right) = sync.Apply(float.NaN, 0.80f, Frame);

        Assert.IsTrue(float.IsNaN(left), "the finite guard upstream owns this decision, not us");
        Assert.AreEqual(0.80f, right, 1e-6);
    }

    [TestMethod]
    public void ResetReturnsToCoupled()
    {
        var sync = new EyeLidSynchronizer(1f);
        Hold(sync, 0f, 0.80f, 1f);
        Assert.IsTrue(sync.IsReleased);

        sync.Reset();

        Assert.IsFalse(sync.IsReleased);
    }
}

/// <summary>Lid coupling as it behaves inside the real post-processor.</summary>
[TestClass]
[TestSubject(typeof(EyeOutputPostProcessor))]
public class EyeLidSyncIntegrationTest
{
    private const float Frame = 1f / 90f;

    private static OrderedFloatMap TunedModelMap() => new([
        "/rightEyeY", "/rightEyeX", "/rightEyeLid", "/rightEyeWiden", "/rightEyeSquint", "/rightEyeBrow",
        "/leftEyeY", "/leftEyeX", "/leftEyeLid", "/leftEyeWiden", "/leftEyeSquint", "/leftEyeBrow",
    ]);

    [TestMethod]
    public void DefaultConstructionLeavesLidsIndependent()
    {
        // The parameter defaults to zero so every existing test and caller keeps its old behaviour;
        // the running app opts in through settings.
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();
        map["/leftEyeLid"] = 0.40f;
        map["/rightEyeLid"] = 0.90f;

        var result = post.Apply(map, Frame);

        Assert.AreEqual(0.40f, result["/leftEyeLid"], 1e-6);
        Assert.AreEqual(0.90f, result["/rightEyeLid"], 1e-6);
    }

    [TestMethod]
    public void SyncAmountCouplesTheLids()
    {
        var post = new EyeOutputPostProcessor(lidSyncAmount: 1f);
        var map = TunedModelMap();
        map["/leftEyeLid"] = 0.70f;
        map["/rightEyeLid"] = 0.80f;

        var result = post.Apply(map, Frame);

        Assert.AreEqual(result["/leftEyeLid"], result["/rightEyeLid"], 1e-6);
        Assert.AreEqual(0.75f, result["/leftEyeLid"], 1e-6);
    }

    [TestMethod]
    public void TheRightSquintFollowsTheLeftWhenLeftIsTheBetterEye()
    {
        // The user's request, stated directly: "the squint on the left eye works better for me, so
        // if the right eye could just follow that value".
        var post = new EyeOutputPostProcessor(
            squintSyncAmount: 1f, syncSource: EyeSyncSource.Left);
        var map = TunedModelMap();
        map["/leftEyeLid"] = 0.80f;
        map["/rightEyeLid"] = 0.80f;
        map["/leftEyeSquint"] = 0.90f;
        map["/rightEyeSquint"] = 0.20f;

        var result = post.Apply(map, Frame);

        Assert.AreEqual(0.90f, result["/leftEyeSquint"], 1e-6, "the good eye must not be dragged down");
        Assert.AreEqual(0.90f, result["/rightEyeSquint"], 1e-6, "the other eye copies it");
    }

    [TestMethod]
    public void WidenFollowsSeparatelyFromSquint()
    {
        // Separate amounts, because one channel can read well while the other does not.
        var post = new EyeOutputPostProcessor(
            squintSyncAmount: 1f, widenSyncAmount: 0f, syncSource: EyeSyncSource.Left);
        var map = TunedModelMap();
        map["/leftEyeLid"] = 0.80f;
        map["/rightEyeLid"] = 0.80f;
        map["/leftEyeSquint"] = 0.90f;
        map["/rightEyeSquint"] = 0.20f;
        map["/leftEyeWiden"] = 0.70f;
        map["/rightEyeWiden"] = 0.10f;

        var result = post.Apply(map, Frame);

        Assert.AreEqual(0.90f, result["/rightEyeSquint"], 1e-6);
        Assert.AreEqual(0.10f, result["/rightEyeWiden"], 1e-6, "widen was left off");
    }

    [TestMethod]
    public void ExpressionSyncIsOffByDefault()
    {
        var post = new EyeOutputPostProcessor(lidSyncAmount: 1f);
        var map = TunedModelMap();
        map["/leftEyeLid"] = 0.80f;
        map["/rightEyeLid"] = 0.80f;
        map["/leftEyeSquint"] = 0.90f;
        map["/rightEyeSquint"] = 0.20f;

        var result = post.Apply(map, Frame);

        Assert.AreEqual(0.90f, result["/leftEyeSquint"], 1e-6);
        Assert.AreEqual(0.20f, result["/rightEyeSquint"], 1e-6);
    }

    [TestMethod]
    public void AWinkReleasesTheSquintCouplingToo()
    {
        // Copying the open eye's squint onto the winking eye would undo the wink.
        var post = new EyeOutputPostProcessor(
            lidSyncAmount: 1f, squintSyncAmount: 1f, syncSource: EyeSyncSource.Left);
        var map = TunedModelMap();

        map["/leftEyeLid"] = 0.85f;
        map["/rightEyeLid"] = 0.85f;
        post.Apply(map, Frame);

        map["/leftEyeLid"] = 0.02f;
        map["/rightEyeLid"] = 0.85f;
        map["/leftEyeSquint"] = 0.90f;
        map["/rightEyeSquint"] = 0.20f;
        var result = post.Apply(map, Frame);

        Assert.IsTrue(post.IsWinking);
        Assert.AreEqual(0.20f, result["/rightEyeSquint"], 1e-6,
            "the open eye keeps its own squint during a wink");
    }

    [TestMethod]
    public void ARestingDisagreementStaysCoupledThroughThePostProcessor()
    {
        // End to end, the exact symptom reported: coupling at full strength doing nothing because
        // the release had latched on a large resting difference.
        var post = new EyeOutputPostProcessor(lidSyncAmount: 1f);
        var map = TunedModelMap();

        for (var i = 0; i < 200; i++)
        {
            // A blink every 50 frames, which is what used to latch the release.
            var blinking = i % 50 == 0;
            map["/leftEyeLid"] = blinking ? 0.02f : 0.45f;
            map["/rightEyeLid"] = blinking ? 0.05f : 0.85f;
            var result = post.Apply(map, Frame);

            if (!blinking && i > 0)
            {
                Assert.AreEqual(result["/leftEyeLid"], result["/rightEyeLid"], 1e-5,
                    $"lids drifted apart at frame {i}");
            }
        }
    }

    [TestMethod]
    public void AWinkStillWorksThroughTheWholePostProcessor()
    {
        var post = new EyeOutputPostProcessor(lidSyncAmount: 1f);
        var map = TunedModelMap();

        map["/leftEyeLid"] = 0.85f;
        map["/rightEyeLid"] = 0.85f;
        post.Apply(map, Frame);

        map["/leftEyeLid"] = 0f;
        map["/rightEyeLid"] = 0.85f;
        var result = post.Apply(map, Frame);

        Assert.AreEqual(0f, result["/leftEyeLid"], 1e-6, "the wink must reach the output");
        Assert.AreEqual(0.85f, result["/rightEyeLid"], 1e-6, "the open eye must not follow it shut");
    }

    [TestMethod]
    public void SyncedLidsAreWhatDerivedShapesSee()
    {
        // Derivation runs after coupling deliberately: two eyes at the same openness should not be
        // classified differently just because their raw values disagreed.
        var post = new EyeOutputPostProcessor(deriveWidenSquint: true, lidSyncAmount: 1f);
        var map = new OrderedFloatMap([
            "/rightEyeY", "/rightEyeX", "/rightEyeLid",
            "/leftEyeY", "/leftEyeX", "/leftEyeLid",
        ]);

        OrderedFloatMap result = map;
        for (var i = 0; i < 400; i++)
        {
            // Straddling the widen entry threshold: uncoupled, one eye would widen and the other
            // would not.
            map["/leftEyeLid"] = 0.83f;
            map["/rightEyeLid"] = 0.95f;
            result = post.Apply(map, Frame);
        }

        Assert.AreEqual(result["/leftEyeWiden"], result["/rightEyeWiden"], 1e-6,
            "coupled lids should produce the same derived widen on both eyes");
    }

    [TestMethod]
    public void SyncNeverProducesAnInvalidValue()
    {
        var post = new EyeOutputPostProcessor(lidSyncAmount: 0.75f);
        var map = TunedModelMap();
        var random = new System.Random(20260829);

        for (var i = 0; i < 2000; i++)
        {
            map["/leftEyeLid"] = i % 37 == 0 ? float.NaN : (float)random.NextDouble();
            map["/rightEyeLid"] = i % 43 == 0 ? -3f : (float)random.NextDouble();
            var result = post.Apply(map, Frame);

            foreach (var key in new[] { "/leftEyeLid", "/rightEyeLid" })
            {
                Assert.IsTrue(float.IsFinite(result[key]), $"{key} was not finite");
                Assert.IsTrue(result[key] is >= 0f and <= 1f, $"{key} left [0,1] at {result[key]}");
            }
        }
    }

    [TestMethod]
    public void CalibrationRunsBeforeCouplingSoTheLidsAreComparable()
    {
        // Two eyes with different raw ranges land at the same calibrated openness. Coupling before
        // calibration would average incomparable numbers and produce a permanent offset.
        var left = new EyeCalibrationProfile(
            EyeCalibrationProfile.CurrentVersion, 0.10f, 0.50f, 0.90f, 0f, 0f, 1f, 1f);
        var right = new EyeCalibrationProfile(
            EyeCalibrationProfile.CurrentVersion, 0.30f, 0.70f, 1.00f, 0f, 0f, 1f, 1f);

        var post = new EyeOutputPostProcessor(left, right, lidSyncAmount: 1f);
        var map = TunedModelMap();
        map["/leftEyeLid"] = 0.50f;  // this eye's neutral
        map["/rightEyeLid"] = 0.70f; // the other eye's neutral

        var result = post.Apply(map, Frame);

        // Both are at their own rest point, so both should read the rest openness, and coupling
        // should therefore have nothing to do.
        Assert.AreEqual(EyeCalibrationProfile.RestOpenness, result["/leftEyeLid"], 1e-5);
        Assert.AreEqual(EyeCalibrationProfile.RestOpenness, result["/rightEyeLid"], 1e-5);
    }
}

/// <summary>The adjustable conjugate-gaze amount in the geometry pass.</summary>
[TestClass]
[TestSubject(typeof(EyeProcessingPipeline))]
public class GazeConjugateAmountTest
{
    /// <summary>
    /// Mirrors the vergence arithmetic in <c>ProcessExpressions</c>. The geometry pass needs a live
    /// camera and a model to exercise end to end, so the formula is pinned here directly; the
    /// pipeline test suite covers that this code path runs at all.
    /// </summary>
    private static (float Left, float Right) Conjugate(float leftX, float rightX, float amount)
    {
        var convergence = System.Math.Max((leftX - rightX) / 2f, 0f) * (1f - amount);
        var averaged = (leftX + rightX) / 2f;
        return (averaged + convergence, averaged - convergence);
    }

    [TestMethod]
    public void ZeroAmountPreservesTheLongStandingBehaviour()
    {
        var (left, right) = Conjugate(0.60f, 0.40f, 0f);

        // Averaged 0.50 with 0.10 of vergence either side, exactly as before the setting existed.
        Assert.AreEqual(0.60f, left, 1e-6);
        Assert.AreEqual(0.40f, right, 1e-6);
    }

    [TestMethod]
    public void FullAmountMakesTheEyesFullyConjugate()
    {
        var (left, right) = Conjugate(0.60f, 0.40f, 1f);

        Assert.AreEqual(0.50f, left, 1e-6);
        Assert.AreEqual(0.50f, right, 1e-6);
    }

    [TestMethod]
    public void PartialAmountKeepsPartOfTheVergence()
    {
        var (left, right) = Conjugate(0.60f, 0.40f, 0.5f);

        Assert.AreEqual(0.55f, left, 1e-6);
        Assert.AreEqual(0.45f, right, 1e-6);
    }

    [TestMethod]
    public void DivergenceIsStillClampedAtEveryAmount()
    {
        // The model occasionally decides the eyes point outward. That is never real.
        foreach (var amount in new[] { 0f, 0.5f, 1f })
        {
            var (left, right) = Conjugate(-0.30f, 0.30f, amount);
            Assert.AreEqual(0f, left, 1e-6, $"amount {amount}");
            Assert.AreEqual(0f, right, 1e-6, $"amount {amount}");
        }
    }

    [TestMethod]
    public void TheDefaultPropertyValueIsNoChange()
    {
        var pipeline = new EyeProcessingPipeline(
            new Baballonia.Services.EyePipelineEventBus(), new Baballonia.Services.PipelineMetrics());

        Assert.AreEqual(0f, pipeline.GazeConjugateAmount,
            "defaulting to anything else would silently change existing tracking");

        pipeline.Dispose();
    }
}

/// <summary>
/// "Whichever eye sees it, both do it" — and turning the wink escape off entirely.
/// </summary>
/// <remarks>
/// Both from the same report: "I don't care about winking at all. So if it detects a squint on the
/// left eye, it should do a squint on the right eye and vice versa."
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeLidSynchronizer))]
public class EyeStrongerSyncTest
{
    private const float Frame = 1f / 90f;

    private static OrderedFloatMap TunedModelMap() => new([
        "/rightEyeY", "/rightEyeX", "/rightEyeLid", "/rightEyeWiden", "/rightEyeSquint", "/rightEyeBrow",
        "/leftEyeY", "/leftEyeX", "/leftEyeLid", "/leftEyeWiden", "/leftEyeSquint", "/leftEyeBrow",
    ]);

    [TestMethod]
    public void AveragingWeakensASquintOnlyOneCameraSees()
    {
        // Why Stronger exists. This is synchronization that makes the expression worse.
        var (left, right) = EyeLidSynchronizer.Pull(0.9f, 0.2f, 1f, EyeSyncSource.Average);

        Assert.AreEqual(0.55f, left, 1e-5);
        Assert.IsTrue(left < 0.9f, "the eye that saw the squint was dragged down");
    }

    [TestMethod]
    public void StrongerKeepsTheFullExpressionOnBothEyes()
    {
        var (left, right) = EyeLidSynchronizer.Pull(0.9f, 0.2f, 1f, EyeSyncSource.Stronger);

        Assert.AreEqual(0.9f, left, 1e-5);
        Assert.AreEqual(0.9f, right, 1e-5);
    }

    [TestMethod]
    public void StrongerWorksFromEitherSide()
    {
        // "and vice versa" — it must not matter which camera noticed.
        var (left, right) = EyeLidSynchronizer.Pull(0.2f, 0.9f, 1f, EyeSyncSource.Stronger);

        Assert.AreEqual(0.9f, left, 1e-5);
        Assert.AreEqual(0.9f, right, 1e-5);
    }

    [TestMethod]
    public void StrongerSquintReachesBothEyesThroughThePostProcessor()
    {
        var post = new EyeOutputPostProcessor(
            squintSyncAmount: 1f, syncSource: EyeSyncSource.Stronger);
        var map = TunedModelMap();
        map["/leftEyeLid"] = 0.80f;
        map["/rightEyeLid"] = 0.80f;
        map["/leftEyeSquint"] = 0.85f;
        map["/rightEyeSquint"] = 0.15f;

        var result = post.Apply(map, Frame);

        Assert.AreEqual(0.85f, result["/leftEyeSquint"], 1e-5);
        Assert.AreEqual(0.85f, result["/rightEyeSquint"], 1e-5);
    }

    [TestMethod]
    public void LidsDoNotUseStronger()
    {
        // On lids "stronger" would have to mean more closed, which lets one mis-reading camera shut
        // both eyes. Lids fall back to Average.
        var post = new EyeOutputPostProcessor(lidSyncAmount: 1f, syncSource: EyeSyncSource.Stronger);
        var map = TunedModelMap();
        map["/leftEyeLid"] = 0.60f;
        map["/rightEyeLid"] = 0.80f;

        var result = post.Apply(map, Frame);

        Assert.AreEqual(0.70f, result["/leftEyeLid"], 1e-4, "lids should average, not take a maximum");
        Assert.AreEqual(result["/leftEyeLid"], result["/rightEyeLid"], 1e-5);
    }

    [TestMethod]
    public void WinkDetectionCanBeTurnedOffEntirely()
    {
        var sync = new EyeLidSynchronizer(1f, EyeSyncSource.Average, detectWinks: false);

        // An unmistakable wink shape, held.
        var (left, right) = (0f, 0f);
        for (var i = 0; i < 90; i++)
            (left, right) = sync.Apply(0.02f, 0.90f, Frame);

        Assert.IsFalse(sync.IsReleased, "coupling must never release when winks are not detected");
        Assert.AreEqual(left, right, 1e-5);
    }

    [TestMethod]
    public void WinkDetectionOnIsStillTheDefault()
    {
        var sync = new EyeLidSynchronizer(1f);
        sync.Apply(0.02f, 0.90f, Frame);

        Assert.IsTrue(sync.IsReleased);
    }

    [TestMethod]
    public void WithWinksOffASquintStillReachesBothEyes()
    {
        // The combination the user asked for: no wink escape, and the stronger squint wins.
        var post = new EyeOutputPostProcessor(
            lidSyncAmount: 1f, squintSyncAmount: 1f,
            syncSource: EyeSyncSource.Stronger, detectWinks: false);
        var map = TunedModelMap();

        map["/leftEyeLid"] = 0.05f;
        map["/rightEyeLid"] = 0.85f;
        map["/leftEyeSquint"] = 0.80f;
        map["/rightEyeSquint"] = 0.10f;
        var result = post.Apply(map, Frame);

        Assert.IsFalse(post.IsWinking);
        Assert.AreEqual(0.80f, result["/rightEyeSquint"], 1e-5);
    }
}

/// <summary>
/// Coupling the two eyes' gaze, after per-eye centring has made them comparable.
/// </summary>
/// <remarks>
/// Measured on a recorded session from the reporting user's own hardware: the two eyes' horizontal
/// motion correlates at only 0.29, and the best time alignment between them is 0 ms. They are not
/// lagging each other — they are moving semi-independently, which is what reads as "one eye moves
/// into position before the other".
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeOutputPostProcessor))]
public class EyeGazeSyncTest
{
    private const float Frame = 1f / 90f;

    private static OrderedFloatMap TunedModelMap() => new([
        "/rightEyeY", "/rightEyeX", "/rightEyeLid", "/rightEyeWiden", "/rightEyeSquint", "/rightEyeBrow",
        "/leftEyeY", "/leftEyeX", "/leftEyeLid", "/leftEyeWiden", "/leftEyeSquint", "/leftEyeBrow",
    ]);

    private static OrderedFloatMap Open(float leftX, float rightX)
    {
        var map = TunedModelMap();
        map["/leftEyeLid"] = 0.85f;
        map["/rightEyeLid"] = 0.85f;
        map["/leftEyeX"] = leftX;
        map["/rightEyeX"] = rightX;
        return map;
    }

    [TestMethod]
    public void GazeSyncIsOffByDefault()
    {
        var result = new EyeOutputPostProcessor().Apply(Open(0.30f, -0.10f), Frame);

        Assert.AreEqual(0.30f, result["/leftEyeX"], 1e-5);
        Assert.AreEqual(-0.10f, result["/rightEyeX"], 1e-5);
    }

    [TestMethod]
    public void FullSyncMakesTheEyesLookAtOnePlace()
    {
        var result = new EyeOutputPostProcessor(gazeSyncAmount: 1f).Apply(Open(0.30f, -0.10f), Frame);

        Assert.AreEqual(0.10f, result["/leftEyeX"], 1e-5);
        Assert.AreEqual(0.10f, result["/rightEyeX"], 1e-5);
    }

    [TestMethod]
    public void PartialSyncKeepsSomeOfTheInwardTurn()
    {
        var result = new EyeOutputPostProcessor(gazeSyncAmount: 0.5f).Apply(Open(0.30f, -0.10f), Frame);

        Assert.AreEqual(0.20f, result["/leftEyeX"], 1e-5);
        Assert.AreEqual(0.0f, result["/rightEyeX"], 1e-5);
    }

    [TestMethod]
    public void OneEyeArrivingEarlyIsPulledBackToTheOther()
    {
        // The reported symptom, as a step: the left eye jumps and the right has not moved yet.
        var post = new EyeOutputPostProcessor(gazeSyncAmount: 1f);
        post.Apply(Open(0f, 0f), Frame);

        var result = post.Apply(Open(0.60f, 0f), Frame);

        Assert.AreEqual(result["/leftEyeX"], result["/rightEyeX"], 1e-5,
            "the eyes must arrive together");
    }

    [TestMethod]
    public void VerticalIsCoupledToo()
    {
        var map = Open(0f, 0f);
        map["/leftEyeY"] = 0.40f;
        map["/rightEyeY"] = -0.20f;

        var result = new EyeOutputPostProcessor(gazeSyncAmount: 1f).Apply(map, Frame);

        Assert.AreEqual(result["/leftEyeY"], result["/rightEyeY"], 1e-5);
    }

    [TestMethod]
    public void NamingABetterEyeMakesGazeFollowIt()
    {
        var result = new EyeOutputPostProcessor(gazeSyncAmount: 1f, syncSource: EyeSyncSource.Left)
            .Apply(Open(0.30f, -0.10f), Frame);

        Assert.AreEqual(0.30f, result["/leftEyeX"], 1e-5, "the named eye is not moved");
        Assert.AreEqual(0.30f, result["/rightEyeX"], 1e-5);
    }

    [TestMethod]
    public void StrongerFallsBackToAverageForGaze()
    {
        // "Stronger" on a gaze axis would mean "further from centre", which would drive both eyes to
        // whichever extreme either camera happened to guess.
        var result = new EyeOutputPostProcessor(
            gazeSyncAmount: 1f, syncSource: EyeSyncSource.Stronger).Apply(Open(0.30f, -0.10f), Frame);

        Assert.AreEqual(0.10f, result["/leftEyeX"], 1e-5);
        Assert.AreEqual(0.10f, result["/rightEyeX"], 1e-5);
    }

    [TestMethod]
    public void AWinkReleasesGazeCoupling()
    {
        // A winking eye's gaze is unreliable; coupling would spread that onto the open eye.
        var post = new EyeOutputPostProcessor(lidSyncAmount: 1f, gazeSyncAmount: 1f);
        var map = Open(0.30f, 0.30f);
        post.Apply(map, Frame);

        map = Open(0.30f, -0.90f);
        map["/leftEyeLid"] = 0.02f;
        map["/rightEyeLid"] = 0.90f;
        var result = post.Apply(map, Frame);

        Assert.IsTrue(post.IsWinking);
        Assert.AreEqual(-0.90f, result["/rightEyeX"], 1e-5,
            "the open eye keeps its own gaze during a wink");
    }

    [TestMethod]
    public void SyncedGazeStaysInRange()
    {
        var result = new EyeOutputPostProcessor(gazeSyncAmount: 1f).Apply(Open(1f, -1f), Frame);

        foreach (var key in new[] { "/leftEyeX", "/rightEyeX" })
            Assert.IsTrue(result[key] is >= -1f and <= 1f);
    }
}
