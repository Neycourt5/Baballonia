using System;
using Baballonia.Services.Calibration;
using Baballonia.Services.Inference;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Inference;

/// <summary>
/// Blink shaping: the closing edge must stay instant, the reopen must stop fluttering.
/// </summary>
/// <remarks>
/// The failure mode worth guarding against is not "blinks look wrong" but "someone made blinks
/// smooth". Smoothing the closing edge is the single easiest way to make tracking feel laggy, and
/// it would be an easy accident for a later change to introduce, so the zero-latency close is
/// asserted directly rather than inferred.
/// </remarks>
[TestClass]
[TestSubject(typeof(BlinkReopenLimiter))]
public class BlinkReopenLimiterTest
{
    private const float Frame = 1f / 90f;

    [TestMethod]
    public void ClosingIsInstantAtAnySpeed()
    {
        var lid = new BlinkReopenLimiter();
        lid.Apply(1f, Frame);

        // Fully open to fully shut in one frame. Any rate limit here is felt as latency.
        Assert.AreEqual(0f, lid.Apply(0f, Frame), 1e-6);
    }

    [TestMethod]
    public void PartialClosingIsAlsoInstant()
    {
        var lid = new BlinkReopenLimiter();
        lid.Apply(1f, Frame);

        Assert.AreEqual(0.4f, lid.Apply(0.4f, Frame), 1e-6);
        Assert.AreEqual(0.1f, lid.Apply(0.1f, Frame), 1e-6);
    }

    [TestMethod]
    public void ReopeningIsRateLimited()
    {
        var lid = new BlinkReopenLimiter();
        lid.Apply(1f, Frame);
        lid.Apply(0f, Frame);

        var afterOneFrame = lid.Apply(1f, Frame);

        Assert.IsTrue(afterOneFrame < 1f, "a full reopen in one frame is the flutter this prevents");
        Assert.AreEqual(BlinkReopenLimiter.RisePerSecond * Frame, afterOneFrame, 1e-5);
    }

    [TestMethod]
    public void AFullReopenTakesAboutOneHundredMilliseconds()
    {
        var lid = new BlinkReopenLimiter();
        lid.Apply(1f, Frame);
        lid.Apply(0f, Frame);

        var elapsed = 0f;
        while (lid.Apply(1f, Frame) < 0.999f)
        {
            elapsed += Frame;
            Assert.IsTrue(elapsed < 0.5f, "the reopen never completed");
        }

        Assert.AreEqual(0.100f, elapsed, 0.020f,
            $"expected roughly a 100 ms recovery, took {elapsed * 1000:F0} ms");
    }

    [TestMethod]
    public void TheFirstFrameDoesNotRampUpFromZero()
    {
        var lid = new BlinkReopenLimiter();

        // Otherwise the eye appears to blink open every time tracking starts or a camera reconnects.
        Assert.AreEqual(1f, lid.Apply(1f, Frame), 1e-6);
    }

    [TestMethod]
    public void TheFirstFrameAfterResetDoesNotRampUpEither()
    {
        var lid = new BlinkReopenLimiter();
        lid.Apply(1f, Frame);
        lid.Apply(0f, Frame);

        lid.Reset();

        Assert.AreEqual(0.9f, lid.Apply(0.9f, Frame), 1e-6);
    }

    [TestMethod]
    public void ASteadyOpenEyeIsNotAltered()
    {
        var lid = new BlinkReopenLimiter();

        for (var i = 0; i < 500; i++)
            Assert.AreEqual(0.75f, lid.Apply(0.75f, Frame), 1e-6);
    }

    [TestMethod]
    public void NonFiniteInputHoldsTheLastValue()
    {
        var lid = new BlinkReopenLimiter();
        lid.Apply(0.6f, Frame);

        Assert.AreEqual(0.6f, lid.Apply(float.NaN, Frame), 1e-6);
        Assert.AreEqual(0.6f, lid.Apply(float.PositiveInfinity, Frame), 1e-6);
    }

    [TestMethod]
    public void NonFiniteFirstInputReadsAsOpen()
    {
        var lid = new BlinkReopenLimiter();

        Assert.AreEqual(1f, lid.Apply(float.NaN, Frame), 1e-6);
    }

    [TestMethod]
    public void ZeroDeltaTimeHoldsRatherThanJumping()
    {
        var lid = new BlinkReopenLimiter();
        lid.Apply(0f, Frame);

        Assert.AreEqual(0f, lid.Apply(1f, 0f), 1e-6);
    }
}

/// <summary>Blink shaping as it behaves inside the real post-processor.</summary>
[TestClass]
[TestSubject(typeof(EyeOutputPostProcessor))]
public class EyeBlinkShapingIntegrationTest
{
    private const float Frame = 1f / 90f;

    private static OrderedFloatMap TunedModelMap() => new([
        "/rightEyeY", "/rightEyeX", "/rightEyeLid", "/rightEyeWiden", "/rightEyeSquint", "/rightEyeBrow",
        "/leftEyeY", "/leftEyeX", "/leftEyeLid", "/leftEyeWiden", "/leftEyeSquint", "/leftEyeBrow",
    ]);

    [TestMethod]
    public void BlinkOnsetReachesOscWithoutDelay()
    {
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();

        map["/leftEyeLid"] = 1f;
        map["/rightEyeLid"] = 1f;
        post.Apply(map, Frame);

        map["/leftEyeLid"] = 0f;
        map["/rightEyeLid"] = 0f;
        var result = post.Apply(map, Frame);

        Assert.AreEqual(0f, result["/leftEyeLid"], 1e-6);
        Assert.AreEqual(0f, result["/rightEyeLid"], 1e-6);
    }

    [TestMethod]
    public void EachEyeReopensOnItsOwnSchedule()
    {
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();

        map["/leftEyeLid"] = 1f;
        map["/rightEyeLid"] = 1f;
        post.Apply(map, Frame);

        // A wink: the left eye closes, the right stays open.
        map["/leftEyeLid"] = 0f;
        var winking = post.Apply(map, Frame);
        Assert.AreEqual(0f, winking["/leftEyeLid"], 1e-6);
        Assert.AreEqual(1f, winking["/rightEyeLid"], 1e-6,
            "shaping one eye must not disturb the other");

        map["/leftEyeLid"] = 1f;
        var reopening = post.Apply(map, Frame);
        Assert.IsTrue(reopening["/leftEyeLid"] < 1f, "the winking eye should be rate-limited");
        Assert.AreEqual(1f, reopening["/rightEyeLid"], 1e-6, "the open eye should be untouched");
    }

    [TestMethod]
    public void ShapingCanBeTurnedOff()
    {
        var post = new EyeOutputPostProcessor(shapeBlinks: false);
        var map = TunedModelMap();

        map["/leftEyeLid"] = 0f;
        post.Apply(map, Frame);
        map["/leftEyeLid"] = 1f;

        Assert.AreEqual(1f, post.Apply(map, Frame)["/leftEyeLid"], 1e-6);
    }

    [TestMethod]
    public void ShapingIsAppliedToDerivedModelsToo()
    {
        var post = new EyeOutputPostProcessor();
        var map = new OrderedFloatMap([
            "/rightEyeY", "/rightEyeX", "/rightEyeLid",
            "/leftEyeY", "/leftEyeX", "/leftEyeLid",
        ]);

        map["/leftEyeLid"] = 1f;
        post.Apply(map, Frame);
        map["/leftEyeLid"] = 0f;
        post.Apply(map, Frame);

        map["/leftEyeLid"] = 1f;
        var result = post.Apply(map, Frame);

        Assert.IsTrue(result["/leftEyeLid"] < 1f,
            "the widened map's lid must be shaped as well, not left raw");
    }

    [TestMethod]
    public void ShapingNeverProducesAnInvalidValue()
    {
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();
        var random = new Random(20260829);

        for (var i = 0; i < 3000; i++)
        {
            map["/leftEyeLid"] = i % 41 == 0 ? float.NaN : (float)random.NextDouble();
            map["/rightEyeLid"] = i % 29 == 0 ? -5f : (float)random.NextDouble();
            var result = post.Apply(map, Frame);

            foreach (var key in new[] { "/leftEyeLid", "/rightEyeLid" })
            {
                Assert.IsTrue(float.IsFinite(result[key]), $"{key} was not finite");
                Assert.IsTrue(result[key] is >= 0f and <= 1f, $"{key} left [0,1] at {result[key]}");
            }
        }
    }

    [TestMethod]
    public void ShapingDoesNotStopTheEyeFromReachingFullyOpen()
    {
        // A rate limit that never lets the value arrive would read as permanently droopy lids.
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();

        map["/leftEyeLid"] = 0f;
        post.Apply(map, Frame);

        // The map is the inference runner's reusable buffer: it is overwritten with fresh model
        // output every frame, and the post-processor writes its result back into it. Re-supplying
        // the input each iteration is what the real pipeline does.
        OrderedFloatMap result = map;
        for (var i = 0; i < 90; i++)
        {
            map["/leftEyeLid"] = 1f;
            result = post.Apply(map, Frame);
        }

        Assert.AreEqual(1f, result["/leftEyeLid"], 1e-6);
    }

    [TestMethod]
    public void AHeldClosureStaysClosed()
    {
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();

        OrderedFloatMap result = map;
        for (var i = 0; i < 450; i++) // five seconds
        {
            map["/leftEyeLid"] = 0f;
            result = post.Apply(map, Frame);
        }

        Assert.AreEqual(0f, result["/leftEyeLid"], 1e-6, "a held closure must not creep open");
    }

    [TestMethod]
    public void ResetClearsShapingStateAlongWithEverythingElse()
    {
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();

        map["/leftEyeLid"] = 0f;
        post.Apply(map, Frame);

        // A camera swap resets the pipeline; the next frame is a fresh start, not a reopen.
        post.Reset();

        map["/leftEyeLid"] = EyeCalibrationProfile.RestOpenness;
        Assert.AreEqual(EyeCalibrationProfile.RestOpenness,
            post.Apply(map, Frame)["/leftEyeLid"], 1e-6);
    }
}
