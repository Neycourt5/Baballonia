using System.Linq;
using Baballonia.Services.Calibration;
using Baballonia.Services.Inference;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Inference;

/// <summary>
/// The widen/squint fallback: what it does, and more importantly what it refuses to do.
/// </summary>
/// <remarks>
/// Openness alone cannot tell a squint from a blink, so the value of this machine is entirely in
/// its restraint - hysteresis, dwell and slew. Tests that only checked "squint appears when the eye
/// narrows" would pass on an implementation that flickers on every frame, which is precisely the
/// implementation nobody wants.
/// </remarks>
[TestClass]
[TestSubject(typeof(DerivedEyeShapeEstimator))]
public class DerivedEyeShapeEstimatorTest
{
    private const float Frame = 1f / 90f;

    /// <summary>Runs the estimator at a realistic frame rate for a given duration.</summary>
    private static void Hold(DerivedEyeShapeEstimator eye, float openness, float seconds)
    {
        for (var elapsed = 0f; elapsed < seconds; elapsed += Frame)
            eye.Update(openness, Frame);
    }

    [TestMethod]
    public void RestingEyeProducesNeitherShape()
    {
        var eye = new DerivedEyeShapeEstimator();

        Hold(eye, EyeCalibrationProfile.RestOpenness, 2f);

        Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Neutral, eye.State);
        Assert.AreEqual(0f, eye.Widen, 1e-6);
        Assert.AreEqual(0f, eye.Squint, 1e-6);
    }

    [TestMethod]
    public void TheWholeNeutralBandProducesNeitherShape()
    {
        // Everything between the squint and widen entry points must be silent. A dead zone that is
        // narrower than advertised shows up as an avatar that is never quite at rest.
        for (var openness = DerivedEyeShapeEstimator.SquintExit;
             openness <= DerivedEyeShapeEstimator.WidenExit;
             openness += 0.01f)
        {
            var eye = new DerivedEyeShapeEstimator();
            Hold(eye, openness, 2f);

            Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Neutral, eye.State,
                $"openness {openness:F2} should be in the neutral dead zone");
            Assert.AreEqual(0f, eye.Widen, 1e-6);
            Assert.AreEqual(0f, eye.Squint, 1e-6);
        }
    }

    [TestMethod]
    public void HeldWideOpenSettlesAndStaysThere()
    {
        var eye = new DerivedEyeShapeEstimator();

        Hold(eye, 1f, 10f);

        Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Widen, eye.State);
        Assert.AreEqual(1f, eye.Widen, 1e-4, "a fully open eye should reach full widen");
        Assert.AreEqual(0f, eye.Squint, 1e-6);

        // Thirty more seconds of the same input must not move it. This is the "held wide 30 s"
        // hardware case, run in a millisecond.
        var settled = eye.Widen;
        Hold(eye, 1f, 30f);
        Assert.AreEqual(settled, eye.Widen, 1e-6, "a held expression drifted");
    }

    [TestMethod]
    public void HeldSquintSettlesAndStaysThere()
    {
        var eye = new DerivedEyeShapeEstimator();

        Hold(eye, 0.30f, 10f);

        Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Squint, eye.State);
        Assert.IsTrue(eye.Squint > 0.5f, $"a clear squint should register strongly, got {eye.Squint}");
        Assert.AreEqual(0f, eye.Widen, 1e-6);

        var settled = eye.Squint;
        Hold(eye, 0.30f, 30f);
        Assert.AreEqual(settled, eye.Squint, 1e-6, "a held squint drifted");
    }

    /// <summary>
    /// The artefact this whole design exists to prevent: a blink reading as a deliberate squint.
    /// </summary>
    [TestMethod]
    public void ABlinkNeverRegistersAsASquint()
    {
        var eye = new DerivedEyeShapeEstimator();
        Hold(eye, EyeCalibrationProfile.RestOpenness, 1f);

        var peakSquint = 0f;

        // A realistic blink: ~80 ms closing, ~40 ms shut, ~120 ms reopening. It sweeps the entire
        // squint band twice on the way through.
        void Sweep(float from, float to, float seconds)
        {
            var steps = (int)(seconds / Frame);
            for (var i = 1; i <= steps; i++)
            {
                eye.Update(from + (to - from) * i / steps, Frame);
                peakSquint = System.Math.Max(peakSquint, eye.Squint);
            }
        }

        Sweep(EyeCalibrationProfile.RestOpenness, 0f, 0.080f);
        Hold(eye, 0f, 0.040f);
        peakSquint = System.Math.Max(peakSquint, eye.Squint);
        Sweep(0f, EyeCalibrationProfile.RestOpenness, 0.120f);

        Assert.IsTrue(peakSquint < 0.10f,
            $"a blink leaked {peakSquint:F3} of squint; dwell and blink suppression should have " +
            "kept it near zero");
    }

    [TestMethod]
    public void RapidBlinkingNeverAccumulatesSquint()
    {
        var eye = new DerivedEyeShapeEstimator();
        var peakSquint = 0f;

        for (var blink = 0; blink < 10; blink++)
        {
            Hold(eye, EyeCalibrationProfile.RestOpenness, 0.100f);
            Hold(eye, 0f, 0.060f);
            peakSquint = System.Math.Max(peakSquint, eye.Squint);
        }

        Assert.IsTrue(peakSquint < 0.10f,
            $"ten blinks in a row accumulated {peakSquint:F3} of squint");
    }

    [TestMethod]
    public void ASingleNoisyFrameChangesNothing()
    {
        var eye = new DerivedEyeShapeEstimator();
        Hold(eye, EyeCalibrationProfile.RestOpenness, 1f);

        // One frame of garbage, well past the widen entry threshold.
        eye.Update(1f, Frame);

        Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Neutral, eye.State,
            "a single frame must not flip state; that is what the dwell timer is for");
        Assert.AreEqual(0f, eye.Widen, 1e-6);
    }

    [TestMethod]
    public void HysteresisStopsOscillationOnTheThreshold()
    {
        var eye = new DerivedEyeShapeEstimator();
        Hold(eye, 1f, 1f);
        Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Widen, eye.State);

        // Sitting between the exit and entry thresholds must not drop the state: that gap is the
        // hysteresis, and without it openness noise here would chatter.
        var between = (DerivedEyeShapeEstimator.WidenExit + DerivedEyeShapeEstimator.WidenEnter) / 2f;
        Hold(eye, between, 2f);

        Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Widen, eye.State,
            "openness inside the hysteresis band should hold the current state");
    }

    [TestMethod]
    public void OutputIsRateLimited()
    {
        var eye = new DerivedEyeShapeEstimator();
        Hold(eye, EyeCalibrationProfile.RestOpenness, 1f);

        // Even after the dwell has committed, the value must arrive as a movement, not a step.
        var previous = eye.Widen;
        for (var i = 0; i < 40; i++)
        {
            eye.Update(1f, Frame);
            Assert.IsTrue(System.Math.Abs(eye.Widen - previous) <= DerivedEyeShapeEstimator.SlewPerSecond * Frame + 1e-5,
                "widen moved faster than the slew limit");
            previous = eye.Widen;
        }
    }

    [TestMethod]
    public void NonFiniteOpennessIsTreatedAsRest()
    {
        var eye = new DerivedEyeShapeEstimator();

        for (var i = 0; i < 200; i++)
            eye.Update(float.NaN, Frame);

        Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Neutral, eye.State);
        Assert.IsTrue(float.IsFinite(eye.Widen) && float.IsFinite(eye.Squint));
        Assert.AreEqual(0f, eye.Widen, 1e-6);
        Assert.AreEqual(0f, eye.Squint, 1e-6);
    }

    [TestMethod]
    public void ZeroDeltaTimeIsHarmless()
    {
        var eye = new DerivedEyeShapeEstimator();

        for (var i = 0; i < 100; i++)
            eye.Update(1f, 0f);

        // No time has passed, so no dwell can elapse and no slew can occur.
        Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Neutral, eye.State);
        Assert.AreEqual(0f, eye.Widen, 1e-6);
    }

    [TestMethod]
    public void ResetReturnsToRest()
    {
        var eye = new DerivedEyeShapeEstimator();
        Hold(eye, 1f, 5f);
        Assert.AreNotEqual(0f, eye.Widen);

        eye.Reset();

        Assert.AreEqual(DerivedEyeShapeEstimator.Shape.Neutral, eye.State);
        Assert.AreEqual(0f, eye.Widen, 1e-6);
        Assert.AreEqual(0f, eye.Squint, 1e-6);
    }
}

/// <summary>
/// How the post-processor chooses between the model's own widen/squint and derived ones.
/// </summary>
[TestClass]
[TestSubject(typeof(EyeOutputPostProcessor))]
public class EyeShapeSourceSelectionTest
{
    private const float Frame = 1f / 90f;

    /// <summary>The user's tuned model: twelve named outputs including widen, squint and brow.</summary>
    private static OrderedFloatMap TunedModelMap() => new([
        "/rightEyeY", "/rightEyeX", "/rightEyeLid", "/rightEyeWiden", "/rightEyeSquint", "/rightEyeBrow",
        "/leftEyeY", "/leftEyeX", "/leftEyeLid", "/leftEyeWiden", "/leftEyeSquint", "/leftEyeBrow",
    ]);

    /// <summary>The stock model: six outputs, no shape channels at all.</summary>
    private static OrderedFloatMap StockModelMap() => new([
        "/rightEyeY", "/rightEyeX", "/rightEyeLid",
        "/leftEyeY", "/leftEyeX", "/leftEyeLid",
    ]);

    [TestMethod]
    public void ModelSuppliedShapesArePassedThroughUntouched()
    {
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();
        map["/leftEyeLid"] = 1f;   // wide open: the estimator would want to invent widen here
        map["/rightEyeLid"] = 1f;
        map["/leftEyeWiden"] = 0.42f;
        map["/rightEyeSquint"] = 0.17f;
        map["/leftEyeBrow"] = 0.33f;

        OrderedFloatMap result = map;
        for (var i = 0; i < 200; i++)
            result = post.Apply(result, Frame);

        Assert.IsFalse(post.IsDerivingWidenSquint,
            "the model emits these channels; deriving them would throw away real predictions");
        Assert.AreEqual(0.42f, result["/leftEyeWiden"], 1e-6);
        Assert.AreEqual(0.17f, result["/rightEyeSquint"], 1e-6);
        Assert.AreEqual(0.33f, result["/leftEyeBrow"], 1e-6, "brow is never derived, only passed on");
    }

    [TestMethod]
    public void ModelSuppliedShapesAreStillClamped()
    {
        var post = new EyeOutputPostProcessor();
        var map = TunedModelMap();
        map["/leftEyeWiden"] = 4f;
        map["/rightEyeSquint"] = -2f;

        var result = post.Apply(map, Frame);

        Assert.AreEqual(1f, result["/leftEyeWiden"], 1e-6);
        Assert.AreEqual(0f, result["/rightEyeSquint"], 1e-6);
    }

    [TestMethod]
    public void AModelWithoutShapesGetsThemDerived()
    {
        var post = new EyeOutputPostProcessor();
        var map = StockModelMap();
        map["/leftEyeLid"] = 1f;
        map["/rightEyeLid"] = 1f;

        OrderedFloatMap result = map;
        for (var i = 0; i < 400; i++)
        {
            map["/leftEyeLid"] = 1f;
            map["/rightEyeLid"] = 1f;
            result = post.Apply(map, Frame);
        }

        Assert.IsTrue(post.IsDerivingWidenSquint);
        Assert.IsTrue(result.ContainsKey("/leftEyeWiden"),
            "the stock model has no widen channel, so the output map must gain one");
        Assert.AreEqual(1f, result["/leftEyeWiden"], 1e-3);
        Assert.AreEqual(0f, result["/leftEyeSquint"], 1e-6);
    }

    [TestMethod]
    public void DerivationCanBeTurnedOff()
    {
        var post = new EyeOutputPostProcessor(deriveWidenSquint: false);
        var map = StockModelMap();
        map["/leftEyeLid"] = 1f;

        var result = post.Apply(map, Frame);

        Assert.IsFalse(post.IsDerivingWidenSquint);
        Assert.AreSame(map, result, "with derivation off the map must not be widened");
        Assert.IsFalse(result.ContainsKey("/leftEyeWiden"));
    }

    [TestMethod]
    public void DerivedOutputCarriesEverySourceValueThrough()
    {
        var post = new EyeOutputPostProcessor();
        var map = StockModelMap();
        map["/leftEyeX"] = 0.5f;
        map["/rightEyeY"] = -0.25f;
        map["/leftEyeLid"] = 0.8f;

        var result = post.Apply(map, Frame);

        // Widening the map must not scramble it: every original key keeps its own value.
        Assert.AreEqual(0.5f, result["/leftEyeX"], 1e-6);
        Assert.AreEqual(-0.25f, result["/rightEyeY"], 1e-6);
        Assert.AreEqual(0.8f, result["/leftEyeLid"], 1e-6);
        foreach (var key in map.Keys)
            Assert.AreEqual(map[key], result[key], 1e-6, $"{key} changed while widening the map");
    }

    [TestMethod]
    public void EachEyeDerivesIndependently()
    {
        var post = new EyeOutputPostProcessor();
        var map = StockModelMap();

        OrderedFloatMap result = map;
        for (var i = 0; i < 400; i++)
        {
            map["/leftEyeLid"] = 1f;     // wide open
            map["/rightEyeLid"] = 0.30f; // squinting
            result = post.Apply(map, Frame);
        }

        Assert.IsTrue(result["/leftEyeWiden"] > 0.9f, "the wide eye should widen");
        Assert.AreEqual(0f, result["/leftEyeSquint"], 1e-6);
        Assert.IsTrue(result["/rightEyeSquint"] > 0.5f, "the narrowed eye should squint");
        Assert.AreEqual(0f, result["/rightEyeWiden"], 1e-6);
    }

    [TestMethod]
    public void AWinkDoesNotDisturbTheOpenEye()
    {
        var post = new EyeOutputPostProcessor();
        var map = StockModelMap();

        OrderedFloatMap result = map;
        for (var i = 0; i < 200; i++)
        {
            map["/leftEyeLid"] = 0f;                                  // winking shut
            map["/rightEyeLid"] = EyeCalibrationProfile.RestOpenness; // resting
            result = post.Apply(map, Frame);
        }

        Assert.AreEqual(0f, result["/leftEyeSquint"], 1e-6, "a closed eye is blinking, not squinting");
        Assert.AreEqual(0f, result["/rightEyeSquint"], 1e-6);
        Assert.AreEqual(0f, result["/rightEyeWiden"], 1e-6);
    }

    [TestMethod]
    public void DerivedValuesAreAlwaysFiniteAndInRange()
    {
        var post = new EyeOutputPostProcessor();
        var map = StockModelMap();
        var random = new System.Random(20260829);

        OrderedFloatMap result = map;
        for (var i = 0; i < 2000; i++)
        {
            // Deliberately hostile: garbage interleaved with plausible values.
            map["/leftEyeLid"] = i % 37 == 0 ? float.NaN : (float)random.NextDouble();
            map["/rightEyeLid"] = i % 53 == 0 ? float.PositiveInfinity : (float)random.NextDouble();
            result = post.Apply(map, Frame);

            foreach (var pair in result)
            {
                Assert.IsTrue(float.IsFinite(pair.Value), $"{pair.Key} was not finite");
                var lower = pair.Key.EndsWith("X") || pair.Key.EndsWith("Y") ? -1f : 0f;
                Assert.IsTrue(pair.Value >= lower && pair.Value <= 1f,
                    $"{pair.Key} left its range at {pair.Value}");
            }
        }

        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void CalibratedOpennessDrivesDerivation()
    {
        // A person whose eyes only ever read 0.30-0.60 raw. Uncalibrated, they would live inside the
        // squint band permanently and never widen. Calibration is what makes fixed thresholds work.
        var profile = new EyeCalibrationProfile(
            EyeCalibrationProfile.CurrentVersion,
            OpennessClosed: 0.30f, OpennessNeutral: 0.45f, OpennessWide: 0.60f,
            GazeCenterX: 0f, GazeCenterY: 0f, GazeGainX: 1f, GazeGainY: 1f);

        var post = new EyeOutputPostProcessor(profile, profile);
        var map = StockModelMap();

        OrderedFloatMap result = map;
        for (var i = 0; i < 400; i++)
        {
            map["/leftEyeLid"] = 0.60f;  // their "wide"
            map["/rightEyeLid"] = 0.45f; // their "rest"
            result = post.Apply(map, Frame);
        }

        Assert.IsTrue(result["/leftEyeWiden"] > 0.9f,
            "their wide-open pose should reach full widen after calibration");
        Assert.AreEqual(0f, result["/rightEyeSquint"], 1e-6,
            "their resting pose should read as rest, not as a permanent squint");
        Assert.AreEqual(0f, result["/rightEyeWiden"], 1e-6);
    }

    [TestMethod]
    public void SwappingToAModelWithShapesStopsDeriving()
    {
        var post = new EyeOutputPostProcessor();

        var stock = StockModelMap();
        stock["/leftEyeLid"] = 1f;
        for (var i = 0; i < 400; i++)
            post.Apply(stock, Frame);
        Assert.IsTrue(post.IsDerivingWidenSquint);

        // Loading the tuned model swaps the runner, and with it the key set.
        var tuned = TunedModelMap();
        tuned["/leftEyeWiden"] = 0.6f;
        var result = post.Apply(tuned, Frame);

        Assert.IsFalse(post.IsDerivingWidenSquint);
        Assert.AreSame(tuned, result);
        Assert.AreEqual(0.6f, result["/leftEyeWiden"], 1e-6);
    }
}
