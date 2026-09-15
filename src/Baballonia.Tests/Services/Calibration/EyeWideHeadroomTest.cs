using Baballonia.Services.Calibration;
using Baballonia.Services.Inference;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Calibration;

/// <summary>
/// What a saturated lid channel does to the top of the openness range.
/// </summary>
/// <remarks>
/// <para>The mapping reserves the output above <see cref="EyeCalibrationProfile.RestOpenness"/> for
/// "wider than relaxed". That is right when the lid channel can report such a state. On a channel
/// that saturates — where relaxed already reads near 1 — it is not: the reserved quarter is
/// unreachable, so the avatar's eyes sit permanently three-quarters open and wide-eye never lands
/// however hard the user opens their eyes.</para>
///
/// <para>Reported as "I'm not able to do wide eye anymore… her eyes would get a surprise expression
/// when it was activated all the way before", immediately after the calibration started applying.</para>
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeCalibrationProfile))]
public class EyeWideHeadroomTest
{
    /// <summary>Synthetic saturated-lid anchors with no reachable wide-eye headroom.</summary>
    private static EyeCalibrationProfile Left => EyeCalibrationProfile.Default with
    {
        OpennessClosed = 0.55f, OpennessNeutral = 0.97f, OpennessWide = 1.02f,
    };

    private static EyeCalibrationProfile Right => EyeCalibrationProfile.Default with
    {
        OpennessClosed = 0.60f, OpennessNeutral = 0.99f, OpennessWide = 1.04f,
    };

    /// <summary>A channel with genuine room above relaxed.</summary>
    private static EyeCalibrationProfile Roomy => EyeCalibrationProfile.Default with
    {
        OpennessClosed = 0.20f, OpennessNeutral = 0.70f, OpennessWide = 0.95f,
    };

    [TestMethod]
    public void WideOpenEyesReachFullyOpen()
    {
        // The regression: reserving the final quarter made full opening unreachable.
        Assert.AreEqual(1f, Left.MapOpenness(1f), 1e-5);
        Assert.AreEqual(1f, Right.MapOpenness(1f), 1e-5);
    }

    [TestMethod]
    public void ASaturatedChannelIsRecognisedAsHavingNoHeadroom()
    {
        Assert.IsFalse(Left.HasWideHeadroom);
        Assert.IsFalse(Right.HasWideHeadroom);
        Assert.IsTrue(Roomy.HasWideHeadroom);
    }

    [TestMethod]
    public void AChannelWithRoomKeepsTheReservedRange()
    {
        // Where the reservation is meaningful it must survive: relaxed still maps to rest, and the
        // top of the range still means "wider than relaxed".
        Assert.AreEqual(EyeCalibrationProfile.RestOpenness, Roomy.MapOpenness(0.70f), 1e-5);
        Assert.AreEqual(1f, Roomy.MapOpenness(0.95f), 1e-5);
        Assert.IsTrue(Roomy.MapOpenness(0.82f) > EyeCalibrationProfile.RestOpenness);
    }

    [TestMethod]
    public void ASyntheticAnchorAboveOneIsNotReachable()
    {
        // Why the reservation was wrong here: the anchor sits outside the input domain entirely,
        // because MapOpenness clamps its input to [0,1].
        Assert.IsTrue(Left.OpennessWide > 1f);
        Assert.AreEqual(1f, Left.ReachableWide, 1e-6);
    }

    [TestMethod]
    public void ClosedStillMapsToShut()
    {
        Assert.AreEqual(0f, Left.MapOpenness(Left.OpennessClosed), 1e-5);
        Assert.AreEqual(0f, Right.MapOpenness(Right.OpennessClosed), 1e-5);
        Assert.AreEqual(0f, Left.MapOpenness(0f), 1e-5);
    }

    [TestMethod]
    public void TheMappingIsStillMonotone()
    {
        foreach (var profile in new[] { Left, Right, Roomy })
        {
            var previous = -1f;
            for (var raw = 0f; raw <= 1f; raw += 0.01f)
            {
                var mapped = profile.MapOpenness(raw);
                Assert.IsTrue(mapped >= previous - 1e-5, $"went backwards at {raw}");
                Assert.IsTrue(mapped is >= 0f and <= 1f, $"left [0,1] at {raw}");
                previous = mapped;
            }
        }
    }

    [TestMethod]
    public void RelaxedStillReadsAsRelaxedNotWide()
    {
        // Full range must not come at the cost of a resting face reading as wide-eyed. Relaxed is
        // the top of this channel's range, so it maps to fully open - but anything below it, which
        // is where a real resting face with any lid movement sits, must be clearly under that.
        Assert.IsTrue(Left.MapOpenness(0.90f) < 0.85f,
            "a slightly-lowered lid must not still read as fully open");
    }
}

/// <summary>Lid coupling surviving the stage that runs after it.</summary>
[TestClass]
[TestSubject(typeof(EyeOutputPostProcessor))]
public class EyeLidResyncTest
{
    private const float Frame = 1f / 90f;

    private static OrderedFloatMap TunedModelMap() => new([
        "/rightEyeY", "/rightEyeX", "/rightEyeLid", "/rightEyeWiden", "/rightEyeSquint", "/rightEyeBrow",
        "/leftEyeY", "/leftEyeX", "/leftEyeLid", "/leftEyeWiden", "/leftEyeSquint", "/leftEyeBrow",
    ]);

    [TestMethod]
    public void BlinkShapingCannotPullTheLidsApartAgain()
    {
        // Reported as "my right eye continues to just eyelid move while the squint's active". Blink
        // shaping keeps one rate limiter per eye; two lids that arrive identical can leave it apart,
        // most visibly while a squint holds them mid-range where the limiters have room to disagree.
        var post = new EyeOutputPostProcessor(lidSyncAmount: 1f, shapeBlinks: true);
        var map = TunedModelMap();

        // Settle, then blink to give the two limiters different histories.
        for (var i = 0; i < 30; i++)
        {
            map["/leftEyeLid"] = 0.85f;
            map["/rightEyeLid"] = 0.85f;
            post.Apply(map, Frame);
        }

        map["/leftEyeLid"] = 0.05f;
        map["/rightEyeLid"] = 0.30f;
        post.Apply(map, Frame);

        // Now hold a squint: mid-range, where the limiters diverge most.
        for (var i = 0; i < 60; i++)
        {
            map["/leftEyeLid"] = 0.45f;
            map["/rightEyeLid"] = 0.55f;
            var result = post.Apply(map, Frame);

            Assert.AreEqual(result["/leftEyeLid"], result["/rightEyeLid"], 1e-5,
                $"lids drifted apart at frame {i} of the squint");
        }
    }

    [TestMethod]
    public void AWinkIsStillNotResynchronized()
    {
        var post = new EyeOutputPostProcessor(lidSyncAmount: 1f, shapeBlinks: true);
        var map = TunedModelMap();

        map["/leftEyeLid"] = 0.85f;
        map["/rightEyeLid"] = 0.85f;
        post.Apply(map, Frame);

        map["/leftEyeLid"] = 0.02f;
        map["/rightEyeLid"] = 0.85f;
        var result = post.Apply(map, Frame);

        Assert.IsTrue(post.IsWinking);
        Assert.IsTrue(result["/rightEyeLid"] > 0.5f, "the open eye must not be dragged shut");
    }

    [TestMethod]
    public void ResyncDoesNotAccelerateTheRecoupleDwell()
    {
        // The re-sync is pull-only on purpose: advancing the state machine twice per frame would
        // halve the dwell it advertises.
        var post = new EyeOutputPostProcessor(lidSyncAmount: 1f, shapeBlinks: true);
        var map = TunedModelMap();

        map["/leftEyeLid"] = 0.02f;
        map["/rightEyeLid"] = 0.85f;
        post.Apply(map, Frame);
        Assert.IsTrue(post.IsWinking);

        // Half the dwell: still released.
        map["/leftEyeLid"] = 0.80f;
        map["/rightEyeLid"] = 0.85f;
        for (var i = 0; i < 5; i++)
            post.Apply(map, Frame);

        Assert.IsTrue(post.IsWinking, "coupling re-engaged in under the advertised dwell");
    }
}
