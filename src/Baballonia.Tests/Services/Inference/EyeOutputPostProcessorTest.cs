using System;
using System.Linq;
using Baballonia.Services.Inference;
using Baballonia.Services.Calibration;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Inference;

[TestClass]
public class EyeOutputPostProcessorTest
{
    [TestMethod]
    public void ClampsGazeToBipolarAndEyeShapesToUnitRange()
    {
        var map = Map(
            ("/leftEyeX", 2.5f),
            ("/leftEyeY", -3f),
            ("/leftEyeLid", 1.4f),
            ("/rightEyeWiden", -0.2f),
            ("/rightEyeSquint", 4f),
            ("/rightEyeBrow", 0.4f));

        new EyeOutputPostProcessor().Apply(map, 1f / 90f);

        Assert.AreEqual(1f, map["/leftEyeX"]);
        Assert.AreEqual(-1f, map["/leftEyeY"]);
        Assert.AreEqual(1f, map["/leftEyeLid"]);
        Assert.AreEqual(0f, map["/rightEyeWiden"]);
        Assert.AreEqual(1f, map["/rightEyeSquint"]);
        Assert.AreEqual(0.4f, map["/rightEyeBrow"]);
    }

    [TestMethod]
    public void FirstInvalidSampleFallsBackToCenteredGazeAndOpenEyes()
    {
        var map = Map(
            ("/leftEyeX", float.NaN),
            ("/leftEyeY", float.PositiveInfinity),
            ("/rightEyeX", float.NegativeInfinity),
            ("/leftEyeLid", float.NaN),
            ("/rightEyeLid", float.PositiveInfinity),
            ("/leftEyeSquint", float.NaN));

        new EyeOutputPostProcessor().Apply(map, 0f);

        Assert.AreEqual(0f, map["/leftEyeX"]);
        Assert.AreEqual(0f, map["/leftEyeY"]);
        Assert.AreEqual(0f, map["/rightEyeX"]);
        Assert.AreEqual(1f, map["/leftEyeLid"]);
        Assert.AreEqual(1f, map["/rightEyeLid"]);
        Assert.AreEqual(0f, map["/leftEyeSquint"]);
        Assert.IsTrue(map.Values.All(float.IsFinite));
    }

    [TestMethod]
    public void LaterInvalidSamplesHoldTheLastGoodValuePerKey()
    {
        var processor = new EyeOutputPostProcessor();
        var map = Map(("/leftEyeX", 0.35f), ("/leftEyeLid", 0.7f));
        processor.Apply(map, 0.01f);

        map["/leftEyeX"] = float.NaN;
        map["/leftEyeLid"] = float.NegativeInfinity;
        processor.Apply(map, 0.01f);

        Assert.AreEqual(0.35f, map["/leftEyeX"]);
        Assert.AreEqual(0.7f, map["/leftEyeLid"]);
    }

    [TestMethod]
    public void ResetDropsLastGoodValuesBackToCanonicalNeutral()
    {
        var processor = new EyeOutputPostProcessor();
        var map = Map(("/leftEyeX", 0.35f), ("/leftEyeLid", 0.7f));
        processor.Apply(map, 0f);

        processor.Reset();
        map["/leftEyeX"] = float.NaN;
        map["/leftEyeLid"] = float.NaN;
        processor.Apply(map, 0f);

        Assert.AreEqual(0f, map["/leftEyeX"]);
        Assert.AreEqual(1f, map["/leftEyeLid"]);
    }

    [TestMethod]
    public void SanitizingPreservesTheModelsKeyOrder()
    {
        var map = Map(
            ("/rightEyeY", 0.1f),
            ("/rightEyeX", 0.2f),
            ("/rightEyeLid", 0.3f),
            ("/leftEyeY", 0.4f),
            ("/leftEyeX", 0.5f),
            ("/leftEyeLid", 0.6f));
        var before = map.Keys.ToArray();

        new EyeOutputPostProcessor().Apply(map, 0f);

        CollectionAssert.AreEqual(before, map.Keys.ToArray());
    }

    [TestMethod]
    public void AppliesIndependentPerEyeOpennessAndGazeCalibration()
    {
        var left = new EyeCalibrationProfile(1, 0.10f, 0.60f, 0.90f,
            0.20f, -0.10f, 2f, 0.5f);
        var right = new EyeCalibrationProfile(1, 0.20f, 0.70f, 0.95f,
            -0.10f, 0.10f, 0.5f, 2f);
        var map = Map(
            ("/leftEyeX", 0.40f), ("/leftEyeY", 0.10f), ("/leftEyeLid", 0.60f),
            ("/rightEyeX", 0.30f), ("/rightEyeY", 0.20f), ("/rightEyeLid", 0.20f));

        new EyeOutputPostProcessor(left, right).Apply(map, 0.01f);

        Assert.AreEqual(0.40f, map["/leftEyeX"], 1e-6f);
        Assert.AreEqual(0.10f, map["/leftEyeY"], 1e-6f);
        Assert.AreEqual(0.75f, map["/leftEyeLid"], 1e-6f);
        Assert.AreEqual(0.20f, map["/rightEyeX"], 1e-6f);
        Assert.AreEqual(0.20f, map["/rightEyeY"], 1e-6f);
        Assert.AreEqual(0f, map["/rightEyeLid"], 1e-6f);
    }

    [TestMethod]
    public void InvalidInputHoldsRawHistoryWithoutDoubleApplyingCalibration()
    {
        var profile = new EyeCalibrationProfile(1, 0.10f, 0.60f, 0.90f,
            0.20f, 0f, 2f, 1f);
        var processor = new EyeOutputPostProcessor(profile, profile);
        var map = Map(("/leftEyeX", 0.40f), ("/leftEyeLid", 0.60f));
        processor.Apply(map, 0.01f);
        var mappedX = map["/leftEyeX"];
        var mappedLid = map["/leftEyeLid"];

        map["/leftEyeX"] = float.NaN;
        map["/leftEyeLid"] = float.NaN;
        processor.Apply(map, 0.01f);

        Assert.AreEqual(mappedX, map["/leftEyeX"], 1e-6f);
        Assert.AreEqual(mappedLid, map["/leftEyeLid"], 1e-6f);
    }

    [TestMethod]
    public void FirstInvalidGazeStillUsesCanonicalCenterWithOffsetProfile()
    {
        var profile = new EyeCalibrationProfile(1, 0.10f, 0.60f, 0.90f,
            0.35f, -0.20f, 2f, 2f);
        var map = Map(("/leftEyeX", float.NaN), ("/leftEyeY", float.PositiveInfinity),
            ("/leftEyeLid", float.NaN));

        new EyeOutputPostProcessor(profile, profile).Apply(map, 0f);

        Assert.AreEqual(0f, map["/leftEyeX"]);
        Assert.AreEqual(0f, map["/leftEyeY"]);
        Assert.AreEqual(1f, map["/leftEyeLid"]);
    }

    private static OrderedFloatMap Map(params (string Key, float Value)[] values)
    {
        var map = new OrderedFloatMap(values.Select(value => value.Key).ToArray());
        foreach (var value in values)
            map[value.Key] = value.Value;
        return map;
    }
}
