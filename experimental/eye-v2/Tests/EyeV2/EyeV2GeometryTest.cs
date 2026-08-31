using Baballonia.Services.EyeV2;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;
using System;

namespace Baballonia.Tests.Services.EyeV2;

[TestClass]
public class EyeV2GeometryTest
{
    [TestMethod]
    public void ClassicExtractor_FindsPupilsAndLidApertureOnDiagnosableCrops()
    {
        using var left = SyntheticEye(aperture: 46);
        using var right = SyntheticEye(aperture: 40);
        using var input = new Mat();
        Cv2.Merge([left, right], input);
        using var extractor = new ClassicEyeGeometryExtractor();

        var result = extractor.Extract(input, DateTime.UnixEpoch.Ticks + TimeSpan.TicksPerSecond);

        Assert.IsTrue(result.Left.Valid, "Left pupil/lids were not detected.");
        Assert.IsTrue(result.Right.Valid, "Right pupil/lids were not detected.");
        Assert.AreEqual(0.5f, result.Left.PupilX, 0.08f);
        Assert.AreEqual(0.5f, result.Left.PupilY, 0.08f);
        Assert.IsTrue(result.Left.NormalizedAperture > result.Right.NormalizedAperture);
        Assert.IsNotNull(result.DebugImage, "The Paper-like annotated crop should be emitted at 10 Hz.");
        Assert.AreEqual(256 * 128 * 3, result.DebugImage!.BgrPixels.Length);
    }

    [TestMethod]
    public void GeometryHybrid_LowConfidenceIsExactlyV2A()
    {
        var calibration = EyeV2CalibrationAndMapperTest.IdentityCalibration();
        var raw = EyeV2CalibrationAndMapperTest.Raw(0.2f, -0.1f, 0.75f, -0.2f, 0.1f, 0.70f);
        var stock = new float[EyeStateLayout.LegacyCount];
        var ticks = DateTime.UnixEpoch.Ticks + TimeSpan.TicksPerSecond;
        var v2A = new EyeV2Mapper(calibration);
        using var hybrid = new EyeV2GeometryMapper(calibration, new MissingExtractor());
        using var frame = new Mat(128, 128, MatType.CV_8UC2, Scalar.All(0));

        hybrid.ObserveFrame(frame, ticks);
        CollectionAssert.AreEqual(v2A.Map(stock, raw, ticks), hybrid.Map(stock, raw, ticks));
    }

    [TestMethod]
    public void GeometryHybrid_VisibleNarrowApertureAddsSquintEvidence()
    {
        var calibration = EyeV2CalibrationAndMapperTest.IdentityCalibration();
        var geometry = new SequenceExtractor();
        using var hybrid = new EyeV2GeometryMapper(calibration, geometry);
        using var frame = new Mat(128, 128, MatType.CV_8UC2, Scalar.All(0));
        var raw = EyeV2CalibrationAndMapperTest.Raw(0, 0, 0.75f, 0, 0, 0.70f);
        var stock = new float[EyeStateLayout.LegacyCount];
        float[] output = [];

        for (var i = 0; i < 16; i++)
        {
            var ticks = DateTime.UnixEpoch.Ticks + i * 20 * TimeSpan.TicksPerMillisecond;
            geometry.Aperture = i < 12 ? 0.32f : 0.20f;
            hybrid.ObserveFrame(frame, ticks);
            output = hybrid.Map(stock, raw, ticks);
        }

        Assert.IsTrue(output[EyeStateLayout.LeftSquint] > 0.25f);
        Assert.IsTrue(output[EyeStateLayout.RightSquint] > 0.25f);
    }

    private static Mat SyntheticEye(int aperture)
    {
        var image = new Mat(128, 128, MatType.CV_8UC1, Scalar.All(220));
        Cv2.Ellipse(image, new Point(64, 64), new Size(16, 20), 0, 0, 360, Scalar.All(15), -1);
        var upper = 64 - aperture / 2;
        var lower = 64 + aperture / 2;
        Cv2.Line(image, new Point(30, upper), new Point(98, upper), Scalar.All(35), 3);
        Cv2.Line(image, new Point(30, lower), new Point(98, lower), Scalar.All(35), 3);
        return image;
    }

    private sealed class MissingExtractor : IEyeGeometryExtractor
    {
        public string Name => "missing";
        public EyeGeometryFrame Extract(Mat transformedEyeFrame, long timestampTicks) =>
            new(timestampTicks, EyeGeometry.Missing, EyeGeometry.Missing);
        public void Reset() { }
        public void Dispose() { }
    }

    private sealed class SequenceExtractor : IEyeGeometryExtractor
    {
        public float Aperture { get; set; } = 0.32f;
        public string Name => "sequence";
        public EyeGeometryFrame Extract(Mat transformedEyeFrame, long timestampTicks)
        {
            var eye = new EyeGeometry(true, 0.5f, 0.5f, 0.1f,
                new EyeGeometryLine(0.2f, 0.35f, 0.8f, 0.35f),
                new EyeGeometryLine(0.2f, 0.65f, 0.8f, 0.65f),
                Aperture, 1f, 1f);
            return new EyeGeometryFrame(timestampTicks, eye, eye);
        }
        public void Reset() { }
        public void Dispose() { }
    }
}
