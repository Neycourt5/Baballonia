using BabblePersonalizer.Core.Calibration;
using BabblePersonalizer.Core.Camera;
using BabblePersonalizer.Core.Models;

namespace BabblePersonalizer.Tests;

[TestClass]
public sealed class CorrectionTypeTests
{
    [TestMethod]
    public void SignedOutputRetainsDirectionAndRange()
    {
        var profile = CreateProfile(FaceParameterType.SignedContinuous, -1, 1);
        var output = new PersonalCorrectionEngine(profile).Apply(new[] { -.6f });
        Assert.AreEqual(-.6f, output[0], .001f);
    }

    [TestMethod]
    public void BinaryAndDisabledOutputsArePassthrough()
    {
        var binary = CreateProfile(FaceParameterType.Binary, 0, 1);
        Assert.AreEqual(.7f, new PersonalCorrectionEngine(binary).Apply(new[] { .7f })[0]);
        var continuous = CreateProfile(FaceParameterType.UnsignedContinuous, 0, 1);
        continuous.Parameters[0].Enabled = false;
        Assert.AreEqual(.7f, new PersonalCorrectionEngine(continuous).Apply(new[] { .7f })[0]);
    }

    private static PersonalCalibrationProfile CreateProfile(FaceParameterType type, float min, float max)
    {
        var tensor = new TensorContract("output", typeof(float).FullName!, new long[] { -1, 1 });
        return new PersonalCalibrationProfile
        {
            ProfileId = "test", SessionId = "test", CreatedUtc = DateTimeOffset.UtcNow,
            ApplicationVersion = "test", StockModelHash = "hash",
            StockInput = new TensorContract("input", typeof(float).FullName!, new long[] { 1, 1, 1, 1 }),
            StockOutput = tensor, ExpressionListHash = "list", CameraConfigurationHash = "camera",
            Camera = new CameraConfiguration(), Parameters = new List<PersonalParameterProfile>
            {
                new()
                {
                    CanonicalName = "Value", OriginalOutputName = "Value", OriginalOutputIndex = 0,
                    Category = "Test", ParameterType = type, ValidMinimum = min, ValidMaximum = max,
                    CalibrationStrategy = CalibrationStrategy.BidirectionalSigned, NeutralMedian = 0,
                    DeadZone = 0, ReliableMinimum = min, ReliableMaximum = max, ActiveRange = 1,
                    Stability = 1, RepetitionConsistency = 1, RampMonotonicity = 1, Confidence = 1,
                    AcceptedSamples = 100, RejectedSamples = 0, Enabled = true,
                    ResponseCurve = new List<ResponseCurvePoint> { new(0, 0), new(1, 1) }
                }
            }
        };
    }
}
