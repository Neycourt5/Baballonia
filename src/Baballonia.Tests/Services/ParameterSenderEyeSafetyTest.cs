using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using OscCore;

namespace Baballonia.Tests.Services;

[TestClass]
public class ParameterSenderEyeSafetyTest
{
    [TestMethod]
    [DataRow(-1f)]
    [DataRow(-0.6f)]
    [DataRow(0f)]
    [DataRow(0.6f)]
    [DataRow(1f)]
    public void LegacyRightEyeCalibrationSendsTheSameHorizontalStrengthAsTheLeft(float gaze)
    {
        var saved = new ConcurrentDictionary<string, CalibrationParameter>();
        saved["/leftEyeX"] = new CalibrationParameter(-1f, 1f, -1f, 1f);
        saved["/rightEyeX"] = new CalibrationParameter();
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<ConcurrentDictionary<string, CalibrationParameter>?>(
            "CalibrationParams", null, false)).Returns(saved);
        var sender = CreateSender(new CalibrationService(settings.Object));

        Invoke(sender, "ProcessEyeExpressionData", EyeMap(0f, gaze, 1f, 0f, gaze, 1f), null!);

        var queue = (ConcurrentQueue<OscMessage>)typeof(ParameterSenderService)
            .GetField("_vrcftQueue", BindingFlags.NonPublic | BindingFlags.Instance)!
            .GetValue(sender)!;
        var left = (float)queue.Single(m => m.Address == "/leftEyeX")[0];
        var right = (float)queue.Single(m => m.Address == "/rightEyeX")[0];
        Assert.AreEqual(gaze, left, 1e-6f);
        Assert.AreEqual(left, right, 1e-6f);
    }

    [TestMethod]
    public void NativeEyeTrackingKeepsUnequalAnatomicalYawInDocumentedOrder()
    {
        var calibration = new Mock<ICalibrationService>();
        calibration.Setup(c=>c.GetExpressionSettings(It.IsAny<string>())).Returns(new CalibrationParameter());
        var queue = new ConcurrentQueue<OscMessage>();
        Invoke(CreateSender(calibration.Object), "ProcessNativeVrcEyeTracking", EyeMap(.2f,.3f,1f,-.4f,-.1f,1f), queue);
        var gaze = queue.Single(m=>m.Address.EndsWith("LeftRightPitchYaw"));
        Assert.AreEqual(18f,(float)gaze[0],1e-5f);
        Assert.AreEqual(-4.5f,(float)gaze[1],1e-5f);
        Assert.AreEqual(-9f,(float)gaze[2],1e-5f);
        Assert.AreEqual(13.5f,(float)gaze[3],1e-5f);
    }
    [TestMethod]
    public void CalibratedEyeOutputIsFiniteAndClampedToItsConfiguredRange()
    {
        var unit = new CalibrationParameter(0.25f, 0.75f, 0f, 1f);
        var gaze = new CalibrationParameter(-0.25f, 0.25f, -1f, 1f);

        Assert.AreEqual(0f, CalibrateAndClamp("/leftEyeLid", -20f, unit));
        Assert.AreEqual(1f, CalibrateAndClamp("/leftEyeLid", 20f, unit));
        Assert.AreEqual(-1f, CalibrateAndClamp("/leftEyeX", -20f, gaze));
        Assert.AreEqual(1f, CalibrateAndClamp("/leftEyeX", 20f, gaze));

        var degenerate = new CalibrationParameter(0.5f, 0.5f, 0f, 1f);
        Assert.AreEqual(1f,
            CalibrateAndClamp("/leftEyeLid", 0.5f, degenerate));
        Assert.AreEqual(0f,
            CalibrateAndClamp("/leftEyeX", 0.5f, degenerate));
    }

    [TestMethod]
    public void NativeEyeTrackingUsesCanonicalSlashPrefixedLidCalibrationKeys()
    {
        var calibration = new Mock<ICalibrationService>();
        calibration.Setup(service => service.GetExpressionSettings(It.IsAny<string>()))
            .Returns(new CalibrationParameter());
        var sender = CreateSender(calibration.Object);
        var nativeQueue = new ConcurrentQueue<OscMessage>();

        Invoke(sender, "ProcessNativeVrcEyeTracking", EyeMap(0f, 0f, 0.7f, 0f, 0f, 0.8f), nativeQueue);

        calibration.Verify(service => service.GetExpressionSettings("/leftEyeLid"), Times.Once);
        calibration.Verify(service => service.GetExpressionSettings("/rightEyeLid"), Times.Once);
        calibration.Verify(service => service.GetExpressionSettings("LeftEyeLid"), Times.Never);
        calibration.Verify(service => service.GetExpressionSettings("RightEyeLid"), Times.Never);
        Assert.AreEqual(2, nativeQueue.Count);
    }

    [TestMethod]
    public void VrcftEyeFanoutCannotEmitCalibrationOvershoot()
    {
        var calibration = new Mock<ICalibrationService>();
        calibration.Setup(service => service.GetExpressionSettings(It.IsAny<string>()))
            .Returns((string key) => key.EndsWith("X", StringComparison.Ordinal)
                || key.EndsWith("Y", StringComparison.Ordinal)
                    ? new CalibrationParameter(-0.25f, 0.25f, -1f, 1f)
                    : new CalibrationParameter(0.25f, 0.75f, 0f, 1f));
        var sender = CreateSender(calibration.Object);

        Invoke(sender, "ProcessEyeExpressionData",
            EyeMap(-50f, 50f, 20f, 50f, -50f, -20f), null!);

        var queue = (ConcurrentQueue<OscMessage>)typeof(ParameterSenderService)
            .GetField("_vrcftQueue", BindingFlags.NonPublic | BindingFlags.Instance)!
            .GetValue(sender)!;
        Assert.AreEqual(6, queue.Count);
        Assert.IsTrue(queue.All(message => float.IsFinite((float)message[0])));
        Assert.IsTrue(queue.Where(message => message.Address.EndsWith("X") || message.Address.EndsWith("Y"))
            .All(message => (float)message[0] is >= -1f and <= 1f));
        Assert.IsTrue(queue.Where(message => message.Address.EndsWith("Lid"))
            .All(message => (float)message[0] is >= 0f and <= 1f));
    }

    private static ParameterSenderService CreateSender(ICalibrationService calibration)
    {
        var loop = (ProcessingLoopService)RuntimeHelpers.GetUninitializedObject(typeof(ProcessingLoopService));
        return new ParameterSenderService(
            vrcftModuleSendService: null!,
            dfrSendService: null!,
            localSettingsService: null!,
            calibrationService: calibration,
            processingLoopService: loop,
            logger: NullLogger<ParameterSenderService>.Instance);
    }

    private static OrderedFloatMap EyeMap(
        float rightY,
        float rightX,
        float rightLid,
        float leftY,
        float leftX,
        float leftLid)
    {
        var map = new OrderedFloatMap([
            "/rightEyeY", "/rightEyeX", "/rightEyeLid",
            "/leftEyeY", "/leftEyeX", "/leftEyeLid",
        ]);
        new[] { rightY, rightX, rightLid, leftY, leftX, leftLid }
            .AsSpan().CopyTo(map.ValuesSpan);
        return map;
    }

    private static void Invoke(object target, string method, params object?[] args) =>
        target.GetType().GetMethod(method, BindingFlags.NonPublic | BindingFlags.Instance)!
            .Invoke(target, args);

    private static float CalibrateAndClamp(
        string key,
        float value,
        CalibrationParameter settings) =>
        (float)typeof(ParameterSenderService)
            .GetMethod("CalibrateAndClampEye", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, [key, value, settings])!;
}
