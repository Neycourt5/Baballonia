using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using OscCore;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Verifies how calibration override targets reach OSC.
///
/// Three properties matter and each has bitten real projects:
/// 1. Commanded targets must bypass the user's calibration remap, or the avatar shows a different
///    value than the one recorded as the training label.
/// 2. Only the face channel is overridden; eye output must keep flowing.
/// 3. Clearing the override restores live tracking immediately.
///
/// Reflection is used to observe the private send queue rather than opening a real UDP socket, so
/// the test asserts on exact addresses and values without touching the network.
/// </summary>
[TestClass]
[TestSubject(typeof(ParameterSenderService))]
public class ParameterSenderOverrideTest
{
    private const int N = PersonalizationSchema.ExpressionCount;

    private ParameterSenderService _service = null!;
    private Mock<ICalibrationService> _calibration = null!;

    /// <summary>
    /// Builds the service without its DI graph: only the loop service is needed (for the event
    /// subscription) and only the calibration service is actually called by the code under test.
    /// </summary>
    [TestInitialize]
    public void Initialize()
    {
        var loop = (ProcessingLoopService)RuntimeHelpers.GetUninitializedObject(typeof(ProcessingLoopService));

        _calibration = new Mock<ICalibrationService>();
        // A deliberately non-identity remap: input 0..1 maps to output 0..0.5. Anything that passes
        // through calibration will be visibly halved, which is how the bypass test detects leakage.
        _calibration.Setup(c => c.GetExpressionSettings(It.IsAny<string>()))
            .Returns(new CalibrationParameter(0f, 1f, 0f, 0.5f));

        _service = new ParameterSenderService(
            vrcftModuleSendService: null!,
            dfrSendService: null!,
            localSettingsService: null!,
            calibrationService: _calibration.Object,
            processingLoopService: loop,
            logger: null!);
    }

    private ConcurrentQueue<OscMessage> Queue =>
        (ConcurrentQueue<OscMessage>)typeof(ParameterSenderService)
            .GetField("_vrcftQueue", BindingFlags.NonPublic | BindingFlags.Instance)!
            .GetValue(_service)!;

    private void SetOverrideActive(bool active) =>
        typeof(ParameterSenderService)
            .GetField("_faceOverrideActive", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(_service, active);

    private void SetPrefix(string prefix) =>
        typeof(ParameterSenderService)
            .GetField("_prefix", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(_service, prefix);

    private void Invoke(string method, params object[] args) =>
        typeof(ParameterSenderService)
            .GetMethod(method, BindingFlags.NonPublic | BindingFlags.Instance)!
            .Invoke(_service, args);

    private async Task InvokeAsync(string method, params object[] args)
    {
        var task = (Task)typeof(ParameterSenderService)
            .GetMethod(method, BindingFlags.NonPublic | BindingFlags.Instance)!
            .Invoke(_service, args)!;
        await task;
    }

    private static float[] Ramp() => Enumerable.Range(0, N).Select(i => i / (float)(N - 1)).ToArray();

    private ParameterSenderService CreateService(VrcftModuleSendService vrcftSender)
    {
        var loop = (ProcessingLoopService)RuntimeHelpers.GetUninitializedObject(typeof(ProcessingLoopService));
        return new ParameterSenderService(
            vrcftModuleSendService: vrcftSender,
            dfrSendService: null!,
            localSettingsService: null!,
            calibrationService: _calibration.Object,
            processingLoopService: loop,
            logger: null!);
    }

    private Dictionary<string, float> DrainByAddress()
    {
        var result = new Dictionary<string, float>();
        foreach (var msg in Queue)
            result[msg.Address] = (float)msg[0];
        return result;
    }

    [TestMethod]
    public void OverrideVector_IsSentVerbatim_WithoutCalibrationRemap()
    {
        var target = Ramp();

        Invoke("EnqueueRawFaceVector", target);

        var sent = DrainByAddress();
        Assert.AreEqual(N, sent.Count, "Every face expression should be commanded.");

        // The mocked calibration would halve everything; values must arrive unscaled.
        Assert.AreEqual(1f, sent["/tongueTwistRight"], 1e-6);
        Assert.AreEqual(target[PersonalizationSchema.IndexOf("JawOpen")], sent["/jawOpen"], 1e-6);
        Assert.AreEqual(target[PersonalizationSchema.IndexOf("MouthSmileLeft")], sent["/mouthSmileLeft"], 1e-6);

        _calibration.Verify(c => c.GetExpressionSettings(It.IsAny<string>()), Times.Never,
            "Commanded ground-truth targets must never pass through user calibration ranges.");
    }

    [TestMethod]
    public void OverrideVector_UsesCanonicalSchemaAddressesInOrder()
    {
        Invoke("EnqueueRawFaceVector", Ramp());

        var addresses = Queue.Select(m => m.Address).ToList();
        var expected = PersonalizationSchema.ExpressionNames
            .Select(n => _service.FaceExpressionMap[n])
            .ToList();

        CollectionAssert.AreEqual(expected, addresses,
            "Override addresses must follow the same positional schema the model and labels use.");
    }

    [TestMethod]
    public void OverrideVector_IsClampedToUnitRange()
    {
        var target = new float[N];
        target[4] = 5f;
        target[5] = -3f;

        Invoke("EnqueueRawFaceVector", target);

        var sent = DrainByAddress();
        Assert.AreEqual(1f, sent["/jawOpen"], 1e-6);
        Assert.AreEqual(0f, sent["/jawForward"], 1e-6);
    }

    [TestMethod]
    public void WhileOverrideActive_TrackedFaceValuesAreSuppressed()
    {
        SetOverrideActive(true);

        Invoke("ProcessFaceExpressionData", Ramp());

        Assert.IsTrue(Queue.IsEmpty,
            "Live tracked face values must not compete with commanded calibration targets.");
    }

    [TestMethod]
    public void WhileOverrideActive_EyeOutputStillFlows()
    {
        SetOverrideActive(true);

        Invoke("ProcessEyeExpressionData", new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f });

        Assert.IsFalse(Queue.IsEmpty,
            "Calibration overrides the face channel only; eye tracking must be unaffected.");
        Assert.IsTrue(Queue.All(m => m.Address.Contains("Eye")),
            "Only eye addresses should be present.");
    }


    [TestMethod]
    public void LegacySixEyeVector_StillUsesDefaultCalibrationPath()
    {
        Invoke("ProcessEyeExpressionData", new[] { 0.2f, 0.4f, 0.6f, 0.8f, 1f, 0.5f });

        var sent = DrainByAddress();
        Assert.AreEqual(6, sent.Count);
        Assert.AreEqual(0.1f, sent["/LeftEyeX"], 1e-6);
        Assert.AreEqual(0.3f, sent["/LeftEyeLid"], 1e-6);
        Assert.IsFalse(sent.ContainsKey("/LeftEyeWiden"));
    }

    [TestMethod]
    public void WhenOverrideInactive_TrackedFaceValuesFlowThroughCalibrationAsBefore()
    {
        SetOverrideActive(false);

        Invoke("ProcessFaceExpressionData", Ramp());

        var sent = DrainByAddress();
        Assert.AreEqual(N, sent.Count);

        // Stock path: the mocked 0..1 -> 0..0.5 remap must still apply, unchanged by this feature.
        Assert.AreEqual(0.5f, sent["/tongueTwistRight"], 1e-6);
        _calibration.Verify(c => c.GetExpressionSettings(It.IsAny<string>()), Times.Exactly(N));
    }

    [TestMethod]
    public void ClearingOverride_RestoresTrackedFaceOutputImmediately()
    {
        SetOverrideActive(true);
        Invoke("ProcessFaceExpressionData", Ramp());
        Assert.IsTrue(Queue.IsEmpty);

        SetOverrideActive(false);
        Invoke("ProcessFaceExpressionData", Ramp());

        Assert.AreEqual(N, Queue.Count, "Live face tracking must resume on the next frame.");
    }

}
