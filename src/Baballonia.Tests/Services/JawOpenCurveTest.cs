using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using OscCore;

namespace Baballonia.Tests.Services;

/// <summary>
/// Bending how fast jaw open climbs, without moving where it ends.
/// </summary>
/// <remarks>
/// Reported as "when I open my mouth a little bit, my whole mouth and jaw will go way down… it feels
/// like I'm making a caveman face for a split second". A small real movement arriving as most of the
/// avatar's range. The user asked for it as an option specifically so nothing else would change,
/// which is why the default is exactly identity.
/// </remarks>
[TestClass]
[TestSubject(typeof(ParameterSenderService))]
public class JawOpenCurveTest
{
    // Reached by reflection, matching ParameterSenderEyeSafetyTest — these stay internal.
    private static float Shape(float v, float curve) =>
        (float)typeof(ParameterSenderService)
            .GetMethod("ShapeExpression", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, [v, curve])!;

    private static string JawOpenKey =>
        (string)typeof(ParameterSenderService)
            .GetField("JawOpenKey", BindingFlags.NonPublic | BindingFlags.Static)!
            .GetRawConstantValue()!;

    [TestMethod]
    public void TheDefaultIsExactlyIdentity()
    {
        // The user's condition for accepting this at all: untouched means unchanged.
        foreach (var v in new[] { 0f, 0.13f, 0.5f, 0.77f, 1f })
            Assert.AreEqual(v, Shape(v, 1f), 0f, "the default must not alter a single value");
    }

    [TestMethod]
    public void ASmallOpeningStopsDroppingTheWholeJaw()
    {
        // A quarter-open mouth reading as a quarter of the avatar's range is the complaint;
        // at 2.0 it becomes a sixteenth.
        Assert.AreEqual(0.0625f, Shape(0.25f, 2f), 1e-5);
        Assert.IsTrue(Shape(0.25f, 1.5f) < 0.25f);
    }

    [TestMethod]
    public void OpeningWideStillReachesFull()
    {
        // The whole reason this is an exponent and not a gain or a ceiling.
        foreach (var curve in new[] { 0.5f, 1f, 1.5f, 2f, 3f })
            Assert.AreEqual(1f, Shape(1f, curve), 1e-6, $"curve {curve} capped a wide opening");
    }

    [TestMethod]
    public void AClosedMouthStaysClosed()
    {
        foreach (var curve in new[] { 0.5f, 1.5f, 3f })
            Assert.AreEqual(0f, Shape(0f, curve), 1e-6);
    }

    [TestMethod]
    public void AGainWouldHaveCappedTheWideOpening()
    {
        // Contrast with the obvious alternative. Scaling by 0.25 tames the small opening the same
        // amount, but the avatar can then never open its mouth past a quarter.
        // Pick the gain that tames a half-open mouth by the same amount the curve does.
        const float gain = 0.5f;
        Assert.AreEqual(Shape(0.5f, 2f), 0.5f * gain, 1e-6, "matched at the small end by construction");

        // At the wide end they diverge completely, and that is the whole argument.
        Assert.AreEqual(0.5f, 1f * gain, 1e-6, "the gain halves a fully open mouth");
        Assert.AreEqual(1f, Shape(1f, 2f), 1e-6, "the curve leaves it fully open");
    }

    [TestMethod]
    public void TheCurveIsMonotone()
    {
        foreach (var curve in new[] { 0.5f, 1.5f, 2f, 3f })
        {
            var previous = -1f;
            for (var v = 0f; v <= 1f; v += 0.01f)
            {
                var shaped = Shape(v, curve);
                Assert.IsTrue(shaped >= previous - 1e-6, $"curve {curve} went backwards at {v}");
                Assert.IsTrue(shaped is >= 0f and <= 1f, $"curve {curve} left [0,1] at {v}");
                previous = shaped;
            }
        }
    }

    [TestMethod]
    public void BelowOneMakesItMoreEager()
    {
        // The opposite complaint, for someone whose jaw barely registers.
        Assert.IsTrue(Shape(0.25f, 0.5f) > 0.25f);
        Assert.AreEqual(1f, Shape(1f, 0.5f), 1e-6);
    }

    [TestMethod]
    public void GarbageIsPassedThroughUntouched()
    {
        Assert.IsTrue(float.IsNaN(Shape(float.NaN, 2f)));
        Assert.AreEqual(0.5f, Shape(0.5f, float.NaN), 1e-6);
        Assert.AreEqual(0.5f, Shape(0.5f, 0f), 1e-6, "a zero exponent must not become a constant 1");
        Assert.AreEqual(0.5f, Shape(0.5f, -2f), 1e-6);
    }

    [TestMethod]
    public void JawKeyMatchesTheRealPipelineSchema()
    {
        var canonicalJaw = PersonalizationSchemaBinding.ExpectedKeys[PersonalizationSchema.IndexOf("JawOpen")];
        Assert.AreEqual("/jawOpen", canonicalJaw);
        Assert.AreEqual(canonicalJaw, JawOpenKey,
            "The curve must recognize the OSC key emitted by the real face pipeline.");
    }

    private bool _curveEnabled;
    private float _savedCurve = 1f;
    private Mock<ILocalSettingsService> _settings = null!;
    private Mock<ICalibrationService> _calibration = null!;
    private ParameterSenderService _sender = null!;

    [TestInitialize]
    public void SetUpSender()
    {
        _curveEnabled = false;
        _savedCurve = 1f;
        _settings = new Mock<ILocalSettingsService>(MockBehavior.Strict);
        _settings.Setup(s => s.ReadSetting("AppSettings_JawOpenCurveEnabled", false, false))
            .Returns(() => _curveEnabled);
        _settings.Setup(s => s.ReadSetting("AppSettings_JawOpenCurve", 1f, false))
            .Returns(() => _savedCurve);
        _calibration = new Mock<ICalibrationService>(MockBehavior.Strict);
        UseCalibration(new CalibrationParameter(0f, 1f, 0f, 1f));
        var loop = (ProcessingLoopService)RuntimeHelpers.GetUninitializedObject(typeof(ProcessingLoopService));
        _sender = new ParameterSenderService(
            vrcftModuleSendService: null!,
            dfrSendService: null!,
            localSettingsService: _settings.Object,
            calibrationService: _calibration.Object,
            processingLoopService: loop,
            logger: NullLogger<ParameterSenderService>.Instance);
    }

    private void UseCalibration(CalibrationParameter range) =>
        _calibration.Setup(c => c.GetExpressionSettings(It.IsAny<string>())).Returns(range);

    private void InvokeSender(string method, object argument) =>
        typeof(ParameterSenderService).GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(_sender, [argument]);

    private Dictionary<string, float> DrainQueue()
    {
        var queue = (ConcurrentQueue<OscMessage>)typeof(ParameterSenderService)
            .GetField("_vrcftQueue", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(_sender)!;
        var sent = new Dictionary<string, float>();
        while (queue.TryDequeue(out var message))
            sent.Add(message.Address, (float)message[0]);
        return sent;
    }

    private Dictionary<string, float> SendFace(float jaw)
    {
        var map = new OrderedFloatMap(PersonalizationSchemaBinding.ExpectedKeys.ToArray());
        map.ValuesSpan.Fill(0.3f);
        map["/jawOpen"] = jaw;
        InvokeSender("ProcessFaceExpressionData", map);
        return DrainQueue();
    }

    private static float ReadEffective(ILocalSettingsService? settings) =>
        (float)typeof(ParameterSenderService)
            .GetMethod("ReadEffectiveJawOpenCurve", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, [settings])!;

    [TestMethod]
    public void NormalOscJawIsCurvedBeforeCalibrationAndOtherChannelsStayUnchanged()
    {
        // Before the curve: 0.5 maps to 0.68. Squaring first: 0.25 maps to 0.38.
        // Squaring after calibration would produce 0.4624, so ordering is observable.
        UseCalibration(new CalibrationParameter(0.1f, 0.6f, 0.2f, 0.8f));
        _savedCurve = 2f;
        var baseline = SendFace(0.5f);
        _curveEnabled = true;
        var curved = SendFace(0.5f);

        Assert.AreEqual(PersonalizationSchema.ExpressionCount, curved.Count);
        Assert.AreEqual(0.68f, baseline["/jawOpen"], 1e-6f);
        Assert.AreEqual(0.38f, curved["/jawOpen"], 1e-6f);
        foreach (var key in PersonalizationSchemaBinding.ExpectedKeys.Where(key => key != "/jawOpen"))
            Assert.AreEqual(baseline[key], curved[key], 0f, $"The jaw setting changed {key}.");
    }

    [TestMethod]
    public void AnExistingNonDefaultSettingRemainsInertUntilExplicitlyEnabled()
    {
        UseCalibration(new CalibrationParameter(0.1f, 0.6f, 0.2f, 0.8f));
        foreach (var saved in new[] { 0.5f, 1.5f, 2f, 3f })
        {
            _savedCurve = saved;
            Assert.AreEqual(0.68f, SendFace(0.5f)["/jawOpen"], 1e-6f,
                "A previously saved slider value must not silently change tracking after upgrade.");
        }
    }

    [TestMethod]
    public void ToggleAndHotEditsApplyOnTheNextFaceBatchWithoutWaitingForTheSenderLoop()
    {
        // The background sender never starts in this fixture. These consecutive batches must
        // nevertheless observe settings edits immediately, without a one-second cached value.
        _savedCurve = 2f;
        Assert.AreEqual(0.5f, SendFace(0.5f)["/jawOpen"], 0f);
        _curveEnabled = true;
        Assert.AreEqual(0.25f, SendFace(0.5f)["/jawOpen"], 1e-6f);
        _savedCurve = 3f;
        Assert.AreEqual(0.125f, SendFace(0.5f)["/jawOpen"], 1e-6f);
        _savedCurve = 0.5f;
        Assert.AreEqual(0.70710678f, SendFace(0.5f)["/jawOpen"], 1e-6f);
        _curveEnabled = false;
        Assert.AreEqual(0.5f, SendFace(0.5f)["/jawOpen"], 0f);
    }

    [TestMethod]
    public void SenderKeepsIdentityAndEndpointsWithTheRealOscKey()
    {
        _curveEnabled = true;
        _savedCurve = 1f;
        foreach (var value in new[] { 0f, 0.13f, 0.5f, 0.77f, 1f })
            Assert.AreEqual(value, SendFace(value)["/jawOpen"], 0f);

        UseCalibration(new CalibrationParameter(0f, 1f, 0.2f, 0.8f));
        foreach (var curve in new[] { 0.5f, 1f, 2f, 3f })
        {
            _savedCurve = curve;
            Assert.AreEqual(0.2f, SendFace(0f)["/jawOpen"], 1e-6f);
            Assert.AreEqual(0.8f, SendFace(1f)["/jawOpen"], 1e-6f);
        }
    }

    [TestMethod]
    public void MissingSettingsAndMissingOptInRemainIdentity()
    {
        Assert.AreEqual(1f, ReadEffective(null), 0f);
        Assert.AreEqual(1f, ReadEffective(new Mock<ILocalSettingsService>().Object), 0f);
        _curveEnabled = true;
        _settings.Setup(s => s.ReadSetting("AppSettings_JawOpenCurve", 1f, false))
            .Returns((string key, float fallback, bool forceLocal) => fallback);
        Assert.AreEqual(0.5f, SendFace(0.5f)["/jawOpen"], 0f,
            "Enabling the feature without a saved exponent must keep the default curve.");
    }

    [TestMethod]
    public void InvalidSavedSettingsCannotPoisonTheNormalJawOutput()
    {
        _curveEnabled = true;
        foreach (var invalid in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            _savedCurve = invalid;
            Assert.AreEqual(0.5f, SendFace(0.5f)["/jawOpen"], 0f);
        }
        foreach (var belowMinimum in new[] { -2f, 0f, 0.1f })
        {
            _savedCurve = belowMinimum;
            Assert.AreEqual(0.70710678f, SendFace(0.5f)["/jawOpen"], 1e-6f);
        }
        foreach (var aboveMaximum in new[] { 4f, float.MaxValue })
        {
            _savedCurve = aboveMaximum;
            Assert.AreEqual(0.125f, SendFace(0.5f)["/jawOpen"], 1e-6f);
        }
    }

    [TestMethod]
    public void SettingsAreCapturedAtTheFaceBatchBoundary()
    {
        _curveEnabled = true;
        _savedCurve = 2f;
        _calibration.Setup(c => c.GetExpressionSettings(It.IsAny<string>()))
            .Returns((string key) =>
            {
                // The jaw is the fifth schema entry, so this edit occurs while earlier channels
                // are already being processed. It must apply to the next batch, not this one.
                _savedCurve = 3f;
                return new CalibrationParameter(0f, 1f, 0f, 1f);
            });

        Assert.AreEqual(0.25f, SendFace(0.5f)["/jawOpen"], 1e-6f);
        Assert.AreEqual(0.125f, SendFace(0.5f)["/jawOpen"], 1e-6f);
    }

    [TestMethod]
    public void CalibrationOverrideBypassesEnabledCurveAndCalibrationRemap()
    {
        _curveEnabled = true;
        _savedCurve = 3f;
        UseCalibration(new CalibrationParameter(0.1f, 0.6f, 0.2f, 0.8f));
        var target = Enumerable.Repeat(0.3f, PersonalizationSchema.ExpressionCount).ToArray();
        target[PersonalizationSchema.IndexOf("JawOpen")] = 0.5f;

        InvokeSender("EnqueueRawFaceVector", target);
        var sent = DrainQueue();

        Assert.AreEqual(target.Length, sent.Count);
        for (var i = 0; i < target.Length; i++)
            Assert.AreEqual(target[i], sent[PersonalizationSchemaBinding.ExpectedKeys[i]], 0f);
        _calibration.Verify(c => c.GetExpressionSettings(It.IsAny<string>()), Times.Never);
    }
}
