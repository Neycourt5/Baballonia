using System.Collections.Concurrent;
using Baballonia.Contracts;
using Baballonia.Models;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace Baballonia.Tests.Services;

[TestClass]
[TestSubject(typeof(CalibrationService))]
public class CalibrationServiceTest
{
    private Mock<ILocalSettingsService> _localSettingsMock;
    private CalibrationService _calibrationService;

    [TestInitialize]
    public void Setup()
    {
        _localSettingsMock = new Mock<ILocalSettingsService>();
        _localSettingsMock
            .Setup(x => x.ReadSetting<ConcurrentDictionary<string, object>>("CalibrationParams", null, false))!
            .Returns((ConcurrentDictionary<string, object>?)null); // simulate empty storage

        _calibrationService = new CalibrationService(_localSettingsMock.Object);
    }


    [TestMethod]
    public void SetExpression_SetsCorrectLowerAndUpperValues()
    {
        var expression = "JawOpenLower";
        var expectedLower = 0.3f;

        _calibrationService.SetExpression(expression, expectedLower);
        var result = _calibrationService.GetExpressionSettings("JawOpen");

        Assert.AreEqual(expectedLower, result.Lower);
        Assert.AreEqual(1.0f, result.Upper); // default upper
    }

    [TestMethod]
    public void SetExpression_UpperValueOverridesCorrectly()
    {
        _calibrationService.SetExpression("JawOpenLower", 0.2f);
        _calibrationService.SetExpression("JawOpenUpper", 0.8f);

        var result = _calibrationService.GetExpressionSettings("JawOpen");

        Assert.AreEqual(0.2f, result.Lower);
        Assert.AreEqual(0.8f, result.Upper);
    }

    [TestMethod]
    public void ResetValues_ResetsAllToDefaults()
    {
        _calibrationService.SetExpression("JawOpenLower", 0.5f);
        _calibrationService.SetExpression("JawOpenUpper", 0.6f);

        _calibrationService.ResetValues();

        var result = _calibrationService.GetExpressionSettings("JawOpen");

        Assert.AreEqual(0f, result.Lower);
        Assert.AreEqual(1f, result.Upper);
    }

    [TestMethod]
    public void RightEyeGazeDefaultsAreNotOverwrittenByTheUnitRangeTable()
    {
        var rightX = _calibrationService.GetExpressionSettings("/rightEyeX");
        var rightY = _calibrationService.GetExpressionSettings("/rightEyeY");

        Assert.AreEqual(-1f, rightX.Lower);
        Assert.AreEqual(1f, rightX.Upper);
        Assert.AreEqual(-1f, rightX.Min);
        Assert.AreEqual(1f, rightX.Max);
        Assert.AreEqual(-1f, rightY.Lower);
        Assert.AreEqual(1f, rightY.Upper);
    }

    [TestMethod]
    public void SavedLegacyGazeDefaultsAreRepairedAndPersistedOnlyOnce()
    {
        var saved = new ConcurrentDictionary<string, CalibrationParameter>();
        saved["/rightEyeX"] = new CalibrationParameter();
        saved["/rightEyeY"] = new CalibrationParameter();
        var leftTrim = saved["/leftEyeX"] = new CalibrationParameter(-0.8f, 0.7f, -1f, 1f);
        var lidTrim = saved["/rightEyeLid"] = new CalibrationParameter(0.1f, 0.9f);
        var unknown = saved["CustomChannel"] = new CalibrationParameter(0.2f, 0.6f);
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<ConcurrentDictionary<string, CalibrationParameter>?>(
            "CalibrationParams", null, false)).Returns(saved);

        var calibration = new CalibrationService(settings.Object);

        foreach (var key in new[] { "/rightEyeX", "/rightEyeY" })
        {
            var repaired = calibration.GetExpressionSettings(key);
            Assert.AreEqual(-1f, repaired.Lower);
            Assert.AreEqual(1f, repaired.Upper);
            Assert.AreEqual(-1f, repaired.Min);
            Assert.AreEqual(1f, repaired.Max);
            Assert.AreSame(repaired, saved[key]);
        }
        Assert.AreSame(leftTrim, saved["/leftEyeX"]);
        Assert.AreSame(lidTrim, saved["/rightEyeLid"]);
        Assert.AreSame(unknown, saved["CustomChannel"]);

        _ = new CalibrationService(settings.Object);
        settings.Verify(s => s.SaveSetting("CalibrationParams", saved, false), Times.Once);
    }

    [TestMethod]
    public void MigrationPreservesExplicitGazeTrims()
    {
        var saved = new ConcurrentDictionary<string, CalibrationParameter>();
        var bipolar = saved["/rightEyeX"] = new CalibrationParameter(-0.6f, 0.8f, -1f, 1f);
        var customizedUnit = saved["/rightEyeY"] = new CalibrationParameter(0.2f, 0.8f);
        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<ConcurrentDictionary<string, CalibrationParameter>?>(
            "CalibrationParams", null, false)).Returns(saved);

        var calibration = new CalibrationService(settings.Object);

        Assert.AreSame(bipolar, calibration.GetExpressionSettings("/rightEyeX"));
        Assert.AreSame(customizedUnit, calibration.GetExpressionSettings("/rightEyeY"));
        settings.Verify(s => s.SaveSetting("CalibrationParams",
            It.IsAny<ConcurrentDictionary<string, CalibrationParameter>>(), false), Times.Never);
    }

    [TestMethod]
    public void SliderSettingPreservesConstructorMinimumAndMaximum()
    {
        var setting = new SliderBindableSetting("EyeX", lower: -0.5f, upper: 0.5f, min: -1f, max: 1f);

        Assert.AreEqual(-1f, setting.Min);
        Assert.AreEqual(1f, setting.Max);
    }
}
