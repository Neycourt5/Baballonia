using Baballonia.Contracts;
using Baballonia.Services.Calibration;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System;

namespace Baballonia.Services;

public class CalibrationService : ICalibrationService
{
    // Grouped by canonical range, not by pipeline ownership: gaze is bipolar; all blendshapes are
    // unit-valued. Keeping each OSC key in exactly one group prevents later loops from overwriting
    // the correct default (the old table listed right-eye X/Y in both groups).
    private readonly Dictionary<string, string> _gazeExpressionMap = new()
    {
        { "LeftEyeX", "/leftEyeX" },
        { "LeftEyeY", "/leftEyeY" },
        { "RightEyeX", "/rightEyeX" },
        { "RightEyeY", "/rightEyeY" },
    };

    private readonly Dictionary<string, string> _unitExpressionMap = new()
    {
        { "LeftEyeLid", "/leftEyeLid" },
        { "LeftEyeWiden", "/leftEyeWiden" },
        { "LeftEyeSquint", "/leftEyeSquint" },
        { "LeftEyeBrow", "/leftEyeBrow" },
        { "RightEyeLid", "/rightEyeLid" },
        { "RightEyeWiden", "/rightEyeWiden" },
        { "RightEyeSquint", "/rightEyeSquint" },
        { "RightEyeBrow", "/rightEyeBrow" },
        { "CheekPuffLeft", "/cheekPuffLeft" },
        { "CheekPuffRight", "/cheekPuffRight" },
        { "CheekSuckLeft", "/cheekSuckLeft" },
        { "CheekSuckRight", "/cheekSuckRight" },
        { "JawOpen", "/jawOpen" },
        { "JawForward", "/jawForward" },
        { "JawLeft", "/jawLeft" },
        { "JawRight", "/jawRight" },
        { "NoseSneerLeft", "/noseSneerLeft" },
        { "NoseSneerRight", "/noseSneerRight" },
        { "MouthFunnel", "/mouthFunnel" },
        { "MouthPucker", "/mouthPucker" },
        { "MouthLeft", "/mouthLeft" },
        { "MouthRight", "/mouthRight" },
        { "MouthRollUpper", "/mouthRollUpper" },
        { "MouthRollLower", "/mouthRollLower" },
        { "MouthShrugUpper", "/mouthShrugUpper" },
        { "MouthShrugLower", "/mouthShrugLower" },
        { "MouthClose", "/mouthClose" },
        { "MouthSmileLeft", "/mouthSmileLeft" },
        { "MouthSmileRight", "/mouthSmileRight" },
        { "MouthFrownLeft", "/mouthFrownLeft" },
        { "MouthFrownRight", "/mouthFrownRight" },
        { "MouthDimpleLeft", "/mouthDimpleLeft" },
        { "MouthDimpleRight", "/mouthDimpleRight" },
        { "MouthUpperUpLeft", "/mouthUpperUpLeft" },
        { "MouthUpperUpRight", "/mouthUpperUpRight" },
        { "MouthLowerDownLeft", "/mouthLowerDownLeft" },
        { "MouthLowerDownRight", "/mouthLowerDownRight" },
        { "MouthPressLeft", "/mouthPressLeft" },
        { "MouthPressRight", "/mouthPressRight" },
        { "MouthStretchLeft", "/mouthStretchLeft" },
        { "MouthStretchRight", "/mouthStretchRight" },
        { "TongueOut", "/tongueOut" },
        { "TongueUp", "/tongueUp" },
        { "TongueDown", "/tongueDown" },
        { "TongueLeft", "/tongueLeft" },
        { "TongueRight", "/tongueRight" },
        { "TongueRoll", "/tongueRoll" },
        { "TongueBendDown", "/tongueBendDown" },
        { "TongueCurlUp", "/tongueCurlUp" },
        { "TongueSquish", "/tongueSquish" },
        { "TongueFlat", "/tongueFlat" },
        { "TongueTwistLeft", "/tongueTwistLeft" },
        { "TongueTwistRight", "/tongueTwistRight" }
    };

    private readonly ConcurrentDictionary<string, CalibrationParameter> _expressionSettings = new();

    private readonly ILocalSettingsService _localSettingsService;
    private CalibrationParameter _DefaultCalibration = new CalibrationParameter();

    public CalibrationService(ILocalSettingsService localSettingsService)
    {
        _localSettingsService = localSettingsService;

        Load();
    }

    public void SetExpression(string expression, float value)
    {
        if (string.IsNullOrEmpty(expression))
            return;

        if (!expression.EndsWith("Lower") && !expression.EndsWith("Upper")) return;

        var isUpper = expression.EndsWith("Upper");
        var parameterName = expression[..^5]; // Remove "Upper"/"Lower", both 5 letters in size :3

        if (!_expressionSettings.TryGetValue(parameterName, out var currentSettings))
            currentSettings = DefaultFor(parameterName);

        var lower = isUpper ? currentSettings.Lower : value;
        var upper = isUpper ? value : currentSettings.Upper;
        var min = currentSettings.Min;
        var max = currentSettings.Max;

        var param = new CalibrationParameter(lower, upper, min, max);
        _expressionSettings[parameterName] = param;
        SaveAsync();
    }

    public CalibrationParameter GetExpressionSettings(string parameterName) // run once per paremeter, per frame
    {
        return _expressionSettings.TryGetValue(parameterName, out var settings) ?
            settings :
            _DefaultCalibration;
    }

    public CalibrationParameter GetNullableExpressionSettings(string parameterName) // run once per paremeter, per frame
    {
        return _expressionSettings.TryGetValue(parameterName, out var settings) ?
            settings :
            null;
    }

    public float GetExpressionSetting(string expression)
    {
        if (!expression.EndsWith("Lower") && !expression.EndsWith("Upper")) return 0;

        var isUpper = expression.EndsWith("Upper");
        var parameterName = expression[..^5]; // Remove "Upper"/"Lower", both 5 letters in size :3

        _expressionSettings.TryGetValue(parameterName, out var currentSettings);

        if (currentSettings == null)
            return 0;

        return isUpper ? currentSettings.Upper : currentSettings.Lower;
    }

    private void SaveAsync()
    {
        _localSettingsService.SaveSetting("CalibrationParams", _expressionSettings);
    }

    private void Load()
    {
        var parameters = _localSettingsService.ReadSetting<ConcurrentDictionary<string, CalibrationParameter>?>("CalibrationParams");
        _expressionSettings.Clear();
        if (parameters == null)
        {
            foreach (var parameterName in _gazeExpressionMap)
            {
                _expressionSettings[parameterName.Value] = new CalibrationParameter(-1, 1f, -1f, 1f);
            }

            foreach (var parameterName in _unitExpressionMap)
            {
                _expressionSettings[parameterName.Value] = new CalibrationParameter(0, 1f, 0f, 1f);
            }
        }
        else
        {
            var repairedLegacyGaze = false;
            var eyeParameterNames = _gazeExpressionMap.Values;
            foreach (var parameterName in eyeParameterNames)
            {
                var param = parameters.GetValueOrDefault(parameterName);
                // Older builds saved right-eye gaze with the blendshape default. Fixing only
                // new-profile defaults left existing users clipping every negative gaze value.
                // Match that exact old default; preserve any explicit calibration trims.
                if (param is { Lower: 0f, Upper: 1f, Min: 0f, Max: 1f })
                {
                    param = new CalibrationParameter(-1f, 1f, -1f, 1f);
                    parameters[parameterName] = param;
                    repairedLegacyGaze = true;
                }
                _expressionSettings[parameterName] = param ?? new CalibrationParameter(-1f, 1f, -1f, 1f);
            }
            var faceParameterNames = _unitExpressionMap.Values;
            foreach (var parameterName in faceParameterNames)
            {
                var param = parameters.GetValueOrDefault(parameterName);
                _expressionSettings[parameterName] = param ?? new CalibrationParameter(0f, 1f, 0f, 1f);
            }
            // Save the original dictionary so this repair also retains unknown/custom keys.
            if (repairedLegacyGaze)
                _localSettingsService.SaveSetting("CalibrationParams", parameters);
        }
    }

    public void ResetValues()
    {
        foreach (var parameter in _expressionSettings.Values)
        {
            parameter.Lower = parameter.Min;
            parameter.Upper = parameter.Max;
        }
        SaveAsync();
    }

    public void ResetMinimums()
    {
        foreach (var parameter in _expressionSettings.Values)
        {
            parameter.Lower = parameter.Min;
        }
        SaveAsync();
    }

    public void ResetMaximums()
    {
        foreach (var parameter in _expressionSettings.Values)
        {
            parameter.Upper = parameter.Max;
        }
        SaveAsync();
    }

    private static CalibrationParameter DefaultFor(string parameterName)
    {
        var normalized = parameterName.TrimStart('/');
        var isGaze = normalized.Equals("leftEyeX", StringComparison.OrdinalIgnoreCase)
            || normalized.Equals("leftEyeY", StringComparison.OrdinalIgnoreCase)
            || normalized.Equals("rightEyeX", StringComparison.OrdinalIgnoreCase)
            || normalized.Equals("rightEyeY", StringComparison.OrdinalIgnoreCase);
        return isGaze
            ? new CalibrationParameter(-1f, 1f, -1f, 1f)
            : new CalibrationParameter(0f, 1f, 0f, 1f);
    }
}
