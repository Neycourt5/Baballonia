using Baballonia.Contracts;
using Microsoft.Extensions.Logging;
using System;

namespace Baballonia.Services.EyeV2;

/// <summary>Owns the Default/V2 selection and the fail-safe mapper hot swap.</summary>
public sealed class EyeV2Manager
{
    public const string ModeSetting = "EyeV2_Mode";

    private readonly ILocalSettingsService _settings;
    private readonly EyeV2CalibrationStore _store;
    private readonly ILogger<EyeV2Manager> _logger;
    private readonly Action<IEyeStateMapper?> _setMapper;

    public EyeTrackingMode Mode { get; private set; } = EyeTrackingMode.DefaultBaballonia;
    public EyeV2Calibration? Calibration { get; private set; }
    public EyeV2Diagnostics? Diagnostics { get; private set; }
    public string Status { get; private set; } = "Default Baballonia eye tracking is active.";

    public event Action? StateChanged;
    public event Action<EyeV2Diagnostics>? DiagnosticsChanged;

    public EyeV2Manager(
        EyePipelineManager pipelineManager,
        ILocalSettingsService settings,
        EyeV2CalibrationStore store,
        ILogger<EyeV2Manager> logger,
        Action<IEyeStateMapper?>? mapperSetter = null)
    {
        _settings = settings;
        _store = store;
        _logger = logger;
        _setMapper = mapperSetter ?? pipelineManager.SetMapper;

        var requested = settings.ReadSetting(ModeSetting, EyeTrackingMode.DefaultBaballonia);
        if (requested == EyeTrackingMode.ExperimentalV2)
            TrySetMode(EyeTrackingMode.ExperimentalV2);
        else
            ApplyDefault(save: false);
    }

    public bool TrySetMode(EyeTrackingMode mode)
    {
        if (mode == EyeTrackingMode.DefaultBaballonia)
        {
            ApplyDefault(save: true);
            return true;
        }

        if (!_store.TryLoad(out var calibration, out var error) || calibration == null)
        {
            _logger.LogWarning("Eye V2 requested but unavailable: {Error}. Falling back to Default.", error);
            ApplyDefault(save: true, status: $"Eye V2 unavailable: {error} Default is active.");
            return false;
        }

        Activate(calibration, saveMode: true);
        return true;
    }

    public void Activate(EyeV2Calibration calibration, bool saveMode = true)
    {
        if (!calibration.IsValid())
        {
            ApplyDefault(save: saveMode, status: "Eye V2 calibration was invalid; Default is active.");
            throw new InvalidOperationException("Eye V2 calibration is invalid.");
        }

        try
        {
            var mapper = new EyeV2Mapper(calibration, UpdateDiagnostics);
            _setMapper(mapper);
            Calibration = calibration;
            Mode = EyeTrackingMode.ExperimentalV2;
            Status = "Experimental Eye V2 is active.";
            if (saveMode) _settings.SaveSetting(ModeSetting, Mode);
            StateChanged?.Invoke();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Eye V2 initialization failed; falling back to Default.");
            ApplyDefault(save: saveMode, status: "Eye V2 initialization failed; Default is active.");
            throw;
        }
    }

    public void ReportStatus(string status)
    {
        Status = status;
        StateChanged?.Invoke();
    }

    private void ApplyDefault(bool save, string? status = null)
    {
        _setMapper(null);
        Mode = EyeTrackingMode.DefaultBaballonia;
        Calibration = null;
        Diagnostics = null;
        Status = status ?? "Default Baballonia eye tracking is active.";
        if (save) _settings.SaveSetting(ModeSetting, Mode);
        StateChanged?.Invoke();
    }

    private void UpdateDiagnostics(EyeV2Diagnostics diagnostics)
    {
        Diagnostics = diagnostics;
        DiagnosticsChanged?.Invoke(diagnostics);
    }
}
