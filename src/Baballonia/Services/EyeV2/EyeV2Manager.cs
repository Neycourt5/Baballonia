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
    private readonly Func<IEyeGeometryExtractor> _geometryFactory;

    public EyeTrackingMode Mode { get; private set; } = EyeTrackingMode.DefaultBaballonia;
    public EyeV2Calibration? Calibration { get; private set; }
    public EyeV2Diagnostics? Diagnostics { get; private set; }
    public string Status { get; private set; } = "Default Baballonia eye tracking is active.";

    public event Action? StateChanged;
    public event Action<EyeV2Diagnostics>? DiagnosticsChanged;
    public event Action<EyeGeometryFrame>? GeometryChanged;

    public EyeV2Manager(
        EyePipelineManager pipelineManager,
        ILocalSettingsService settings,
        EyeV2CalibrationStore store,
        ILogger<EyeV2Manager> logger,
        Action<IEyeStateMapper?>? mapperSetter = null,
        Func<IEyeGeometryExtractor>? geometryFactory = null)
    {
        _settings = settings;
        _store = store;
        _logger = logger;
        _setMapper = mapperSetter ?? pipelineManager.SetMapper;
        _geometryFactory = geometryFactory ?? (() => new ClassicEyeGeometryExtractor());

        var requested = settings.ReadSetting(ModeSetting, EyeTrackingMode.DefaultBaballonia);
        if (requested is EyeTrackingMode.ExperimentalV2 or EyeTrackingMode.GeometryHybridV2B)
            TrySetMode(requested);
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

        try
        {
            ActivateMode(calibration, mode, saveMode: true);
            return true;
        }
        catch when (mode == EyeTrackingMode.GeometryHybridV2B)
        {
            // A geometry implementation/load failure has a better fallback than Default: the
            // already-validated V2-A mapper using the same personal calibration.
            ActivateMode(calibration, EyeTrackingMode.ExperimentalV2, saveMode: true,
                status: "V2-B geometry was unavailable; V2-A is active instead.");
            return false;
        }
    }

    public void Activate(EyeV2Calibration calibration, bool saveMode = true)
    {
        var requested = Mode == EyeTrackingMode.GeometryHybridV2B
            ? EyeTrackingMode.GeometryHybridV2B
            : EyeTrackingMode.ExperimentalV2;
        ActivateMode(calibration, requested, saveMode);
    }

    private void ActivateMode(
        EyeV2Calibration calibration,
        EyeTrackingMode mode,
        bool saveMode,
        string? status = null)
    {
        if (!calibration.IsValid())
        {
            ApplyDefault(save: saveMode, status: "Eye V2 calibration was invalid; Default is active.");
            throw new InvalidOperationException("Eye V2 calibration is invalid.");
        }

        try
        {
            IEyeStateMapper mapper = mode == EyeTrackingMode.GeometryHybridV2B
                ? new EyeV2GeometryMapper(
                    calibration, _geometryFactory(), UpdateDiagnostics, UpdateGeometry)
                : new EyeV2Mapper(calibration, UpdateDiagnostics);
            _setMapper(mapper);
            Calibration = calibration;
            Mode = mode;
            Status = status ?? (mode == EyeTrackingMode.GeometryHybridV2B
                ? "Eye V2-B Geometry Hybrid is active; low-confidence frames fall back to V2-A."
                : "Eye V2-A personal mapping is active.");
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

    private void UpdateGeometry(EyeGeometryFrame geometry) => GeometryChanged?.Invoke(geometry);
}
