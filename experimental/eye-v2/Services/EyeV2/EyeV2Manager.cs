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
    private IEyeStateMapper? _activeMapper;

    public EyeTrackingMode Mode { get; private set; } = EyeTrackingMode.DefaultBaballonia;
    public EyeTrackingMode RequestedMode { get; private set; } = EyeTrackingMode.DefaultBaballonia;
    public EyeV2Calibration? Calibration { get; private set; }
    public EyeV2Calibration? StoredCalibration { get; private set; }
    public EyeV2Diagnostics? Diagnostics { get; private set; }
    public string? FallbackReason { get; private set; }
    public string? RuntimeNotice { get; private set; }
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

        if (_store.TryLoad(out var stored, out _) && stored?.IsValid() == true)
            StoredCalibration = stored;

        var requested = settings.ReadSetting(ModeSetting, EyeTrackingMode.DefaultBaballonia);
        if (requested is EyeTrackingMode.ExperimentalV2 or EyeTrackingMode.GeometryHybridV2B)
            TrySetMode(requested);
        else
            ApplyDefault(save: false);
    }

    public bool TrySetMode(EyeTrackingMode mode)
    {
        RequestedMode = mode;
        FallbackReason = null;
        RuntimeNotice = null;
        if (mode == EyeTrackingMode.DefaultBaballonia)
        {
            ApplyDefault(save: true);
            return true;
        }

        if (!_store.TryLoad(out var calibration, out var error) || calibration == null)
        {
            _logger.LogWarning("Eye V2 requested but unavailable: {Error}. Falling back to Default.", error);
            FallbackReason = error;
            ApplyDefault(save: true, preserveRequest: true);
            return false;
        }

        StoredCalibration = calibration;
        try
        {
            ActivateMode(calibration, mode, saveMode: true);
            return true;
        }
        catch when (mode == EyeTrackingMode.GeometryHybridV2B)
        {
            // A geometry implementation/load failure has a better fallback than Default: the
            // already-validated V2-A mapper using the same personal calibration.
            FallbackReason = "V2-B geometry runtime was unavailable.";
            ActivateMode(calibration, EyeTrackingMode.ExperimentalV2, saveMode: true,
                preserveRequest: true);
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
        bool preserveRequest = false)
    {
        if (!calibration.IsValid())
            throw new InvalidOperationException("Eye V2 calibration is invalid.");

        IEyeStateMapper? candidate = null;
        var previousMapper = _activeMapper;
        try
        {
            candidate = mode == EyeTrackingMode.GeometryHybridV2B
                ? new EyeV2GeometryMapper(
                    calibration, _geometryFactory(), UpdateDiagnostics, UpdateGeometry)
                : new EyeV2Mapper(calibration, UpdateDiagnostics);

            // Install first, then persist the selection, but do not publish any manager state until
            // both operations succeed. A settings failure therefore has a concrete mapper to roll
            // back to and observers never see a half-activated calibration.
            _setMapper(candidate);
            if (saveMode) _settings.SaveSetting(ModeSetting, mode);
        }
        catch (Exception ex)
        {
            try
            {
                _setMapper(previousMapper);
            }
            catch (Exception rollbackError)
            {
                _logger.LogCritical(rollbackError,
                    "Eye V2 mapper rollback failed after activation error");
                throw new AggregateException(
                    "Eye V2 activation failed and its previous mapper could not be restored.",
                    ex, rollbackError);
            }

            (candidate as IDisposable)?.Dispose();
            _logger.LogError(ex,
                "Eye V2 activation failed; the previous mapper and manager state were restored.");
            throw;
        }

        _activeMapper = candidate;
        Calibration = calibration;
        StoredCalibration = calibration;
        Mode = mode;
        if (!preserveRequest) RequestedMode = mode;
        FallbackReason = null;
        RuntimeNotice = null;
        Status = BuildStatus();
        (previousMapper as IDisposable)?.Dispose();
        NotifyStateChanged();
    }

    public void ReportStatus(string status)
    {
        RuntimeNotice = status;
        Status = BuildStatus();
        NotifyStateChanged();
    }

    private void ApplyDefault(bool save, bool preserveRequest = false)
    {
        var previousMapper = _activeMapper;
        try
        {
            _setMapper(null);
            if (save) _settings.SaveSetting(ModeSetting, EyeTrackingMode.DefaultBaballonia);
        }
        catch
        {
            _setMapper(previousMapper);
            throw;
        }

        _activeMapper = null;
        Mode = EyeTrackingMode.DefaultBaballonia;
        Calibration = null;
        Diagnostics = null;
        if (!preserveRequest)
        {
            RequestedMode = EyeTrackingMode.DefaultBaballonia;
            FallbackReason = null;
        }
        Status = BuildStatus();
        (previousMapper as IDisposable)?.Dispose();
        NotifyStateChanged();
    }

    private void NotifyStateChanged()
    {
        try
        {
            StateChanged?.Invoke();
        }
        catch (Exception ex)
        {
            // UI/diagnostic listeners are outside the activation transaction. Their failure must
            // not undo an otherwise committed runtime + settings swap.
            _logger.LogError(ex, "An Eye V2 state listener failed");
        }
    }

    private string BuildStatus()
    {
        static string Name(EyeTrackingMode mode) => mode switch
        {
            EyeTrackingMode.ExperimentalV2 => "Eye V2-A — Personal mapping",
            EyeTrackingMode.GeometryHybridV2B => "Eye V2-B — Geometry Hybrid",
            _ => "Default Baballonia",
        };

        var saved = StoredCalibration;
        var lines = new System.Collections.Generic.List<string>
        {
            $"Requested: {Name(RequestedMode)}",
            $"Actually active: {Name(Mode)}",
            saved == null
                ? "Calibration: Not loaded"
                : $"Calibration: Loaded — captured {saved.Capture.CreatedUtc.ToLocalTime():g}",
            $"Gaze mapper: {(Mode == EyeTrackingMode.DefaultBaballonia ? "Inactive (legacy path)" : "Active")}",
            $"Left eye: {(saved?.Left.IsValid() == true ? "Valid" : "Not calibrated")}",
            $"Right eye: {(saved?.Right.IsValid() == true ? "Valid" : "Not calibrated")}",
        };
        if (!string.IsNullOrWhiteSpace(FallbackReason))
            lines.Add($"Fallback: {FallbackReason}");
        if (!string.IsNullOrWhiteSpace(RuntimeNotice))
            lines.Add(RuntimeNotice);
        return string.Join(Environment.NewLine, lines);
    }

    private void UpdateDiagnostics(EyeV2Diagnostics diagnostics)
    {
        Diagnostics = diagnostics;
        DiagnosticsChanged?.Invoke(diagnostics);
    }

    private void UpdateGeometry(EyeGeometryFrame geometry) => GeometryChanged?.Invoke(geometry);
}
