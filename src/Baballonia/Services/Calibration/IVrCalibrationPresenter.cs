using System;

namespace Baballonia.Services.Calibration;

/// <summary>The behavioral state shown by the headset calibration surface.</summary>
public enum VrCalibrationPhase
{
    Preparing,
    Sampling,
    Hold,
    Relax,
    Target,
    Error,
    Complete,
}

/// <summary>An explicit action selected from inside the headset.</summary>
public enum VrCalibrationAction
{
    None,
    Retry,
    Skip,
    Cancel,
}

/// <summary>
/// One immutable presentation state. Calibration services publish these at their own state
/// boundaries; the presenter owns pixels and controller/mouse input, never sample timing.
/// </summary>
public sealed record VrCalibrationFrame(
    string Title,
    string Instruction,
    VrCalibrationPhase Phase,
    double OverallProgress,
    double PhaseProgress = 0,
    double? CountdownSeconds = null,
    float? TargetX = null,
    float? TargetY = null,
    float? Intensity = null,
    int Repetition = 0,
    int RepetitionCount = 0,
    bool AllowRetry = false,
    bool AllowSkip = false,
    bool AllowCancel = true)
{
    public VrCalibrationFrame Clamp() => this with
    {
        OverallProgress = Math.Clamp(OverallProgress, 0, 1),
        PhaseProgress = Math.Clamp(PhaseProgress, 0, 1),
        TargetX = TargetX is null ? null : Math.Clamp(TargetX.Value, -1f, 1f),
        TargetY = TargetY is null ? null : Math.Clamp(TargetY.Value, -1f, 1f),
        Intensity = Intensity is null ? null : Math.Clamp(Intensity.Value, 0f, 1f),
    };
}

public sealed record VrPresenterStartResult(bool Started, string Message)
{
    public static VrPresenterStartResult Success(string message = "VR presenter ready.") => new(true, message);
    public static VrPresenterStartResult Failure(string message) => new(false, message);
}

/// <summary>
/// A reusable, true headset-facing presenter. Implementations may use OpenVR, OpenXR or a test
/// recorder; consumers remain responsible for calibration order and for deciding when samples count.
/// </summary>
public interface IVrCalibrationPresenter : IDisposable
{
    bool IsAvailable { get; }
    bool IsPresenting { get; }
    /// <summary>
    /// True only while the most recently requested frame is known to be visible and input polling
    /// remains usable. Consumers must stop sampling when this becomes false: <see cref="Status"/>
    /// describes the terminal presenter error.
    /// </summary>
    bool IsHealthy => IsPresenting;
    string Status { get; }

    VrPresenterStartResult Begin(string sessionTitle);
    void Present(VrCalibrationFrame frame);
    VrCalibrationAction ConsumeAction();
    void End(string? completionMessage = null);
}

/// <summary>Headless fallback used by tests and platforms without a headset overlay backend.</summary>
public sealed class NullVrCalibrationPresenter : IVrCalibrationPresenter
{
    public bool IsAvailable => false;
    public bool IsPresenting => false;
    public bool IsHealthy => false;
    public string Status => "A true in-headset calibration presenter is unavailable on this platform.";

    public VrPresenterStartResult Begin(string sessionTitle) => VrPresenterStartResult.Failure(Status);
    public void Present(VrCalibrationFrame frame) { }
    public VrCalibrationAction ConsumeAction() => VrCalibrationAction.None;
    public void End(string? completionMessage = null) { }
    public void Dispose() { }
}
