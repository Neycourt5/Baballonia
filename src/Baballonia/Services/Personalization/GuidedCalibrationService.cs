using System;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Microsoft.Extensions.Logging;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Runs a guided calibration session: drives the avatar, records the camera, stamps the commanded
/// target onto every frame.
///
/// The three parts have to start and stop together or the data is worthless - a recording without
/// cue stamps is an unlabelled session, and cues without a recording are a light show. Owning all
/// three here is what makes that impossible to get wrong.
///
/// Every exit path leads through <see cref="StopAsync"/>: normal completion, user abort, an
/// exception, or the caller simply navigating away. The override service has its own deadman on top
/// of that, so even a crash here releases the avatar within a second rather than freezing its face.
/// </summary>
public sealed class GuidedCalibrationService : IDisposable
{
    /// <summary>
    /// The VRCFT module matches bare OSC addresses (<c>/jawOpen</c>). With a prefix configured, every
    /// commanded value is sent to an address nothing is listening on: the avatar sits still, the
    /// recording fills with frames of a neutral face labelled as expressions, and the resulting
    /// dataset is confidently wrong. Refusing to start is the only safe response, and it is why this
    /// preflight exists rather than a log line.
    /// </summary>
    public const string OscPrefixSetting = "AppSettings_OSCPrefix";

    private readonly ExpressionOverrideService _override;
    private readonly CueStateSource _cueState;
    private readonly DatasetRecorderService _recorder;
    private readonly ILocalSettingsService _settings;
    private readonly ILogger<GuidedCalibrationService> _logger;

    private GuidedCaptureRoutine? _routine;
    private string? _sessionId;

    public GuidedCalibrationService(
        ExpressionOverrideService overrideService,
        CueStateSource cueState,
        DatasetRecorderService recorder,
        ILocalSettingsService settings,
        ILogger<GuidedCalibrationService> logger)
    {
        _override = overrideService;
        _cueState = cueState;
        _recorder = recorder;
        _settings = settings;
        _logger = logger;
    }

    public bool IsRunning => _routine != null;

    /// <summary>Progress, instruction text and step position for the UI.</summary>
    public GuidedProgress Progress => _routine is null
        ? new GuidedProgress(false, 0, "", 0, 0)
        : new GuidedProgress(!_routine.IsFinished, _routine.Progress, _routine.Instruction,
                             _routine.StepIndex + 1, _routine.StepCount);

    public sealed record GuidedProgress(
        bool Active, double Fraction, string Instruction, int Step, int TotalSteps);

    public sealed record StartResult(bool Started, string Message);

    /// <summary>Checks everything that would make a session produce misleading data.</summary>
    public StartResult Preflight()
    {
        if (IsRunning)
            return new StartResult(false, "A calibration session is already running.");

        if (_recorder.IsRecording)
            return new StartResult(false, "Stop the current recording first.");

        var prefix = _settings.ReadSetting<string>(OscPrefixSetting, "");
        if (!string.IsNullOrWhiteSpace(prefix))
        {
            return new StartResult(false,
                $"OSC prefix is set to \"{prefix}\". Guided calibration drives your avatar through " +
                "bare addresses, so with a prefix set the avatar would not move and the recording " +
                "would be labelled wrong. Clear the prefix in Settings and try again.");
        }

        return new StartResult(true, "");
    }

    /// <summary>
    /// Starts a guided session. The caller must then call <see cref="Tick"/> regularly - that is
    /// both what advances the routine and what keeps the override's deadman satisfied.
    /// </summary>
    public StartResult Start(GuidedCaptureRoutine routine, string? notes = null)
    {
        var preflight = Preflight();
        if (!preflight.Started)
            return preflight;

        try
        {
            _sessionId = _recorder.StartSession(SessionType.Guided, camera: null, notes: notes);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Guided calibration: could not start recording");
            return new StartResult(false, "Could not start recording. Is the face camera running?");
        }

        _routine = routine;
        _override.Activate();

        _logger.LogInformation("Guided calibration started: {Session}, {Steps} steps, {Seconds:F0}s",
            _sessionId, routine.StepCount, routine.TotalSeconds);

        return new StartResult(true, "");
    }

    /// <summary>
    /// Advances the routine one tick. Returns true while the session is still running.
    /// </summary>
    /// <remarks>
    /// Calls <see cref="ExpressionOverrideService.KeepAlive"/> unconditionally: if this stops being
    /// called for any reason - the UI thread stalls, the page closes, an exception escapes - the
    /// override lapses on its own and live tracking resumes. That is the safety property, so it must
    /// not be conditional on anything.
    /// </remarks>
    public bool Tick()
    {
        var routine = _routine;
        if (routine is null)
            return false;

        _override.KeepAlive();

        var phase = routine.Tick();
        if (phase is not null)
        {
            _override.PushPhase(phase);
            _cueState.SetPhase(phase);
        }

        return !routine.IsFinished;
    }

    /// <summary>Ends the session, releasing the avatar and closing the recording.</summary>
    public async Task<DatasetRecorderService.SessionSummary?> StopAsync()
    {
        _routine = null;

        // Order matters: stop stamping cues before the avatar is released, so no frame is labelled
        // with a target that is no longer being commanded.
        _cueState.Clear();
        _override.Deactivate();

        if (_sessionId is null)
            return null;

        _sessionId = null;

        try
        {
            return await _recorder.StopSessionAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Guided calibration: could not close the recording cleanly");
            return null;
        }
    }

    public void Dispose()
    {
        // Page teardown mid-session must not leave the avatar frozen mid-expression.
        if (_routine is not null || _override.IsActive)
        {
            _cueState.Clear();
            _override.Deactivate();
            _routine = null;
        }
    }
}
