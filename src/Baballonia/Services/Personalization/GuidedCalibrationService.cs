using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services.Calibration;
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
    private readonly IVrCalibrationPresenter? _presenter;
    private readonly Func<bool> _cameraFrameReady;

    private GuidedCaptureRoutine? _routine;
    private string? _sessionId;
    private readonly List<(GuidedAttemptIdentity Attempt, string Reason)> _invalidAttempts = [];
    private bool _cancelledFromVr;

    public GuidedCalibrationService(
        ExpressionOverrideService overrideService,
        CueStateSource cueState,
        DatasetRecorderService recorder,
        ILocalSettingsService settings,
        ILogger<GuidedCalibrationService> logger,
        IVrCalibrationPresenter? presenter = null,
        Func<bool>? cameraFrameReady = null)
    {
        _override = overrideService;
        _cueState = cueState;
        _recorder = recorder;
        _settings = settings;
        _logger = logger;
        _presenter = presenter;
        _cameraFrameReady = cameraFrameReady ?? (() => recorder.HasRecentSourceFrame);
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

        if (!_cameraFrameReady())
        {
            return new StartResult(false,
                "No fresh face-camera inference frame has reached the training recorder. Start " +
                "the face camera, wait for the live preview to move, then try again.");
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

        var presenterStarted = false;
        if (_presenter != null)
        {
            var presentation = _presenter.Begin("GUIDED FACE CALIBRATION");
            if (!presentation.Started)
                return new StartResult(false,
                    $"Guided calibration needs the in-headset presenter. {presentation.Message} " +
                    "Start SteamVR, confirm the headset is connected, and try again.");
            if (!_presenter.IsHealthy || !_presenter.IsPresenting)
            {
                _presenter.End();
                return new StartResult(false,
                    $"The in-headset presenter did not become healthy. {_presenter.Status}");
            }
            presenterStarted = true;
        }

        try
        {
            _sessionId = _recorder.StartSession(SessionType.Guided, camera: null, notes: notes);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Guided calibration: could not start recording");
            if (presenterStarted) _presenter?.End();
            return new StartResult(false, "Could not start recording. Is the face camera running?");
        }

        _routine = routine;
        _invalidAttempts.Clear();
        _cancelledFromVr = false;
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

        // Keep cue stamping and avatar control fail-closed together. If the UI was stalled long
        // enough for the sender deadman to lapse, the recorder's CueStateSource has already stopped
        // returning labels. Abort this session instead of silently resuming with an avatar that is
        // back under live tracking, and conservatively reject the interrupted attempt.
        if (routine.Current is not null && !_override.IsCommandHealthy)
        {
            InvalidateCurrentAttempt(routine, "override-deadman-lapsed");
            _cancelledFromVr = true;
            return false;
        }

        if (!PresenterIsHealthy())
            return AbortForPresenterFailure(routine);

        _override.KeepAlive();

        VrCalibrationAction action;
        try
        {
            action = _presenter?.ConsumeAction() ?? VrCalibrationAction.None;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Guided calibration presenter input failed");
            return AbortForPresenterFailure(routine);
        }
        if (!PresenterIsHealthy())
            return AbortForPresenterFailure(routine);

        switch (action)
        {
            case VrCalibrationAction.Cancel:
                InvalidateCurrentAttempt(routine, "cancelled-in-vr");
                _cancelledFromVr = true;
                return false;

            case VrCalibrationAction.Retry when routine.Current is { Phase: "hold" }:
            {
                var invalid = routine.CurrentAttempt;
                var replay = routine.RetryCurrentAttempt();
                if (invalid is { } attempt)
                    AddInvalidAttempt(attempt, "retried-in-vr");
                if (replay != null)
                {
                    if (!PresentCurrent(routine))
                        return AbortForPresenterFailure(routine);
                    _override.PushPhase(replay);
                    _cueState.SetPhase(replay);
                }
                return true;
            }

            case VrCalibrationAction.Skip when routine.Current is { Phase: "hold" }:
            {
                var invalid = routine.CurrentAttempt;
                var next = routine.SkipCurrentAttempt();
                if (invalid is { } attempt)
                    AddInvalidAttempt(attempt, "skipped-in-vr");
                if (next != null)
                {
                    if (!PresentCurrent(routine))
                        return AbortForPresenterFailure(routine);
                    _override.PushPhase(next);
                    _cueState.SetPhase(next);
                }
                return !routine.IsFinished;
            }
        }

        var phase = routine.Tick();
        if (phase is not null)
        {
            // The headset state is synchronously uploaded before the commanded/recorded phase is
            // published, preventing a one-frame "HOLD label while the headset says RELAX" seam.
            if (!PresentCurrent(routine))
                return AbortForPresenterFailure(routine);
            _override.PushPhase(phase);
            _cueState.SetPhase(phase);
        }
        else
        {
            // Presented on every tick on purpose. The presenter discards frames whose content has
            // not changed before it draws anything, so an idle tick costs a comparison rather than
            // a texture upload - and keeping the call unconditional means a dead headset is still
            // discovered on the very next tick instead of one refresh interval later.
            if (!PresentCurrent(routine))
                return AbortForPresenterFailure(routine);
        }

        return !routine.IsFinished;
    }

    /// <summary>Ends the session, releasing the avatar and closing the recording.</summary>
    public async Task<DatasetRecorderService.SessionSummary?> StopAsync()
    {
        var routine = _routine;
        var completed = routine?.IsFinished == true && !_cancelledFromVr;
        if (routine is { IsFinished: false })
            InvalidateCurrentAttempt(routine,
                _cancelledFromVr ? "cancelled-in-vr" : "stopped-early");
        _routine = null;

        // Order matters: stop stamping cues before the avatar is released, so no frame is labelled
        // with a target that is no longer being commanded.
        _cueState.Clear();
        _override.Deactivate();

        if (_sessionId is null)
        {
            _presenter?.End();
            return null;
        }

        var sessionDirectory = PersonalizationPaths.SessionDirectory(_sessionId);
        _sessionId = null;

        try
        {
            // The invalid-attempt verdict is independent of recorder finalization. Persist it from
            // the known session id first, so a later writer/drain failure cannot turn a cancelled or
            // retried hold back into trusted training data.
            if (_invalidAttempts.Count > 0)
            {
                try
                {
                    await WriteQualityOverridesAsync(sessionDirectory).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex,
                        "Guided calibration: could not persist quality overrides for {Directory}",
                        sessionDirectory);
                }
            }

            return await _recorder.StopSessionAsync().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Guided calibration: could not close the recording cleanly");
            return null;
        }
        finally
        {
            _presenter?.End(completed
                ? "Guided face calibration was saved. Review its quality before training."
                : null);
        }
    }

    private bool PresentCurrent(GuidedCaptureRoutine routine)
    {
        if (_presenter == null) return true;
        if (!PresenterIsHealthy() || routine.Current is not { } step) return false;

        var isLeadIn = step.CueId.EndsWith("LeadIn", StringComparison.Ordinal);
        var movingTowardRest = step.Phase == "transition" && step.To.Sum() < step.From.Sum();
        var phase = step.Phase switch
        {
            "hold" => VrCalibrationPhase.Hold,
            "rest" when !isLeadIn => VrCalibrationPhase.Relax,
            "transition" when movingTowardRest => VrCalibrationPhase.Relax,
            _ => VrCalibrationPhase.Preparing,
        };

        try
        {
            _presenter.Present(new VrCalibrationFrame(
                string.IsNullOrWhiteSpace(step.DisplayName) ? step.CueId : step.DisplayName,
                step.Instruction,
                phase,
                routine.Progress,
                routine.CurrentStepProgress,
                phase == VrCalibrationPhase.Preparing ? routine.CurrentStepSecondsRemaining : null,
                Intensity: step.Level > 0 ? step.Level : null,
                Repetition: isLeadIn ? 0 : step.Repetition + 1,
                RepetitionCount: isLeadIn ? 0 : step.RepetitionCount,
                AllowRetry: step.Phase == "hold",
                AllowSkip: step.Phase == "hold",
                AllowCancel: true));
            return PresenterIsHealthy();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Guided calibration presenter update failed");
            return false;
        }
    }

    private bool PresenterIsHealthy() =>
        _presenter == null || (_presenter.IsPresenting && _presenter.IsHealthy);

    private bool AbortForPresenterFailure(GuidedCaptureRoutine routine)
    {
        InvalidateCurrentAttempt(routine, "vr-presenter-failed");
        _cancelledFromVr = true;

        // Do not wait for the UI's asynchronous Stop call: the face pipeline can publish another
        // frame immediately after Tick returns. Clearing supervision and releasing the avatar here
        // makes those interim frames unlabelled and therefore unusable as false guided targets.
        _cueState.Clear();
        _override.Deactivate();
        _logger.LogError("Guided calibration aborted because the headset presenter became unhealthy: {Status}",
            _presenter?.Status ?? "unknown presenter error");
        return false;
    }

    private async Task WriteQualityOverridesAsync(string sessionDirectory)
    {
        var attempts = _invalidAttempts
            .Distinct()
            .Select(item => new
            {
                cue = item.Attempt.CueId,
                rep = item.Attempt.Repetition,
                attempt = item.Attempt.Attempt,
                valid = false,
                reason = item.Reason,
            })
            .ToArray();
        var path = Path.Combine(sessionDirectory, "guided_quality_overrides.json");
        await File.WriteAllTextAsync(path,
            JsonSerializer.Serialize(new { attempts }, PersonalizationPaths.IndentedJson),
            PersonalizationPaths.Utf8NoBom).ConfigureAwait(false);
    }

    private void InvalidateCurrentAttempt(GuidedCaptureRoutine routine, string reason)
    {
        if (routine.CurrentAttempt is { } attempt)
            AddInvalidAttempt(attempt, reason);
    }

    private void AddInvalidAttempt(GuidedAttemptIdentity attempt, string reason)
    {
        // One explicit verdict per attempt keeps the override file deterministic. The earliest
        // reason is the useful one (for example, "retried" before a later session cancellation).
        if (_invalidAttempts.Any(item => item.Attempt == attempt))
            return;

        _invalidAttempts.Add((attempt, reason));
    }

    public void Dispose()
    {
        // Page teardown is a real stop, not just an avatar release: finalizing through StopAsync
        // persists retry/skip/cancel quality overrides before the recorder is closed. Its awaits do
        // not capture the UI context, so this synchronous IDisposable bridge cannot deadlock it.
        if (_routine is not null || _sessionId is not null || _override.IsActive)
        {
            try
            {
                Task.Run(StopAsync).GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Guided calibration teardown could not finalize the recording");
                _cueState.Clear();
                _override.Deactivate();
                _routine = null;
                _presenter?.End();
            }
        }
    }
}
