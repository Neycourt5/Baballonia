using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Baballonia.Services.Calibration;
using Baballonia.Services.Inference;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Baballonia.Services.Personalization.Eye;

/// <summary>Why a guided eye capture could not start.</summary>
public sealed record EyeCaptureStartResult(bool Started, string Message);

/// <summary>
/// Runs the guided eye routine: drives the dot, stamps cues onto recorded frames, and stops
/// everything together.
/// </summary>
/// <remarks>
/// <para>The three parts have to start and stop as one or the data is worthless — frames recorded
/// while the headset shows a different pose are confidently mislabelled, which is worse than no
/// data at all. So this owns the routine, the presenter and the recorder together, and any failure
/// tears down all three.</para>
///
/// <para>Unlike the face routine there is no avatar override: the dot in the headset is the teacher,
/// so nothing needs to be sent to VRChat and there is no OSC path to fail. That removes the whole
/// deadman-timer layer the face guided capture needs.</para>
///
/// <para><see cref="Tick"/> is pumped by the UI at ~50 ms. It is deliberately not self-driving: a
/// timer that outlived a closed page would keep recording into a session nobody is watching.</para>
/// </remarks>
public sealed class EyeGuidedCaptureService
{
    private readonly EyeDatasetRecorder _recorder;
    private readonly EyePipelineManager _pipelineManager;
    private readonly IVrCalibrationPresenter? _presenter;
    private readonly ILogger _logger;

    private GuidedCaptureRoutine? _routine;
    private readonly List<EyePoseQuality> _attempts = [];
    private readonly Dictionary<GuidedAttemptIdentity, int> _heldFrames = [];
    private GuidedAttemptIdentity? _currentHold;
    private bool _cameraDisturbed;
    private bool _usingHeadset;

    public EyeGuidedCaptureService(
        EyeDatasetRecorder recorder,
        EyePipelineManager pipelineManager,
        IVrCalibrationPresenter? presenter = null,
        ILogger<EyeGuidedCaptureService>? logger = null)
    {
        _recorder = recorder;
        _pipelineManager = pipelineManager;
        _presenter = presenter;
        _logger = logger ?? NullLogger<EyeGuidedCaptureService>.Instance;
    }

    public bool IsRunning => _routine is not null;

    /// <summary>Where the dot is being shown, for the desktop fallback panel to mirror.</summary>
    public VrCalibrationFrame? CurrentFrame { get; private set; }

    public double Progress => _routine?.Progress ?? 0;

    public string Instruction => _routine?.Instruction ?? "Ready.";

    /// <summary>How long the whole routine takes, so the user knows what they are agreeing to.</summary>
    public static double RoutineSeconds => EyeGuidedCues.TotalSeconds();

    /// <summary>
    /// Starts a capture, refusing rather than producing a dataset that cannot be trained on.
    /// </summary>
    public EyeCaptureStartResult Start(bool preferHeadset = true)
    {
        if (IsRunning)
            return new EyeCaptureStartResult(false, "A guided eye capture is already running.");

        if (_recorder.IsRecording)
            return new EyeCaptureStartResult(false, "Something else is already recording.");

        var running = _pipelineManager.Slots
            .Where(slot => slot.Target is not null)
            .ToArray();
        if (running.Length == 0 || running.Any(slot => slot.State != CameraState.Running))
        {
            return new EyeCaptureStartResult(false,
                "Start the eye cameras before recording a guided capture.");
        }

        var modelPath = _pipelineManager.ResolveActiveEyeModelPath();
        if (!File.Exists(modelPath))
            return new EyeCaptureStartResult(false, "The eye model file could not be found.");

        _usingHeadset = preferHeadset && _presenter is { IsAvailable: true };
        if (_usingHeadset)
        {
            var begun = _presenter!.Begin("Eye personalization capture");
            if (!begun.Started)
            {
                // Falling back is better than refusing: the desktop dot still produces usable
                // expression data, and the trainer down-weights its gaze.
                _logger.LogInformation(
                    "Headset presenter unavailable ({Reason}); using the desktop dot", begun.Message);
                _usingHeadset = false;
            }
        }

        var metadata = BuildMetadata(modelPath);
        _attempts.Clear();
        _heldFrames.Clear();
        _currentHold = null;
        _cameraDisturbed = false;

        _recorder.Start(metadata);
        _routine = new GuidedCaptureRoutine(EyeGuidedCues.BuildSteps());

        _logger.LogInformation(
            "Eye capture {Session} started ({Presentation}, {Seconds:F0}s)",
            metadata.sessionId, metadata.presentation, RoutineSeconds);

        return new EyeCaptureStartResult(true,
            $"Recording — about {RoutineSeconds:F0} seconds. Follow the prompts.");
    }

    /// <summary>
    /// Advances the routine. Returns false once it has finished, so the caller can stop pumping.
    /// </summary>
    public bool Tick()
    {
        if (_routine is not { } routine)
            return false;

        // A camera that drops mid-routine invalidates the attempt in progress: the frames after it
        // recovers are not the pose the user was holding.
        if (_pipelineManager.Slots.Any(slot => slot.Target is not null &&
                                               slot.State != CameraState.Running))
        {
            _cameraDisturbed = true;
            InvalidateCurrentHold("camera-interrupted");
        }

        var changed = routine.Tick();
        if (changed is not null)
            OnPhaseChanged(routine, changed);
        else if (routine.Current is { Phase: "hold" } && _currentHold is { } hold)
            _heldFrames[hold] = _recorder.FrameCount;

        Present(routine);
        ConsumePresenterAction();

        return !routine.IsFinished;
    }

    private void OnPhaseChanged(GuidedCaptureRoutine routine, CuePhase phase)
    {
        if (phase.PhaseName != "hold")
            CloseCurrentHold();

        _recorder.CurrentCue = new EyeCueLabel(
            phase.CueId, phase.PhaseName, phase.RepetitionIndex, phase.AttemptIndex);

        if (phase.PhaseName != "hold")
            return;

        _currentHold = new GuidedAttemptIdentity(phase.CueId, phase.RepetitionIndex, phase.AttemptIndex);
        // Frame counts are recorded as a span: start now, end when the hold closes.
        _heldFrames[_currentHold.Value] = _recorder.FrameCount;
        _holdStartedFrames = _recorder.FrameCount;
    }

    private int _holdStartedFrames;

    private void CloseCurrentHold()
    {
        if (_currentHold is not { } hold)
            return;

        var held = Math.Max(0, _recorder.FrameCount - _holdStartedFrames);
        var alreadyInvalid = _attempts.Any(a =>
            a.id == hold.CueId && a.repetition == hold.Repetition && a.attempt == hold.Attempt && !a.valid);

        if (!alreadyInvalid)
        {
            _attempts.Add(new EyePoseQuality(
                hold.CueId, hold.Repetition, hold.Attempt, held,
                valid: held >= MinimumHeldFrames,
                reason: held >= MinimumHeldFrames ? null : "too-few-frames"));
        }

        _currentHold = null;
    }

    /// <summary>
    /// Below this many frames in a hold there is not enough of the pose to learn from. At ~90 Hz a
    /// 2.5 s hold yields roughly 220, so this only trips when something genuinely went wrong.
    /// </summary>
    public const int MinimumHeldFrames = 60;

    private void InvalidateCurrentHold(string reason)
    {
        if (_currentHold is not { } hold)
            return;

        if (_attempts.Any(a => a.id == hold.CueId && a.repetition == hold.Repetition &&
                               a.attempt == hold.Attempt && !a.valid))
            return;

        _attempts.Add(new EyePoseQuality(hold.CueId, hold.Repetition, hold.Attempt,
            Math.Max(0, _recorder.FrameCount - _holdStartedFrames), valid: false, reason: reason));
    }

    private void Present(GuidedCaptureRoutine routine)
    {
        var step = routine.Current;
        if (step is null)
        {
            CurrentFrame = null;
            return;
        }

        var pose = EyeGuidedCues.Poses.FirstOrDefault(p => p.Id == step.CueId);
        var showDot = pose is { DotXDegrees: not null, DotYDegrees: not null } &&
                      step.Phase is "transition" or "hold";

        var frame = new VrCalibrationFrame(
            Title: "Eye personalization",
            Instruction: step.Instruction,
            Phase: step.Phase switch
            {
                // Target suppresses the surrounding chrome: text beside the dot steals the very
                // fixation being recorded.
                "hold" when showDot => VrCalibrationPhase.Target,
                "hold" => VrCalibrationPhase.Hold,
                "transition" => VrCalibrationPhase.Settling,
                "rest" => VrCalibrationPhase.Relax,
                _ => VrCalibrationPhase.Preparing,
            },
            OverallProgress: routine.Progress,
            PhaseProgress: routine.CurrentStepProgress,
            CountdownSeconds: routine.CurrentStepSecondsRemaining,
            TargetX: showDot ? EyeGuidedCues.PanelFromDegrees(pose!.DotXDegrees!.Value) : null,
            TargetY: showDot ? EyeGuidedCues.PanelYFromDegrees(pose!.DotYDegrees!.Value) : null,
            Repetition: step.Repetition + 1,
            RepetitionCount: Math.Max(1, step.RepetitionCount),
            AllowRetry: step.Phase == "hold",
            AllowSkip: step.Phase == "hold",
            AllowCancel: true).Clamp();

        CurrentFrame = frame;

        if (_usingHeadset)
            _presenter?.Present(frame);
    }

    private void ConsumePresenterAction()
    {
        if (!_usingHeadset || _presenter is null || _routine is null)
            return;

        switch (_presenter.ConsumeAction())
        {
            case VrCalibrationAction.Retry:
                Retry();
                break;
            case VrCalibrationAction.Skip:
                Skip();
                break;
            case VrCalibrationAction.Cancel:
                _ = StopAsync(completed: false);
                break;
        }
    }

    /// <summary>Replays the current pose, marking the abandoned attempt invalid.</summary>
    public void Retry()
    {
        if (_routine is not { } routine)
            return;

        InvalidateCurrentHold("retried");
        _currentHold = null;
        var phase = routine.RetryCurrentAttempt();
        if (phase is not null)
            OnPhaseChanged(routine, phase);
    }

    /// <summary>Abandons the current pose and moves on, marking it invalid.</summary>
    public void Skip()
    {
        if (_routine is not { } routine)
            return;

        InvalidateCurrentHold("skipped");
        _currentHold = null;
        var phase = routine.SkipCurrentAttempt();
        if (phase is not null)
            OnPhaseChanged(routine, phase);
    }

    /// <summary>Stops everything together and writes the quality sidecar.</summary>
    public async Task<EyeSessionMetadata?> StopAsync(bool completed)
    {
        if (_routine is null)
            return null;

        if (!completed)
            InvalidateCurrentHold("stopped-early");

        CloseCurrentHold();
        _routine = null;
        CurrentFrame = null;

        if (_usingHeadset)
        {
            _presenter?.End(completed ? "Capture complete." : "Capture cancelled.");
            _usingHeadset = false;
        }

        return await _recorder
            .StopAsync(new EyeSessionQuality(_attempts.ToArray(), _cameraDisturbed, completed))
            .ConfigureAwait(false);
    }

    private EyeSessionMetadata BuildMetadata(string modelPath) => new(
        schemaVersion: EyePersonalizationSchema.Version,
        sessionId: EyeDatasetPaths.NewSessionId(DateTime.UtcNow),
        startedUtc: DateTime.UtcNow.ToString("o"),
        endedUtc: null,
        appVersion: Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "unknown",
        eyeSchemaNames: EyePersonalizationSchema.ExpressionNames.ToArray(),
        eyeSchemaSha256: EyePersonalizationSchema.Sha256,
        gazeRangeDegrees: EyePersonalizationSchema.GazeRangeDegrees,
        baseEyeModelPath: modelPath,
        baseEyeModelMd5: ComputeMd5(modelPath),
        presentation: _usingHeadset ? "headset" : "desktop",
        poses: EyeGuidedCues.Poses.Select(pose => new EyePoseRecord(
            pose.Id, pose.DisplayName, pose.DotXDegrees, pose.DotYDegrees,
            pose.HoldSeconds, pose.Repetitions,
            pose.Supervision
                .Select(c => new EyeChannelSupervisionRecord(c.Dim, c.Target, c.Weight))
                .ToArray())).ToArray(),
        frameCount: 0,
        effectiveFps: 0);

    private static string ComputeMd5(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(MD5.HashData(stream)).ToLowerInvariant();
    }
}
