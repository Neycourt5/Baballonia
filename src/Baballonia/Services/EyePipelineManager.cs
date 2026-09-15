using Baballonia.Contracts;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.BlinkGuard;
using Baballonia.Services.Inference.Filters;
using Baballonia.Services.Inference.Models;
using Baballonia.Services.Inference.VideoSources;
using Baballonia.Services.Calibration;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.Services;

/// <summary>
/// This class should be the only place where direct Pipeline modifications happen
/// </summary>
public class EyePipelineManager : ICameraSlotHost
{
    private readonly ILogger<EyePipelineManager> _logger;
    private readonly EyeProcessingPipeline _pipeline;
    private readonly ILocalSettingsService _localSettings;
    private readonly InferenceFactory _inferenceFactory;
    private readonly SingleCameraSourceFactory _singleCameraSourceFactory;

    private string? _currentLeftAddress;
    private string? _currentRightAddress;

    private enum EyeSide { Left, Right, Both }

    private readonly SemaphoreSlim _lifecycle = new(1, 1);
    private int _generation;
    private CameraTarget? _leftTarget;
    private CameraTarget? _rightTarget;
    private volatile CameraState _leftState = CameraState.Stopped;
    private volatile CameraState _rightState = CameraState.Stopped;
    private DateTime? _leftInstalledAtUtc;
    private DateTime? _rightInstalledAtUtc;
    private readonly EyeCameraSlot _leftSlot;
    private readonly EyeCameraSlot _rightSlot;
    private readonly EyeCameraSlot _sharedSlot;
    private readonly IReadOnlyList<IRecoverableCameraSlot> _independentSlots;
    private readonly IReadOnlyList<IRecoverableCameraSlot> _sharedSlots;

    public EyePipelineManager(ILogger<EyePipelineManager> logger, EyeProcessingPipeline pipeline,
        ILocalSettingsService localSettings, InferenceFactory inferenceFactory,
        SingleCameraSourceFactory singleCameraSourceFactory)
    {
        _logger = logger;
        _pipeline = pipeline;
        _localSettings = localSettings;
        _inferenceFactory = inferenceFactory;
        _singleCameraSourceFactory = singleCameraSourceFactory;
        _leftSlot = new EyeCameraSlot(this, EyeSide.Left, "Left eye");
        _rightSlot = new EyeCameraSlot(this, EyeSide.Right, "Right eye");
        _sharedSlot = new EyeCameraSlot(this, EyeSide.Both, "Eyes");
        _independentSlots = [_leftSlot, _rightSlot];
        _sharedSlots = [_sharedSlot];

        InitializePipeline();
    }

    public IRecoverableCameraSlot LeftSlot => _leftSlot;
    public IRecoverableCameraSlot RightSlot => _rightSlot;

    public IReadOnlyList<IRecoverableCameraSlot> Slots =>
        HasSharedTarget ? _sharedSlots : _independentSlots;

    public bool IsTotalOutage => AllTargetedSlotsReconnecting(Slots);

    public static bool AllTargetedSlotsReconnecting(
        IReadOnlyList<IRecoverableCameraSlot> slots)
    {
        var targeted = false;
        foreach (var slot in slots)
        {
            if (slot.Target is null)
                continue;
            targeted = true;
            if (slot.State != CameraState.Reconnecting)
                return false;
        }
        return targeted;
    }

    private bool HasSharedTarget =>
        _leftTarget is { } left && _rightTarget is { } right &&
        string.Equals(left.Address, right.Address, StringComparison.OrdinalIgnoreCase);

    private sealed class EyeCameraSlot(EyePipelineManager owner, EyeSide side, string name)
        : IRecoverableCameraSlot
    {
        public string Name => name;
        public CameraTarget? Target => owner.GetTarget(side);
        public CameraState State => owner.GetState(side);
        public event Action<CameraState>? StateChanged
        {
            add => owner.AddStateHandler(side, value);
            remove => owner.RemoveStateHandler(side, value);
        }
        public DateTime? SourceInstalledAtUtc => owner.GetInstalledAtUtc(side);
        public TimeSpan? TimeSinceLastFrame => owner.GetTimeSinceLastFrame(side);
        public bool IsTargetPresent(string address) => owner._singleCameraSourceFactory.IsDevicePresent(address);
        public Task<bool> RecoverAsync(TimeSpan firstFrameTimeout, CancellationToken cancellationToken) =>
            owner.RecoverSlotAsync(side, firstFrameTimeout, cancellationToken);
    }

    private event Action<CameraState>? LeftStateChanged;
    private event Action<CameraState>? RightStateChanged;

    private void AddStateHandler(EyeSide side, Action<CameraState>? handler)
    {
        if (side is EyeSide.Left or EyeSide.Both) LeftStateChanged += handler;
        if (side is EyeSide.Right or EyeSide.Both) RightStateChanged += handler;
    }

    private void RemoveStateHandler(EyeSide side, Action<CameraState>? handler)
    {
        if (side is EyeSide.Left or EyeSide.Both) LeftStateChanged -= handler;
        if (side is EyeSide.Right or EyeSide.Both) RightStateChanged -= handler;
    }

    private CameraTarget? GetTarget(EyeSide side) => side switch
    {
        EyeSide.Left => _leftTarget,
        EyeSide.Right => _rightTarget,
        _ => HasSharedTarget ? _leftTarget : null,
    };

    private CameraState GetState(EyeSide side) => side switch
    {
        EyeSide.Left => _leftState,
        EyeSide.Right => _rightState,
        _ when _leftState == _rightState => _leftState,
        _ when _leftState == CameraState.Reconnecting || _rightState == CameraState.Reconnecting =>
            CameraState.Reconnecting,
        _ when _leftState == CameraState.Starting || _rightState == CameraState.Starting =>
            CameraState.Starting,
        _ => CameraState.Stopped,
    };

    private DateTime? GetInstalledAtUtc(EyeSide side) => side switch
    {
        EyeSide.Left => _leftInstalledAtUtc,
        EyeSide.Right => _rightInstalledAtUtc,
        _ => _leftInstalledAtUtc ?? _rightInstalledAtUtc,
    };

    private TimeSpan? GetTimeSinceLastFrame(EyeSide side)
    {
        lock (_pipeline.SyncRoot)
        {
            return (_pipeline.VideoSource, side) switch
            {
                (SingleCameraSource single, _) => single.TimeSinceLastHealthyFrame,
                (DualCameraSource { LeftCam: SingleCameraSource left }, EyeSide.Left) =>
                    left.TimeSinceLastHealthyFrame,
                (DualCameraSource { RightCam: SingleCameraSource right }, EyeSide.Right) =>
                    right.TimeSinceLastHealthyFrame,
                _ => null,
            };
        }
    }

    private void SetState(EyeSide side, CameraState state)
    {
        if ((side is EyeSide.Left or EyeSide.Both) && _leftState != state)
        {
            _leftState = state;
            LeftStateChanged?.Invoke(state);
        }
        if ((side is EyeSide.Right or EyeSide.Both) && _rightState != state)
        {
            _rightState = state;
            RightStateChanged?.Invoke(state);
        }
    }


    public void InitializePipeline()
    {
        _pipeline.ImageConverter = new MatToFloatTensorConverter();
        var dualTransformer = new DualImageTransformer();
        dualTransformer.LeftTransformer.TargetSize = new Size(128, 128);
        dualTransformer.RightTransformer.TargetSize = new Size(128, 128);
        _pipeline.ImageTransformer = dualTransformer;

        _ = LoadInferenceAsync();
        LoadFilter();
        LoadEyeStabilization();
        LoadEyePostProcessor();
        LoadBlinkGuard();
        LoadSplitEyeSwap();
    }

    /// <summary>
    /// Raised after the eye model is swapped. The personalized corrector listens so it can re-check
    /// that its base model is still the one it was trained against — retraining in VR replaces the
    /// model underneath it, and the learned corrections would then be noise applied to real
    /// tracking.
    /// </summary>
    public event Action? InferenceReloaded;

    /// <summary>
    /// The eye model file actually in use, resolved exactly as <see cref="CreateInference"/>
    /// resolves it. Absolute settings values (how the tuned model is configured) pass through
    /// unchanged.
    /// </summary>
    public string ResolveActiveEyeModelPath()
    {
        var name = _localSettings.ReadSetting<string>("EyeHome_EyeModel", DefaultEyeModelName);
        if (string.IsNullOrWhiteSpace(name))
            name = DefaultEyeModelName;

        var path = Path.Combine(AppContext.BaseDirectory, name);
        return File.Exists(path) ? path : Path.Combine(AppContext.BaseDirectory, DefaultEyeModelName);
    }

    /// <summary>
    /// Installs (or removes) the personalized eye correction stage. The pipeline's own lock makes
    /// the swap wait for any in-flight frame, so the caller may dispose the outgoing corrector once
    /// this returns.
    /// </summary>
    public void SetCorrector(IExpressionCorrector? corrector)
    {
        _pipeline.Corrector = corrector;
    }

    public async Task LoadInferenceAsync()
    {
        var inf = await Task.Run(CreateInference);
        lock (_pipeline.SyncRoot)
            _pipeline.InferenceService = inf;
        InferenceReloaded?.Invoke();
    }

    private const string DefaultEyeModelName = "eyeModel.onnx";

    private DefaultInferenceRunner CreateInference()
    {
        const string defaultEyeModelName = DefaultEyeModelName;
        var eyeModelName = _localSettings.ReadSetting<string>("EyeHome_EyeModel", defaultEyeModelName);
        var eyeModelPath = Path.Combine(AppContext.BaseDirectory, eyeModelName);

        if (File.Exists(eyeModelPath)) return _inferenceFactory.Create(eyeModelPath);
        _logger.LogError("{} Does not exists, Loading default...", eyeModelPath);

        eyeModelName = defaultEyeModelName;
        eyeModelPath = Path.Combine(AppContext.BaseDirectory, eyeModelName);

        return _inferenceFactory.Create(eyeModelPath);
    }


    public void LoadInference()
    {
        var inf = CreateInference();
        lock (_pipeline.SyncRoot)
            _pipeline.InferenceService = inf;
        InferenceReloaded?.Invoke();
    }

    public void LoadFilter()
    {
        var enabled = _localSettings.ReadSetting<bool>("AppSettings_OneEuroEnabled");
        var cutoff = _localSettings.ReadSetting<float>("AppSettings_OneEuroMinFreqCutoff");
        var speedCutoff = _localSettings.ReadSetting<float>("AppSettings_OneEuroSpeedCutoff");

        lock (_pipeline.SyncRoot)
        {
            _pipeline.Filter = enabled
                ? new OneEuroFilter(minCutoff: cutoff, beta: speedCutoff,
                    pairedKeys: OneEuroFilter.EyeGazePairs)
                : null;
        }
    }

    public void LoadEyeStabilization()
    {
        var stabilizeEyes = _localSettings.ReadSetting<bool>("AppSettings_StabilizeEyes", true);
        _pipeline.StabilizeEyes = stabilizeEyes;
        // Default 0 keeps the long-standing vergence behaviour exactly; raise it if the eyes still
        // read as crossing or wandering apart.
        _pipeline.GazeConjugateAmount = Math.Clamp(
            _localSettings.ReadSetting<float>("AppSettings_EyeGazeConjugateAmount", 0f), 0f, 1f);
    }

    /// <summary>
    /// BlinkGuard's live instance, shared with the settings screen and the diagnostics panel.
    /// </summary>
    /// <remarks>
    /// Always present so the UI has counters and state to show; whether it does anything is
    /// <see cref="BlinkGuardSettings.Enabled"/>, which defaults off. A new feature that changes live
    /// gaze should be opted into, not discovered.
    /// </remarks>
    public BlinkGuardFilter BlinkGuard { get; } = new(
        BlinkGuardSettings.Balanced with { Enabled = false });

    /// <summary>Re-reads BlinkGuard's settings and pushes them into the running filter.</summary>
    public void LoadBlinkGuard()
    {
        var enabled = _localSettings.ReadSetting("AppSettings_BlinkGuardEnabled", false);
        var presetName = _localSettings.ReadSetting("AppSettings_BlinkGuardPreset", nameof(BlinkGuardPreset.Balanced));
        var preset = Enum.TryParse<BlinkGuardPreset>(presetName, ignoreCase: true, out var parsed)
            ? parsed
            : BlinkGuardPreset.Balanced;

        var settings = BlinkGuardSettings.ForPreset(preset) with { Enabled = enabled };

        // Advanced values override the preset, so tuning one number does not silently discard the
        // rest of the preset it started from.
        if (preset == BlinkGuardPreset.Custom)
        {
            settings = settings with
            {
                ClosedThreshold = _localSettings.ReadSetting("AppSettings_BlinkGuardClosedThreshold", 0.60f),
                ReopenThreshold = _localSettings.ReadSetting("AppSettings_BlinkGuardReopenThreshold", 0.30f),
                MinimumHoldSeconds = _localSettings.ReadSetting("AppSettings_BlinkGuardMinimumHoldMs", 30f) / 1000f,
                RequiredStableSamples = _localSettings.ReadSetting("AppSettings_BlinkGuardStableSamples", 3),
                StabilityToleranceDegrees = _localSettings.ReadSetting("AppSettings_BlinkGuardToleranceDegrees", 5f),
                ReacquisitionTimeoutSeconds = _localSettings.ReadSetting("AppSettings_BlinkGuardTimeoutMs", 175f) / 1000f,
                TransitionSeconds = _localSettings.ReadSetting("AppSettings_BlinkGuardTransitionMs", 80f) / 1000f,
                UseConfidence = _localSettings.ReadSetting("AppSettings_BlinkGuardUseConfidence", false),
                NormalSpikeGuardEnabled = _localSettings.ReadSetting("AppSettings_BlinkGuardNormalSpikeGuard", false),
            };
        }

        BlinkGuard.UpdateSettings(settings);
    }

    public void LoadEyePostProcessor()
    {
        var enabled = _localSettings.ReadSetting<bool>("AppSettings_EyePostProcessing", true);
        // Only ever consulted for a model that emits no widen/squint of its own; the tuned model
        // does, and its channels are always preferred over anything derived from openness.
        var derive = _localSettings.ReadSetting<bool>("AppSettings_EyeDerivedWidenSquint", true);
        var shapeBlinks = _localSettings.ReadSetting<bool>("AppSettings_EyeBlinkShaping", true);
        // The two cameras disagree about lid position in ways that are not information. Coupling
        // them is on by default; a deliberate wink still releases the coupling immediately.
        var lidSync = Math.Clamp(
            _localSettings.ReadSetting<float>("AppSettings_EyeLidSyncAmount", 0.75f), 0f, 1f);
        // Squint and widen are off by default: which eye reads an expression better is personal,
        // so there is no default that is right for everyone.
        var squintSync = Math.Clamp(
            _localSettings.ReadSetting<float>("AppSettings_EyeSquintSyncAmount", 0f), 0f, 1f);
        var widenSync = Math.Clamp(
            _localSettings.ReadSetting<float>("AppSettings_EyeWidenSyncAmount", 0f), 0f, 1f);
        var gazeSync = Math.Clamp(
            _localSettings.ReadSetting<float>("AppSettings_EyeGazeSyncAmount", 0f), 0f, 1f);
        var syncSource = Enum.TryParse<EyeSyncSource>(
            _localSettings.ReadSetting<string>("AppSettings_EyeSyncSource", nameof(EyeSyncSource.Average)),
            ignoreCase: true, out var parsed)
            ? parsed
            : EyeSyncSource.Average;
        var detectWinks = _localSettings.ReadSetting<bool>("AppSettings_EyeWinkDetection", true);
        var left = LoadCalibrationProfile(EyeCalibrationSettings.LeftKey);
        var right = LoadCalibrationProfile(EyeCalibrationSettings.RightKey);
        lock (_pipeline.SyncRoot)
        {
            var post = enabled
                ? new EyeOutputPostProcessor(
                    left, right, derive, shapeBlinks, lidSync, squintSync, widenSync, gazeSync,
                    syncSource, detectWinks)
                : null;

            // The filter instance is kept across reloads so a settings edit does not throw away the
            // blink state mid-blink, and so the diagnostics counters survive tuning.
            if (post is not null)
                post.BlinkGuard = BlinkGuard;

            _pipeline.PostProcessor = post;
        }
    }

    private EyeCalibrationProfile LoadCalibrationProfile(string key)
    {
        var profile = _localSettings.ReadSetting<EyeCalibrationProfile?>(
            key, EyeCalibrationProfile.Default);
        if (profile is not null && profile.Version == EyeCalibrationProfile.CurrentVersion)
            return profile;

        _logger.LogWarning("Ignoring unsupported eye calibration profile in {Key}", key);
        return EyeCalibrationProfile.Default;
    }

    public void LoadSplitEyeSwap()
    {
        // Default unswapped: split devices like BSB2E expect the left/right halves as-is.
        _pipeline.SwapSplitEyes = _localSettings.ReadSetting<bool>("AppSettings_SplitEyeVideoSwap", false);
    }

    public void SetLeftTransformation(CameraSettings cameraSettings)
    {
        lock (_pipeline.SyncRoot)
        {
            if (_pipeline.ImageTransformer is DualImageTransformer dualImageTransformer)
            {
                dualImageTransformer.LeftTransformer.Transformation = cameraSettings;
            }
        }
    }
    public void SetRightTransformation(CameraSettings cameraSettings)
    {
        lock (_pipeline.SyncRoot)
        {
            if (_pipeline.ImageTransformer is DualImageTransformer dualImageTransformer)
            {
                dualImageTransformer.RightTransformer.Transformation = cameraSettings;
            }
        }
    }

    public async Task<bool> StartLeftVideoSource(string cameraAddress, string preferredBackend)
    {
        if (string.IsNullOrEmpty(cameraAddress))
            return false;

        var generation = Interlocked.Increment(ref _generation);
        _leftTarget = new CameraTarget(cameraAddress, preferredBackend);
        SetState(EyeSide.Left, CameraState.Starting);

        await _lifecycle.WaitAsync().ConfigureAwait(false);
        try
        {
            var opened = await StartLeftVideoSourceCore(cameraAddress, preferredBackend, generation)
                .ConfigureAwait(false);
            if (Volatile.Read(ref _generation) == generation && _leftTarget is not null)
            {
                var side = HasSharedTarget && _pipeline.VideoSource is SingleCameraSource
                    ? EyeSide.Both : EyeSide.Left;
                if (opened)
                {
                    var installed = DateTime.UtcNow;
                    if (side is EyeSide.Left or EyeSide.Both) _leftInstalledAtUtc = installed;
                    if (side is EyeSide.Right or EyeSide.Both) _rightInstalledAtUtc = installed;
                }
                SetState(side, opened ? CameraState.Running : CameraState.Reconnecting);
            }
            else if (_leftTarget is not null && _leftState == CameraState.Starting)
            {
                SetState(EyeSide.Left, CameraState.Reconnecting);
            }
            return opened;
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    private async Task<bool> StartLeftVideoSourceCore(
        string cameraAddress, string preferredBackend, int generation)
    {
        if (_pipeline.VideoSource == null)
        {
            SingleCameraSource cam;
            if (string.IsNullOrEmpty(preferredBackend))
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress);
            else
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);

            if (cam == null)
                return false;
            if (DiscardIfSuperseded(cam, generation))
                return false;

            var source = new DualCameraSource();
            source.LeftCam = cam;
            lock (_pipeline.SyncRoot)
            {
                _pipeline.VideoSource = source;
                _pipeline.ResetTemporalHistory();
            }
            _currentLeftAddress = cameraAddress;
            return true;
        }

        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
            if (cameraAddress == _currentRightAddress && _currentRightAddress != null)
            {
                lock (_pipeline.SyncRoot)
                {
                    var shared = dualCameraSource.RightCam;
                    if (shared == null)
                        return false;

                    var replaced = dualCameraSource.LeftCam;
                    dualCameraSource.LeftCam = null;
                    dualCameraSource.RightCam = null;
                    _pipeline.VideoSource = shared;
                    if (!ReferenceEquals(replaced, shared))
                        replaced?.Dispose();
                    dualCameraSource.Dispose();
                    _pipeline.ResetTemporalHistory();
                }
                _currentLeftAddress = cameraAddress;
                return true;
            }
            else
            {
                lock (_pipeline.SyncRoot)
                {
                    if (dualCameraSource.LeftCam != null)
                    {
                        dualCameraSource.LeftCam.Dispose();
                        dualCameraSource.LeftCam = null;
                    }
                    dualCameraSource.InvalidateLeftCache();
                    _pipeline.ResetTemporalHistory();
                }

                var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
                if (cam == null)
                    return false;
                if (DiscardIfSuperseded(cam, generation))
                    return false;
                lock (_pipeline.SyncRoot)
                {
                    dualCameraSource.LeftCam = cam;
                    dualCameraSource.InvalidateLeftCache();
                    _pipeline.ResetTemporalHistory();
                }
                _currentLeftAddress = cameraAddress;
                return true;
            }

        if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
        {
            if (_currentLeftAddress == cameraAddress && _currentLeftAddress != null)
                return true;

            var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
            if (cam == null)
                return false;
            if (DiscardIfSuperseded(cam, generation))
                return false;

            var source = new DualCameraSource();
            source.LeftCam = cam;
            source.RightCam = singleCameraSource;
            lock (_pipeline.SyncRoot)
            {
                _pipeline.VideoSource = source;
                _pipeline.ResetTemporalHistory();
            }

            _currentLeftAddress = cameraAddress;
            return true;
        }

        return true;
    }

    public async Task<bool> StartRightVideoSource(string cameraAddress, string preferredBackend)
    {
        if (string.IsNullOrEmpty(cameraAddress))
            return false;

        var generation = Interlocked.Increment(ref _generation);
        _rightTarget = new CameraTarget(cameraAddress, preferredBackend);
        SetState(EyeSide.Right, CameraState.Starting);

        await _lifecycle.WaitAsync().ConfigureAwait(false);
        try
        {
            var opened = await StartRightVideoSourceCore(cameraAddress, preferredBackend, generation)
                .ConfigureAwait(false);
            if (Volatile.Read(ref _generation) == generation && _rightTarget is not null)
            {
                var side = HasSharedTarget && _pipeline.VideoSource is SingleCameraSource
                    ? EyeSide.Both : EyeSide.Right;
                if (opened)
                {
                    var installed = DateTime.UtcNow;
                    if (side is EyeSide.Left or EyeSide.Both) _leftInstalledAtUtc = installed;
                    if (side is EyeSide.Right or EyeSide.Both) _rightInstalledAtUtc = installed;
                }
                SetState(side, opened ? CameraState.Running : CameraState.Reconnecting);
            }
            else if (_rightTarget is not null && _rightState == CameraState.Starting)
            {
                SetState(EyeSide.Right, CameraState.Reconnecting);
            }
            return opened;
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    private async Task<bool> StartRightVideoSourceCore(
        string cameraAddress, string preferredBackend, int generation)
    {
        if (_pipeline.VideoSource == null)
        {
            SingleCameraSource cam;
            if (string.IsNullOrEmpty(preferredBackend))
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress);
            else
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);

            if (cam == null)
                return false;
            if (DiscardIfSuperseded(cam, generation))
                return false;

            var source = new DualCameraSource();
            source.RightCam = cam;
            lock (_pipeline.SyncRoot)
            {
                _pipeline.VideoSource = source;
                _pipeline.ResetTemporalHistory();
            }
            _currentRightAddress = cameraAddress;
            return true;
        }

        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
            if (cameraAddress == _currentLeftAddress && _currentLeftAddress != null)
            {
                lock (_pipeline.SyncRoot)
                {
                    var shared = dualCameraSource.LeftCam;
                    if (shared == null)
                        return false;

                    var replaced = dualCameraSource.RightCam;
                    dualCameraSource.LeftCam = null;
                    dualCameraSource.RightCam = null;
                    _pipeline.VideoSource = shared;
                    if (!ReferenceEquals(replaced, shared))
                        replaced?.Dispose();
                    dualCameraSource.Dispose();
                    _pipeline.ResetTemporalHistory();
                }
                _currentRightAddress = cameraAddress;
                return true;
            }
            else
            {
                lock (_pipeline.SyncRoot)
                {
                    if (dualCameraSource.RightCam != null)
                    {
                        dualCameraSource.RightCam.Dispose();
                        dualCameraSource.RightCam = null;
                    }
                    dualCameraSource.InvalidateRightCache();
                    _pipeline.ResetTemporalHistory();
                }

                var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
                if (cam == null)
                    return false;
                if (DiscardIfSuperseded(cam, generation))
                    return false;
                lock (_pipeline.SyncRoot)
                {
                    dualCameraSource.RightCam = cam;
                    dualCameraSource.InvalidateRightCache();
                    _pipeline.ResetTemporalHistory();
                }
                _currentRightAddress = cameraAddress;
                return true;
            }

        if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
        {
            if (_currentRightAddress == cameraAddress && _currentRightAddress != null)
                return true;

            var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
            if (cam == null)
                return false;
            if (DiscardIfSuperseded(cam, generation))
                return false;

            var source = new DualCameraSource();
            source.RightCam = cam;
            source.LeftCam = singleCameraSource;
            lock (_pipeline.SyncRoot)
            {
                _pipeline.VideoSource = source;
                _pipeline.ResetTemporalHistory();
            }

            _currentRightAddress = cameraAddress;
            return true;
        }

        return true;
    }

    private bool DiscardIfSuperseded(SingleCameraSource camera, int generation)
    {
        if (Volatile.Read(ref _generation) == generation)
            return false;

        _logger.LogDebug("Discarding an eye camera opened for a superseded request");
        camera.Dispose();
        return true;
    }

    public async Task<bool> TryStartLeftIfNotRunning(string cameraAddress, string preferredBackend)
    {
        switch (_pipeline.VideoSource)
        {
            case SingleCameraSource when _leftState == CameraState.Running &&
                                         string.Equals(_leftTarget?.Address, cameraAddress, StringComparison.OrdinalIgnoreCase):
            case DualCameraSource { LeftCam: not null } when _leftState == CameraState.Running &&
                                                           string.Equals(_leftTarget?.Address, cameraAddress, StringComparison.OrdinalIgnoreCase):
                return true;
            default:
                return await StartLeftVideoSource(cameraAddress, preferredBackend);
        }
    }
    public async Task<bool> TryStartRightIfNotRunning(string cameraAddress, string preferredBackend)
    {
        switch (_pipeline.VideoSource)
        {
            case SingleCameraSource when _rightState == CameraState.Running &&
                                         string.Equals(_rightTarget?.Address, cameraAddress, StringComparison.OrdinalIgnoreCase):
            case DualCameraSource { RightCam: not null } when _rightState == CameraState.Running &&
                                                            string.Equals(_rightTarget?.Address, cameraAddress, StringComparison.OrdinalIgnoreCase):
                return true;
            default:
                return await StartRightVideoSource(cameraAddress, preferredBackend);
        }
    }
    public void StopLeftCamera()
    {
        Interlocked.Increment(ref _generation);
        var shared = HasSharedTarget;
        _leftTarget = null;
        _leftInstalledAtUtc = null;
        _currentLeftAddress = null;
        lock (_pipeline.SyncRoot)
        {
            if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
            {
                dualCameraSource.LeftCam?.Dispose();
                dualCameraSource.LeftCam = null;
                dualCameraSource.InvalidateLeftCache();
                _pipeline.ResetTemporalHistory();
            }

            if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
            {
                singleCameraSource.Dispose();
                _pipeline.VideoSource = null;
                _currentRightAddress = null;
                _rightTarget = null;
                _rightInstalledAtUtc = null;
                _pipeline.ResetTemporalHistory();
            }
        }
        SetState(shared ? EyeSide.Both : EyeSide.Left, CameraState.Stopped);
    }

    public void StopRightCamera()
    {
        Interlocked.Increment(ref _generation);
        var shared = HasSharedTarget;
        _rightTarget = null;
        _rightInstalledAtUtc = null;
        _currentRightAddress = null;
        lock (_pipeline.SyncRoot)
        {
            if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
            {
                dualCameraSource.RightCam?.Dispose();
                dualCameraSource.RightCam = null;
                dualCameraSource.InvalidateRightCache();
                _pipeline.ResetTemporalHistory();
            }

            if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
            {
                singleCameraSource.Dispose();
                _pipeline.VideoSource = null;
                _currentLeftAddress = null;
                _leftTarget = null;
                _leftInstalledAtUtc = null;
                _pipeline.ResetTemporalHistory();
            }
        }
        SetState(shared ? EyeSide.Both : EyeSide.Right, CameraState.Stopped);
    }

    public void StopAllCameras()
    {
        Interlocked.Increment(ref _generation);
        _leftTarget = null;
        _rightTarget = null;
        _leftInstalledAtUtc = null;
        _rightInstalledAtUtc = null;
        _currentRightAddress = null;
        _currentLeftAddress = null;
        lock (_pipeline.SyncRoot)
        {
            _pipeline.VideoSource?.Dispose();
            _pipeline.VideoSource = null;
            _pipeline.ResetTemporalHistory();
        }
        SetState(EyeSide.Both, CameraState.Stopped);
    }

    /// <summary>Drops failed eye captures while preserving the user's targets for recovery.</summary>
    public void FaultEyeCameras()
    {
        Interlocked.Increment(ref _generation);
        lock (_pipeline.SyncRoot)
        {
            _pipeline.VideoSource?.Dispose();
            _pipeline.VideoSource = null;
            _pipeline.ResetTemporalHistory();
        }
        _leftInstalledAtUtc = null;
        _rightInstalledAtUtc = null;
        SetState(EyeSide.Left,
            _leftTarget is null ? CameraState.Stopped : CameraState.Reconnecting);
        SetState(EyeSide.Right,
            _rightTarget is null ? CameraState.Stopped : CameraState.Reconnecting);
    }

    private async Task<bool> RecoverSlotAsync(
        EyeSide side,
        TimeSpan firstFrameTimeout,
        CancellationToken cancellationToken)
    {
        var target = GetTarget(side);
        if (target is null)
            return false;
        var generation = Volatile.Read(ref _generation);

        await _lifecycle.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!IsRecoveryCurrent(side, target, generation))
                return false;

            SetState(side, CameraState.Reconnecting);
            TearDownSlot(side);

            if (!_singleCameraSourceFactory.IsDevicePresent(target.Address))
                return false;

            SingleCameraSource? camera;
            try
            {
                camera = string.IsNullOrEmpty(target.Backend)
                    ? await _singleCameraSourceFactory.CreateStart(
                        target.Address, firstFrameTimeout, cancellationToken, quiet: true).ConfigureAwait(false)
                    : await _singleCameraSourceFactory.CreateStart(
                        target.Address, target.Backend, firstFrameTimeout, cancellationToken, quiet: true)
                        .ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                return false;
            }

            if (camera is null)
                return false;
            if (!IsRecoveryCurrent(side, target, generation))
            {
                camera.Dispose();
                return false;
            }

            InstallRecoveredSlot(side, target, camera);
            SetState(side, CameraState.Running);
            return true;
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    private bool IsRecoveryCurrent(EyeSide side, CameraTarget target, int generation)
    {
        if (Volatile.Read(ref _generation) != generation)
            return false;

        bool Matches(CameraTarget? candidate) => candidate is not null &&
            string.Equals(candidate.Address, target.Address, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(candidate.Backend, target.Backend, StringComparison.Ordinal);

        return side switch
        {
            EyeSide.Left => Matches(_leftTarget),
            EyeSide.Right => Matches(_rightTarget),
            _ => Matches(_leftTarget) && Matches(_rightTarget),
        };
    }

    private void TearDownSlot(EyeSide side)
    {
        lock (_pipeline.SyncRoot)
        {
            if (side == EyeSide.Both || _pipeline.VideoSource is SingleCameraSource)
            {
                _pipeline.VideoSource?.Dispose();
                _pipeline.VideoSource = null;
                _leftInstalledAtUtc = null;
                _rightInstalledAtUtc = null;
            }
            else if (_pipeline.VideoSource is DualCameraSource dual)
            {
                if (side == EyeSide.Left)
                {
                    var camera = dual.LeftCam;
                    dual.LeftCam = null;
                    if (!ReferenceEquals(camera, dual.RightCam)) camera?.Dispose();
                    dual.InvalidateLeftCache();
                    _leftInstalledAtUtc = null;
                }
                else
                {
                    var camera = dual.RightCam;
                    dual.RightCam = null;
                    if (!ReferenceEquals(camera, dual.LeftCam)) camera?.Dispose();
                    dual.InvalidateRightCache();
                    _rightInstalledAtUtc = null;
                }
            }
            _pipeline.ResetTemporalHistory();
        }
    }

    private void InstallRecoveredSlot(EyeSide side, CameraTarget target, SingleCameraSource camera)
    {
        lock (_pipeline.SyncRoot)
        {
            if (side == EyeSide.Both)
            {
                _pipeline.VideoSource?.Dispose();
                _pipeline.VideoSource = camera;
                _currentLeftAddress = target.Address;
                _currentRightAddress = target.Address;
                var installed = DateTime.UtcNow;
                _leftInstalledAtUtc = installed;
                _rightInstalledAtUtc = installed;
            }
            else
            {
                DualCameraSource dual;
                if (_pipeline.VideoSource is DualCameraSource current)
                {
                    dual = current;
                }
                else
                {
                    dual = new DualCameraSource();
                    if (_pipeline.VideoSource is SingleCameraSource existing)
                    {
                        if (side == EyeSide.Left) dual.RightCam = existing;
                        else dual.LeftCam = existing;
                    }
                    _pipeline.VideoSource = dual;
                }

                if (side == EyeSide.Left)
                {
                    dual.LeftCam = camera;
                    dual.InvalidateLeftCache();
                    _currentLeftAddress = target.Address;
                    _leftInstalledAtUtc = DateTime.UtcNow;
                }
                else
                {
                    dual.RightCam = camera;
                    dual.InvalidateRightCache();
                    _currentRightAddress = target.Address;
                    _rightInstalledAtUtc = DateTime.UtcNow;
                }
            }
            _pipeline.ResetTemporalHistory();
        }
    }

    /// <summary>
    /// Stops only the eye feed(s) backed by a serial camera, releasing the serial handle so the
    /// firmware page can open it. For a DualCameraSource each eye is evaluated independently so a
    /// non-serial eye keeps running. UVC/IP eyes are left untouched.
    /// </summary>
    public bool StopSerialCameras()
    {
        var stopLeft = IsSerialAddress(_leftTarget?.Address);
        var stopRight = IsSerialAddress(_rightTarget?.Address);
        if (!stopLeft || !stopRight)
        {
            lock (_pipeline.SyncRoot)
            {
                switch (_pipeline.VideoSource)
                {
                    case SingleCameraSource single when IsSerialAddress(single.Capture?.Source):
                        stopLeft = true;
                        stopRight = true;
                        break;
                    case DualCameraSource dual:
                        stopLeft |= dual.LeftCam is SingleCameraSource left &&
                            IsSerialAddress(left.Capture?.Source);
                        stopRight |= dual.RightCam is SingleCameraSource right &&
                            IsSerialAddress(right.Capture?.Source);
                        break;
                }
            }
        }
        if (!stopLeft && !stopRight)
            return false;

        Interlocked.Increment(ref _generation);
        if (stopLeft)
        {
            _leftTarget = null;
            _leftInstalledAtUtc = null;
            _currentLeftAddress = null;
        }
        if (stopRight)
        {
            _rightTarget = null;
            _rightInstalledAtUtc = null;
            _currentRightAddress = null;
        }

        lock (_pipeline.SyncRoot)
        {
            switch (_pipeline.VideoSource)
            {
                case SingleCameraSource single:
                    single.Dispose();
                    _pipeline.VideoSource = null;
                    break;
                case DualCameraSource dual:
                    if (stopLeft)
                    {
                        var left = dual.LeftCam;
                        dual.LeftCam = null;
                        if (!ReferenceEquals(left, dual.RightCam)) left?.Dispose();
                        dual.InvalidateLeftCache();
                    }
                    if (stopRight)
                    {
                        var right = dual.RightCam;
                        dual.RightCam = null;
                        if (!ReferenceEquals(right, dual.LeftCam)) right?.Dispose();
                        dual.InvalidateRightCache();
                    }
                    if (dual.LeftCam == null && dual.RightCam == null)
                    {
                        dual.Dispose();
                        _pipeline.VideoSource = null;
                    }
                    break;
            }
            _pipeline.ResetTemporalHistory();
        }
        if (stopLeft) SetState(EyeSide.Left, CameraState.Stopped);
        if (stopRight) SetState(EyeSide.Right, CameraState.Stopped);
        return true;
    }

    // Mirrors SerialCameraCaptureFactory.CanConnect: serial camera addresses are COM* / /dev/tty* / /dev/cu*.
    private static bool IsSerialAddress(string? address)
    {
        if (string.IsNullOrEmpty(address)) return false;
        var a = address.ToLowerInvariant();
        return a.StartsWith("com") || a.StartsWith("/dev/tty") || a.StartsWith("/dev/cu");
    }

    public bool IsUsingSameCamera()
    {
        return HasSharedTarget;
    }

    public void SetFilter(IFilter? filter)
    {
        lock (_pipeline.SyncRoot)
            _pipeline.Filter = filter;
    }
}
