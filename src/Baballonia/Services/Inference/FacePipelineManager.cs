using Baballonia.Contracts;
using Baballonia.Services.Inference.Filters;
using Baballonia.Services.Inference.Models;
using Baballonia.Services.Inference.VideoSources;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.Services.Inference;

/// <summary>
/// This class should be the only place where direct Pipeline modifications happen
/// </summary>
public class FacePipelineManager : IRecoverableCameraSlot, ICameraSlotHost
{
    private readonly ILogger<FacePipelineManager> _logger;
    private readonly FaceProcessingPipeline _pipeline;
    private readonly ILocalSettingsService _localSettings;
    private readonly InferenceFactory _inferenceFactory;
    private readonly SingleCameraSourceFactory _singleCameraSourceFactory;
    private readonly IReadOnlyList<IRecoverableCameraSlot> _slots;
    private Task _initialInferenceLoad = Task.CompletedTask;

    public FacePipelineManager(ILogger<FacePipelineManager> logger, FaceProcessingPipeline pipeline,
        ILocalSettingsService localSettings, InferenceFactory inferenceFactory,
        SingleCameraSourceFactory singleCameraSourceFactory)
    {
        _logger = logger;
        _pipeline = pipeline;
        _localSettings = localSettings;
        _inferenceFactory = inferenceFactory;
        _singleCameraSourceFactory = singleCameraSourceFactory;
        _slots = [this];

        InitializePipeline();
    }

    public void InitializePipeline()
    {
        _pipeline.ImageConverter = new MatToFloatTensorConverter();
        _pipeline.ImageTransformer = new ImageTransformer();

        // Keep the task so startup services can order dependent model loading after the stock
        // runner is actually installed. Fire-and-forget made embedding adapters race this load.
        _initialInferenceLoad = LoadInferenceAsync();
        LoadFilter();
    }

    /// <summary>
    /// Completes when the constructor-triggered stock (or embedding-enabled) runner is installed.
    /// A personal adapter that consumes that runner's embedding must not validate before this task.
    /// </summary>
    public Task InitialInferenceLoad => _initialInferenceLoad;

    public async Task LoadInferenceAsync()
    {
        var inf = await Task.Run(CreateInference);
        DisposeReplacedInference(_pipeline.SwapInferenceService(inf));
    }

    public void LoadInference()
    {
        DisposeReplacedInference(_pipeline.SwapInferenceService(CreateInference()));
    }

    private void DisposeReplacedInference(IInferenceRunner? previous)
    {
        try
        {
            (previous as IDisposable)?.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not dispose the replaced face inference session");
        }
    }

    /// <summary>
    /// True when the loaded face model also emits the stock visual embedding that model C needs.
    /// </summary>
    /// <remarks>
    /// Read by <see cref="PersonalModelManager"/> before accepting an embedding-based adapter. An
    /// adapter that needs features from a runner that is not producing them would run as a slow
    /// passthrough while appearing to work, so it is refused at load instead.
    /// </remarks>
    public bool EmbeddingAvailable =>
        (_pipeline.InferenceService as DefaultInferenceRunner)?.GetEmbedding() != null;

    public string InferenceProvider =>
        (_pipeline.InferenceService as DefaultInferenceRunner)?.ExecutionProvider ?? "Unknown";

    /// <summary>
    /// Installs (or removes) the personal correction stage. The pipeline's own lock makes the swap
    /// wait for any in-flight frame, so the caller may dispose the outgoing corrector on return.
    /// </summary>
    public void SetCorrector(IExpressionCorrector? corrector)
    {
        _pipeline.Corrector = corrector;
    }

    public DefaultInferenceRunner CreateInference()
    {
        const string defaultFaceModel = "faceModel.onnx";
        var stockPath = Path.Combine(AppContext.BaseDirectory, defaultFaceModel);

        // Opt-in and off by default. When enabled, this swaps in a derived copy of the stock graph
        // that exposes the 1280-d visual embedding as a second output. The weights are identical and
        // the expression output is bit-identical, so the only risk is loading a derived model built
        // from a *different* stock file - which the store's MD5 check refuses outright, because an
        // embedding from another network would produce confident nonsense rather than an error.
        if (_localSettings.ReadSetting<bool>(PersonalModelManager.EmbeddingRunnerSetting))
        {
            var validation = EmbeddingModelStore.TryGetValid(stockPath);
            if (validation is { Valid: true, Path: not null })
            {
                try
                {
                    var runner = _inferenceFactory.Create(validation.Path,
                        secondaryOutputName: EmbeddingModelStore.EmbeddingOutputName);
                    _logger.LogInformation("Face model loaded with visual embedding output");
                    return runner;
                }
                catch (Exception ex)
                {
                    // Any failure here falls through to stock: personalization is optional, tracking
                    // is not.
                    _logger.LogWarning(ex, "Could not load the embedding model; using stock");
                }
            }
            else
            {
                _logger.LogWarning("Embedding runner requested but unavailable: {Reason}",
                    validation.Message);
            }
        }

        return _inferenceFactory.Create(stockPath);
    }

    public void LoadFilter()
    {
        var enabled = _localSettings.ReadSetting<bool>("AppSettings_OneEuroEnabled");
        var cutoff = _localSettings.ReadSetting<float>("AppSettings_OneEuroMinFreqCutoff");
        var speedCutoff = _localSettings.ReadSetting<float>("AppSettings_OneEuroSpeedCutoff");

        lock (_pipeline.SyncRoot)
        {
            _pipeline.Filter = enabled
                ? new OneEuroFilter(minCutoff: cutoff, beta: speedCutoff)
                : null;
        }
    }

    public Personalization.C2.C2CandidateCorrector? SwapC2(Personalization.C2.C2CandidateCorrector? candidate,
        Func<bool>? contextMatches = null) => _pipeline.SwapC2(candidate, contextMatches);

    public CameraSettings? ActiveTransformation => (_pipeline.ImageTransformer as ImageTransformer)?.Transformation;

    // ---- camera lifecycle -----------------------------------------------------------------------

    private readonly SemaphoreSlim _lifecycle = new(1, 1);
    private int _generation;
    private CameraTarget? _target;
    private volatile CameraState _state = CameraState.Stopped;

    public string Name => "Face";
    public CameraTarget? Target => _target;
    public CameraState State => _state;
    public event Action<CameraState>? StateChanged;
    public DateTime? SourceInstalledAtUtc { get; private set; }
    public IReadOnlyList<IRecoverableCameraSlot> Slots => _slots;

    public TimeSpan? TimeSinceLastFrame
    {
        get
        {
            lock (_pipeline.SyncRoot)
                return (_pipeline.VideoSource as SingleCameraSource)?.TimeSinceLastHealthyFrame;
        }
    }

    public bool IsTargetPresent(string address) => _singleCameraSourceFactory.IsDevicePresent(address);

    private void SetState(CameraState state)
    {
        if (_state == state)
            return;
        _state = state;
        StateChanged?.Invoke(state);
    }

    public void FaultCamera()
    {
        Interlocked.Increment(ref _generation);
        lock (_pipeline.SyncRoot)
        {
            _pipeline.VideoSource?.Dispose();
            _pipeline.VideoSource = null;
            SourceInstalledAtUtc = null;
        }
        SetState(_target is null ? CameraState.Stopped : CameraState.Reconnecting);
    }

    public void StopCamera()
    {
        Interlocked.Increment(ref _generation);
        _target = null;
        lock (_pipeline.SyncRoot)
        {
            _pipeline.VideoSource?.Dispose();
            _pipeline.VideoSource = null;
            SourceInstalledAtUtc = null;
        }
        SetState(CameraState.Stopped);
    }

    /// <summary>
    /// Stops the running video source ONLY if it is backed by a serial camera (its capture
    /// address looks like a COM/tty serial port), releasing the exclusive serial handle so the
    /// firmware page can open it. UVC (/dev/videoN) and IP feeds are left running.
    /// </summary>
    public bool StopSerialCameras()
    {
        var isSerial = IsSerialAddress(_target?.Address);
        if (!isSerial)
        {
            lock (_pipeline.SyncRoot)
                isSerial = _pipeline.VideoSource is SingleCameraSource single &&
                    IsSerialAddress(single.Capture?.Source);
        }
        if (!isSerial) return false;

        Interlocked.Increment(ref _generation);
        _target = null;
        lock (_pipeline.SyncRoot)
        {
            _pipeline.VideoSource?.Dispose();
            _pipeline.VideoSource = null;
            SourceInstalledAtUtc = null;
        }
        SetState(CameraState.Stopped);
        return true;
    }

    // Mirrors SerialCameraCaptureFactory.CanConnect (the main project can't reference that type):
    // serial camera addresses are COM* / /dev/tty* / /dev/cu*.
    internal static bool IsSerialAddress(string? address)
    {
        if (string.IsNullOrEmpty(address)) return false;
        var a = address.ToLowerInvariant();
        return a.StartsWith("com") || a.StartsWith("/dev/tty") || a.StartsWith("/dev/cu");
    }

    public void SetVideoSource(IVideoSource videoSource)
    {
        lock (_pipeline.SyncRoot)
        {
            _pipeline.VideoSource = videoSource;
            SourceInstalledAtUtc = DateTime.UtcNow;
        }
    }

    public void SetTransformation(CameraSettings cameraSettings)
    {
        lock (_pipeline.SyncRoot)
        {
            if (_pipeline.ImageTransformer is ImageTransformer dualImageTransformer)
            {
                dualImageTransformer.Transformation = cameraSettings;
            }
        }
    }

    public async Task<bool> StartVideoSource(string cameraAddress, string preferredBackend)
    {
        if (string.IsNullOrEmpty(cameraAddress))
            return false;

        var generation = Interlocked.Increment(ref _generation);
        _target = new CameraTarget(cameraAddress, preferredBackend);

        await _lifecycle.WaitAsync().ConfigureAwait(false);
        try
        {
            SetState(CameraState.Starting);
            var opened = await OpenLocked(
                cameraAddress, preferredBackend, generation,
                SingleCameraSourceFactory.DefaultFirstFrameTimeout,
                CancellationToken.None).ConfigureAwait(false);

            if (Volatile.Read(ref _generation) == generation && _target is not null)
                SetState(opened ? CameraState.Running : CameraState.Reconnecting);
            return opened;
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    private async Task<bool> OpenLocked(
        string cameraAddress,
        string preferredBackend,
        int generation,
        TimeSpan firstFrameTimeout,
        CancellationToken cancellationToken,
        bool quiet = false)
    {
        lock (_pipeline.SyncRoot)
        {
            _pipeline.VideoSource?.Dispose();
            _pipeline.VideoSource = null;
            SourceInstalledAtUtc = null;
        }

        SingleCameraSource? cam;
        try
        {
            cam = string.IsNullOrEmpty(preferredBackend)
                ? await _singleCameraSourceFactory.CreateStart(
                    cameraAddress, firstFrameTimeout, cancellationToken, quiet).ConfigureAwait(false)
                : await _singleCameraSourceFactory.CreateStart(
                    cameraAddress, preferredBackend, firstFrameTimeout, cancellationToken, quiet).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return false;
        }

        if (cam == null)
            return false;

        lock (_pipeline.SyncRoot)
        {
            if (Volatile.Read(ref _generation) != generation || _target is null)
            {
                _logger.LogDebug("Discarding a face camera opened for a superseded request");
                cam.Dispose();
                return false;
            }

            _pipeline.VideoSource = cam;
            SourceInstalledAtUtc = DateTime.UtcNow;
        }
        return true;
    }

    public async Task<bool> RecoverAsync(
        TimeSpan firstFrameTimeout,
        CancellationToken cancellationToken)
    {
        var target = _target;
        if (target is null)
            return false;
        var generation = Volatile.Read(ref _generation);

        await _lifecycle.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (Volatile.Read(ref _generation) != generation || _target is null)
                return false;

            SetState(CameraState.Reconnecting);
            lock (_pipeline.SyncRoot)
            {
                _pipeline.VideoSource?.Dispose();
                _pipeline.VideoSource = null;
                SourceInstalledAtUtc = null;
            }

            if (!_singleCameraSourceFactory.IsDevicePresent(target.Address))
                return false;

            var opened = await OpenLocked(
                target.Address, target.Backend, generation, firstFrameTimeout,
                cancellationToken, quiet: true).ConfigureAwait(false);
            if (opened)
                SetState(CameraState.Running);
            return opened;
        }
        catch (OperationCanceledException)
        {
            return false;
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    public async Task<bool> TryStartIfNotRunning(string cameraAddress, string preferredBackend)
    {
        lock (_pipeline.SyncRoot)
        {
            if (_pipeline.VideoSource != null &&
                State == CameraState.Running &&
                string.Equals(_target?.Address, cameraAddress, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return await StartVideoSource(cameraAddress, preferredBackend);
    }

    public void SetFilter(IFilter? filter)
    {
        lock (_pipeline.SyncRoot)
            _pipeline.Filter = filter;
    }

    public static string GenerateMD5(string filepath)
    {
        using var stream = File.OpenRead(filepath);
        using var md5 = MD5.Create();
        var hash = md5.ComputeHash(stream);
        return BitConverter.ToString(hash).Replace("-", "");
    }
}
