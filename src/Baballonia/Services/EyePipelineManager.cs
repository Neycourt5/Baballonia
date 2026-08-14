using Baballonia.Contracts;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Filters;
using Baballonia.Services.Inference.Models;
using Baballonia.Services.Inference.VideoSources;
using Baballonia.Services.EyeV2;
using Microsoft.Extensions.Logging;
using OpenCvSharp;
using System;
using System.IO;
using System.Threading.Tasks;

namespace Baballonia.Services;

/// <summary>
/// This class should be the only place where direct Pipeline modifications happen
/// </summary>
public class EyePipelineManager
{
    private readonly ILogger<EyePipelineManager> _logger;
    private readonly EyeProcessingPipeline _pipeline;
    private readonly ILocalSettingsService _localSettings;
    private readonly InferenceFactory _inferenceFactory;
    private readonly SingleCameraSourceFactory _singleCameraSourceFactory;

    private string? _currentLeftAddress;
    private string? _currentRightAddress;

    public EyePipelineManager(ILogger<EyePipelineManager> logger, EyeProcessingPipeline pipeline,
        ILocalSettingsService localSettings, InferenceFactory inferenceFactory,
        SingleCameraSourceFactory singleCameraSourceFactory)
    {
        _logger = logger;
        _pipeline = pipeline;
        _localSettings = localSettings;
        _inferenceFactory = inferenceFactory;
        _singleCameraSourceFactory = singleCameraSourceFactory;

        InitializePipeline();
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
    }

    public async Task LoadInferenceAsync()
    {
        var inf = await Task.Run(CreateInference);
        var previous = _pipeline.InferenceService;

        _pipeline.InferenceService = inf;

        // The new model gets a clean temporal stack: frames captured for the old one describe a
        // different input contract, and splicing them across the swap feeds it a history it never
        // saw. Ordering matters - reset after the swap, so no tick can refill the queue from the
        // outgoing runner.
        _pipeline.ResetTemporalState();

        // The outgoing session holds native memory and, on Windows, a file lock on the model.
        // Hot-reloading repeatedly (which the calibration flow does) otherwise accumulates both.
        (previous as IDisposable)?.Dispose();
    }

    private DefaultInferenceRunner CreateInference()
    {
        const string defaultEyeModelName = "eyeModel.onnx";
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
        var previous = _pipeline.InferenceService;

        _pipeline.InferenceService = CreateInference();
        _pipeline.ResetTemporalState();

        (previous as IDisposable)?.Dispose();
    }

    public void LoadFilter()
    {
        var enabled = _localSettings.ReadSetting<bool>("AppSettings_OneEuroEnabled");
        var cutoff = _localSettings.ReadSetting<float>("AppSettings_OneEuroMinFreqCutoff");
        var speedCutoff = _localSettings.ReadSetting<float>("AppSettings_OneEuroSpeedCutoff");

        if (!enabled)
            return;

        var eyeArray = new float[Utils.EyeRawExpressions];
        var eyeFilter = new OneEuroFilter(
            eyeArray,
            minCutoff: cutoff,
            beta: speedCutoff
        );

        _pipeline.Filter = eyeFilter;
    }

    public void LoadEyeStabilization()
    {
        var stabilizeEyes = _localSettings.ReadSetting<bool>("AppSettings_StabilizeEyes", true);
        _pipeline.StabilizeEyes = stabilizeEyes;
    }

    public void SetLeftTransformation(CameraSettings cameraSettings)
    {
        if (_pipeline.ImageTransformer is DualImageTransformer dualImageTransformer)
        {
            dualImageTransformer.LeftTransformer.Transformation = cameraSettings;
        }
    }
    public void SetRightTransformation(CameraSettings cameraSettings)
    {
        if (_pipeline.ImageTransformer is DualImageTransformer dualImageTransformer)
        {
            dualImageTransformer.RightTransformer.Transformation = cameraSettings;
        }
    }

    public async Task<bool> StartLeftVideoSource(string cameraAddress, string preferredBackend)
    {
        if (string.IsNullOrEmpty(cameraAddress))
            return false;

        if (_pipeline.VideoSource == null)
        {
            SingleCameraSource cam;
            if (string.IsNullOrEmpty(preferredBackend))
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress);
            else
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);

            if (cam == null)
                return false;

            var source = new DualCameraSource();
            source.LeftCam = cam;
            _pipeline.VideoSource = source;
            _currentLeftAddress = cameraAddress;
            return true;
        }

        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
            if (cameraAddress == _currentRightAddress && _currentRightAddress != null)
            {
                var tmp = dualCameraSource.RightCam;
                _pipeline.VideoSource = tmp;
                _currentLeftAddress = cameraAddress;
                return true;
            }
            else
            {
                if (dualCameraSource.LeftCam != null)
                {
                    dualCameraSource.LeftCam.Dispose();
                    dualCameraSource.LeftCam = null;
                }

                var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
                if (cam == null)
                    return false;
                dualCameraSource.LeftCam = cam;
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

            var tmp = singleCameraSource;
            _pipeline.VideoSource = null;
            var source = new DualCameraSource();
            source.LeftCam = cam;
            source.RightCam = tmp;

            _currentLeftAddress = cameraAddress;
            return true;
        }

        return true;
    }

    public async Task<bool> StartRightVideoSource(string cameraAddress, string preferredBackend)
    {
        if (string.IsNullOrEmpty(cameraAddress))
            return false;

        if (_pipeline.VideoSource == null)
        {
            SingleCameraSource cam;
            if (string.IsNullOrEmpty(preferredBackend))
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress);
            else
                cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);

            if (cam == null)
                return false;

            var source = new DualCameraSource();
            source.RightCam = cam;
            _pipeline.VideoSource = source;
            _currentRightAddress = cameraAddress;
            return true;
        }

        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
            if (cameraAddress == _currentLeftAddress && _currentLeftAddress != null)
            {
                var tmp = dualCameraSource.LeftCam;
                _pipeline.VideoSource = tmp;
                _currentRightAddress = cameraAddress;
                return true;
            }
            else
            {
                if (dualCameraSource.RightCam != null)
                {
                    dualCameraSource.RightCam.Dispose();
                    dualCameraSource.RightCam = null;
                }

                var cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);
                if (cam == null)
                    return false;
                dualCameraSource.RightCam = cam;
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

            var tmp = singleCameraSource;
            _pipeline.VideoSource = null;
            var source = new DualCameraSource();
            source.RightCam = cam;
            source.LeftCam = tmp;

            _currentRightAddress = cameraAddress;
            return true;
        }

        return true;
    }

    public async Task<bool> TryStartLeftIfNotRunning(string cameraAddress, string preferredBackend)
    {
        switch (_pipeline.VideoSource)
        {
            case SingleCameraSource singleCameraSource:
            case DualCameraSource { LeftCam: not null }:
                return true;
            default:
                return await StartLeftVideoSource(cameraAddress, preferredBackend);
        }
    }
    public async Task<bool> TryStartRightIfNotRunning(string cameraAddress, string preferredBackend)
    {
        switch (_pipeline.VideoSource)
        {
            case SingleCameraSource singleCameraSource:
            case DualCameraSource { RightCam: not null }:
                return true;
            default:
                return await StartRightVideoSource(cameraAddress, preferredBackend);
        }
    }
    public void StopLeftCamera()
    {
        _currentLeftAddress = null;
        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
        {
            dualCameraSource.LeftCam?.Dispose();
            dualCameraSource.LeftCam = null;
        }

        if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
        {
            singleCameraSource.Dispose();
            _pipeline.VideoSource = null;
            _currentRightAddress = null;
        }
    }

    public void StopRightCamera()
    {
        _currentRightAddress = null;
        if (_pipeline.VideoSource is DualCameraSource dualCameraSource)
        {
            dualCameraSource.RightCam?.Dispose();
            dualCameraSource.RightCam = null;
        }

        if (_pipeline.VideoSource is SingleCameraSource singleCameraSource)
        {
            singleCameraSource.Dispose();
            _pipeline.VideoSource = null;
            _currentLeftAddress = null;
        }
    }

    public void StopAllCameras()
    {
        _currentRightAddress = null;
        _currentLeftAddress = null;
        _pipeline.VideoSource?.Dispose();
        _pipeline.VideoSource = null;
    }

    public bool IsUsingSameCamera()
    {
        return _currentLeftAddress == _currentRightAddress && _currentLeftAddress != null;
    }

    public void SetFilter(IFilter? filter)
    {
        _pipeline.Filter = filter;
    }

    /// <summary>Atomically installs, clears or hot-swaps the optional personal eye mapper.</summary>
    public void SetMapper(IEyeStateMapper? mapper)
    {
        mapper?.Reset();
        var previous = _pipeline.Mapper;
        _pipeline.Mapper = mapper;
        (previous as IDisposable)?.Dispose();
    }
}
