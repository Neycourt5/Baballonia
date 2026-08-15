using Baballonia.Contracts;
using Baballonia.Services.Inference.Filters;
using Baballonia.Services.Inference.Models;
using Baballonia.Services.Inference.VideoSources;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Security.Cryptography;
using System.Threading.Tasks;

namespace Baballonia.Services.Inference;

/// <summary>
/// This class should be the only place where direct Pipeline modifications happen
/// </summary>
public class FacePipelineManager
{
    private readonly ILogger<FacePipelineManager> _logger;
    private readonly FaceProcessingPipeline _pipeline;
    private readonly ILocalSettingsService _localSettings;
    private readonly InferenceFactory _inferenceFactory;
    private readonly SingleCameraSourceFactory _singleCameraSourceFactory;

    public FacePipelineManager(ILogger<FacePipelineManager> logger, FaceProcessingPipeline pipeline,
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
        _pipeline.ImageTransformer = new ImageTransformer();

        _ = LoadInferenceAsync();
        LoadFilter();
    }

    public async Task LoadInferenceAsync()
    {
        var inf = await Task.Run(CreateInference);
        _pipeline.InferenceService = inf;
    }

    public void LoadInference()
    {
        _pipeline.InferenceService = CreateInference();
    }

    /// <summary>
    /// True when the loaded face model also emits the stock visual embedding that model C needs.
    /// </summary>
    /// <remarks>
    /// Read by <see cref="PersonalModelManager"/> before accepting an embedding-based adapter. An
    /// adapter that needs features from a runner that is not producing them would run as a slow
    /// passthrough while appearing to work, so it is refused at load instead.
    /// </remarks>
    public bool EmbeddingAvailable => _pipeline.InferenceService is IEmbeddingSource;
    public string InferenceProvider =>
        (_pipeline.InferenceService as DefaultInferenceRunner)?.ExecutionProvider ?? "Unknown";

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

        if (!enabled)
            return;

        var faceArray = new float[Utils.FaceRawExpressions];
        var faceFilter = new OneEuroFilter(
            faceArray,
            minCutoff: cutoff,
            beta: speedCutoff
        );

        _pipeline.Filter = faceFilter;
    }

    public void StopCamera()
    {
        _pipeline.VideoSource?.Dispose();
        _pipeline.VideoSource = null;
    }

    public void SetVideoSource(IVideoSource videoSource)
    {
        _pipeline.VideoSource = videoSource;
    }

    public void SetTransformation(CameraSettings cameraSettings)
    {
        if (_pipeline.ImageTransformer is ImageTransformer dualImageTransformer)
        {
            dualImageTransformer.Transformation = cameraSettings;
        }
    }

    public async Task<bool> StartVideoSource(string cameraAddress, string preferredBackend)
    {
        if (string.IsNullOrEmpty(cameraAddress))
            return false;

        if (_pipeline.VideoSource != null)
        {
            _pipeline.VideoSource.Dispose();
            _pipeline.VideoSource = null;
        }

        SingleCameraSource cam;
        if (string.IsNullOrEmpty(preferredBackend))
            cam = await _singleCameraSourceFactory.CreateStart(cameraAddress);
        else
            cam = await _singleCameraSourceFactory.CreateStart(cameraAddress, preferredBackend);

        if (cam == null)
            return false;

        _pipeline.VideoSource = cam;
        return true;
    }

    public async Task<bool> TryStartIfNotRunning(string cameraAddress, string preferredBackend)
    {
        if (_pipeline.VideoSource != null)
            return true;

        return await StartVideoSource(cameraAddress, preferredBackend);
    }

    public void SetFilter(IFilter? filter)
    {
        _pipeline.Filter = filter;
    }

    /// <summary>
    /// Installs (or, with null, removes) the personal correction stage. Passing null must restore
    /// stock behavior immediately - unlike <see cref="LoadFilter"/>, which historically returns
    /// early when disabled and leaves a previously installed filter in place.
    /// </summary>
    public void SetCorrector(IExpressionCorrector? corrector)
    {
        _pipeline.Corrector = corrector;
    }

    public static string GenerateMD5(string filepath)
    {
        using var stream = File.OpenRead(filepath);
        using var md5 = MD5.Create();
        var hash = md5.ComputeHash(stream);
        return BitConverter.ToString(hash).Replace("-", "");
    }
}
