using Baballonia.Contracts;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Baballonia.Services;

public class DefaultInferenceRunner(ILoggerFactory loggerFactory) :
    IInferenceRunner, INamedInferenceOutput, IEmbeddingSource
{
    public Size InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public string ExecutionProvider { get; private set; } = "Uninitialized";
    public string ModelPath { get; private set; } = "";
    public DenseTensor<float> InputTensor;
    private ILogger _logger;
    private string _inputName;
    private InferenceSession _session;
    private string[] _outputExpressionNames;
    private bool _isOldEyeModel;

    /// <summary>
    /// Names supplied by the model for each primary-output element, when available.
    /// Eye models use these to project extended layouts onto the legacy gaze/lid contract.
    /// </summary>
    public IReadOnlyList<string>? OutputNames => _outputExpressionNames;

    /// <summary>
    /// Name of a second graph output to capture alongside the primary one, or null for the usual
    /// single-output behavior.
    /// </summary>
    /// <remarks>
    /// Set before <see cref="Setup"/>. This exists for the derived face model, which exposes the
    /// stock network's internal 1280-d visual embedding as an extra output so a personal adapter can
    /// reuse it. ONNX Runtime computes every declared output in the one pass it was already making,
    /// so capturing it costs nothing measurable.
    ///
    /// When null, <see cref="Run"/> behaves exactly as it always has. The eye pipeline and the plain
    /// stock face path never set it and are therefore untouched by this.
    /// </remarks>
    public string SecondaryOutputName { get; set; }

    private string _primaryOutputName;
    private DenseTensor<float> _secondaryTensor;
    private bool _secondaryAvailable;


    /// <summary>
    /// Loads/reloads the ONNX model and setups the environment
    /// </summary>
    public void Setup(string modelPath, bool useGpu = true)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"{modelPath} does not exist");

        _logger = loggerFactory.CreateLogger(this.GetType().Name + "." + Path.GetFileName(modelPath));
        ModelPath = Path.GetFullPath(modelPath);

        SessionOptions sessionOptions = SetupSessionOptions();
        if (useGpu)
            ConfigurePlatformSpecificGpu(sessionOptions, modelPath);
        else
        {
            sessionOptions.AppendExecutionProvider_CPU();
            ExecutionProvider = "CPU";
        }

        _session = new InferenceSession(modelPath, sessionOptions);
        _inputName = _session.InputMetadata.Keys.First();
        var dimensions = _session.InputMetadata.Values.First().Dimensions;
        InputSize = new Size(dimensions[2], dimensions[3]);

        InputTensor = new DenseTensor<float>([1, dimensions[1], dimensions[2], dimensions[3]]);

        ConfigureSecondaryOutput();
        InitializeModelMetadata();

        _logger.LogInformation("{} initialization finished", modelPath);
    }

    /// <summary>
    /// Resolves the secondary output, if one was requested and the loaded model has it.
    /// </summary>
    /// <remarks>
    /// A missing secondary output is not an error: the caller falls back to plain stock behavior,
    /// which is exactly what should happen if a derived model was replaced by an ordinary one.
    /// The primary is identified as "the output that is not the secondary" rather than by index, so
    /// a graph that lists them in the other order still returns expressions from <see cref="Run"/>.
    /// </remarks>
    private void ConfigureSecondaryOutput()
    {
        _secondaryAvailable = false;
        _secondaryTensor = null;
        _primaryOutputName = null;

        if (string.IsNullOrEmpty(SecondaryOutputName))
            return;

        if (!_session.OutputMetadata.TryGetValue(SecondaryOutputName, out var metadata))
        {
            _logger.LogWarning(
                "Model has no '{Output}' output; continuing without it", SecondaryOutputName);
            return;
        }

        _primaryOutputName = _session.OutputMetadata.Keys
            .FirstOrDefault(name => name != SecondaryOutputName);

        if (_primaryOutputName == null)
        {
            _logger.LogWarning("Model exposes only '{Output}'; ignoring it", SecondaryOutputName);
            return;
        }

        var dimensions = metadata.Dimensions.Select(d => d > 0 ? d : 1).ToArray();
        _secondaryTensor = new DenseTensor<float>(dimensions);
        _secondaryAvailable = true;

        _logger.LogInformation("Capturing secondary output '{Output}' ({Length} values)",
            SecondaryOutputName, _secondaryTensor.Length);
    }

    /// <inheritdoc />
    public DenseTensor<float> GetEmbedding() => _secondaryAvailable ? _secondaryTensor : null;

    /// <summary>
    /// Reads and caches model metadata once during initialization
    /// </summary>
    private void InitializeModelMetadata()
    {
        _isOldEyeModel = _session.ModelMetadata.CustomMetadataMap.Count() == 0;

        if (!_isOldEyeModel)
        {
            var metadataJson = _session.ModelMetadata.CustomMetadataMap["blendshape_names"];
            _outputExpressionNames = JsonConvert.DeserializeObject<string[]>(metadataJson)!;
        }
    }

    /// <summary>
    /// Per-platform hardware accel. detection/activation
    /// </summary>
    /// <param name="sessionOptions"></param>
    /// <param name="modelName"></param>
    private void ConfigurePlatformSpecificGpu(SessionOptions sessionOptions, string modelName)
    {
        // "The Android Neural Networks API (NNAPI) is an Android C API designed for
        // running computationally intensive operations for machine learning on Android devices."
        // It was added in Android 8.1 and will be deprecated in Android 15
        if (OperatingSystem.IsAndroid() &&
            OperatingSystem.IsAndroidVersionAtLeast(8, 1) && // At least 8.1
            !OperatingSystem.IsAndroidVersionAtLeast(15)) // At most 15
        {
            sessionOptions.AppendExecutionProvider_Nnapi();
            ExecutionProvider = "NNAPI";
            _logger.LogInformation("Initialized ExecutionProvider: nnAPI for {ModelName}", modelName);
            return;
        }

        if (OperatingSystem.IsIOS() ||
            OperatingSystem.IsMacCatalyst() ||
            OperatingSystem.IsMacOS() ||
            OperatingSystem.IsWatchOS() ||
            OperatingSystem.IsTvOS())
        {
            sessionOptions.AppendExecutionProvider_CoreML();
            ExecutionProvider = "CoreML";
            _logger.LogInformation("Initialized ExecutionProvider: CoreML for {ModelName}", modelName);
            return;
        }

        if (OperatingSystem.IsWindows())
        {
            // If DirectML is supported on the user's system, try using it first.
            // This has support for both AMD and Nvidia GPUs, and uses less memory in my testing
            try
            {
                sessionOptions.AppendExecutionProvider_DML();
                ExecutionProvider = "DirectML";
                _logger.LogInformation("Initialized ExecutionProvider: DirectML for {ModelName}", modelName);
                return;
            }
            catch (Exception)
            {
                _logger.LogWarning("Failed to create DML Execution Provider on Windows. Falling back to CUDA...");
            }
        }

        // If the user's system does not support DirectML (for whatever reason,
        // it's shipped with Windows 10, version 1903(10.0; Build 18362)+
        // Fallback on good ol' CUDA
        try
        {
            sessionOptions.AppendExecutionProvider_CUDA();
            ExecutionProvider = "CUDA";
            _logger.LogInformation("Initialized ExecutionProvider: CUDA for {ModelName}", modelName);
            return;
        }
        catch (Exception)
        {
            _logger.LogWarning("Failed to create CUDA Execution Provider.");
        }

        // And, if CUDA fails (or we have an AMD card)
        // Try one more time with MiGraphX
        try
        {
            sessionOptions.AppendExecutionProvider_MIGraphX();
            ExecutionProvider = "MIGraphX";
            _logger.LogInformation("Initialized ExecutionProvider: MIGraphX for {ModelName}", modelName);
            return;
        }
        catch (Exception)
        {
            _logger.LogWarning("Failed to create MIGraphX Execution Provider.");
        }

        // Finally, try OpenVINO (for Intel CPUs/GPUs)
        try
        {
            sessionOptions.AppendExecutionProvider_OpenVINO();
            ExecutionProvider = "OpenVINO";
            _logger.LogInformation("Initialized ExecutionProvider: OpenVINO for {ModelName}", modelName);
            return;
        }
        catch (Exception)
        {
            _logger.LogWarning("Failed to create OpenVINO Execution Provider.");
        }

        _logger.LogWarning("No GPU acceleration will be applied.");
        sessionOptions.AppendExecutionProvider_CPU();
        ExecutionProvider = "CPU";
    }

    /// <summary>
    /// Make our SessionOptions *fancy*
    /// </summary>
    /// <returns></returns>
    private SessionOptions SetupSessionOptions()
    {
        // Random environment variable(s) to speed up webcam opening on the MSMF backend.
        // https://github.com/opencv/opencv/issues/17687
        Environment.SetEnvironmentVariable("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0");
        Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "1");

        // Setup inference backend
        var sessionOptions = new SessionOptions();
        sessionOptions.InterOpNumThreads = 1;
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        // ~3% savings worth ~6ms avg latency. Not noticeable at 60fps?
        sessionOptions.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        sessionOptions.EnableMemoryPattern = true;
        return sessionOptions;
    }

    /// <summary>
    /// Runs inference on current InputTensor
    /// </summary>
    /// <returns></returns>
    public float[] Run()
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, InputTensor)
        };

        using var results = _session.Run(inputs);

        // Fast path, byte for byte what this method has always done. Anything that does not opt into
        // a secondary output - the eye pipeline, the plain stock face model - takes this branch.
        if (!_secondaryAvailable)
        {
            var single = results[0].AsEnumerable<float>().ToArray();
            OutputSize = single.Length;
            return single;
        }

        float[] output = null;
        foreach (var result in results)
        {
            if (result.Name == SecondaryOutputName)
            {
                // Copied into our own buffer because `results` is disposed on leaving this method.
                var span = _secondaryTensor.Buffer.Span;
                var index = 0;
                foreach (var value in result.AsEnumerable<float>())
                {
                    if (index >= span.Length) break;
                    span[index++] = value;
                }
            }
            else if (result.Name == _primaryOutputName)
            {
                output = result.AsEnumerable<float>().ToArray();
            }
        }

        if (output == null)
        {
            // Should be unreachable, but returning stale expressions would be far worse than a null.
            _logger.LogWarning("Primary output '{Output}' missing from results", _primaryOutputName);
            return null;
        }

        OutputSize = output.Length;
        return output;
    }

    // Dictionary mapping eye output names to their indices
    private static readonly Dictionary<string, int> OutputIndexMap = new()
    {
        { "leftEyePitch", 0 },
        { "leftEyeYaw", 1 },
        { "leftEyeLid", 2 },
        { "leftEyeWiden", 3 },
        { "leftBrow", 4 },
        { "rightEyePitch", 5 },
        { "rightEyeYaw", 6 },
        { "rightEyeLid", 7 },
        { "rightEyeWiden", 8 },
        { "rightBrow", 9 }
    };


    public DenseTensor<float> GetInputTensor()
    {
        return InputTensor;
    }
}
