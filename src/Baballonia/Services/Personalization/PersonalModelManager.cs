using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services.Inference;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;

namespace Baballonia.Services.Personalization;

/// <summary>Metadata embedded in an exported personal model by the Python trainer.</summary>
public sealed record PersonalModelMetadata(
    int AdapterVersion,
    string AdapterType,
    string SchemaSha256,
    string InputNormalization,
    int InputSize,
    string? TrainedUtc,
    string? BaseModelMd5,
    string? RoiSettings)
{
    public static PersonalModelMetadata FromSession(InferenceSession session)
    {
        var map = session.ModelMetadata.CustomMetadataMap;

        string? Get(string key) => map.TryGetValue(key, out var value) ? value : null;

        return new PersonalModelMetadata(
            AdapterVersion: int.TryParse(Get("personal_adapter_version"), out var v) ? v : 0,
            AdapterType: Get("adapter_type") ?? "unknown",
            SchemaSha256: Get("expression_schema_sha256") ?? "",
            InputNormalization: Get("input_normalization") ?? "",
            InputSize: int.TryParse(Get("input_size"), out var s) ? s : 0,
            TrainedUtc: Get("trained_utc"),
            BaseModelMd5: Get("base_model_md5"),
            RoiSettings: Get("roi_settings"));
    }
}

/// <summary>Why a personal model was refused, for logging and UI.</summary>
public sealed record PersonalModelLoadResult(bool Success, string Message)
{
    public static PersonalModelLoadResult Ok(string message) => new(true, message);
    public static PersonalModelLoadResult Fail(string message) => new(false, message);
}

/// <summary>
/// Owns the lifecycle of the personal model: settings, loading, validation, hot reload.
///
/// The guiding rule is that personalization is strictly optional. Missing file, corrupt file, wrong
/// schema, wrong tensor shapes, or a runtime exception all resolve to plain stock behavior with a
/// logged warning - never to a crash and never to silently wrong expressions.
/// </summary>
public sealed class PersonalModelManager : IDisposable
{
    public const string EnabledSetting = "PersonalModel_Enabled";
    public const string PathSetting = "PersonalModel_Path";
    public const string BlendSetting = "PersonalModel_Blend";

    /// <summary>Adapter formats this build knows how to feed.</summary>
    public const int SupportedAdapterVersion = 1;

    private readonly FacePipelineManager _facePipelineManager;
    private readonly ILocalSettingsService _settings;
    private readonly ILogger<PersonalModelManager> _logger;
    private readonly SemaphoreSlim _reloadLock = new(1, 1);

    private PersonalModelCorrector? _corrector;

    public PersonalModelManager(
        FacePipelineManager facePipelineManager,
        ILocalSettingsService settings,
        ILogger<PersonalModelManager> logger)
    {
        _facePipelineManager = facePipelineManager;
        _settings = settings;
        _logger = logger;
    }

    /// <summary>Result of the most recent load attempt, for display on the personalization page.</summary>
    public PersonalModelLoadResult? LastResult { get; private set; }

    public PersonalModelMetadata? LoadedMetadata => _corrector?.Metadata;

    public bool IsActive => _corrector is { HasFailed: false };

    public string ModelPath
    {
        get
        {
            var configured = _settings.ReadSetting<string>(PathSetting);
            return string.IsNullOrWhiteSpace(configured)
                ? PersonalizationPaths.DefaultPersonalModelPath
                : configured;
        }
    }

    public bool Enabled => _settings.ReadSetting<bool>(EnabledSetting);

    public float Blend
    {
        get
        {
            var value = _settings.ReadSetting<float>(BlendSetting);
            // A brand-new settings file reads 0, which would make an enabled model look broken.
            return value <= 0f ? 1f : Math.Clamp(value, 0f, 1f);
        }
        set
        {
            var clamped = Math.Clamp(value, 0f, 1f);
            _settings.SaveSetting(BlendSetting, clamped);
            if (_corrector != null)
                _corrector.Blend = clamped;
        }
    }

    /// <summary>
    /// Applies current settings: loads and installs the model, or uninstalls it. Safe to call
    /// repeatedly; loading happens off the UI thread.
    /// </summary>
    public async Task<PersonalModelLoadResult> ReloadAsync()
    {
        await _reloadLock.WaitAsync();
        try
        {
            Uninstall();

            if (!Enabled)
            {
                LastResult = PersonalModelLoadResult.Ok("Personalization disabled; using stock model.");
                return LastResult;
            }

            var path = ModelPath;
            var result = await Task.Run(() => TryLoad(path));
            LastResult = result;

            if (result.Success)
                _logger.LogInformation("Personal model active: {Message}", result.Message);
            else
                _logger.LogWarning("Personal model not loaded: {Message}", result.Message);

            return result;
        }
        finally
        {
            _reloadLock.Release();
        }
    }

    private PersonalModelLoadResult TryLoad(string path)
    {
        if (!File.Exists(path))
            return PersonalModelLoadResult.Fail($"No personal model at {path}.");

        InferenceSession? session = null;
        try
        {
            var options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.InterOpNumThreads = 1;
            options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
            // CPU deliberately: the adapter is tiny, and a second GPU session would contend with
            // the stock model for the same device.
            options.AppendExecutionProvider_CPU();

            session = new InferenceSession(path, options);

            var validation = Validate(session);
            if (!validation.Success)
            {
                session.Dispose();
                return validation;
            }

            var metadata = PersonalModelMetadata.FromSession(session);
            var corrector = new PersonalModelCorrector(session, metadata, _logger) { Blend = Blend };

            _corrector = corrector;
            _facePipelineManager.SetCorrector(corrector);

            return PersonalModelLoadResult.Ok(
                $"{metadata.AdapterType} (trained {metadata.TrainedUtc ?? "unknown"}), blend {Blend:P0}");
        }
        catch (Exception ex)
        {
            session?.Dispose();
            return PersonalModelLoadResult.Fail($"Could not load {Path.GetFileName(path)}: {ex.Message}");
        }
    }

    /// <summary>
    /// Rejects models this build cannot feed correctly. The schema check is the important one: a
    /// model trained against a different expression ordering would apply every learned correction
    /// to the wrong expression, which looks like erratic tracking rather than an error.
    /// </summary>
    private PersonalModelLoadResult Validate(InferenceSession session)
    {
        var metadata = PersonalModelMetadata.FromSession(session);
        var n = PersonalizationSchema.ExpressionCount;

        if (metadata.AdapterVersion > SupportedAdapterVersion)
        {
            return PersonalModelLoadResult.Fail(
                $"Model format v{metadata.AdapterVersion} is newer than this build supports " +
                $"(v{SupportedAdapterVersion}). Update Baballonia or re-export the model.");
        }

        if (string.IsNullOrEmpty(metadata.SchemaSha256))
        {
            return PersonalModelLoadResult.Fail(
                "Model has no expression_schema_sha256 metadata, so its output order cannot be " +
                "verified. Re-export it with the current trainer.");
        }

        if (!string.Equals(metadata.SchemaSha256, PersonalizationSchema.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            return PersonalModelLoadResult.Fail(
                "Model was trained against a different expression schema " +
                $"({Shorten(metadata.SchemaSha256)} vs {Shorten(PersonalizationSchema.Sha256)}). " +
                "Retrain it against the current expression list.");
        }

        var inputs = session.InputMetadata;

        if (!inputs.TryGetValue(PersonalModelCorrector.ImageInputName, out var image))
            return PersonalModelLoadResult.Fail($"Model has no '{PersonalModelCorrector.ImageInputName}' input.");

        if (!inputs.TryGetValue(PersonalModelCorrector.StockInputName, out var stock))
            return PersonalModelLoadResult.Fail($"Model has no '{PersonalModelCorrector.StockInputName}' input.");

        if (!ShapeMatches(image.Dimensions, [1, 1, metadata.InputSize, metadata.InputSize]))
        {
            return PersonalModelLoadResult.Fail(
                $"Image input shape [{string.Join(",", image.Dimensions)}] does not match the " +
                $"declared input_size {metadata.InputSize}.");
        }

        if (!ShapeMatches(stock.Dimensions, [1, n]))
        {
            return PersonalModelLoadResult.Fail(
                $"Stock input shape [{string.Join(",", stock.Dimensions)}] should be [1,{n}].");
        }

        if (!session.OutputMetadata.ContainsKey(PersonalModelCorrector.OutputName))
        {
            return PersonalModelLoadResult.Fail(
                $"Model has no '{PersonalModelCorrector.OutputName}' output " +
                $"(found: {string.Join(", ", session.OutputMetadata.Keys)}).");
        }

        // Not fatal: the ROI may legitimately have been nudged since recording. Worth saying out
        // loud though, because a large change is a good explanation for a model behaving oddly.
        if (!string.IsNullOrEmpty(metadata.RoiSettings))
            _logger.LogInformation("Personal model was trained with ROI {Roi}", metadata.RoiSettings);

        return PersonalModelLoadResult.Ok("valid");
    }

    private static bool ShapeMatches(IReadOnlyList<int> actual, IReadOnlyList<int> expected)
    {
        if (actual.Count != expected.Count)
            return false;

        // Negative dimensions are dynamic axes and match anything.
        return !actual.Where((dim, i) => dim >= 0 && dim != expected[i]).Any();
    }

    private static string Shorten(string hash) => hash.Length <= 12 ? hash : hash[..12] + "...";

    /// <summary>Removes the corrector so the pipeline returns to pure stock output.</summary>
    public void Uninstall()
    {
        // Clear the pipeline first, then dispose: the tick may be mid-Correct on another thread.
        _facePipelineManager.SetCorrector(null);

        var previous = _corrector;
        _corrector = null;
        previous?.Dispose();
    }

    public void Dispose()
    {
        Uninstall();
        _reloadLock.Dispose();
    }
}
