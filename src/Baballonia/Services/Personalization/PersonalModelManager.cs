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
    string? RoiSettings,
    bool RequiresEmbedding = false,
    int EmbeddingDim = 0)
{
    /// <summary>
    /// "Model A" / "Model B" plus what that means, for the status line. The adapter type is the
    /// only reliable record of which architecture is actually installed - the training dropdown
    /// shows what the *next* run will use, which is not necessarily what is loaded right now.
    /// </summary>
    public string DisplayName => TrainingModelChoice.DisplayName(AdapterType);

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
            RoiSettings: Get("roi_settings"),
            RequiresEmbedding: Get("requires_embedding") == "1",
            EmbeddingDim: int.TryParse(Get("embedding_dim"), out var e) ? e : 0);
    }
}

/// <summary>Why a personal model was refused, for logging and UI.</summary>
public sealed record PersonalModelLoadResult(bool Success, string Message)
{
    public static PersonalModelLoadResult Ok(string message) => new(true, message);
    public static PersonalModelLoadResult Fail(string message) => new(false, message);
}

public enum PersonalModelArtifactSource
{
    CanonicalSlot,
    LegacySlot,
    ConfiguredPath,
    PreviousFallback,
    TrainingRun
}

/// <summary>One independently selectable personal-model artifact.</summary>
public sealed record AvailablePersonalModel(
    string Kind,
    string Label,
    string Path,
    PersonalModelMetadata Metadata,
    PersonalModelArtifactSource Source = PersonalModelArtifactSource.CanonicalSlot,
    PersonalTrainingRun? TrainingRun = null);

/// <summary>
/// Owns the lifecycle of the personal model: settings, loading, validation, hot reload.
///
/// The guiding rule is that personalization is strictly optional. Missing file, corrupt file, wrong
/// schema, wrong tensor shapes, or a runtime exception all resolve to plain stock behavior with a
/// logged warning - never to a crash and never to silently wrong expressions.
/// </summary>
public sealed class PersonalModelManager : IDisposable
{
    private sealed record ModelCandidate(
        string Path,
        PersonalModelArtifactSource Source,
        PersonalTrainingRun? TrainingRun = null);

    private sealed record PreparedModel(
        PersonalModelLoadResult Result,
        IPersonalCorrector? Corrector = null,
        string? FullPath = null);

    public const string EnabledSetting = "PersonalModel_Enabled";
    public const string PathSetting = "PersonalModel_Path";
    public const string BlendSetting = "PersonalModel_Blend";

    /// <summary>
    /// Whether the face pipeline should load the derived model that also outputs the stock visual
    /// embedding. Off by default: model C is unproven on real data, and the derived graph's
    /// behaviour under DirectML has not yet been verified on hardware.
    /// </summary>
    public const string EmbeddingRunnerSetting = "PersonalModel_UseEmbeddingRunner";

    /// <summary>
    /// Adapter formats this build knows how to feed.
    /// <para>
    /// v1 takes (image, stock); v2 takes (stock, embedding) and needs the embedding runner enabled.
    /// The version is what makes a build lacking that support refuse a model C file outright rather
    /// than feed it something plausible and ship whatever comes back.
    /// </para>
    /// </summary>
    public const int SupportedAdapterVersion = 2;

    /// <summary>Adapter version that requires the stock visual embedding.</summary>
    public const int EmbeddingAdapterVersion = 2;

    private readonly FacePipelineManager _facePipelineManager;
    private readonly ILocalSettingsService _settings;
    private readonly ILogger<PersonalModelManager> _logger;
    private readonly SemaphoreSlim _reloadLock = new(1, 1);

    private IPersonalCorrector? _corrector;
    private string? _modelCatalogFingerprint;
    private IReadOnlyList<AvailablePersonalModel> _modelCatalog = [];

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

    /// <summary>The configured artifact the user requested, even if runtime loading fell back.</summary>
    public string RequestedModelPath => ModelPath;

    /// <summary>The exact ONNX file feeding the installed corrector; null on stock/failure.</summary>
    public string? ActiveModelPath { get; private set; }

    /// <summary>Why Requested and Active differ, without hiding the actually running model.</summary>
    public string? FallbackReason { get; private set; }

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

    public bool EmbeddingRunnerEnabled => _settings.ReadSetting<bool>(EmbeddingRunnerSetting);

    /// <summary>True when the currently loaded stock face runner exposes model C's features.</summary>
    public bool EmbeddingRunnerAvailable => _facePipelineManager.EmbeddingAvailable;
    public string StockInferenceProvider => _facePipelineManager.InferenceProvider;

    /// <summary>
    /// Persists the on/off state. Does not load or unload by itself - call
    /// <see cref="ReloadAsync"/> afterwards to act on the change.
    /// </summary>
    public void SetEnabled(bool enabled) => _settings.SaveSetting(EnabledSetting, enabled);

    public void SetModelPath(string path) => _settings.SaveSetting(PathSetting, path);

    /// <summary>
    /// Finds independently installed A/B/C slots, rollback files and every immutable completed
    /// training artifact. Candidates are deduplicated by exact path only: several B runs are several
    /// genuinely selectable models, not one family entry that silently hides history.
    /// </summary>
    public IReadOnlyList<AvailablePersonalModel> DiscoverAvailableModels(
        string? trainingRunsDirectory = null,
        string? datasetRoot = null)
    {
        var effectiveRunsDirectory = trainingRunsDirectory ??
                                     PersonalizationEnvironment.TrainingRunsDirectory;
        var effectiveDatasetRoot = datasetRoot ?? PersonalizationPaths.DatasetRoot;
        var primaryCandidates = TrainingModelChoice.Options
            .Select(option => PersonalizationPaths.PersonalModelPath(option.Kind))
            .Append(PersonalizationPaths.DefaultPersonalModelPath)
            .Append(ModelPath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var installedCandidates = primaryCandidates
            // Earlier builds kept the outgoing trained adapter as .previous.onnx. Treat it as an
            // available model too so an existing A/B pair immediately appears in the new selector.
            .SelectMany(path => new[] { path, PreviousModelPath(path) })
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var fingerprint = string.Join("|", installedCandidates.Select(FileFingerprint)) + "|" +
                          TrainingRunHistory.Fingerprint(effectiveRunsDirectory, effectiveDatasetRoot);

        // Reading ONNX metadata creates an inference session. Cache the catalog until one of the
        // candidate or provenance files changes so merely refreshing the page cannot burn CPU.
        if (string.Equals(fingerprint, _modelCatalogFingerprint, StringComparison.Ordinal))
            return _modelCatalog;

        var candidates = new List<ModelCandidate>();
        var candidateIndices = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        void AddCandidate(
            string path,
            PersonalModelArtifactSource source,
            PersonalTrainingRun? trainingRun = null)
        {
            string fullPath;
            try { fullPath = Path.GetFullPath(path); }
            catch (Exception) { return; }

            if (candidateIndices.TryGetValue(fullPath, out var existingIndex))
            {
                // A selected historical path is also the configured path. Keep its richer immutable
                // provenance while retaining its original position in the selector.
                if (trainingRun != null)
                    candidates[existingIndex] = new ModelCandidate(
                        fullPath, PersonalModelArtifactSource.TrainingRun, trainingRun);
                return;
            }

            candidateIndices.Add(fullPath, candidates.Count);
            candidates.Add(new ModelCandidate(fullPath, source, trainingRun));
        }

        foreach (var option in TrainingModelChoice.Options)
            AddCandidate(PersonalizationPaths.PersonalModelPath(option.Kind),
                PersonalModelArtifactSource.CanonicalSlot);
        AddCandidate(PersonalizationPaths.DefaultPersonalModelPath,
            PersonalModelArtifactSource.LegacySlot);
        AddCandidate(ModelPath, PersonalModelArtifactSource.ConfiguredPath);

        foreach (var path in primaryCandidates)
            AddCandidate(PreviousModelPath(path), PersonalModelArtifactSource.PreviousFallback);

        // Discover() is independently useful to the normal UI and returns newest first. Appending
        // in that order preserves the familiar canonical A/B/C entries while making all historical
        // artifacts available immediately afterwards.
        foreach (var run in TrainingRunHistory.Discover(effectiveRunsDirectory, effectiveDatasetRoot))
            AddCandidate(run.ModelPath, PersonalModelArtifactSource.TrainingRun, run);

        var catalog = new List<AvailablePersonalModel>();

        foreach (var path in candidates.Where(candidate => File.Exists(candidate.Path)))
        {
            try
            {
                using var session = new InferenceSession(path.Path);
                var metadata = PersonalModelMetadata.FromSession(session);
                var option = TrainingModelChoice.ForAdapterType(metadata.AdapterType);
                if (option == null)
                    continue;

                // Full structural/schema validation, without treating a temporarily disabled Model
                // C embedding runner as an incompatible artifact. The selector knows how to enable
                // that runner before Use; malformed tensors or stale expression ordering stay out.
                var validation = Validate(session, requireEmbeddingRunner: false);
                if (!validation.Success)
                {
                    _logger.LogDebug(
                        "Ignoring incompatible personal-model candidate {Path}: {Reason}",
                        path.Path, validation.Message);
                    continue;
                }

                catalog.Add(new AvailablePersonalModel(
                    option.Kind,
                    CatalogLabel(option, path),
                    path.Path,
                    metadata,
                    path.Source,
                    path.TrainingRun));
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Ignoring unreadable personal-model candidate {Path}", path.Path);
            }
        }

        // The normal picker is a history browser, so chronology wins over canonical-slot grouping.
        // A/B/C remain explicit in every label and the exact artifact path remains the selection key.
        _modelCatalog = catalog
            .OrderByDescending(ModelSortTimestamp)
            .ThenByDescending(model => model.TrainingRun?.RunId ?? string.Empty, StringComparer.Ordinal)
            .ThenBy(model => model.Path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        _modelCatalogFingerprint = fingerprint;
        return _modelCatalog;
    }

    public async Task<PersonalModelLoadResult> SelectModelAsync(string path)
    {
        await _reloadLock.WaitAsync();
        try
        {
            // A picker choice is a proposed replacement, not permission to tear down the model
            // that is already working. Open and validate it first so a corrupt, incompatible or
            // temporarily unavailable artifact cannot persist a bad path or drop the live pipeline
            // back to stock behavior.
            var prepared = await Task.Run(() => PrepareModel(path));
            if (!prepared.Result.Success)
            {
                LastResult = prepared.Result;
                _logger.LogWarning(
                    "Personal model selection rejected; keeping the active model: {Message}",
                    prepared.Result.Message);
                return prepared.Result;
            }

            var previousConfiguredPath = _settings.ReadSetting<string>(PathSetting) ?? "";
            var previousEnabled = Enabled;
            try
            {
                SetModelPath(path);
                SetEnabled(true);
                InstallPreparedModel(prepared);
            }
            catch (Exception ex)
            {
                RestoreSelectionSettings(previousConfiguredPath, previousEnabled);
                try
                {
                    prepared.Corrector?.Dispose();
                }
                catch (Exception disposeException)
                {
                    _logger.LogWarning(disposeException,
                        "Could not dispose a personal-model candidate after a failed selection");
                }

                LastResult = PersonalModelLoadResult.Fail(
                    $"Could not select {Path.GetFileName(path)}: {ex.Message}");
                _logger.LogWarning(ex,
                    "Personal model selection failed while committing; keeping the active model");
                return LastResult;
            }

            FallbackReason = null;
            LastResult = prepared.Result;
            _logger.LogInformation("Personal model active: {Message}", LastResult.Message);
            return LastResult;
        }
        finally
        {
            _reloadLock.Release();
        }
    }

    /// <summary>
    /// Switches the stock face runner as well as persisting the setting, then reloads the personal
    /// adapter against that runner. This keeps the checkbox from claiming C is enabled while the
    /// old inference session is still live.
    /// </summary>
    public async Task<PersonalModelLoadResult> SetEmbeddingRunnerEnabledAsync(bool enabled)
    {
        _settings.SaveSetting(EmbeddingRunnerSetting, enabled);
        await _facePipelineManager.LoadInferenceAsync();
        return await ReloadAsync();
    }

    public float Blend
    {
        get
        {
            // The explicit default distinguishes a brand-new settings file from a deliberately
            // persisted 0% blend. Zero is the exact Stock side of the UI comparison and must remain
            // zero after restart rather than silently becoming 100% Personal.
            var value = _settings.ReadSetting<float>(BlendSetting, 1f);
            return Math.Clamp(value, 0f, 1f);
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
            FallbackReason = null;

            if (!Enabled)
            {
                LastResult = PersonalModelLoadResult.Ok("Personalization disabled; using stock model.");
                return LastResult;
            }

            var path = ModelPath;
            var result = await Task.Run(() => TryLoad(path));

            // One rung of fallback before giving up on personalization entirely. The installer keeps
            // the outgoing model as .previous.onnx, so the common failure - a model C adapter
            // installed while the embedding runner is off - lands on the model that was working an
            // hour ago rather than dropping the user all the way back to stock without explanation.
            if (!result.Success)
            {
                var previous = PreviousModelPath(path);
                if (File.Exists(previous))
                {
                    _logger.LogWarning("Personal model not loaded ({Reason}); trying the previous one",
                        result.Message);

                    var fallback = await Task.Run(() => TryLoad(previous));
                    if (fallback.Success)
                    {
                        FallbackReason = result.Message;
                        LastResult = PersonalModelLoadResult.Ok(
                            $"{fallback.Message} (using the previous model: {result.Message})");
                        _logger.LogInformation("Personal model active: {Message}", LastResult.Message);
                        return LastResult;
                    }
                }
            }

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
        var prepared = PrepareModel(path);
        if (!prepared.Result.Success)
            return prepared.Result;

        try
        {
            InstallPreparedModel(prepared);
            return prepared.Result;
        }
        catch (Exception ex)
        {
            try
            {
                prepared.Corrector?.Dispose();
            }
            catch (Exception disposeException)
            {
                _logger.LogWarning(disposeException,
                    "Could not dispose a personal model after installation failed");
            }

            return PersonalModelLoadResult.Fail(
                $"Could not load {Path.GetFileName(path)}: {ex.Message}");
        }
    }

    private PreparedModel PrepareModel(string path)
    {
        if (!File.Exists(path))
            return new PreparedModel(PersonalModelLoadResult.Fail($"No personal model at {path}."));

        InferenceSession? session = null;
        try
        {
            var fullPath = Path.GetFullPath(path);
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
                session = null;
                return new PreparedModel(validation);
            }

            var metadata = PersonalModelMetadata.FromSession(session);
            var blend = Blend;

            IPersonalCorrector corrector = RequiresEmbedding(metadata)
                ? new EmbeddingModelCorrector(session, metadata, _logger) { Blend = blend }
                : new PersonalModelCorrector(session, metadata, _logger) { Blend = blend };
            session = null; // The corrector now owns the inference session.

            return new PreparedModel(
                PersonalModelLoadResult.Ok(
                    $"{metadata.AdapterType} (trained {metadata.TrainedUtc ?? "unknown"}), " +
                    $"blend {blend:P0}"),
                corrector,
                fullPath);
        }
        catch (Exception ex)
        {
            session?.Dispose();
            return new PreparedModel(PersonalModelLoadResult.Fail(
                $"Could not load {Path.GetFileName(path)}: {ex.Message}"));
        }
    }

    private void InstallPreparedModel(PreparedModel prepared)
    {
        var corrector = prepared.Corrector
            ?? throw new InvalidOperationException("A successful model preparation had no corrector.");
        var fullPath = prepared.FullPath
            ?? throw new InvalidOperationException("A successful model preparation had no path.");

        // Publish the replacement before disposing the old session. SetCorrector is a volatile
        // assignment, so the next pipeline tick sees either complete corrector, never a stock gap.
        var previous = _corrector;
        _facePipelineManager.SetCorrector(corrector);
        _corrector = corrector;
        ActiveModelPath = fullPath;

        if (previous == null || ReferenceEquals(previous, corrector))
            return;

        try
        {
            previous.Dispose();
        }
        catch (Exception ex)
        {
            // The new corrector is already live. A cleanup failure must not roll that successful
            // swap back or make the picker claim selection failed.
            _logger.LogWarning(ex, "Could not dispose the replaced personal-model session");
        }
    }

    private void RestoreSelectionSettings(string path, bool enabled)
    {
        try
        {
            _settings.SaveSetting(PathSetting, path);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Could not restore the personal-model path after a failed selection commit");
        }

        try
        {
            _settings.SaveSetting(EnabledSetting, enabled);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Could not restore the personal-model enabled state after a failed selection commit");
        }
    }

    /// <summary>
    /// Rejects models this build cannot feed correctly. The schema check is the important one: a
    /// model trained against a different expression ordering would apply every learned correction
    /// to the wrong expression, which looks like erratic tracking rather than an error.
    /// </summary>
    private PersonalModelLoadResult Validate(
        InferenceSession session,
        bool requireEmbeddingRunner = true)
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

        if (!inputs.TryGetValue(PersonalModelCorrector.StockInputName, out var stock))
            return PersonalModelLoadResult.Fail($"Model has no '{PersonalModelCorrector.StockInputName}' input.");

        if (!ShapeMatches(stock.Dimensions, [1, n]))
        {
            return PersonalModelLoadResult.Fail(
                $"Stock input shape [{string.Join(",", stock.Dimensions)}] should be [1,{n}].");
        }

        if (RequiresEmbedding(metadata))
        {
            if (!inputs.TryGetValue(EmbeddingModelCorrector.EmbeddingInputName, out var embedding))
            {
                return PersonalModelLoadResult.Fail(
                    $"Model declares itself embedding-based but has no " +
                    $"'{EmbeddingModelCorrector.EmbeddingInputName}' input.");
            }

            if (!ShapeMatches(embedding.Dimensions, [1, metadata.EmbeddingDim]))
            {
                return PersonalModelLoadResult.Fail(
                    $"Embedding input shape [{string.Join(",", embedding.Dimensions)}] should be " +
                    $"[1,{metadata.EmbeddingDim}].");
            }

            // Refused rather than run degraded: without the runner there is no embedding to feed it,
            // and a model C adapter with no features is just a slower passthrough pretending to work.
            if (requireEmbeddingRunner && !_facePipelineManager.EmbeddingAvailable)
            {
                return PersonalModelLoadResult.Fail(
                    "This model needs the stock visual embedding, which is not switched on. " +
                    "Enable the embedding runner under Advanced, or install a model A/B instead.");
            }
        }
        else
        {
            if (!inputs.TryGetValue(PersonalModelCorrector.ImageInputName, out var image))
                return PersonalModelLoadResult.Fail($"Model has no '{PersonalModelCorrector.ImageInputName}' input.");

            if (!ShapeMatches(image.Dimensions, [1, 1, metadata.InputSize, metadata.InputSize]))
            {
                return PersonalModelLoadResult.Fail(
                    $"Image input shape [{string.Join(",", image.Dimensions)}] does not match the " +
                    $"declared input_size {metadata.InputSize}.");
            }
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

    /// <summary>
    /// The rollback copy the installer leaves behind. Matches the name used when installing, so the
    /// two cannot drift apart silently.
    /// </summary>
    public static string PreviousModelPath(string modelPath) =>
        Path.ChangeExtension(modelPath, ".previous.onnx");

    /// <summary>Whether a model consumes the stock network's visual features rather than the frame.</summary>
    private static bool RequiresEmbedding(PersonalModelMetadata metadata) =>
        metadata.RequiresEmbedding || metadata.AdapterVersion >= EmbeddingAdapterVersion;

    private static bool ShapeMatches(IReadOnlyList<int> actual, IReadOnlyList<int> expected)
    {
        if (actual.Count != expected.Count)
            return false;

        // Negative dimensions are dynamic axes and match anything.
        return !actual.Where((dim, i) => dim >= 0 && dim != expected[i]).Any();
    }

    private static string Shorten(string hash) => hash.Length <= 12 ? hash : hash[..12] + "...";

    private static string FileFingerprint(string path)
    {
        try
        {
            var fullPath = Path.GetFullPath(path);
            var file = new FileInfo(fullPath);
            return file.Exists
                ? $"{fullPath}:{file.Length}:{file.LastWriteTimeUtc.Ticks}"
                : $"{fullPath}:missing";
        }
        catch (Exception)
        {
            return $"{path}:invalid";
        }
    }

    private static DateTimeOffset ModelSortTimestamp(AvailablePersonalModel model)
    {
        if (model.TrainingRun?.TrainedUtc is { } runTimestamp)
            return runTimestamp;
        if (DateTimeOffset.TryParse(model.Metadata.TrainedUtc, out var metadataTimestamp))
            return metadataTimestamp;
        try
        {
            return new DateTimeOffset(File.GetLastWriteTimeUtc(model.Path));
        }
        catch (Exception)
        {
            return DateTimeOffset.MinValue;
        }
    }

    private static string CatalogLabel(TrainingModelChoice.Option option, ModelCandidate candidate)
    {
        if (candidate.TrainingRun is { } run)
        {
            var trained = run.TrainedUtc is { } timestamp
                ? timestamp.UtcDateTime.ToString("yyyy-MM-dd HH:mm 'UTC'")
                : "time unknown";
            return $"{option.Kind.ToUpperInvariant()} — {trained} — {run.RunId}";
        }

        return candidate.Source switch
        {
            PersonalModelArtifactSource.LegacySlot => $"{option.Label} — Legacy slot",
            PersonalModelArtifactSource.ConfiguredPath => $"{option.Label} — Configured file",
            PersonalModelArtifactSource.PreviousFallback =>
                $"{option.Label} — Previous fallback ({Path.GetFileName(candidate.Path)})",
            _ => option.Label
        };
    }

    /// <summary>Removes the corrector so the pipeline returns to pure stock output.</summary>
    public void Uninstall()
    {
        // Clear the pipeline first, then dispose: the tick may be mid-Correct on another thread.
        _facePipelineManager.SetCorrector(null);

        var previous = _corrector;
        _corrector = null;
        ActiveModelPath = null;
        previous?.Dispose();
    }

    public void Dispose()
    {
        Uninstall();
        _reloadLock.Dispose();
    }
}
