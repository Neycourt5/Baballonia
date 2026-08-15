using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;

namespace Baballonia.Services.Personalization;

/// <summary>How one immutable training run divided the recorded corpus.</summary>
public sealed record TrainingRunDatasetSlice(
    IReadOnlyList<string> SessionIds,
    int SessionCount,
    int ResolvedSessionCount,
    int? FrameCount,
    IReadOnlyList<string> MissingSessionIds,
    bool? IncludesGuided,
    bool? IncludesCorrection)
{
    public bool HasUnresolvedSessions => MissingSessionIds.Count > 0;
}

/// <summary>
/// Provenance and measured outcome for one completed trainer output directory.
/// </summary>
/// <remarks>
/// The paths and split come from the run itself, not from recomputing today's default split. That
/// distinction matters after more recordings have been added: history must continue to say exactly
/// what an already-exported model did and did not learn from.
/// </remarks>
public sealed record PersonalTrainingRun(
    string RunId,
    string AdapterFamily,
    string AdapterType,
    string ModelPath,
    DateTimeOffset? TrainedUtc,
    TrainingRunDatasetSlice Train,
    TrainingRunDatasetSlice HeldOut,
    TrainingSummary Summary,
    PersonalModelMetadata ModelMetadata)
{
    public IReadOnlyList<string> TrainSessionIds => Train.SessionIds;
    public IReadOnlyList<string> HeldOutSessionIds => HeldOut.SessionIds;
    public int TrainSessionCount => Train.SessionCount;
    public int HeldOutSessionCount => HeldOut.SessionCount;
    public int? TrainFrameCount => Train.FrameCount;
    public int? HeldOutFrameCount => HeldOut.FrameCount;
    public bool? GuidedInTraining => Train.IncludesGuided;
    public bool? GuidedInHeldOut => HeldOut.IncludesGuided;
    public bool? CorrectionInTraining => Train.IncludesCorrection;
    public bool? CorrectionInHeldOut => HeldOut.IncludesCorrection;
    public string Verdict => Summary.Verdict;
    public bool ValidatedOnHeldOutSessions => Summary.ValidatedOnHeldOutSessions;
    public int ExpressionsImproved => Summary.ExpressionsImproved;
    public int ExpressionsRegressed => Summary.ExpressionsRegressed;
    public double? MeanStockMae => Summary.MeanStockMae;
    public double? MeanPersonalMae => Summary.MeanPersonalMae;
}

/// <summary>
/// Reads completed personal-training runs without modifying them.
/// </summary>
/// <remarks>
/// A completed run must contain the manifest, measured summary and exported ONNX. A checkpoint by
/// itself is deliberately not history the user can select. Malformed JSON, contradictory adapter
/// identities and unreadable ONNX files are omitted rather than allowed to poison the whole catalog.
/// Compatibility with this build is a separate concern handled by <see cref="PersonalModelManager"/>:
/// an honest history can still contain an older-schema artifact that the current runtime must refuse.
/// </remarks>
public static class TrainingRunHistory
{
    public const string RunManifestFileName = "run.json";
    public const string SummaryFileName = "summary.json";
    public const string ExportedModelFileName = "personalFaceModel.onnx";

    private sealed record SessionInventory(string Type, int Frames);
    private sealed record RunSessionInventory(string Type, int Frames);
    private sealed record OrderedRun(PersonalTrainingRun Run, DateTimeOffset SortTimestamp);

    /// <summary>Discovers every complete run, newest first.</summary>
    public static IReadOnlyList<PersonalTrainingRun> Discover(
        string? runsRoot = null,
        string? datasetRoot = null)
    {
        var effectiveRunsRoot = runsRoot ?? PersonalizationEnvironment.TrainingRunsDirectory;
        var effectiveDatasetRoot = datasetRoot ?? PersonalizationPaths.DatasetRoot;
        if (!Directory.Exists(effectiveRunsRoot))
            return [];

        var sessions = ReadSessionInventory(effectiveDatasetRoot);
        var runs = new List<OrderedRun>();

        IEnumerable<string> directories;
        try
        {
            directories = Directory.EnumerateDirectories(effectiveRunsRoot).ToArray();
        }
        catch (Exception)
        {
            return [];
        }

        foreach (var directory in directories)
        {
            if (TryReadRun(directory, sessions, out var run, out var sortTimestamp))
                runs.Add(new OrderedRun(run!, sortTimestamp));
        }

        return runs
            .OrderByDescending(item => item.SortTimestamp)
            .ThenByDescending(item => item.Run.RunId, StringComparer.Ordinal)
            .Select(item => item.Run)
            .ToArray();
    }

    /// <summary>
    /// Cheap cache key for consumers that would otherwise reopen every ONNX on each page refresh.
    /// Includes dataset metadata because resolved frame totals can change when recordings are moved.
    /// </summary>
    public static string Fingerprint(string? runsRoot = null, string? datasetRoot = null)
    {
        var effectiveRunsRoot = Path.GetFullPath(
            runsRoot ?? PersonalizationEnvironment.TrainingRunsDirectory);
        var effectiveDatasetRoot = Path.GetFullPath(datasetRoot ?? PersonalizationPaths.DatasetRoot);
        var builder = new StringBuilder()
            .Append(effectiveRunsRoot)
            .Append('|')
            .Append(effectiveDatasetRoot);

        AppendDirectoryFiles(builder, effectiveRunsRoot,
            [RunManifestFileName, SummaryFileName, ExportedModelFileName]);
        AppendDirectoryFiles(builder, effectiveDatasetRoot, ["session.json"]);
        return builder.ToString();
    }

    private static bool TryReadRun(
        string directory,
        IReadOnlyDictionary<string, SessionInventory> sessions,
        out PersonalTrainingRun? run,
        out DateTimeOffset sortTimestamp)
    {
        run = null;
        sortTimestamp = DateTimeOffset.MinValue;

        var manifestPath = Path.Combine(directory, RunManifestFileName);
        var summaryPath = Path.Combine(directory, SummaryFileName);
        var modelPath = Path.Combine(directory, ExportedModelFileName);
        if (!File.Exists(manifestPath) || !File.Exists(summaryPath) || !File.Exists(modelPath))
            return false;

        try
        {
            using var manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
            using var summaryDocument = JsonDocument.Parse(File.ReadAllText(summaryPath));
            var manifest = manifestDocument.RootElement;
            var summaryRoot = summaryDocument.RootElement;

            if (manifest.ValueKind != JsonValueKind.Object ||
                summaryRoot.ValueKind != JsonValueKind.Object ||
                !TryReadString(manifest, "adapter_type", out var adapterType) ||
                !TryReadString(summaryRoot, "adapter_type", out var summaryAdapterType) ||
                !string.Equals(adapterType, summaryAdapterType, StringComparison.Ordinal) ||
                !TryReadStringArray(manifest, "train_sessions", out var trainIds) ||
                !TryReadStringArray(manifest, "val_sessions", out var heldOutIds) ||
                trainIds.Count == 0 ||
                trainIds.Count != trainIds.Distinct(StringComparer.Ordinal).Count() ||
                heldOutIds.Count != heldOutIds.Distinct(StringComparer.Ordinal).Count() ||
                trainIds.Intersect(heldOutIds, StringComparer.Ordinal).Any())
            {
                return false;
            }

            // Optional in older runs. New trainers freeze the type/frame identity here so deleting
            // a recording cannot silently rewrite an already-trained model's provenance.
            if (!TryReadRunInventory(manifest, "train_inventory", trainIds, out var trainInventory) ||
                !TryReadRunInventory(manifest, "val_inventory", heldOutIds, out var heldOutInventory))
            {
                return false;
            }

            using var onnx = new InferenceSession(modelPath);
            var metadata = PersonalModelMetadata.FromSession(onnx);
            if (!string.Equals(metadata.AdapterType, adapterType, StringComparison.Ordinal))
                return false;

            var summary = TrainingSummaryReader.Parse(summaryRoot.GetRawText());
            var option = TrainingModelChoice.ForAdapterType(adapterType);
            var family = option?.Kind.ToUpperInvariant() ?? "Unknown";
            var trainedUtc = ParseTimestamp(metadata.TrainedUtc);
            sortTimestamp = trainedUtc ?? new DateTimeOffset(File.GetLastWriteTimeUtc(modelPath));

            run = new PersonalTrainingRun(
                RunId: Path.GetFileName(directory),
                AdapterFamily: family,
                AdapterType: adapterType,
                ModelPath: Path.GetFullPath(modelPath),
                TrainedUtc: trainedUtc,
                Train: ResolveSlice(trainIds, sessions, trainInventory),
                HeldOut: ResolveSlice(heldOutIds, sessions, heldOutInventory),
                Summary: summary,
                ModelMetadata: metadata);
            return true;
        }
        catch (Exception)
        {
            // One corrupt/stale artifact must not hide the other completed models.
            return false;
        }
    }

    private static TrainingRunDatasetSlice ResolveSlice(
        IReadOnlyList<string> ids,
        IReadOnlyDictionary<string, SessionInventory> sessions,
        IReadOnlyDictionary<string, RunSessionInventory> frozenInventory)
    {
        var frames = 0;
        var framesKnown = true;
        var resolved = 0;
        var guided = false;
        var correction = false;
        var typesKnown = true;
        var missing = new List<string>();

        foreach (var id in ids)
        {
            var existsNow = sessions.TryGetValue(id, out var current);
            if (!existsNow)
            {
                missing.Add(id);
            }
            else
            {
                resolved++;
            }

            if (!frozenInventory.TryGetValue(id, out var frozen) && current == null)
            {
                // An old run only stored IDs. Once its source is deleted, inventing zero frames or
                // saying it was not Guided/Correction would be false precision.
                framesKnown = false;
                typesKnown = false;
                continue;
            }

            var type = frozen?.Type ?? current!.Type;
            var frameCount = frozen?.Frames ?? current!.Frames;
            frames += frameCount;
            guided |= string.Equals(type, "Guided", StringComparison.OrdinalIgnoreCase);
            correction |= string.Equals(type, "Correction", StringComparison.OrdinalIgnoreCase);
        }

        return new TrainingRunDatasetSlice(
            SessionIds: Array.AsReadOnly(ids.ToArray()),
            SessionCount: ids.Count,
            ResolvedSessionCount: resolved,
            FrameCount: framesKnown ? frames : null,
            MissingSessionIds: missing.AsReadOnly(),
            IncludesGuided: guided ? true : typesKnown ? false : null,
            IncludesCorrection: correction ? true : typesKnown ? false : null);
    }

    private static bool TryReadRunInventory(
        JsonElement manifest,
        string propertyName,
        IReadOnlyList<string> expectedIds,
        out IReadOnlyDictionary<string, RunSessionInventory> inventory)
    {
        inventory = new Dictionary<string, RunSessionInventory>(StringComparer.Ordinal);
        if (!manifest.TryGetProperty(propertyName, out var element))
            return true; // Backward-compatible run.json written before the frozen inventory field.
        if (element.ValueKind != JsonValueKind.Array)
            return false;

        var expected = expectedIds.ToHashSet(StringComparer.Ordinal);
        var parsed = new Dictionary<string, RunSessionInventory>(StringComparer.Ordinal);
        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object ||
                !TryReadString(item, "session_id", out var id) ||
                !expected.Contains(id) ||
                !TryReadString(item, "session_type", out var type) ||
                !Enum.TryParse<SessionType>(type, ignoreCase: true, out _) ||
                !item.TryGetProperty("frame_count", out var frameCountElement) ||
                !frameCountElement.TryGetInt32(out var frameCount) || frameCount < 0 ||
                !parsed.TryAdd(id, new RunSessionInventory(type, frameCount)))
            {
                return false;
            }
        }

        if (parsed.Count != expected.Count)
            return false;

        inventory = parsed;
        return true;
    }

    private static IReadOnlyDictionary<string, SessionInventory> ReadSessionInventory(string root)
    {
        var result = new Dictionary<string, SessionInventory>(StringComparer.Ordinal);
        if (!Directory.Exists(root))
            return result;

        IEnumerable<string> directories;
        try
        {
            directories = Directory.EnumerateDirectories(root).ToArray();
        }
        catch (Exception)
        {
            return result;
        }

        foreach (var directory in directories)
        {
            var metadataPath = Path.Combine(directory, "session.json");
            if (!File.Exists(metadataPath))
                continue;

            try
            {
                using var document = JsonDocument.Parse(File.ReadAllText(metadataPath));
                var metadata = document.RootElement;
                if (!TryReadString(metadata, "SessionType", out var type) ||
                    !metadata.TryGetProperty("FrameCount", out var framesElement) ||
                    !framesElement.TryGetInt32(out var frames) || frames < 0)
                {
                    continue;
                }

                result[Path.GetFileName(directory)] = new SessionInventory(type, frames);
            }
            catch (Exception)
            {
                // A moved/truncated recording is represented as a missing resolved session on the run.
            }
        }

        return result;
    }

    private static bool TryReadString(JsonElement root, string name, out string value)
    {
        value = "";
        if (!root.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.String)
            return false;

        value = element.GetString() ?? "";
        return !string.IsNullOrWhiteSpace(value);
    }

    private static bool TryReadStringArray(
        JsonElement root,
        string name,
        out IReadOnlyList<string> values)
    {
        values = [];
        if (!root.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Array)
            return false;

        var parsed = new List<string>();
        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(item.GetString()))
                return false;
            parsed.Add(item.GetString()!);
        }

        values = parsed.AsReadOnly();
        return true;
    }

    private static DateTimeOffset? ParseTimestamp(string? text) =>
        DateTimeOffset.TryParse(
            text,
            CultureInfo.InvariantCulture,
            DateTimeStyles.AllowWhiteSpaces | DateTimeStyles.AssumeUniversal |
            DateTimeStyles.AdjustToUniversal,
            out var value)
            ? value
            : null;

    private static void AppendDirectoryFiles(
        StringBuilder builder,
        string root,
        IReadOnlyList<string> fileNames)
    {
        if (!Directory.Exists(root))
        {
            builder.Append('|').Append(root).Append(":missing");
            return;
        }

        try
        {
            foreach (var directory in Directory.EnumerateDirectories(root)
                         .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            {
                builder.Append('|').Append(Path.GetFileName(directory));
                foreach (var fileName in fileNames)
                {
                    var file = new FileInfo(Path.Combine(directory, fileName));
                    builder.Append(':').Append(fileName).Append('=');
                    if (file.Exists)
                        builder.Append(file.Length).Append('@').Append(file.LastWriteTimeUtc.Ticks);
                    else
                        builder.Append("missing");
                }
            }
        }
        catch (Exception)
        {
            builder.Append('|').Append(root).Append(":unreadable");
        }
    }
}
