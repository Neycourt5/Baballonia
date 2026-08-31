using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Baballonia.Services.Personalization;

/// <summary>Health of one prerequisite, phrased for a user rather than a developer.</summary>
/// <param name="Ready">Whether this prerequisite is satisfied.</param>
/// <param name="Summary">Short friendly line, e.g. "Training tools: Ready".</param>
/// <param name="Detail">Optional technical specifics for the Advanced section and logs.</param>
public sealed record SetupItem(bool Ready, string Summary, string? Detail = null);

/// <summary>How many sessions of each type have been recorded, and whether that is enough.</summary>
public sealed record DatasetStatus(
    int NeutralSessions,
    int SpeechSessions,
    int GuidedSessions,
    int TotalFrames,
    int CorrectionSessions = 0)
{
    /// <summary>
    /// Two of a type is the real threshold, because validation holds out a whole session: with only
    /// one, the trainer has nothing honest to measure against and refuses to make a claim.
    /// </summary>
    public const int RecommendedPerType = 2;

    /// <summary>
    /// Corrections are excluded on purpose. They are a handful of seconds each and supervise one
    /// expression, so counting them here would let "you have enough recordings to train" become true
    /// on evidence that cannot teach the model what a resting face looks like.
    /// </summary>
    public int TotalSessions => NeutralSessions + SpeechSessions + GuidedSessions;

    /// <summary>Enough to train something meaningful at all.</summary>
    public bool CanTrain => NeutralSessions >= 1 && TotalSessions >= 2;

    /// <summary>Enough for the results to be trustworthy.</summary>
    public bool IsRecommended =>
        NeutralSessions >= RecommendedPerType && SpeechSessions >= RecommendedPerType;

    /// <summary>What to record next, or null when the recommended set is complete.</summary>
    public string? NextRecommendation
    {
        get
        {
            if (NeutralSessions < RecommendedPerType)
            {
                var need = RecommendedPerType - NeutralSessions;
                return $"{need} more Neutral recording{(need == 1 ? "" : "s")} recommended.";
            }

            if (SpeechSessions < RecommendedPerType)
            {
                var need = RecommendedPerType - SpeechSessions;
                return $"{need} more Speech recording{(need == 1 ? "" : "s")} recommended.";
            }

            return null;
        }
    }
}

/// <summary>One session as the trainer will see it, without loading camera pixels into memory.</summary>
public sealed record TrainingDatasetSession(
    string Id,
    SessionType Type,
    int Frames,
    IReadOnlySet<int> GuidedPositiveDims,
    IReadOnlySet<int> CorrectedDims);

/// <summary>Counts for one evidence type, split exactly like <c>dataset.split_sessions</c>.</summary>
public sealed record TrainingDatasetTypeSummary(
    SessionType Type,
    int DiscoveredSessions,
    int DiscoveredFrames,
    int TrainingSessions,
    int TrainingFrames,
    int HeldOutSessions,
    int HeldOutFrames);

/// <summary>
/// A cheap, read-only preview of the cumulative corpus and the default session-level split.
/// This is intentionally a C# mirror of Python's <c>split_sessions</c>: sorted chronologically,
/// hold out the newest session of each type only when that type has at least two sessions.
/// Tests on both sides pin the rule so the normal UI can say what will actually be optimized.
/// </summary>
public sealed record TrainingDatasetInventory(
    IReadOnlyList<TrainingDatasetSession> Discovered,
    IReadOnlyList<TrainingDatasetSession> Training,
    IReadOnlyList<TrainingDatasetSession> HeldOut,
    IReadOnlyList<TrainingDatasetTypeSummary> ByType,
    IReadOnlyList<string> GuidedTrainingExpressions,
    IReadOnlyDictionary<string, int> CorrectionCounts,
    string? LatestGuidedSessionId,
    bool LatestGuidedIsTraining)
{
    public int DiscoveredFrames => Discovered.Sum(x => x.Frames);
    public int TrainingFrames => Training.Sum(x => x.Frames);
    public int HeldOutFrames => HeldOut.Sum(x => x.Frames);
}

/// <summary>Coverage of model C's visual-feature sidecars across recorded sessions.</summary>
public sealed record EmbeddingDatasetStatus(int Sessions, int ReadySessions, string Summary)
{
    public bool Ready => Sessions > 0 && ReadySessions == Sessions;
    public int MissingSessions => Math.Max(0, Sessions - ReadySessions);
}

/// <summary>Everything the Personalization page needs to describe the current state.</summary>
public sealed record PersonalizationSetup(
    SetupItem Camera,
    SetupItem Dataset,
    SetupItem TrainingTools,
    SetupItem Model,
    DatasetStatus DatasetStatus,
    string? PythonPath,
    string? TrainingRoot)
{
    /// <summary>Everything needed to press Train.</summary>
    public bool CanTrain => TrainingTools.Ready && DatasetStatus.CanTrain;
}

/// <summary>
/// Locates the training tooling and reports what is and is not ready.
///
/// Pure detection - nothing here creates, installs or modifies anything, so it is safe to call
/// whenever the page refreshes. Creating the environment is <see cref="PersonalTrainingService"/>'s
/// job.
/// </summary>
public sealed class PersonalizationEnvironment
{
    /// <summary>Recommended venv location: outside the repo, and outside OneDrive.</summary>
    public static string DefaultVenvDirectory => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "babble-train-venv");

    /// <summary>Where trained runs are kept. Beside the app's other data, never inside the repo.</summary>
    public static string TrainingRunsDirectory =>
        Path.Combine(Utils.PersistentDataDirectory, "PersonalTraining");

    /// <summary>Python executable inside a venv, per-platform.</summary>
    public static string VenvPython(string venvDirectory) =>
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? Path.Combine(venvDirectory, "Scripts", "python.exe")
            : Path.Combine(venvDirectory, "bin", "python");

    /// <summary>
    /// Finds the `training/` package by walking up from the app directory.
    ///
    /// The app normally runs out of bin/Debug|Release inside the repo, so the package sits a few
    /// levels up. A published build placed elsewhere will not find it, which is reported honestly
    /// rather than guessed at.
    /// </summary>
    public static string? FindTrainingRoot(string? startDirectory = null)
    {
        var directory = new DirectoryInfo(startDirectory ?? AppContext.BaseDirectory);

        for (var depth = 0; directory != null && depth < 8; depth++, directory = directory.Parent)
        {
            var candidate = Path.Combine(directory.FullName, "training");
            if (Directory.Exists(Path.Combine(candidate, "babble_personal")))
                return candidate;
        }

        return null;
    }

    /// <summary>Counts recorded sessions by type. Missing/partial sessions are simply not counted.</summary>
    public static DatasetStatus InspectDataset(string? datasetRoot = null)
    {
        var root = datasetRoot ?? PersonalizationPaths.DatasetRoot;
        if (!Directory.Exists(root))
            return new DatasetStatus(0, 0, 0, 0);

        int neutral = 0, speech = 0, guided = 0, correction = 0, frames = 0;

        foreach (var directory in Directory.EnumerateDirectories(root))
        {
            var metadataPath = Path.Combine(directory, "session.json");
            if (!File.Exists(metadataPath))
                continue;

            // The folder name already encodes the type (…_neutral); reading it avoids parsing JSON
            // for what is only a checklist count, and tolerates a truncated metadata file.
            var name = Path.GetFileName(directory).ToLowerInvariant();
            if (name.EndsWith("_neutral", StringComparison.Ordinal)) neutral++;
            else if (name.EndsWith("_speech", StringComparison.Ordinal)) speech++;
            else if (name.EndsWith("_guided", StringComparison.Ordinal)) guided++;
            else if (name.EndsWith("_correction", StringComparison.Ordinal)) correction++;
            else continue;

            var framesDirectory = Path.Combine(directory, "frames");
            if (Directory.Exists(framesDirectory))
                frames += Directory.EnumerateFiles(framesDirectory, "*.jpg").Count();
        }

        return new DatasetStatus(neutral, speech, guided, frames, correction);
    }

    /// <summary>
    /// Inspects every cumulative session and previews the trainer's default split. No recording is
    /// modified, excluded, or moved. A malformed session is skipped here just as it cannot be
    /// meaningfully described in the UI; the Python loader still remains the final validation gate.
    /// </summary>
    public static TrainingDatasetInventory InspectTrainingInventory(string? datasetRoot = null)
    {
        var root = datasetRoot ?? PersonalizationPaths.DatasetRoot;
        if (!Directory.Exists(root))
            return EmptyInventory();

        var sessions = new List<TrainingDatasetSession>();
        foreach (var directory in Directory.EnumerateDirectories(root).OrderBy(Path.GetFileName,
                     StringComparer.Ordinal))
        {
            var metadataPath = Path.Combine(directory, "session.json");
            if (!File.Exists(metadataPath) || !TryReadSessionType(directory, metadataPath, out var type))
                continue;

            var frames = CountUsableFrames(directory);
            var guidedDims = type == SessionType.Guided
                ? ReadGuidedPositiveDims(directory)
                : new HashSet<int>();
            var correctedDims = type == SessionType.Correction
                ? ReadCorrectionDims(directory)
                : new HashSet<int>();

            sessions.Add(new TrainingDatasetSession(
                Path.GetFileName(directory), type, frames, guidedDims, correctedDims));
        }

        if (sessions.Count == 0)
            return EmptyInventory();

        var heldOutIds = sessions
            .GroupBy(x => x.Type)
            .Where(group => group.Count() > 1)
            .Select(group => group.Last().Id)
            .ToHashSet(StringComparer.Ordinal);
        var training = sessions.Where(x => !heldOutIds.Contains(x.Id)).ToArray();
        var heldOut = sessions.Where(x => heldOutIds.Contains(x.Id)).ToArray();

        var byType = Enum.GetValues<SessionType>()
            .Select(type =>
            {
                var discovered = sessions.Where(x => x.Type == type).ToArray();
                var train = training.Where(x => x.Type == type).ToArray();
                var val = heldOut.Where(x => x.Type == type).ToArray();
                return new TrainingDatasetTypeSummary(
                    type,
                    discovered.Length, discovered.Sum(x => x.Frames),
                    train.Length, train.Sum(x => x.Frames),
                    val.Length, val.Sum(x => x.Frames));
            })
            .ToArray();

        var guidedTrainingExpressions = training
            .Where(x => x.Type == SessionType.Guided)
            .SelectMany(x => x.GuidedPositiveDims)
            .Where(dim => dim >= 0 && dim < PersonalizationSchema.ExpressionCount)
            .Distinct()
            .Order()
            .Select(dim => PersonalizationSchema.ExpressionNames[dim])
            .ToArray();
        var correctionCounts = sessions
            .Where(x => x.Type == SessionType.Correction)
            .SelectMany(x => x.CorrectedDims)
            .Where(dim => dim >= 0 && dim < PersonalizationSchema.ExpressionCount)
            .GroupBy(dim => PersonalizationSchema.ExpressionNames[dim])
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal);
        var latestGuided = sessions.LastOrDefault(x => x.Type == SessionType.Guided);

        return new TrainingDatasetInventory(
            sessions, training, heldOut, byType, guidedTrainingExpressions, correctionCounts,
            latestGuided?.Id,
            latestGuided != null && !heldOutIds.Contains(latestGuided.Id));
    }

    private static TrainingDatasetInventory EmptyInventory() => new(
        [], [], [], [], [], new Dictionary<string, int>(), null, false);

    private static bool TryReadSessionType(string directory, string metadataPath, out SessionType type)
    {
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(metadataPath));
            if (document.RootElement.TryGetProperty("SessionType", out var value) &&
                Enum.TryParse(value.GetString(), ignoreCase: true, out type))
                return true;
        }
        catch (Exception)
        {
            // The folder suffix remains a safe fallback for older/truncated recorder metadata.
        }

        var suffix = Path.GetFileName(directory).Split('_').LastOrDefault();
        return Enum.TryParse(suffix, ignoreCase: true, out type);
    }

    private static int CountUsableFrames(string sessionDirectory)
    {
        var labels = Path.Combine(sessionDirectory, "labels.jsonl");
        var frames = Path.Combine(sessionDirectory, "frames");
        if (!File.Exists(labels) || !Directory.Exists(frames))
            return Directory.Exists(frames) ? Directory.EnumerateFiles(frames, "*.jpg").Count() : 0;

        var count = 0;
        foreach (var line in File.ReadLines(labels))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                using var document = JsonDocument.Parse(line.TrimStart('\uFEFF'));
                if (!document.RootElement.TryGetProperty("i", out var index) ||
                    index.ValueKind != JsonValueKind.Number ||
                    !index.TryGetInt32(out var frameIndex) || frameIndex < 0)
                    continue;
                if (File.Exists(Path.Combine(frames, $"{frameIndex:D6}.jpg"))) count++;
            }
            catch (Exception)
            {
                // Inventory is a cleanup aid, not the strict training validator. One scuffed line
                // must not take down the page that gives the user its Show-in-Folder button. The
                // Python loader will report the corrupt line precisely if training is attempted.
            }
        }

        return count;
    }

    private static IReadOnlySet<int> ReadGuidedPositiveDims(string sessionDirectory)
    {
        var result = new HashSet<int>();
        var labels = Path.Combine(sessionDirectory, "labels.jsonl");
        if (!File.Exists(labels)) return result;

        foreach (var line in File.ReadLines(labels))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                using var document = JsonDocument.Parse(line.TrimStart('\uFEFF'));
                if (!document.RootElement.TryGetProperty("cue", out var cue) ||
                    cue.ValueKind != JsonValueKind.Object ||
                    !cue.TryGetProperty("phase", out var phase) ||
                    phase.ValueKind != JsonValueKind.String ||
                    !string.Equals(phase.GetString(), "hold", StringComparison.OrdinalIgnoreCase) ||
                    !cue.TryGetProperty("dims", out var dims) ||
                    dims.ValueKind != JsonValueKind.Array ||
                    !cue.TryGetProperty("target", out var target) ||
                    target.ValueKind != JsonValueKind.Array)
                    continue;

                foreach (var dimElement in dims.EnumerateArray())
                {
                    if (dimElement.ValueKind != JsonValueKind.Number ||
                        !dimElement.TryGetInt32(out var dim) ||
                        dim < 0 || dim >= target.GetArrayLength())
                        continue;
                    var targetElement = target[dim];
                    if (targetElement.ValueKind == JsonValueKind.Number &&
                        targetElement.TryGetSingle(out var targetValue) && targetValue > 0f)
                        result.Add(dim);
                }
            }
            catch (Exception)
            {
                // Keep the inventory usable; training itself remains strict.
            }
        }

        return result;
    }

    private static IReadOnlySet<int> ReadCorrectionDims(string sessionDirectory)
    {
        var result = new HashSet<int>();
        var path = Path.Combine(sessionDirectory, "correction.json");
        if (!File.Exists(path)) return result;

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            var root = document.RootElement;
            if (!root.TryGetProperty("CorrectedDims", out var dims) &&
                !root.TryGetProperty("corrected_dims", out dims))
                return result;
            foreach (var dim in dims.EnumerateArray()) result.Add(dim.GetInt32());
        }
        catch (Exception)
        {
            // A malformed correction is never invented into the summary or treated as all-neutral.
        }

        return result;
    }

    /// <summary>
    /// Verifies that every non-empty recording has the complete visual-feature sidecar consumed by
    /// model C. The optional hash also prevents features from an older derived face graph being
    /// accepted after the stock model changes.
    /// </summary>
    public static EmbeddingDatasetStatus InspectEmbeddingDataset(
        string? datasetRoot = null,
        string? expectedModelMd5 = null)
    {
        var root = datasetRoot ?? PersonalizationPaths.DatasetRoot;
        if (!Directory.Exists(root))
            return new EmbeddingDatasetStatus(0, 0, "No recordings are available to prepare.");

        var sessions = 0;
        var ready = 0;

        foreach (var directory in Directory.EnumerateDirectories(root))
        {
            if (!File.Exists(Path.Combine(directory, "session.json")))
                continue;

            var framesDirectory = Path.Combine(directory, "frames");
            var frameCount = Directory.Exists(framesDirectory)
                ? Directory.EnumerateFiles(framesDirectory, "*.jpg").Count()
                : 0;
            if (frameCount == 0)
                continue;

            sessions++;
            if (EmbeddingSidecarIsComplete(directory, frameCount, expectedModelMd5))
                ready++;
        }

        var summary = sessions == 0
            ? "No recordings are available to prepare."
            : ready == sessions
                ? $"Visual features are ready for all {sessions} recording{(sessions == 1 ? "" : "s")}."
                : $"Generate visual features for {sessions - ready} of {sessions} recordings before training C.";

        return new EmbeddingDatasetStatus(sessions, ready, summary);
    }

    private static bool EmbeddingSidecarIsComplete(
        string sessionDirectory,
        int frameCount,
        string? expectedModelMd5)
    {
        var binary = Path.Combine(sessionDirectory, "embeddings.bin");
        var metadata = Path.Combine(sessionDirectory, "embeddings.json");
        if (!File.Exists(binary) || !File.Exists(metadata))
            return false;

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(metadata));
            var root = document.RootElement;

            if (!root.TryGetProperty("dim", out var dim) || dim.GetInt32() != 1280 ||
                !root.TryGetProperty("count", out var count) || count.GetInt32() != frameCount ||
                !root.TryGetProperty("dtype", out var dtype) || dtype.GetString() != "float16")
                return false;

            if (!string.IsNullOrEmpty(expectedModelMd5) &&
                (!root.TryGetProperty("model_md5", out var modelMd5) ||
                 !string.Equals(modelMd5.GetString(), expectedModelMd5, StringComparison.OrdinalIgnoreCase)))
                return false;

            return new FileInfo(binary).Length == (long)frameCount * 1280 * sizeof(ushort);
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>True when the venv exists and has the packages training needs.</summary>
    public static bool IsTrainingEnvironmentReady(string venvDirectory)
    {
        var python = VenvPython(venvDirectory);
        if (!File.Exists(python))
            return false;

        // Checking for the packages' presence on disk avoids launching Python just to render the
        // page. Layout differs slightly per platform, so accept either.
        var sitePackages = new[]
        {
            Path.Combine(venvDirectory, "Lib", "site-packages"),
            Path.Combine(venvDirectory, "lib"),
        };

        foreach (var root in sitePackages.Where(Directory.Exists))
        {
            if (Directory.Exists(Path.Combine(root, "torch")))
                return true;

            // Unix venvs nest under lib/pythonX.Y/site-packages.
            foreach (var nested in Directory.EnumerateDirectories(root))
            {
                if (Directory.Exists(Path.Combine(nested, "site-packages", "torch")))
                    return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Finds a Python interpreter able to create the venv. Windows' `py` launcher first, since it
    /// resolves versions properly; then the usual names.
    /// </summary>
    public static string? FindHostPython()
    {
        var candidates = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? new[] { "py", "python", "python3" }
            : new[] { "python3", "python" };

        return candidates.FirstOrDefault(CommandExists);
    }

    private static bool CommandExists(string command)
    {
        var pathVariable = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrEmpty(pathVariable))
            return false;

        var extensions = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? new[] { ".exe", ".cmd", ".bat" }
            : new[] { "" };

        return pathVariable
            .Split(Path.PathSeparator)
            .Where(directory => !string.IsNullOrWhiteSpace(directory))
            .Any(directory => extensions.Any(extension =>
            {
                try
                {
                    return File.Exists(Path.Combine(directory, command + extension));
                }
                catch
                {
                    return false; // malformed PATH entries are common and harmless
                }
            }));
    }

    /// <summary>Builds the full status shown on the Personalization page.</summary>
    public PersonalizationSetup Inspect(
        bool cameraRunning,
        PersonalModelManager modelManager,
        string? venvDirectory = null,
        string? datasetRoot = null)
    {
        var venv = venvDirectory ?? DefaultVenvDirectory;
        var trainingRoot = FindTrainingRoot();
        var dataset = InspectDataset(datasetRoot);

        var camera = cameraRunning
            ? new SetupItem(true, "Face camera: Running")
            : new SetupItem(false, "Face camera: Not running",
                "Start the face camera on the Home page. Recording and the live preview need it.");

        var datasetItem = dataset.TotalSessions == 0
            ? new SetupItem(false, "Recordings: None yet",
                $"Sessions are stored in {PersonalizationPaths.DatasetRoot}")
            : new SetupItem(dataset.IsRecommended,
                dataset.IsRecommended
                    ? $"Recordings: {dataset.TotalSessions} sessions"
                    : $"Recordings: {dataset.TotalSessions} sessions ({dataset.NextRecommendation})",
                $"{dataset.NeutralSessions} neutral, {dataset.SpeechSessions} speech, " +
                $"{dataset.GuidedSessions} guided, {dataset.TotalFrames} frames total");

        SetupItem tools;
        string? python = null;

        if (trainingRoot == null)
        {
            tools = new SetupItem(false, "Training tools: Not found",
                "The training scripts could not be located. They live in the 'training' folder of " +
                "the Baballonia source tree; this build appears to be running outside it.");
        }
        else if (!IsTrainingEnvironmentReady(venv))
        {
            var host = FindHostPython();
            tools = host == null
                ? new SetupItem(false, "Training tools: Python not installed",
                    "Python 3 was not found. Install Python 3.11 or newer, then use Set Up " +
                    "Training Tools.")
                : new SetupItem(false, "Training tools: Not set up yet",
                    $"Will be installed to {venv} using '{host}'. This downloads PyTorch " +
                    "(about 200 MB) and runs once.");
        }
        else
        {
            python = VenvPython(venv);
            tools = new SetupItem(true, "Training tools: Ready", $"{venv}");
        }

        var model = BuildModelItem(modelManager);

        return new PersonalizationSetup(camera, datasetItem, tools, model, dataset, python, trainingRoot);
    }

    private static SetupItem BuildModelItem(PersonalModelManager modelManager)
    {
        var result = modelManager.LastResult;

        if (modelManager.IsActive)
        {
            var metadata = modelManager.LoadedMetadata;
            return metadata == null
                ? new SetupItem(true, "Personal model: Active", modelManager.ModelPath)
                : new SetupItem(true, $"Personal model: Active - {metadata.DisplayName}",
                    $"{metadata.AdapterType}, trained {metadata.TrainedUtc ?? "unknown"}");
        }

        if (!modelManager.Enabled)
        {
            return File.Exists(modelManager.ModelPath)
                ? new SetupItem(false, "Personal model: Installed but turned off",
                    modelManager.ModelPath)
                : new SetupItem(false, "Personal model: Not trained yet",
                    "Record some sessions, then train.");
        }

        return new SetupItem(false,
            result is { Success: false } ? "Personal model: Could not be loaded" : "Personal model: Not installed",
            result?.Message ?? modelManager.ModelPath);
    }
}
