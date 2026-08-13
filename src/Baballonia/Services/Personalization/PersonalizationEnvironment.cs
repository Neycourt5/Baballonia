using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

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
    int TotalFrames)
{
    /// <summary>
    /// Two of a type is the real threshold, because validation holds out a whole session: with only
    /// one, the trainer has nothing honest to measure against and refuses to make a claim.
    /// </summary>
    public const int RecommendedPerType = 2;

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

        int neutral = 0, speech = 0, guided = 0, frames = 0;

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
            else continue;

            var framesDirectory = Path.Combine(directory, "frames");
            if (Directory.Exists(framesDirectory))
                frames += Directory.EnumerateFiles(framesDirectory, "*.jpg").Count();
        }

        return new DatasetStatus(neutral, speech, guided, frames);
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
            return new SetupItem(true, "Personal model: Active",
                metadata == null
                    ? modelManager.ModelPath
                    : $"{metadata.AdapterType}, trained {metadata.TrainedUtc ?? "unknown"}");
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
