using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Baballonia.Services.Personalization;

public enum TrainingStage
{
    Idle,
    SettingUpTools,
    PreparingData,
    Training,
    Evaluating,
    Exporting,
    Installing,
    Done,
    Failed
}

/// <param name="Stage">Coarse step, for the progress line.</param>
/// <param name="Message">Friendly description of what is happening.</param>
public sealed record TrainingProgress(TrainingStage Stage, string Message);

/// <summary>
/// How one watched expression behaved, stock versus personal (summary.json v2 and later).
///
/// A false-activation rate alone cannot describe the user's actual complaint: one jaw that hangs
/// open for two seconds and forty single-frame flickers score the same rate but feel nothing alike.
/// <see cref="LongestFalseRunSeconds"/> is what separates them.
///
/// <see cref="RangeRetention"/> is the guardrail in the other direction. Any model can win every
/// false-positive number by refusing to move the expression at all, which reads as a dead face; a
/// value well below 1 means the improvement was bought by flattening real movement.
/// </summary>
public sealed record WatchedExpressionSummary(
    string Name,
    int ClosedFrames,
    double? StockFalseActivation,
    double? PersonalFalseActivation,
    double? StockLongestFalseRunSeconds,
    double? PersonalLongestFalseRunSeconds,
    double? RangeRetention,
    bool SuppressionWarning);

/// <summary>Measured outcome of a training run, read from the trainer's summary.json.</summary>
/// <remarks>
/// Every field beyond the v1 set is nullable and read defensively, so a summary written by an older
/// trainer still loads and simply reports nothing for the newer metrics.
/// </remarks>
public sealed record TrainingSummary(
    string AdapterType,
    int Parameters,
    bool ValidatedOnHeldOutSessions,
    int ExpressionsImproved,
    int ExpressionsRegressed,
    double? MeanStockMae,
    double? MeanPersonalMae,
    double? NeutralStockFalseActivation,
    double? NeutralPersonalFalseActivation,
    double? NeutralStockJitter,
    double? NeutralPersonalJitter,
    string Verdict,
    IReadOnlyList<string> WorstRegressions,
    int SummaryVersion = 1,
    WatchedExpressionSummary? JawOpen = null,
    WatchedExpressionSummary? TongueOut = null,
    double? CrossTalkStock = null,
    double? CrossTalkPersonal = null,
    int HardExampleFrames = 0);

/// <param name="Success">Whether a model was trained, exported and installed.</param>
/// <param name="Message">Friendly headline, safe to show directly.</param>
/// <param name="Remedy">Suggested action when something went wrong, e.g. "Repair Training Tools".</param>
public sealed record TrainingResult(
    bool Success,
    string Message,
    TrainingSummary? Summary = null,
    string? Remedy = null);

/// <summary>
/// Drives the existing Python tooling end to end so the user does not have to.
///
/// This deliberately orchestrates rather than reimplements: training, evaluation and ONNX export all
/// stay in <c>training/babble_personal</c>, which remains fully usable from a terminal. All this
/// adds is running those steps in order, moving the result into place, and translating failures into
/// something actionable.
/// </summary>
public sealed class PersonalTrainingService(
    PersonalModelManager modelManager,
    ILogger<PersonalTrainingService> logger)
{
    private readonly List<string> _detailLog = [];
    private readonly object _logLock = new();
    private int _running;

    /// <summary>True while a setup or training run is in progress.</summary>
    public bool IsBusy => Volatile.Read(ref _running) != 0;

    /// <summary>Full captured output of the last operation, for the Show Details view and bug reports.</summary>
    public string DetailLog
    {
        get { lock (_logLock) return string.Join(Environment.NewLine, _detailLog); }
    }

    private void ClearLog()
    {
        lock (_logLock) _detailLog.Clear();
    }

    private void AppendLog(string line)
    {
        lock (_logLock)
        {
            _detailLog.Add(line);
            // Keep memory bounded on a long pip install; the tail is what matters when diagnosing.
            if (_detailLog.Count > 4000)
                _detailLog.RemoveRange(0, 1000);
        }
    }

    /// <summary>
    /// Creates the training virtualenv and installs the pinned requirements.
    ///
    /// Runs once, takes a few minutes, and downloads roughly 200 MB of PyTorch. Everything lands in
    /// <see cref="PersonalizationEnvironment.DefaultVenvDirectory"/>, outside the repository, so a
    /// synced Documents folder never sees it.
    /// </summary>
    public async Task<TrainingResult> SetUpTrainingToolsAsync(
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (Interlocked.CompareExchange(ref _running, 1, 0) != 0)
            return new TrainingResult(false, "Something is already running. Wait for it to finish.");

        try
        {
            ClearLog();

            var trainingRoot = PersonalizationEnvironment.FindTrainingRoot();
            if (trainingRoot == null)
            {
                return new TrainingResult(false,
                    "The training scripts could not be found. They live in the 'training' folder of " +
                    "the Baballonia source tree, and this build appears to be running outside it.");
            }

            var hostPython = PersonalizationEnvironment.FindHostPython();
            if (hostPython == null)
            {
                return new TrainingResult(false,
                    "Python 3 was not found on this computer. Install Python 3.11 or newer, then try again.",
                    Remedy: "Install Python");
            }

            var venv = PersonalizationEnvironment.DefaultVenvDirectory;
            var venvPython = PersonalizationEnvironment.VenvPython(venv);

            if (!File.Exists(venvPython))
            {
                progress?.Report(new TrainingProgress(TrainingStage.SettingUpTools,
                    "Creating the training environment..."));

                var create = await RunAsync(hostPython, ["-m", "venv", venv], trainingRoot, cancellationToken);
                if (!create.Success || !File.Exists(venvPython))
                {
                    return new TrainingResult(false,
                        "Could not create the Python environment. See details for the exact error.",
                        Remedy: "Show Details");
                }
            }

            progress?.Report(new TrainingProgress(TrainingStage.SettingUpTools,
                "Downloading PyTorch (about 200 MB, this happens once)..."));

            // The CPU wheel index matters: the default PyPI torch pulls a multi-gigabyte CUDA build.
            var torch = await RunAsync(venvPython,
                ["-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.7.1"],
                trainingRoot, cancellationToken);

            if (!torch.Success)
            {
                return new TrainingResult(false,
                    "Could not download PyTorch. Check your internet connection and try again.",
                    Remedy: "Show Details");
            }

            progress?.Report(new TrainingProgress(TrainingStage.SettingUpTools,
                "Installing the remaining training packages..."));

            var requirements = Path.Combine(trainingRoot, "requirements.txt");
            var rest = await RunAsync(venvPython,
                ["-m", "pip", "install", "-r", requirements], trainingRoot, cancellationToken);

            if (!rest.Success)
            {
                return new TrainingResult(false,
                    "Could not install the training packages. See details for the exact error.",
                    Remedy: "Show Details");
            }

            progress?.Report(new TrainingProgress(TrainingStage.Done, "Training tools are ready."));
            return new TrainingResult(true, "Training tools are ready.");
        }
        catch (OperationCanceledException)
        {
            return new TrainingResult(false, "Setup was cancelled.");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Personalization: training tool setup failed");
            AppendLog(ex.ToString());
            return new TrainingResult(false, $"Setup failed: {ex.Message}", Remedy: "Show Details");
        }
        finally
        {
            Interlocked.Exchange(ref _running, 0);
        }
    }

    /// <summary>
    /// Trains, exports, installs and activates a personal model.
    ///
    /// Defaults are chosen for someone who has not read the plan: the output-only adapter (cheapest,
    /// and the honest baseline), the standard dataset location, and the trainer's own defaults for
    /// epochs, shrinkage and the session-level split.
    /// </summary>
    public async Task<TrainingResult> TrainAsync(
        string modelKind = "a",
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (Interlocked.CompareExchange(ref _running, 1, 0) != 0)
            return new TrainingResult(false, "Training is already running.");

        try
        {
            ClearLog();

            var trainingRoot = PersonalizationEnvironment.FindTrainingRoot();
            if (trainingRoot == null)
            {
                return new TrainingResult(false,
                    "The training scripts could not be found. They live in the 'training' folder of " +
                    "the Baballonia source tree.");
            }

            var venv = PersonalizationEnvironment.DefaultVenvDirectory;
            if (!PersonalizationEnvironment.IsTrainingEnvironmentReady(venv))
            {
                return new TrainingResult(false,
                    "The Python training environment is missing or incomplete.",
                    Remedy: "Set Up Training Tools");
            }

            var dataset = PersonalizationEnvironment.InspectDataset();
            if (!dataset.CanTrain)
            {
                return new TrainingResult(false,
                    "There are not enough recordings yet. Record at least one Neutral session and " +
                    "one more session of any type.",
                    Remedy: "Record");
            }

            var python = PersonalizationEnvironment.VenvPython(venv);
            var runsDirectory = PersonalizationEnvironment.TrainingRunsDirectory;
            Directory.CreateDirectory(runsDirectory);

            // --- train -------------------------------------------------------------------------
            progress?.Report(new TrainingProgress(TrainingStage.PreparingData, "Preparing your recordings..."));

            var training = await RunAsync(python,
                [
                    "-m", "babble_personal.train",
                    "--data", PersonalizationPaths.DatasetRoot,
                    "--model", modelKind,
                    "--out", runsDirectory
                ],
                trainingRoot, cancellationToken,
                onLine: line => ReportStage(line, progress));

            if (!training.Success)
                return new TrainingResult(false, DescribeTrainingFailure(training.Output), Remedy: "Show Details");

            var runDirectory = FindNewestRun(runsDirectory);
            if (runDirectory == null)
            {
                return new TrainingResult(false,
                    "Training finished but produced no output folder. See details.",
                    Remedy: "Show Details");
            }

            var checkpoint = Path.Combine(runDirectory, "model.pt");
            if (!File.Exists(checkpoint))
            {
                return new TrainingResult(false,
                    "Training finished but no model file was produced. See details.",
                    Remedy: "Show Details");
            }

            // --- export ------------------------------------------------------------------------
            progress?.Report(new TrainingProgress(TrainingStage.Exporting, "Exporting the model..."));

            var stockModel = Path.Combine(AppContext.BaseDirectory, "faceModel.onnx");
            string[] exportArgs =
            [
                "-m", "babble_personal.export",
                "--checkpoint", checkpoint,
                .. File.Exists(stockModel) ? new[] { "--base-model", stockModel } : []
            ];

            var export = await RunAsync(python, exportArgs, trainingRoot, cancellationToken);
            if (!export.Success)
            {
                return new TrainingResult(false,
                    "The model trained successfully but could not be exported. See details.",
                    Remedy: "Show Details");
            }

            var exported = Path.Combine(runDirectory, "personalFaceModel.onnx");
            if (!File.Exists(exported))
            {
                return new TrainingResult(false,
                    "Export reported success but produced no model file. See details.",
                    Remedy: "Show Details");
            }

            // --- install -----------------------------------------------------------------------
            progress?.Report(new TrainingProgress(TrainingStage.Installing, "Installing the model..."));

            var summary = ReadSummary(runDirectory);
            var installed = InstallModel(exported);
            if (installed != null)
                return new TrainingResult(false, installed, summary, Remedy: "Show Details");

            modelManager.SetEnabled(true);
            var reload = await modelManager.ReloadAsync();

            if (!reload.Success)
            {
                return new TrainingResult(false,
                    $"The model was trained but could not be loaded: {reload.Message}",
                    summary, Remedy: "Show Details");
            }

            progress?.Report(new TrainingProgress(TrainingStage.Done, "Your personal model is ready."));
            return new TrainingResult(true, "Your personal model is ready.", summary);
        }
        catch (OperationCanceledException)
        {
            return new TrainingResult(false, "Training was cancelled.");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Personalization: training failed");
            AppendLog(ex.ToString());
            return new TrainingResult(false, $"Training failed: {ex.Message}", Remedy: "Show Details");
        }
        finally
        {
            Interlocked.Exchange(ref _running, 0);
        }
    }

    /// <summary>
    /// Copies the exported model into the app's Models folder, keeping one backup.
    ///
    /// The live model must be unloaded first: an active InferenceSession holds the file open on
    /// Windows, so overwriting it would fail with a sharing violation.
    /// </summary>
    private string? InstallModel(string exportedPath)
    {
        try
        {
            var destination = PersonalizationPaths.DefaultPersonalModelPath;
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);

            modelManager.Uninstall();

            if (File.Exists(destination))
            {
                var backup = Path.ChangeExtension(destination, ".previous.onnx");
                File.Copy(destination, backup, overwrite: true);
            }

            File.Copy(exportedPath, destination, overwrite: true);
            return null;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Personalization: could not install personal model");
            AppendLog(ex.ToString());
            return $"Could not install the model file: {ex.Message}";
        }
    }

    private static string? FindNewestRun(string runsDirectory) =>
        !Directory.Exists(runsDirectory)
            ? null
            : new DirectoryInfo(runsDirectory)
                .EnumerateDirectories()
                .OrderByDescending(d => d.LastWriteTimeUtc)
                .FirstOrDefault()?.FullName;

    private TrainingSummary? ReadSummary(string runDirectory)
    {
        var path = Path.Combine(runDirectory, "summary.json");
        if (!File.Exists(path))
            return null;

        try
        {
            return TrainingSummaryReader.Parse(File.ReadAllText(path));
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Personalization: could not read training summary");
            return null;
        }
    }

    private static void ReportStage(string line, IProgress<TrainingProgress>? progress)
    {
        if (progress == null || !line.StartsWith("[stage] ", StringComparison.Ordinal))
            return;

        var stage = line["[stage] ".Length..].Trim();
        var mapped = stage switch
        {
            "preparing" => new TrainingProgress(TrainingStage.PreparingData, "Preparing your recordings..."),
            "training" => new TrainingProgress(TrainingStage.Training, "Learning your face..."),
            "evaluating" => new TrainingProgress(TrainingStage.Evaluating, "Checking the results..."),
            _ => null
        };

        if (mapped != null)
            progress.Report(mapped);
    }

    /// <summary>Turns the trainer's own error text into something the user can act on.</summary>
    private static string DescribeTrainingFailure(string output)
    {
        if (output.Contains("no supervised cells", StringComparison.OrdinalIgnoreCase))
        {
            return "There was nothing to learn from. Record a Neutral session - those frames are " +
                   "what teach the model your resting face.";
        }

        if (output.Contains("Every session was assigned to validation", StringComparison.OrdinalIgnoreCase))
            return "Record at least two sessions of each type so one can be held back for testing.";

        if (output.Contains("schema mismatch", StringComparison.OrdinalIgnoreCase))
        {
            return "Some recordings were made with a different version of Baballonia and cannot be " +
                   "mixed with the newer ones. Move the older session folders aside and try again.";
        }

        if (output.Contains("No sessions found", StringComparison.OrdinalIgnoreCase))
            return "No recordings were found. Record a Neutral session first.";

        if (output.Contains("ModuleNotFoundError", StringComparison.OrdinalIgnoreCase) ||
            output.Contains("ImportError", StringComparison.OrdinalIgnoreCase))
        {
            return "The Python training environment is incomplete.";
        }

        if (output.Contains("MemoryError", StringComparison.OrdinalIgnoreCase) ||
            output.Contains("out of memory", StringComparison.OrdinalIgnoreCase))
        {
            return "Training ran out of memory. Try training with fewer recordings, or use the " +
                   "output-only model.";
        }

        return "Training did not finish. See details for the exact error.";
    }

    private sealed record ProcessOutcome(bool Success, int ExitCode, string Output);

    /// <summary>
    /// Runs a child process, streaming both streams into the detail log.
    ///
    /// stderr is captured too and treated as ordinary output: pip and PyTorch write progress there
    /// routinely, so treating it as failure would produce constant false alarms. Success is decided
    /// by the exit code alone.
    /// </summary>
    private async Task<ProcessOutcome> RunAsync(
        string fileName,
        IReadOnlyList<string> arguments,
        string workingDirectory,
        CancellationToken cancellationToken,
        Action<string>? onLine = null)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = fileName,
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        foreach (var argument in arguments)
            startInfo.ArgumentList.Add(argument);

        // Force UTF-8 and unbuffered output so progress lines arrive as they happen.
        startInfo.Environment["PYTHONIOENCODING"] = "utf-8";
        startInfo.Environment["PYTHONUNBUFFERED"] = "1";

        AppendLog($"$ {fileName} {string.Join(' ', arguments)}");
        logger.LogInformation("Personalization: running {File} {Args}", fileName, string.Join(' ', arguments));

        using var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        var output = new StringBuilder();

        void Handle(string? line)
        {
            if (line == null)
                return;

            output.AppendLine(line);
            AppendLog(line);
            onLine?.Invoke(line);
        }

        process.OutputDataReceived += (_, e) => Handle(e.Data);
        process.ErrorDataReceived += (_, e) => Handle(e.Data);

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        try
        {
            await process.WaitForExitAsync(cancellationToken);
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); } catch { /* already gone */ }
            throw;
        }

        AppendLog($"[exit code {process.ExitCode}]");
        return new ProcessOutcome(process.ExitCode == 0, process.ExitCode, output.ToString());
    }
}
