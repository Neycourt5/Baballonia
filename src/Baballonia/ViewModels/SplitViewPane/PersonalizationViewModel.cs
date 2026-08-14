using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using Baballonia.Contracts;
using Baballonia.Models;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.Audio;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using OpenCvSharp;

namespace Baballonia.ViewModels.SplitViewPane;

/// <summary>
/// The Personalization workspace: record, train, compare.
///
/// The whole point of this page is that personalizing tracking should not require knowing what a
/// venv, a checkpoint or an ONNX file is. It orchestrates the existing tooling rather than
/// duplicating it - the Python CLI stays fully usable for anyone who wants it.
/// </summary>
public partial class PersonalizationViewModel : ViewModelBase, IDisposable
{
    private readonly DatasetRecorderService _recorder;
    private readonly PersonalModelManager _modelManager;
    private readonly PersonalTrainingService _trainingService;
    private readonly PersonalizationEnvironment _environment;
    private readonly ILocalSettingsService _settings;
    private readonly IFacePipelineEventBus _faceEventBus;
    private readonly ILogger<PersonalizationViewModel> _logger;

    /// <summary>Optional so the view model still constructs in tests that do not care about it.</summary>
    private readonly HardExampleService? _hardExamples;
    private readonly GuidedCalibrationService? _guided;
    private readonly AudioAssistService? _audio;

    /// <summary>
    /// Ticks the cue engine while a guided session runs. Faster than the status timer because it
    /// also drives the override's keep-alive, and because a 0.75 s transition commanded on a 250 ms
    /// timer would land visibly late.
    /// </summary>
    private readonly DispatcherTimer _guidedTimer;

    private readonly Action<FacePipelineEvents.NewTransformedFrameEvent> _frameHandler;
    private readonly Action<FacePipelineEvents.NewRawExpressionsEvent> _rawHandler;
    private readonly Action<FacePipelineEvents.NewCorrectedExpressionsEvent> _correctedHandler;
    private readonly DispatcherTimer _statusTimer;

    private WriteableBitmap? _backingBitmap;
    private CancellationTokenSource? _workCancellation;

    // Latest values, written on the processing tick and drained by the timer. Binding at the tick
    // rate would swamp the UI; a few times a second is plenty to watch values move.
    private readonly float[] _latestStock = new float[PersonalizationSchema.ExpressionCount];
    private readonly float[] _latestPersonal = new float[PersonalizationSchema.ExpressionCount];
    private volatile bool _hasCorrectedValues;
    private volatile bool _cameraSeenRecently;
    private DateTime _lastFrameUtc = DateTime.MinValue;

    // ---- live preview -------------------------------------------------------------------------
    [ObservableProperty] private WriteableBitmap? _preview;

    // ---- setup --------------------------------------------------------------------------------
    [ObservableProperty] private string _cameraStatus = "Face camera: checking...";
    [ObservableProperty] private bool _cameraReady;
    [ObservableProperty] private string _toolsStatus = "Training tools: checking...";
    [ObservableProperty] private bool _toolsReady;
    [ObservableProperty] private string _modelStatus = "Personal model: checking...";
    [ObservableProperty] private bool _modelReady;
    [ObservableProperty] private bool _canSetUpTools;

    // ---- recordings ---------------------------------------------------------------------------
    [ObservableProperty] private int _neutralCount;
    [ObservableProperty] private int _speechCount;
    [ObservableProperty] private string _recordingAdvice = "";
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private string _recordingStatus = "";
    [ObservableProperty] private string _notes = "";

    // ---- audio expression assist ----------------------------------------------------------------

    /// <summary>
    /// Whether microphone loudness may amplify mouth movement the tracker already sees.
    /// </summary>
    /// <remarks>
    /// Off by default and strictly additive. The camera remains the only thing that decides what the
    /// face is doing - audio can make a real movement bigger, never create one - so the worst case
    /// for a user who leaves this off, or has no microphone, is exactly the tracking they had.
    /// </remarks>
    [ObservableProperty] private bool _audioAssistEnabled;

    [ObservableProperty] private double _audioStrength = 50;
    [ObservableProperty] private string _audioStatus = "";
    [ObservableProperty] private string _audioDiagnostics = "";

    // ---- guided calibration -------------------------------------------------------------------

    [ObservableProperty] private bool _isGuidedRunning;
    [ObservableProperty] private string _guidedInstruction = "";
    [ObservableProperty] private double _guidedProgress;
    [ObservableProperty] private string _guidedStatus = "";
    [ObservableProperty] private bool _canStartGuided;

    // ---- quick correction ---------------------------------------------------------------------

    /// <summary>
    /// How far back "my mouth was closed" reaches. Five seconds by default: a mistake is noticed a
    /// beat after it happens, and the window has to cover both the noticing and the reaching.
    /// </summary>
    public IReadOnlyList<string> CorrectionWindows { get; } = ["Last 2 seconds", "Last 5 seconds", "Last 10 seconds"];

    [ObservableProperty] private int _selectedCorrectionWindowIndex = 1;
    [ObservableProperty] private string _correctionStatus = "";
    [ObservableProperty] private int _savedCorrectionCount;
    [ObservableProperty] private bool _canFlagCorrection;

    private TimeSpan SelectedCorrectionWindow =>
        HardExampleService.WindowChoices[
            Math.Clamp(SelectedCorrectionWindowIndex, 0, HardExampleService.WindowChoices.Count - 1)];

    // ---- training -----------------------------------------------------------------------------

    /// <summary>
    /// Which adapter to train. 0 = model A (output-only), 1 = model B (image-conditioned).
    ///
    /// A is the default because it is the honest baseline: it sees only the stock 45 values, so it
    /// can fix systematic bias and cross-talk but nothing that needs to look at the face. B also
    /// sees the frame, which is the only way to fix errors where the stock outputs are ambiguous -
    /// at the cost of a slower, more memory-hungry training run and more capacity to overfit. Train
    /// both and keep whichever wins on your own held-out recordings.
    /// </summary>
    [ObservableProperty] private int _selectedModelIndex;

    [ObservableProperty] private string _modelChoiceDescription = OutputOnlyDescription;

    private const string OutputOnlyDescription =
        "Learns from the 45 expression values only. Fast to train, small, and hard to overfit - " +
        "it is very good at removing a constant bias, but it never sees your face.";

    private const string ImageConditionedDescription =
        "Also looks at the camera frame, so it can correct errors the 45 values alone cannot " +
        "explain. Training takes longer and needs a few GB of RAM. Experimental: check the result " +
        "against model A before trusting it.";

    /// <summary>The trainer's --model flag for the current selection.</summary>
    private string SelectedModelKind => TrainingModelChoice.KindForIndex(SelectedModelIndex);

    partial void OnSelectedModelIndexChanged(int value) =>
        ModelChoiceDescription = value == TrainingModelChoice.ImageConditionedIndex
            ? ImageConditionedDescription
            : OutputOnlyDescription;

    [ObservableProperty] private bool _isBusy;
    [ObservableProperty] private string _busyMessage = "";
    [ObservableProperty] private bool _canTrain;
    [ObservableProperty] private string _resultHeadline = "";
    [ObservableProperty] private string _resultDetail = "";
    [ObservableProperty] private bool _hasResult;
    [ObservableProperty] private string? _remedyAction;
    [ObservableProperty] private bool _showDetails;
    [ObservableProperty] private string _detailLog = "";

    // ---- comparison ---------------------------------------------------------------------------

    /// <summary>
    /// Which adapter is loaded right now, read from the installed model's own metadata. The Train
    /// dropdown cannot answer this: it holds the choice for the *next* run, so after switching it
    /// without retraining the two would disagree. This is the authoritative one.
    /// </summary>
    [ObservableProperty] private string _activeModelName = "";

    [ObservableProperty] private bool _personalModelEnabled;
    [ObservableProperty] private double _personalStrength = 100;
    [ObservableProperty] private bool _showAdvanced;
    [ObservableProperty] private bool _sortByDelta = true;

    /// <summary>
    /// Whether the face pipeline loads the derived model that also emits the stock visual embedding.
    /// </summary>
    /// <remarks>
    /// Advanced and off by default, deliberately. Model C is unproven against real recordings, and
    /// the derived graph has not yet been exercised on a GPU. Turning this on changes which face
    /// model file is loaded - the weights are identical and the expression output is verified
    /// bit-identical, so the risk is not accuracy but a load failure, which falls back to stock.
    /// </remarks>
    [ObservableProperty] private bool _useEmbeddingRunner;

    [ObservableProperty] private string _embeddingRunnerStatus = "";

    public ObservableCollection<ExpressionComparisonRow> Comparison { get; } = [];

    /// <summary>Guided sessions need the cue engine, which is a later phase.</summary>
    public IReadOnlyList<string> SessionTypes { get; } =
        [nameof(SessionType.Neutral), nameof(SessionType.Speech)];

    [ObservableProperty] private string _selectedSessionType = nameof(SessionType.Neutral);

    public PersonalizationViewModel(
        DatasetRecorderService recorder,
        PersonalModelManager modelManager,
        PersonalTrainingService trainingService,
        PersonalizationEnvironment environment,
        ILocalSettingsService settings,
        IFacePipelineEventBus faceEventBus,
        ILogger<PersonalizationViewModel> logger,
        HardExampleService? hardExamples = null,
        GuidedCalibrationService? guided = null,
        AudioAssistService? audio = null)
    {
        _recorder = recorder;
        _modelManager = modelManager;
        _trainingService = trainingService;
        _environment = environment;
        _settings = settings;
        _faceEventBus = faceEventBus;
        _logger = logger;
        _hardExamples = hardExamples;
        _guided = guided;
        _audio = audio;

        if (audio != null)
        {
            _audioAssistEnabled = audio.Enabled;
            _audioStrength = audio.Strength * 100.0;
            _audioStatus = audio.StatusMessage;
        }

        _guidedTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
        _guidedTimer.Tick += (_, _) => OnGuidedTick();

        foreach (var (name, index) in PersonalizationSchema.ExpressionNames.Select((n, i) => (n, i)))
            Comparison.Add(new ExpressionComparisonRow(index, name));

        _frameHandler = OnTransformedFrame;
        _rawHandler = OnRawExpressions;
        _correctedHandler = OnCorrectedExpressions;
        _faceEventBus.Subscribe(_frameHandler);
        _faceEventBus.Subscribe(_rawHandler);
        _faceEventBus.Subscribe(_correctedHandler);

        _personalModelEnabled = _modelManager.Enabled;
        _personalStrength = _modelManager.Blend * 100.0;
        _useEmbeddingRunner = _settings.ReadSetting<bool>(PersonalModelManager.EmbeddingRunnerSetting);

        _statusTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _statusTimer.Tick += (_, _) => OnTick();
        _statusTimer.Start();

        _ = InitializeAsync();
    }

    private async Task InitializeAsync()
    {
        await _modelManager.ReloadAsync();
        RefreshSetup();
    }

    // =============================================================================================
    // Setup
    // =============================================================================================

    [RelayCommand]
    private void RefreshSetup()
    {
        var setup = _environment.Inspect(_cameraSeenRecently, _modelManager);

        CameraStatus = setup.Camera.Summary;
        CameraReady = setup.Camera.Ready;
        ToolsStatus = setup.TrainingTools.Summary;
        ToolsReady = setup.TrainingTools.Ready;
        ModelStatus = setup.Model.Summary;
        ModelReady = setup.Model.Ready;

        var loaded = _modelManager.LoadedMetadata;
        ActiveModelName = loaded == null
            ? ""
            : $"Currently loaded: {loaded.DisplayName}";

        // Offer setup only when it can actually succeed: scripts located and Python available.
        CanSetUpTools = !setup.TrainingTools.Ready
                        && setup.TrainingRoot != null
                        && PersonalizationEnvironment.FindHostPython() != null;

        NeutralCount = setup.DatasetStatus.NeutralSessions;
        SpeechCount = setup.DatasetStatus.SpeechSessions;
        RecordingAdvice = setup.DatasetStatus.NextRecommendation
                          ?? "You have enough recordings to train a good model.";

        SavedCorrectionCount = HardExampleService.CountSaved();

        // Nothing to save unless frames are arriving, so the button says so rather than failing.
        CanFlagCorrection = _hardExamples != null && _cameraSeenRecently;

        CanStartGuided = _guided != null && _cameraSeenRecently && !IsGuidedRunning && !IsRecording;

        CanTrain = setup.CanTrain && !IsBusy;
    }

    [RelayCommand]
    private async Task SetUpToolsAsync()
    {
        await RunWorkAsync(
            "Setting up training tools...",
            (progress, token) => _trainingService.SetUpTrainingToolsAsync(progress, token));
    }

    // =============================================================================================
    // Recording
    // =============================================================================================

    [RelayCommand]
    private void StartRecording()
    {
        if (_recorder.IsRecording)
            return;

        try
        {
            var type = Enum.Parse<SessionType>(SelectedSessionType);
            _recorder.StartSession(type, camera: null,
                notes: string.IsNullOrWhiteSpace(Notes) ? null : Notes);

            IsRecording = true;
            RecordingStatus = type == SessionType.Neutral
                ? "Recording. Let your face rest - breathe normally for about 45 seconds."
                : "Recording. Talk naturally for a minute or so.";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Personalization: failed to start recording");
            RecordingStatus = $"Could not start recording: {ex.Message}";
        }
    }

    [RelayCommand]
    private async Task StopRecordingAsync()
    {
        if (!_recorder.IsRecording)
            return;

        try
        {
            var summary = await _recorder.StopSessionAsync();
            IsRecording = false;

            RecordingStatus = summary == null
                ? ""
                : $"Saved {summary.FrameCount} frames at {summary.EffectiveFps:F0} fps." +
                  (summary.DroppedFrames > 0 ? $" ({summary.DroppedFrames} dropped)" : "");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Personalization: failed to stop recording");
            IsRecording = false;
            RecordingStatus = $"Error finishing the recording: {ex.Message}";
        }

        RefreshSetup();
    }

    [RelayCommand]
    private void RecordNeutral()
    {
        SelectedSessionType = nameof(SessionType.Neutral);
        StartRecording();
    }

    [RelayCommand]
    private void RecordSpeech()
    {
        SelectedSessionType = nameof(SessionType.Speech);
        StartRecording();
    }

    [RelayCommand]
    private void OpenDatasetFolder()
    {
        System.IO.Directory.CreateDirectory(PersonalizationPaths.DatasetRoot);
        Utils.OpenUrl(PersonalizationPaths.DatasetRoot);
    }

    /// <summary>
    /// Switches the face pipeline between the stock model and the derived one that also emits the
    /// visual embedding, building the derived model first if it is missing or stale.
    /// </summary>
    partial void OnUseEmbeddingRunnerChanged(bool value)
    {
        _settings.SaveSetting(PersonalModelManager.EmbeddingRunnerSetting, value);
        _ = ApplyEmbeddingRunnerAsync(value);
    }

    private async Task ApplyEmbeddingRunnerAsync(bool enabled)
    {
        if (!enabled)
        {
            EmbeddingRunnerStatus = "";
            await _modelManager.ReloadAsync();
            RefreshSetup();
            return;
        }

        var stockModel = System.IO.Path.Combine(AppContext.BaseDirectory, "faceModel.onnx");
        var validation = EmbeddingModelStore.TryGetValid(stockModel);

        if (!validation.Valid)
        {
            EmbeddingRunnerStatus = "Building the embedding model...";
            var built = await _trainingService.RegenerateEmbeddingModelAsync();
            if (!built.Success)
            {
                EmbeddingRunnerStatus = built.Message;
                // Leave the setting on: the pipeline falls back to stock on its own, and flipping
                // the toggle back here would fight the user rather than explain the problem.
                return;
            }
        }

        EmbeddingRunnerStatus = "Embedding model active.";
        RefreshSetup();
    }

    // =============================================================================================
    // Audio expression assist
    // =============================================================================================

    partial void OnAudioAssistEnabledChanged(bool value)
    {
        if (_audio is null)
        {
            AudioStatus = "Audio assist is unavailable in this build.";
            return;
        }

        _audio.SetEnabled(value);
        AudioStatus = _audio.StatusMessage;

        if (!value)
            AudioDiagnostics = "";
    }

    partial void OnAudioStrengthChanged(double value)
    {
        if (_audio != null)
            _audio.Strength = (float)(value / 100.0);
    }

    /// <summary>Live audio readout, drained on the status timer like the expression table.</summary>
    private void UpdateAudioDiagnostics()
    {
        if (_audio is null || !AudioAssistEnabled || !ShowAdvanced)
            return;

        var features = _audio.Features;
        var gain = _audio.CurrentGain;

        AudioDiagnostics =
            $"{(features.IsVoiced ? "speaking" : "quiet")}   " +
            $"energy {features.SpeechEnergy:F2}   " +
            $"pitch {(features.PitchHz > 0 ? $"{features.PitchHz:F0} Hz" : "-")}   " +
            $"boost x{gain:F2}";
    }

    // =============================================================================================
    // Guided calibration
    // =============================================================================================

    /// <summary>
    /// Runs the jaw routine: the avatar performs a known sequence, the user copies it, and the
    /// commanded value becomes the label.
    /// </summary>
    /// <remarks>
    /// This is the only source of non-zero supervision in the project. Neutral sessions teach the
    /// model what a resting face is; nothing else teaches it what a correct *open* jaw is, which is
    /// why "be quieter at rest" was previously the only thing it could learn.
    /// </remarks>
    [RelayCommand]
    private void StartGuidedJawCalibration()
    {
        if (_guided is null)
        {
            GuidedStatus = "Guided calibration is unavailable in this build.";
            return;
        }

        var routine = new GuidedCaptureRoutine(GuidedCaptureRoutine.BuildJawOpenRoutine());
        var result = _guided.Start(routine, notes: "Guided JawOpen calibration");

        GuidedStatus = result.Started
            ? "Watch your avatar and copy what it does."
            : result.Message;

        if (!result.Started)
            return;

        IsGuidedRunning = true;
        GuidedProgress = 0;
        _guidedTimer.Start();
    }

    [RelayCommand]
    private async Task StopGuidedCalibrationAsync()
    {
        _guidedTimer.Stop();
        IsGuidedRunning = false;

        if (_guided is null)
            return;

        var summary = await _guided.StopAsync();
        GuidedStatus = summary is null
            ? "Calibration stopped."
            : $"Saved {summary.FrameCount} frames. Press Train My Face Model to use them.";

        GuidedProgress = 0;
        GuidedInstruction = "";
        RefreshSetup();
    }

    private void OnGuidedTick()
    {
        if (_guided is null)
            return;

        var stillRunning = _guided.Tick();
        var progress = _guided.Progress;

        GuidedInstruction = progress.Instruction;
        GuidedProgress = progress.Fraction * 100;

        if (!stillRunning)
            _ = StopGuidedCalibrationAsync();
    }

    // =============================================================================================
    // Quick correction
    // =============================================================================================

    /// <summary>
    /// Saves the last few seconds as evidence that the jaw was wrong.
    /// </summary>
    /// <remarks>
    /// Deliberately does not retrain. Corrections are worth most in batches - one is a single
    /// noisy example, a dozen describe a pattern - and an automatic retrain after every press would
    /// swap the running model constantly and make it impossible to tell which change helped.
    /// </remarks>
    [RelayCommand]
    private async Task FlagMouthClosedAsync()
    {
        if (_hardExamples is null)
        {
            CorrectionStatus = "Quick correction is unavailable in this build.";
            return;
        }

        var result = await _hardExamples.FlagAsync(CorrectionKind.MouthClosed, SelectedCorrectionWindow);
        CorrectionStatus = result.Message;

        if (result.Success)
        {
            SavedCorrectionCount = HardExampleService.CountSaved();
            RefreshSetup();
        }
    }

    // =============================================================================================
    // Training
    // =============================================================================================

    [RelayCommand]
    private async Task TrainAsync()
    {
        var kind = SelectedModelKind;

        await RunWorkAsync(
            "Starting...",
            (progress, token) => _trainingService.TrainAsync(kind, progress, token));

        PersonalModelEnabled = _modelManager.Enabled;
    }

    [RelayCommand]
    private void CancelWork() => _workCancellation?.Cancel();

    [RelayCommand]
    private void ToggleDetails()
    {
        ShowDetails = !ShowDetails;
        if (ShowDetails)
            DetailLog = _trainingService.DetailLog;
    }

    /// <summary>Shared plumbing for the two long-running operations: busy state, progress, results.</summary>
    private async Task RunWorkAsync(
        string initialMessage,
        Func<IProgress<TrainingProgress>, CancellationToken, Task<TrainingResult>> work)
    {
        if (IsBusy)
            return;

        _workCancellation?.Dispose();
        _workCancellation = new CancellationTokenSource();

        IsBusy = true;
        CanTrain = false;
        HasResult = false;
        RemedyAction = null;
        BusyMessage = initialMessage;

        var progress = new Progress<TrainingProgress>(p => BusyMessage = p.Message);

        try
        {
            var result = await work(progress, _workCancellation.Token);
            PresentResult(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Personalization: background work failed");
            ResultHeadline = "Something went wrong.";
            ResultDetail = ex.Message;
            RemedyAction = "Show Details";
            HasResult = true;
        }
        finally
        {
            IsBusy = false;
            BusyMessage = "";
            DetailLog = _trainingService.DetailLog;
            RefreshSetup();
        }
    }

    /// <summary>
    /// Turns a measured result into plain language.
    ///
    /// Nothing here is invented: the wording is driven by the trainer's own verdict, which stays
    /// "unclear" whenever there was no held-out session to judge against. Saying "we cannot tell
    /// yet" is more useful than a confident number that means nothing.
    /// </summary>
    private void PresentResult(TrainingResult result)
    {
        HasResult = true;
        RemedyAction = result.Remedy;

        if (!result.Success)
        {
            ResultHeadline = result.Message;
            ResultDetail = result.Remedy == "Show Details"
                ? "Open Details below for the exact error."
                : "";
            return;
        }

        var summary = result.Summary;
        if (summary == null)
        {
            ResultHeadline = result.Message;
            ResultDetail = "";
            return;
        }

        var lines = new List<string>();

        // Which architecture actually produced this result. Without it the dropdown is the only
        // hint, and that shows the *next* run's choice rather than what was just trained.
        if (!string.IsNullOrEmpty(summary.AdapterType))
            lines.Add($"Trained: {TrainingModelChoice.DisplayName(summary.AdapterType)}.");

        if (summary.NeutralStockFalseActivation is { } stockNeutral &&
            summary.NeutralPersonalFalseActivation is { } personalNeutral)
        {
            lines.Add($"Expressions firing while your face is at rest: " +
                      $"{stockNeutral:P1} before, {personalNeutral:P1} after.");
        }

        if (summary.MeanStockMae is { } stockMae && summary.MeanPersonalMae is { } personalMae)
            lines.Add($"Average expression error: {stockMae:F3} before, {personalMae:F3} after.");

        // The "my jaw wiggles while resting" number. A model can score a perfect false-activation
        // rate and still shiver just under the threshold, so this is reported on its own.
        if (summary.NeutralStockJitter is { } stockJitter &&
            summary.NeutralPersonalJitter is { } personalJitter)
        {
            lines.Add($"Twitchiness while resting: {stockJitter:F4} before, {personalJitter:F4} after " +
                      $"({(personalJitter <= stockJitter ? "steadier" : "shakier")}).");
        }

        // The specific complaint this phase exists to fix, called out by name rather than left for
        // the user to find in a 45-row table. Absent from older summaries, hence the null checks.
        AppendWatchedExpression(lines, summary.JawOpen, "jaw-open");
        AppendWatchedExpression(lines, summary.TongueOut, "tongue-out");

        lines.Add($"{summary.ExpressionsImproved} expressions improved, " +
                  $"{summary.ExpressionsRegressed} got worse" +
                  (summary.WorstRegressions.Count > 0
                      ? $" ({string.Join(", ", summary.WorstRegressions.Take(3))})."
                      : "."));

        ResultHeadline = summary.Verdict switch
        {
            "better" => "Personal model ready - it beat the stock tracker on your held-out recordings.",
            "worse" => "Personal model trained, but it did worse than stock on your held-out recordings.",
            _ when !summary.ValidatedOnHeldOutSessions =>
                "Personal model ready, but the results cannot be trusted yet.",
            _ => "Personal model ready, but the results are not clearly better than stock yet."
        };

        if (!summary.ValidatedOnHeldOutSessions)
        {
            lines.Add("There was no spare recording to test against, so these numbers describe data " +
                      "the model already learned from. Record two sessions of each type for an " +
                      "honest comparison.");
        }
        else if (summary.Verdict != "better")
        {
            lines.Add("More recordings usually help, especially Neutral ones from separate sittings.");
        }

        lines.Add("Use Stock / Personal below to hear and see the difference for yourself.");
        ResultDetail = string.Join(Environment.NewLine + Environment.NewLine, lines);
    }

    /// <summary>
    /// Reports one watched expression in plain language: how often it fired wrongly, how long the
    /// worst episode lasted, and whether the fix cost real movement.
    /// </summary>
    /// <remarks>
    /// Duration is reported alongside rate because they answer different questions. Twenty
    /// single-frame flickers and one two-second hang produce the same rate; only the second is the
    /// thing the user notices. The suppression warning is reported in the same breath so an
    /// improvement that came from deadening the expression cannot read as a clean win.
    /// </remarks>
    private static void AppendWatchedExpression(
        List<string> lines, WatchedExpressionSummary? watched, string label)
    {
        if (watched is null || watched.ClosedFrames == 0)
            return;

        if (watched.StockFalseActivation is not { } stock ||
            watched.PersonalFalseActivation is not { } personal)
            return;

        var text = $"False {label} while your mouth was closed: {stock:P1} before, {personal:P1} after.";

        if (watched.StockLongestFalseRunSeconds is { } stockRun &&
            watched.PersonalLongestFalseRunSeconds is { } personalRun &&
            (stockRun > 0 || personalRun > 0))
        {
            text += $" Longest single episode: {stockRun:F1}s before, {personalRun:F1}s after.";
        }

        if (watched.SuppressionWarning)
        {
            text += " Careful: it also moves noticeably less during speech, which can look flat - " +
                    "check the Stock / Personal switch while talking.";
        }

        lines.Add(text);
    }

    // =============================================================================================
    // Comparison
    // =============================================================================================

    [RelayCommand]
    private async Task UseStockAsync()
    {
        PersonalModelEnabled = false;
        await Task.CompletedTask;
    }

    [RelayCommand]
    private async Task UsePersonalAsync()
    {
        PersonalModelEnabled = true;
        await Task.CompletedTask;
    }

    [RelayCommand]
    private async Task ReloadModelAsync()
    {
        await _modelManager.ReloadAsync();
        RefreshSetup();
    }

    partial void OnPersonalModelEnabledChanged(bool value)
    {
        _modelManager.SetEnabled(value);
        _ = ReloadModelAsync();
    }

    partial void OnPersonalStrengthChanged(double value)
    {
        // Live evaluation control: apply immediately so A/B comparison is instant.
        _modelManager.Blend = (float)(value / 100.0);
    }

    // =============================================================================================
    // Live data
    // =============================================================================================

    private void OnRawExpressions(FacePipelineEvents.NewRawExpressionsEvent e)
    {
        if (_hasCorrectedValues)
            return;

        var count = Math.Min(e.rawResult.Length, _latestStock.Length);
        Array.Copy(e.rawResult, _latestStock, count);
        Array.Copy(e.rawResult, _latestPersonal, count);
    }

    private void OnCorrectedExpressions(FacePipelineEvents.NewCorrectedExpressionsEvent e)
    {
        _hasCorrectedValues = true;
        var count = Math.Min(e.rawResult.Length, _latestStock.Length);
        Array.Copy(e.rawResult, _latestStock, count);
        Array.Copy(e.correctedResult, _latestPersonal, Math.Min(e.correctedResult.Length, count));
    }

    private void OnTick()
    {
        if (_recorder.IsRecording)
            RecordingStatus = $"Recording... {_recorder.FramesWritten} frames captured.";

        for (var i = 0; i < Comparison.Count; i++)
            Comparison[i].Update(_latestStock[i], _latestPersonal[i]);

        if (SortByDelta)
            SortComparisonByDelta();

        UpdateAudioDiagnostics();

        // The camera is "running" if a frame arrived recently; there is no event when it stops.
        var seen = (DateTime.UtcNow - _lastFrameUtc).TotalSeconds < 2;
        if (seen != _cameraSeenRecently)
        {
            _cameraSeenRecently = seen;
            RefreshSetup();
        }
    }

    private void SortComparisonByDelta()
    {
        var ordered = Comparison.OrderByDescending(r => r.AbsoluteDelta).ToList();
        for (var target = 0; target < ordered.Count; target++)
        {
            var current = Comparison.IndexOf(ordered[target]);
            if (current != target)
                Comparison.Move(current, target);
        }
    }

    private void OnTransformedFrame(FacePipelineEvents.NewTransformedFrameEvent e)
    {
        if (e.image is null || e.image.Empty())
            return;

        _lastFrameUtc = DateTime.UtcNow;

        try
        {
            UpdateBitmap(e.image);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Personalization: preview update failed");
        }
    }

    private void UpdateBitmap(Mat image)
    {
        if (_backingBitmap is null ||
            _backingBitmap.PixelSize.Width != image.Width ||
            _backingBitmap.PixelSize.Height != image.Height)
        {
            _backingBitmap = new WriteableBitmap(
                new PixelSize(image.Width, image.Height),
                new Vector(96, 96),
                image.Channels() == 3 ? PixelFormats.Bgr24 : PixelFormats.Gray8,
                AlphaFormat.Opaque);
        }

        var source = image.IsContinuous() ? image : image.Clone();

        using (var buffer = _backingBitmap.Lock())
        {
            var size = source.Rows * source.Cols * source.ElemSize();
            unsafe
            {
                Buffer.MemoryCopy(source.Data.ToPointer(), buffer.Address.ToPointer(), size, size);
            }
        }

        if (!ReferenceEquals(source, image))
            source.Dispose();

        // The bitmap instance is unchanged, so the binding needs a null round-trip to repaint.
        Preview = null;
        Preview = _backingBitmap;
    }

    public void Dispose()
    {
        _statusTimer.Stop();
        _guidedTimer.Stop();
        _faceEventBus.Unsubscribe(_frameHandler);
        _faceEventBus.Unsubscribe(_rawHandler);
        _faceEventBus.Unsubscribe(_correctedHandler);

        _workCancellation?.Cancel();
        _workCancellation?.Dispose();

        // Navigating away mid-calibration must hand the avatar back to live tracking rather than
        // leaving it frozen in whatever expression was being commanded.
        _guided?.Dispose();

        if (_recorder.IsRecording)
            _recorder.StopSessionAsync().GetAwaiter().GetResult();
    }
}
