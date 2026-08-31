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
    private readonly GrimacePreviewService? _grimace;

    /// <summary>
    /// Ticks the cue engine while a guided session runs. Faster than the status timer because it
    /// also drives the override's keep-alive, and because a 0.75 s transition commanded on a 250 ms
    /// timer would land visibly late.
    /// </summary>
    private readonly DispatcherTimer _guidedTimer;
    private readonly DispatcherTimer _grimaceTimer;

    private readonly Action<FacePipelineEvents.NewTransformedFrameEvent> _frameHandler;
    private readonly Action<FacePipelineEvents.NewRawExpressionsEvent> _rawHandler;
    private readonly Action<FacePipelineEvents.NewCorrectedExpressionsEvent> _correctedHandler;
    private readonly DispatcherTimer _statusTimer;

    private WriteableBitmap? _backingBitmap;
    private CancellationTokenSource? _workCancellation;
    private bool _lastObservedModelActive;
    private bool _lastRecorderFrameReady;

    // Latest values, written on the processing tick and drained by the timer. Binding at the tick
    // rate would swamp the UI; a few times a second is plenty to watch values move.
    private readonly ExpressionComparisonBuffer _comparisonValues =
        new(PersonalizationSchema.ExpressionCount);
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
    [ObservableProperty] private int _guidedCount;
    [ObservableProperty] private string _recordingAdvice = "";
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private bool _canStartRecording;
    [ObservableProperty] private bool _canOpenDatasetFolder;
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
    public ObservableCollection<string> AudioInputLabels { get; } = [];
    private IReadOnlyList<AudioInputDevice> _audioInputDevices = [];
    private bool _changingAudioInput;
    [ObservableProperty] private int _selectedAudioInputIndex;
    [ObservableProperty] private double _audioInputLevel;
    [ObservableProperty] private string _audioVoiceStatus = "Voice: Not detected";
    [ObservableProperty] private string _audioSelectedDevice = "Selected: unavailable";

    partial void OnSelectedAudioInputIndexChanged(int value)
    {
        if (_changingAudioInput || _audio is null) return;
        var id = value <= 0 || value > _audioInputDevices.Count
            ? null
            : _audioInputDevices[value - 1].Id;
        _audio.SelectInputDevice(id);
        AudioStatus = _audio.StatusMessage;
        UpdateAudioDiagnostics();
    }

    // ---- guided calibration -------------------------------------------------------------------

    [ObservableProperty] private bool _isGuidedRunning;
    [ObservableProperty] private string _guidedInstruction = "";
    [ObservableProperty] private double _guidedProgress;
    [ObservableProperty] private string _guidedStatus = "";
    [ObservableProperty] private bool _canStartGuided;

    /// <summary>
    /// What a guided session will cover. The full pass is the useful default; single expressions
    /// exist so a specific weakness can be topped up without sitting through everything again.
    /// </summary>
    public ObservableCollection<GuidedRoutineChoice> GuidedRoutines { get; } =
        new(GuidedRoutineChoice.All);

    [ObservableProperty] private int _selectedGuidedRoutineIndex;
    [ObservableProperty] private string _guidedRoutineDescription = "";

    partial void OnSelectedGuidedRoutineIndexChanged(int value) => UpdateGuidedRoutineDescription();

    private GuidedRoutineChoice SelectedGuidedRoutine =>
        GuidedRoutines[Math.Clamp(SelectedGuidedRoutineIndex, 0, GuidedRoutines.Count - 1)];

    private void UpdateGuidedRoutineDescription()
    {
        var choice = SelectedGuidedRoutine;
        var minutes = choice.EstimatedSeconds / 60.0;
        var length = minutes >= 1
            ? $"about {minutes:F0} minute{(minutes < 1.5 ? "" : "s")}"
            : $"about {choice.EstimatedSeconds:F0} seconds";

        GuidedRoutineDescription = $"{choice.Description} Takes {length}.";
    }

    // ---- Grimace candidate lab ---------------------------------------------------------------

    [ObservableProperty] private bool _isGrimacePreviewing;
    [ObservableProperty] private bool _canStartGrimacePreview;
    [ObservableProperty] private bool _canConfirmGrimace;
    [ObservableProperty] private bool _hasConfirmedGrimace;
    [ObservableProperty] private string _grimaceStatus =
        "Preview the candidate poses on your avatar before enabling Grimace training.";
    [ObservableProperty] private string _grimaceCandidateDetails =
        "No candidate is active. Nothing is being recorded.";
    [ObservableProperty] private string _grimaceConfirmationStatus =
        "Grimace is not trainable until you preview and explicitly confirm one candidate.";
    private string? _appliedGrimaceCandidateId;

    // ---- quick correction ---------------------------------------------------------------------

    /// <summary>
    /// How far back "my mouth was closed" reaches. Five seconds by default: a mistake is noticed a
    /// beat after it happens, and the window has to cover both the noticing and the reaching.
    /// </summary>
    public IReadOnlyList<string> CorrectionWindows { get; } = ["Last 2 seconds", "Last 5 seconds", "Last 10 seconds"];

    [ObservableProperty] private bool _quickCorrectionEnabled = true;
    [ObservableProperty] private string _quickCorrectionAvailability = "";
    [ObservableProperty] private int _selectedCorrectionWindowIndex = 1;
    [ObservableProperty] private string _correctionStatus = "";
    [ObservableProperty] private int _savedCorrectionCount;
    [ObservableProperty] private bool _canFlagCorrection;
    [ObservableProperty] private bool _canDeleteSavedCorrections;
    [ObservableProperty] private bool _showDeleteCorrectionsConfirmation;

    private TimeSpan SelectedCorrectionWindow =>
        HardExampleService.WindowChoices[
            Math.Clamp(SelectedCorrectionWindowIndex, 0, HardExampleService.WindowChoices.Count - 1)];

    // ---- training -----------------------------------------------------------------------------

    /// <summary>
    /// Which adapter to train next. This is intentionally separate from the model currently in use.
    /// A is the simple reference, B is the proven real-world baseline, and C is the shared-feature
    /// experiment that must be explicitly prepared before it can train.
    /// </summary>
    [ObservableProperty] private int _selectedModelIndex;

    public IReadOnlyList<string> TrainingModelLabels { get; } =
        TrainingModelChoice.Options.Select(option => option.Label).ToArray();

    [ObservableProperty] private string _modelChoiceDescription =
        TrainingModelChoice.Options[TrainingModelChoice.OutputOnlyIndex].Description;
    [ObservableProperty] private bool _isModelCSelected;
    [ObservableProperty] private string _modelCStatus = "";
    [ObservableProperty] private bool _canPrepareModelC;
    [ObservableProperty] private string _trainButtonText = "Train Model A";
    [ObservableProperty] private string _trainingDataSummary = "No training data discovered.";
    [ObservableProperty] private string _trainingSplitSummary = "";
    [ObservableProperty] private string _guidedCoverageSummary = "";

    /// <summary>The trainer's --model flag for the current selection.</summary>
    private string SelectedModelKind => TrainingModelChoice.KindForIndex(SelectedModelIndex);

    partial void OnSelectedModelIndexChanged(int value) => RefreshSetup();

    [ObservableProperty] private bool _isBusy;
    [ObservableProperty] private string _busyMessage = "";
    [ObservableProperty] private bool _canTrain;
    [ObservableProperty] private string _resultHeadline = "";
    [ObservableProperty] private string _resultDetail = "";
    [ObservableProperty] private bool _hasResult;
    [ObservableProperty] private string? _remedyAction;
    [ObservableProperty] private bool _showDetails;
    [ObservableProperty] private string _detailLog = "";
    [ObservableProperty] private bool _canOpenLastRunFolder;
    [ObservableProperty] private bool _canUseLastCreatedModel;
    private string? _lastRunDirectory;
    private string? _lastCreatedModelPath;

    // ---- comparison ---------------------------------------------------------------------------

    /// <summary>
    /// Which adapter is loaded right now, read from the installed model's own metadata. The Train
    /// dropdown cannot answer this: it holds the choice for the *next* run, so after switching it
    /// without retraining the two would disagree. This is the authoritative one.
    /// </summary>
    [ObservableProperty] private string _activeModelName = "";
    [ObservableProperty] private string _activeModelProvenance = "";
    [ObservableProperty] private string _modelActionStatus = "";
    [ObservableProperty] private string _selectedModelProvenance = "";
    [ObservableProperty] private bool _canOpenSelectedModelFolder;

    [ObservableProperty] private bool _personalModelEnabled;
    [ObservableProperty] private bool _canUseStock;
    [ObservableProperty] private bool _canUsePersonal;
    public ObservableCollection<string> ComparisonModelLabels { get; } = [];
    private IReadOnlyList<AvailablePersonalModel> _comparisonModels = [];
    [ObservableProperty] private int _selectedComparisonModelIndex;
    [ObservableProperty] private string _useSelectedModelButtonText = "Use Selected Model";
    [ObservableProperty] private bool _canUseSelectedModel;
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

    partial void OnSelectedComparisonModelIndexChanged(int value) => UpdateComparisonSelection();

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
        AudioAssistService? audio = null,
        GrimacePreviewService? grimace = null)
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
        _grimace = grimace;

        _quickCorrectionEnabled = hardExamples?.Enabled ?? false;

        if (audio != null)
        {
            _audioAssistEnabled = audio.Enabled;
            _audioStrength = audio.Strength * 100.0;
            _audioStatus = audio.StatusMessage;
            RefreshAudioInputs();
        }

        _guidedTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
        _guidedTimer.Tick += (_, _) => OnGuidedTick();
        _grimaceTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
        _grimaceTimer.Tick += (_, _) => OnGrimaceTick();

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

        UpdateGuidedRoutineDescription();
        RefreshGrimaceState();

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
        ActiveModelName = _modelManager.IsActive && loaded != null
            ? $"Running now: Personal — {loaded.DisplayName}"
            : "Running now: Stock Baballonia face model";
        CanUseStock = _modelManager.Enabled || _modelManager.IsActive;
        CanUsePersonal = !_modelManager.IsActive && System.IO.File.Exists(_modelManager.ModelPath);
        RefreshComparisonModels();
        UpdateActiveModelStatus();
        _lastObservedModelActive = _modelManager.IsActive;
        _lastRecorderFrameReady = _recorder.HasRecentSourceFrame;

        // Offer setup only when it can actually succeed: scripts located and Python available.
        CanSetUpTools = !setup.TrainingTools.Ready
                        && setup.TrainingRoot != null
                        && PersonalizationEnvironment.FindHostPython() != null;

        NeutralCount = setup.DatasetStatus.NeutralSessions;
        SpeechCount = setup.DatasetStatus.SpeechSessions;
        GuidedCount = setup.DatasetStatus.GuidedSessions;
        RecordingAdvice = setup.DatasetStatus.NextRecommendation
                          ?? "You have enough recordings to train a good model.";

        UpdateTrainingDataSummary(PersonalizationEnvironment.InspectTrainingInventory());

        SavedCorrectionCount = HardExampleService.CountSaved();
        var captureOrPreviewActive = _recorder.IsRecording || IsRecording || IsGuidedRunning ||
                                     (_grimace?.IsPreviewing ?? false);
        CanDeleteSavedCorrections = _hardExamples != null && SavedCorrectionCount > 0 &&
                                    !IsBusy && !captureOrPreviewActive;
        if (SavedCorrectionCount == 0) ShowDeleteCorrectionsConfirmation = false;

        // Nothing to save unless capture is enabled and frames are arriving, so the button and
        // status say which prerequisite is missing rather than failing after the press.
        CanFlagCorrection = _hardExamples != null && QuickCorrectionEnabled && _cameraSeenRecently &&
                            !IsBusy && !captureOrPreviewActive;
        QuickCorrectionAvailability = _hardExamples == null
            ? "Quick correction is unavailable in this build."
            : !QuickCorrectionEnabled
                ? "Quick correction is off. No rolling frames are being copied or retained."
                : !_cameraSeenRecently
                    ? "Start the face camera on the Home page to use quick corrections."
                    : "";

        CanStartRecording = !IsBusy && !captureOrPreviewActive && _recorder.HasRecentSourceFrame;
        CanOpenDatasetFolder = !IsBusy && !captureOrPreviewActive;
        CanStartGuided = _guided != null && _cameraSeenRecently && !IsBusy &&
                         !captureOrPreviewActive;
        RefreshGrimaceState();

        var choice = TrainingModelChoice.ForIndex(SelectedModelIndex);
        ModelChoiceDescription = choice.Description;
        TrainButtonText = $"Train Model {choice.Kind.ToUpperInvariant()}";
        IsModelCSelected = choice.RequiresEmbedding;

        var modelCReady = true;
        if (choice.RequiresEmbedding)
        {
            var readiness = _trainingService.InspectModelCReadiness();
            modelCReady = readiness.Ready;
            ModelCStatus = readiness.Message;
            CanPrepareModelC = setup.CanTrain && !IsBusy && !captureOrPreviewActive &&
                               !readiness.Ready;
        }
        else
        {
            ModelCStatus = "";
            CanPrepareModelC = false;
        }

        CanTrain = !IsBusy && !captureOrPreviewActive && TrainingModelChoice.TrainingUnavailableReason(
            choice.Kind, setup.CanTrain, modelCReady) == null;
    }

    private void UpdateTrainingDataSummary(TrainingDatasetInventory inventory)
    {
        if (inventory.Discovered.Count == 0)
        {
            TrainingDataSummary = "No usable recording sessions were discovered.";
            TrainingSplitSummary = "";
            GuidedCoverageSummary = "";
            return;
        }

        string Row(SessionType type, string label)
        {
            var row = inventory.ByType.First(x => x.Type == type);
            return $"{label,-12} {row.DiscoveredSessions,2} sessions  {row.DiscoveredFrames,6:N0} frames";
        }

        TrainingDataSummary = string.Join(Environment.NewLine,
            Row(SessionType.Neutral, "Neutral"),
            Row(SessionType.Speech, "Speech"),
            Row(SessionType.Guided, "Guided"),
            Row(SessionType.Correction, "Corrections"));

        TrainingSplitSummary =
            $"Optimization split: {inventory.Training.Count} sessions / " +
            $"{inventory.TrainingFrames:N0} matched image-label frame pairs. " +
            $"Held out for validation: {inventory.HeldOut.Count} sessions / " +
            $"{inventory.HeldOutFrames:N0} frame pairs. Some transition, preparation, or " +
            "quality-suppressed rows intentionally receive zero training weight.";

        var coverage = inventory.GuidedTrainingExpressions.Count == 0
            ? "none"
            : string.Join(", ", inventory.GuidedTrainingExpressions);
        var latest = inventory.LatestGuidedSessionId == null
            ? "No Guided session recorded yet."
            : inventory.LatestGuidedIsTraining
                ? $"Newest Guided ({inventory.LatestGuidedSessionId}) is assigned to the " +
                  "optimization split; per-attempt quality still controls its effective weight."
                : $"Newest Guided ({inventory.LatestGuidedSessionId}) is held out for validation, not optimization.";
        var corrections = inventory.CorrectionCounts.Count == 0
            ? "No saved corrections in the discovered corpus."
            : "Corrections present across the discovered corpus (only named dimensions are labelled): " +
              string.Join(", ", inventory.CorrectionCounts.Select(x => $"{x.Key} x{x.Value}")) + ".";

        GuidedCoverageSummary =
            $"Raw Guided hold dimensions present in optimization-split recordings: {coverage}. " +
            $"The trainer's per-attempt quality report decides their actual weight. {latest} {corrections}";
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
        if (IsBusy)
        {
            RecordingStatus = "Wait for training/setup to finish before starting a recording.";
            return;
        }
        if (_recorder.IsRecording || IsGuidedRunning || (_grimace?.IsPreviewing ?? false))
        {
            RecordingStatus = "Another recording or avatar preview is already active.";
            return;
        }
        if (!_recorder.HasRecentSourceFrame)
        {
            RecordingStatus = "No fresh face-camera inference frame is available. Start the face " +
                              "camera, wait for tracking to move, then try again.";
            CanStartRecording = false;
            return;
        }

        try
        {
            var type = Enum.Parse<SessionType>(SelectedSessionType);
            _recorder.StartSession(type, camera: null,
                notes: string.IsNullOrWhiteSpace(Notes) ? null : Notes,
                requireRecentSourceFrame: true);

            IsRecording = true;
            RecordingStatus = type == SessionType.Neutral
                ? "Recording. Let your face rest - breathe normally for about 45 seconds."
                : "Recording. Talk naturally for a minute or so.";
            RefreshGrimaceState();
            CanStartRecording = false;
            CanOpenDatasetFolder = false;
            CanStartGuided = false;
            CanTrain = false;
            CanPrepareModelC = false;
            CanDeleteSavedCorrections = false;
            CanFlagCorrection = false;
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
        if (IsBusy || _recorder.IsRecording || IsRecording || IsGuidedRunning ||
            (_grimace?.IsPreviewing ?? false))
        {
            RecordingStatus = "Finish the active recording, calibration, preview, or training " +
                              "operation before opening the dataset for file management.";
            return;
        }

        System.IO.Directory.CreateDirectory(PersonalizationPaths.DatasetRoot);
        Utils.OpenUrl(PersonalizationPaths.DatasetRoot);
    }

    /// <summary>
    /// Switches the face pipeline between the stock model and the derived one that also emits the
    /// visual embedding, building the derived model first if it is missing or stale.
    /// </summary>
    partial void OnUseEmbeddingRunnerChanged(bool value)
    {
        _ = ApplyEmbeddingRunnerAsync(value);
    }

    private async Task ApplyEmbeddingRunnerAsync(bool enabled)
    {
        if (!enabled)
        {
            await _modelManager.SetEmbeddingRunnerEnabledAsync(false);
            EmbeddingRunnerStatus = "Shared visual-feature runner is off.";
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
                _useEmbeddingRunner = false;
                OnPropertyChanged(nameof(UseEmbeddingRunner));
                return;
            }
        }

        var load = await _modelManager.SetEmbeddingRunnerEnabledAsync(true);
        EmbeddingRunnerStatus = _modelManager.EmbeddingRunnerAvailable
            ? "Shared visual-feature runner is active."
            : $"The shared visual-feature runner could not be enabled: {load.Message}";
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

    [RelayCommand]
    private void RefreshAudioInputs()
    {
        if (_audio is null) return;
        _audioInputDevices = _audio.GetInputDevices();

        _changingAudioInput = true;
        AudioInputLabels.Clear();
        AudioInputLabels.Add("Automatic (first available input)");
        foreach (var device in _audioInputDevices)
            AudioInputLabels.Add(device.IsDefault
                ? $"{device.DisplayName} (current Windows default)"
                : device.DisplayName);

        var preferred = _audio.PreferredDeviceId;
        SelectedAudioInputIndex = string.IsNullOrWhiteSpace(preferred)
            ? 0
            : Math.Max(0, _audioInputDevices.ToList().FindIndex(device =>
                string.Equals(device.Id, preferred, StringComparison.Ordinal)) + 1);
        _changingAudioInput = false;
        UpdateAudioDiagnostics();
    }

    [RelayCommand]
    private void RestartAudioInput()
    {
        if (_audio is null) return;
        _audio.RestartAudioCapture();
        RefreshAudioInputs();
    }

    /// <summary>Live audio readout, drained on the status timer like the expression table.</summary>
    private void UpdateAudioDiagnostics()
    {
        if (_audio is null)
            return;

        var features = _audio.Features;
        var gain = _audio.CurrentGain;
        AudioInputLevel = AudioAssistEnabled ? _audio.InputLevel : 0;
        AudioVoiceStatus = _audio.IsVoiceDetected ? "Voice: Detected" : "Voice: Not detected";
        AudioSelectedDevice = _audio.ActiveDevice is { } device
            ? $"Selected: {device.DisplayName}" +
              (_audio.IsUsingDeviceFallback ? " (temporary fallback)" : "")
            : "Selected: no active microphone";
        AudioStatus = _audio.StatusMessage;

        AudioDiagnostics = ShowAdvanced
            ? $"{(features.IsVoiced ? "speaking" : "quiet")}   " +
              $"energy {features.SpeechEnergy:F2}   " +
              $"pitch {(features.PitchHz > 0 ? $"{features.PitchHz:F0} Hz" : "-")}   " +
              $"boost x{gain:F2}"
            : "";
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
        if (IsBusy)
        {
            GuidedStatus = "Wait for training/setup to finish before starting guided calibration.";
            return;
        }
        if (_recorder.IsRecording || IsRecording || (_grimace?.IsPreviewing ?? false))
        {
            GuidedStatus = "Stop the current recording or candidate preview first.";
            return;
        }

        var choice = SelectedGuidedRoutine;
        var routine = new GuidedCaptureRoutine(choice.Build());
        var result = _guided.Start(routine, notes: $"Guided calibration: {choice.DisplayName}");

        GuidedStatus = result.Started
            ? "Follow the explicit headset cue and copy your avatar. Retry/Skip/Cancel are available in VR."
            : result.Message;

        if (!result.Started)
            return;

        IsGuidedRunning = true;
        GuidedProgress = 0;
        _guidedTimer.Start();
        RefreshGrimaceState();
        CanStartRecording = false;
        CanOpenDatasetFolder = false;
        CanTrain = false;
        CanPrepareModelC = false;
        CanDeleteSavedCorrections = false;
        CanFlagCorrection = false;
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
            : $"Saved {summary.FrameCount} frames. Choose a model below, then press its Train button.";

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
    // Grimace candidate preview
    // =============================================================================================

    [RelayCommand]
    private void BeginGrimacePreview()
    {
        if (_grimace is null)
        {
            GrimaceStatus = "The Grimace candidate preview is unavailable in this build.";
            return;
        }

        if (IsBusy)
        {
            GrimaceStatus = "Wait for the current training/setup operation to finish first.";
            return;
        }

        var result = _grimace.BeginPreview();
        if (result.Success)
            _grimaceTimer.Start();
        RefreshGrimaceState();
        RefreshSetup();
        GrimaceStatus = result.Message;
    }

    [RelayCommand]
    private void PreviousGrimaceCandidate()
    {
        if (_grimace is null) return;
        var result = _grimace.ShowPreviousCandidate();
        RefreshGrimaceState();
        GrimaceStatus = result.Message;
    }

    [RelayCommand]
    private void NextGrimaceCandidate()
    {
        if (_grimace is null) return;
        var result = _grimace.ShowNextCandidate();
        RefreshGrimaceState();
        GrimaceStatus = result.Message;
    }

    [RelayCommand]
    private void ConfirmGrimaceCandidate()
    {
        if (_grimace is null) return;
        var result = _grimace.ConfirmCurrentCandidate();
        if (!_grimace.IsPreviewing)
            _grimaceTimer.Stop();
        RefreshGrimaceState();
        RefreshSetup();
        GrimaceStatus = result.Message;
    }

    [RelayCommand]
    private void CancelGrimacePreview()
    {
        if (_grimace is null) return;
        _grimace.CancelPreview();
        _grimaceTimer.Stop();
        RefreshGrimaceState();
        RefreshSetup();
    }

    [RelayCommand]
    private void ClearGrimaceConfirmation()
    {
        if (_grimace is null) return;
        _grimace.ClearConfirmation();
        RefreshGrimaceState();
        RefreshSetup();
    }

    private void OnGrimaceTick()
    {
        if (_grimace is null)
        {
            _grimaceTimer.Stop();
            return;
        }

        var previousIndex = _grimace.CurrentCandidateIndex;
        var stillPreviewing = _grimace.Tick();
        if (!stillPreviewing)
            _grimaceTimer.Stop();

        // Avoid rebinding the picker/status fifty times per second. Only controller actions or a
        // cancelled preview change observable state; the tick otherwise just feeds the deadman.
        if (!stillPreviewing || previousIndex != _grimace.CurrentCandidateIndex)
        {
            RefreshGrimaceState();
            RefreshSetup();
        }
    }

    private void RefreshGrimaceState()
    {
        var confirmedChoice = _grimace?.ConfirmedRoutineChoice;
        var confirmedCandidateId = _grimace?.ConfirmedCandidate?.Id;
        if (!string.Equals(
                confirmedCandidateId, _appliedGrimaceCandidateId, StringComparison.Ordinal))
        {
            for (var i = GuidedRoutines.Count - 1; i >= 0; i--)
            {
                if (string.Equals(GuidedRoutines[i].Id, "grimace-confirmed", StringComparison.Ordinal))
                    GuidedRoutines.RemoveAt(i);
            }
            if (confirmedChoice != null)
                GuidedRoutines.Add(confirmedChoice);
            _appliedGrimaceCandidateId = confirmedCandidateId;
        }

        if (GuidedRoutines.Count > 0 && SelectedGuidedRoutineIndex >= GuidedRoutines.Count)
            SelectedGuidedRoutineIndex = GuidedRoutines.Count - 1;
        UpdateGuidedRoutineDescription();

        if (_grimace is null)
        {
            IsGrimacePreviewing = false;
            CanStartGrimacePreview = false;
            CanConfirmGrimace = false;
            HasConfirmedGrimace = false;
            GrimaceConfirmationStatus = "The Grimace candidate lab is unavailable in this build.";
            return;
        }

        IsGrimacePreviewing = _grimace.IsPreviewing;
        CanStartGrimacePreview = !IsBusy && !IsRecording && !IsGuidedRunning &&
                                 !_grimace.IsPreviewing;
        CanConfirmGrimace = _grimace.IsPreviewing && _grimace.CurrentCandidate != null;
        GrimaceStatus = _grimace.Status;

        var current = _grimace.CurrentCandidate;
        GrimaceCandidateDetails = current == null
            ? "No candidate is active. Nothing is being recorded."
            : $"{current.DisplayName}\n{current.Description}\n" +
              "Exact non-zero components: " + string.Join(", ", current.Components
                  .OrderBy(component => component.Key)
                  .Select(component =>
                      $"{PersonalizationSchema.ExpressionNames[component.Key]}={component.Value:P0}"));

        var confirmed = _grimace.ConfirmedCandidate;
        HasConfirmedGrimace = confirmed != null;
        GrimaceConfirmationStatus = confirmed == null
            ? "Not confirmed: no Grimace candidate can enter a guided recording or become a label."
            : $"Confirmed: {confirmed.DisplayName}. Only this exact schema/fingerprint is available " +
              "as the separate Grimace guided routine.";
    }

    // =============================================================================================
    // Quick correction
    // =============================================================================================

    partial void OnQuickCorrectionEnabledChanged(bool value)
    {
        if (_hardExamples is null)
        {
            CorrectionStatus = "Quick correction is unavailable in this build.";
            return;
        }

        _hardExamples.SetEnabled(value);
        CorrectionStatus = value
            ? "Quick correction is on and collecting a short in-memory history."
            : "Quick correction is off; its rolling history has been released.";
        RefreshSetup();
    }

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
        if (IsBusy || _recorder.IsRecording || IsGuidedRunning ||
            (_grimace?.IsPreviewing ?? false))
        {
            CorrectionStatus =
                "Wait for the current training, recording, calibration, or preview to finish.";
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

    [RelayCommand]
    private void RequestDeleteSavedCorrections()
    {
        if (CanDeleteSavedCorrections)
            ShowDeleteCorrectionsConfirmation = true;
    }

    [RelayCommand]
    private void CancelDeleteSavedCorrections() => ShowDeleteCorrectionsConfirmation = false;

    [RelayCommand]
    private async Task ConfirmDeleteSavedCorrectionsAsync()
    {
        ShowDeleteCorrectionsConfirmation = false;
        if (_hardExamples is null) return;
        if (IsBusy || _recorder.IsRecording || IsGuidedRunning ||
            (_grimace?.IsPreviewing ?? false))
        {
            CorrectionStatus =
                "Stop training, recording, calibration, or preview before deleting correction data.";
            return;
        }

        CanDeleteSavedCorrections = false;
        var result = await _hardExamples.DeleteSavedCorrectionsAsync();
        SavedCorrectionCount = HardExampleService.CountSaved();
        CorrectionStatus = result.Failed == 0
            ? result.Deleted == 0
                ? "There were no saved quick corrections to delete."
                : $"Deleted {result.Deleted} saved quick correction{(result.Deleted == 1 ? "" : "s")}. " +
                  "Your installed model is unchanged; retrain it without those examples if needed."
            : $"Deleted {result.Deleted} corrections, but {result.Failed} could not be removed. " +
              "Your installed model is unchanged.";
        RefreshSetup();
    }

    // =============================================================================================
    // Training
    // =============================================================================================

    [RelayCommand]
    private async Task PrepareModelCAsync()
    {
        await RunWorkAsync(
            "Preparing Model C...",
            (progress, token) => _trainingService.PrepareModelCAsync(progress, token));

        // Preparation enables the runner through the manager. Synchronize the advanced checkbox
        // without invoking its change handler and doing the expensive reload a second time.
        _useEmbeddingRunner = _modelManager.EmbeddingRunnerEnabled;
        OnPropertyChanged(nameof(UseEmbeddingRunner));
    }

    [RelayCommand]
    private async Task TrainAsync()
    {
        var kind = SelectedModelKind;

        await RunWorkAsync(
            "Starting...",
            (progress, token) => _trainingService.TrainAsync(kind, progress, token));

        _personalModelEnabled = _modelManager.Enabled;
        OnPropertyChanged(nameof(PersonalModelEnabled));
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
        if (_recorder.IsRecording || IsRecording || IsGuidedRunning ||
            (_grimace?.IsPreviewing ?? false))
        {
            ResultHeadline = "Finish the active capture first.";
            ResultDetail = "Training/setup cannot read the dataset while a recording, guided " +
                           "calibration, or Grimace preview is active.";
            HasResult = true;
            return;
        }

        _workCancellation?.Dispose();
        _workCancellation = new CancellationTokenSource();

        IsBusy = true;
        CanTrain = false;
        CanPrepareModelC = false;
        CanStartRecording = false;
        CanOpenDatasetFolder = false;
        CanStartGuided = false;
        CanStartGrimacePreview = false;
        CanDeleteSavedCorrections = false;
        CanFlagCorrection = false;
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
        _lastRunDirectory = result.RunDirectory;
        _lastCreatedModelPath = result.ExportedModelPath;
        CanOpenLastRunFolder = !string.IsNullOrWhiteSpace(_lastRunDirectory) &&
                               System.IO.Directory.Exists(_lastRunDirectory);
        CanUseLastCreatedModel = !string.IsNullOrWhiteSpace(_lastCreatedModelPath) &&
                                 System.IO.File.Exists(_lastCreatedModelPath);

        if (!result.Success)
        {
            ResultHeadline = result.TrainingSucceeded
                ? "Training succeeded, but activation did not."
                : result.Message;
            var failureLines = new List<string>();
            if (result.TrainingSucceeded)
            {
                failureLines.Add(result.Message);
                if (result.ExportedModelPath != null)
                    failureLines.Add($"Created: {result.ExportedModelPath}");
                failureLines.Add($"Still actually active: " +
                                 (_modelManager.IsActive
                                     ? $"{_modelManager.LoadedMetadata?.DisplayName} — {_modelManager.ActiveModelPath}"
                                     : "Default Baballonia (Stock)"));
            }
            if (result.Remedy == "Show Details")
                failureLines.Add("Open Details below for the exact error.");
            ResultDetail = string.Join(Environment.NewLine + Environment.NewLine, failureLines);
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

        if (result.ExportedModelPath != null)
            lines.Add($"Created: {System.IO.Path.GetFileName(result.ExportedModelPath)}\n" +
                      $"Immutable artifact: {result.ExportedModelPath}");
        lines.Add($"Installed as active: {(result.ActivationSucceeded ? "Yes" : "No")}");

        // Which architecture actually produced this result. Without it the dropdown is the only
        // hint, and that shows the *next* run's choice rather than what was just trained.
        if (!string.IsNullOrEmpty(summary.AdapterType))
            lines.Add($"Trained: {TrainingModelChoice.DisplayName(summary.AdapterType)}.");

        if (result.Run is { } run)
        {
            lines.Add($"Run: {run.RunId}\n" +
                      $"Training corpus: {run.TrainSessionCount} sessions / {FormatFrameCount(run.TrainFrameCount)} frames " +
                      $"(Guided {FormatEvidenceState(run.GuidedInTraining, "included", "not included")}, " +
                      $"Corrections {FormatEvidenceState(run.CorrectionInTraining, "included", "not included")}).\n" +
                      $"Held out: {run.HeldOutSessionCount} sessions / {FormatFrameCount(run.HeldOutFrameCount)} frames." +
                      FormatMissingSessions(run));
        }

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

        lines.Add("Use the Face model selection below to try Stock or any trained model.");
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

    private void RefreshComparisonModels()
    {
        _comparisonModels = _modelManager.DiscoverAvailableModels();
        var activePath = _modelManager.IsActive ? _modelManager.ActiveModelPath : null;
        var labels = new[] { "Default Baballonia (Stock)" }
            .Concat(_comparisonModels.Select(model => model.Label +
                (string.Equals(model.Path, activePath, StringComparison.OrdinalIgnoreCase)
                    ? " — Current"
                    : "")))
            .ToArray();

        if (!ComparisonModelLabels.SequenceEqual(labels))
        {
            ComparisonModelLabels.Clear();
            foreach (var label in labels) ComparisonModelLabels.Add(label);
        }

        _selectedComparisonModelIndex = activePath == null
            ? 0
            : Math.Max(0, _comparisonModels.ToList().FindIndex(model =>
                string.Equals(model.Path, activePath, StringComparison.OrdinalIgnoreCase)) + 1);
        OnPropertyChanged(nameof(SelectedComparisonModelIndex));
        UpdateComparisonSelection();
    }

    private void UpdateComparisonSelection()
    {
        var index = Math.Clamp(SelectedComparisonModelIndex, 0, _comparisonModels.Count);
        if (index == 0)
        {
            UseSelectedModelButtonText = "Use Stock";
            CanUseSelectedModel = _modelManager.IsActive || _modelManager.Enabled;
            SelectedModelProvenance = "Default Baballonia face inference. No personal adapter is applied.";
            CanOpenSelectedModelFolder = false;
            return;
        }

        var model = _comparisonModels[index - 1];
        UseSelectedModelButtonText = $"Use Model {model.Kind.ToUpperInvariant()}";
        CanUseSelectedModel = !_modelManager.IsActive ||
            !string.Equals(_modelManager.ActiveModelPath, model.Path, StringComparison.OrdinalIgnoreCase);
        SelectedModelProvenance = DescribeModel(model);
        CanOpenSelectedModelFolder = true;
    }

    private void UpdateActiveModelStatus()
    {
        var requestedPath = _modelManager.Enabled ? _modelManager.RequestedModelPath : null;
        var loadedPath = _modelManager.ActiveModelPath;
        var activePath = _modelManager.IsActive ? loadedPath : null;
        var requested = requestedPath == null ? null : _comparisonModels.FirstOrDefault(model =>
            string.Equals(model.Path, requestedPath, StringComparison.OrdinalIgnoreCase));
        var active = activePath == null ? null : _comparisonModels.FirstOrDefault(model =>
            string.Equals(model.Path, activePath, StringComparison.OrdinalIgnoreCase));

        var requestedName = !_modelManager.Enabled
            ? "Default Baballonia (Stock)"
            : requested != null
                ? $"Model {requested.Kind.ToUpperInvariant()} — {System.IO.Path.GetFileName(requested.Path)}"
                : System.IO.Path.GetFileName(_modelManager.RequestedModelPath);
        var activeMetadata = _modelManager.LoadedMetadata;
        var activeName = _modelManager.IsActive && activeMetadata != null
            ? $"{activeMetadata.DisplayName} — {System.IO.Path.GetFileName(activePath)}"
            : "Default Baballonia (Stock)";

        ActiveModelName = $"Requested: {requestedName}{Environment.NewLine}Actually active: {activeName}";
        var details = new List<string>();
        if (active != null) details.Add(DescribeModel(active));
        else if (_modelManager.IsActive && activeMetadata != null)
        {
            details.Add($"Adapter: {activeMetadata.AdapterType}");
            details.Add($"Exact file: {activePath}");
            details.Add($"Trained: {activeMetadata.TrainedUtc ?? "unknown"}");
        }
        if (!string.IsNullOrWhiteSpace(_modelManager.FallbackReason))
            details.Add($"FALLBACK: {_modelManager.FallbackReason}");
        if (!_modelManager.IsActive && !string.IsNullOrWhiteSpace(loadedPath))
            details.Add($"RUNTIME FAILURE: {System.IO.Path.GetFileName(loadedPath)} stopped; " +
                        "Stock is actually active. Reload it or choose another model.");
        details.Add($"Stock perception provider: {_modelManager.StockInferenceProvider}");
        details.Add($"Personal adapter provider: {(_modelManager.IsActive ? "CPU" : "inactive")}");
        details.Add($"Embedding runner: {(_modelManager.EmbeddingRunnerAvailable ? "active" : "inactive")}");
        details.Add($"Blend strength: {_modelManager.Blend:P0}");
        ActiveModelProvenance = string.Join(Environment.NewLine, details);
    }

    private static string DescribeModel(AvailablePersonalModel model)
    {
        var metadata = model.Metadata;
        var lines = new List<string>
        {
            $"Family: Model {model.Kind.ToUpperInvariant()}",
            $"Adapter: {metadata.AdapterType}",
            $"File: {System.IO.Path.GetFileName(model.Path)}",
            $"Exact path: {model.Path}",
            $"Trained: {metadata.TrainedUtc ?? "unknown"}",
            $"Requires embedding: {(metadata.RequiresEmbedding ? "yes" : "no")}",
            $"Derived stock MD5: {metadata.BaseModelMd5 ?? "not recorded"}",
        };

        if (model.TrainingRun is { } run)
        {
            lines.Add($"Run: {run.RunId}");
            lines.Add($"Optimization: {run.TrainSessionCount} sessions / {FormatFrameCount(run.TrainFrameCount)} frames");
            lines.Add($"Held out: {run.HeldOutSessionCount} sessions / {FormatFrameCount(run.HeldOutFrameCount)} frames");
            lines.Add($"Guided used to learn: {FormatEvidenceState(run.GuidedInTraining)}; " +
                      $"Corrections used to learn: {FormatEvidenceState(run.CorrectionInTraining)}");
            var missing = FormatMissingSessions(run).Trim();
            if (missing.Length > 0) lines.Add(missing);
            lines.Add($"Validation verdict: {run.Verdict}; improved {run.ExpressionsImproved}, " +
                      $"regressed {run.ExpressionsRegressed}");
        }
        else
        {
            lines.Add($"Artifact source: {model.Source}");
        }

        return string.Join(Environment.NewLine, lines);
    }

    private static string FormatFrameCount(int? count) =>
        count is { } known ? known.ToString("N0") : "unknown";

    private static string FormatEvidenceState(
        bool? state,
        string yes = "yes",
        string no = "no") =>
        state switch
        {
            true => yes,
            false => no,
            null => "unknown (source session is missing)",
        };

    private static string FormatMissingSessions(PersonalTrainingRun run)
    {
        var parts = new List<string>();
        if (run.Train.MissingSessionIds.Count > 0)
            parts.Add("Training recordings no longer on disk: " +
                      string.Join(", ", run.Train.MissingSessionIds));
        if (run.HeldOut.MissingSessionIds.Count > 0)
            parts.Add("Held-out recordings no longer on disk: " +
                      string.Join(", ", run.HeldOut.MissingSessionIds));
        return parts.Count == 0
            ? ""
            : Environment.NewLine + string.Join(Environment.NewLine, parts) + ".";
    }

    [RelayCommand]
    private void RefreshModelLibrary() => RefreshSetup();

    [RelayCommand]
    private void OpenSelectedModelFolder()
    {
        var index = Math.Clamp(SelectedComparisonModelIndex, 0, _comparisonModels.Count);
        if (index == 0) return;
        var directory = System.IO.Path.GetDirectoryName(_comparisonModels[index - 1].Path);
        if (!string.IsNullOrWhiteSpace(directory)) Utils.OpenUrl(directory);
    }

    [RelayCommand]
    private void OpenLastRunFolder()
    {
        if (!string.IsNullOrWhiteSpace(_lastRunDirectory)) Utils.OpenUrl(_lastRunDirectory);
    }

    [RelayCommand]
    private async Task UseLastCreatedModelAsync()
    {
        if (string.IsNullOrWhiteSpace(_lastCreatedModelPath)) return;
        var result = await _modelManager.SelectModelAsync(_lastCreatedModelPath);
        RefreshSetup();
        ModelActionStatus = result.Success
            ? result.Message
            : $"Could not activate the new artifact; the previous model is still in use. {result.Message}";
    }

    public async Task SelectManualModelAsync(string path)
    {
        var result = await _modelManager.SelectModelAsync(path);
        RefreshSetup();
        ModelActionStatus = result.Success
            ? result.Message
            : $"Could not use {System.IO.Path.GetFileName(path)}; the previous model is still in use. " +
              result.Message;
    }

    [RelayCommand]
    private async Task UseSelectedModelAsync()
    {
        var index = Math.Clamp(SelectedComparisonModelIndex, 0, _comparisonModels.Count);
        if (index == 0)
        {
            await UseStockAsync();
            return;
        }

        var model = _comparisonModels[index - 1];
        if (model.Kind == "c" && !_modelManager.EmbeddingRunnerAvailable)
        {
            var stock = System.IO.Path.Combine(AppContext.BaseDirectory, "faceModel.onnx");
            var derived = EmbeddingModelStore.TryGetValid(stock);
            if (!derived.Valid)
            {
                ModelActionStatus = "Model C is trained, but its shared-feature runner is not ready. " +
                                    "Select C under Train and use Prepare Model C first.";
                return;
            }

            await _modelManager.SetEmbeddingRunnerEnabledAsync(true);
        }

        var result = await _modelManager.SelectModelAsync(model.Path);
        _personalModelEnabled = _modelManager.Enabled;
        OnPropertyChanged(nameof(PersonalModelEnabled));
        RefreshSetup();
        ModelActionStatus = result.Success
            ? result.Message
            : $"Could not use Model {model.Kind.ToUpperInvariant()}; the previous model is still in use. " +
              result.Message;
    }

    [RelayCommand]
    private async Task UseStockAsync()
    {
        _modelManager.SetEnabled(false);
        var result = await _modelManager.ReloadAsync();
        _personalModelEnabled = false;
        OnPropertyChanged(nameof(PersonalModelEnabled));
        RefreshSetup();
        ModelActionStatus = result.Message;
    }

    [RelayCommand]
    private async Task UsePersonalAsync()
    {
        _modelManager.SetEnabled(true);
        var result = await _modelManager.ReloadAsync();
        _personalModelEnabled = _modelManager.Enabled;
        OnPropertyChanged(nameof(PersonalModelEnabled));
        RefreshSetup();
        ModelActionStatus = result.Success
            ? result.Message
            : $"Could not activate the requested personal model. {result.Message}";
    }

    [RelayCommand]
    private async Task ReloadModelAsync()
    {
        var result = await _modelManager.ReloadAsync();
        RefreshSetup();
        ModelActionStatus = result.Success
            ? result.Message
            : $"Reload failed; Stock is actually active. {result.Message}";
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
        UpdateActiveModelStatus();
    }

    // =============================================================================================
    // Live data
    // =============================================================================================

    private void OnRawExpressions(FacePipelineEvents.NewRawExpressionsEvent e)
    {
        _comparisonValues.AcceptRaw(e.rawResult, _modelManager.IsActive);
    }

    private void OnCorrectedExpressions(FacePipelineEvents.NewCorrectedExpressionsEvent e)
    {
        _comparisonValues.AcceptCorrected(e.rawResult, e.correctedResult);
    }

    private void OnTick()
    {
        if (_recorder.IsRecording)
            RecordingStatus = $"Recording... {_recorder.FramesWritten} frames captured.";

        // A corrector can fail inside the inference thread after loading successfully. Poll the
        // manager's effective state so the picker drops its Current badge and the status switches
        // to Stock without requiring a page navigation or manual refresh.
        var modelActive = _modelManager.IsActive;
        if (modelActive != _lastObservedModelActive)
        {
            _lastObservedModelActive = modelActive;
            RefreshComparisonModels();
            UpdateActiveModelStatus();
        }

        for (var i = 0; i < Comparison.Count && i < _comparisonValues.Count; i++)
            Comparison[i].Update(_comparisonValues.StockAt(i), _comparisonValues.PersonalAt(i));

        if (SortByDelta)
            SortComparisonByDelta();

        UpdateAudioDiagnostics();

        // The camera is "running" if a frame arrived recently; there is no event when it stops.
        var seen = (DateTime.UtcNow - _lastFrameUtc).TotalSeconds < 2;
        var recorderFrameReady = _recorder.HasRecentSourceFrame;
        if (seen != _cameraSeenRecently || recorderFrameReady != _lastRecorderFrameReady)
        {
            _cameraSeenRecently = seen;
            _lastRecorderFrameReady = recorderFrameReady;
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
        _grimaceTimer.Stop();
        _faceEventBus.Unsubscribe(_frameHandler);
        _faceEventBus.Unsubscribe(_rawHandler);
        _faceEventBus.Unsubscribe(_correctedHandler);

        _workCancellation?.Cancel();
        _workCancellation?.Dispose();

        // Navigating away mid-calibration must hand the avatar back to live tracking rather than
        // leaving it frozen in whatever expression was being commanded.
        _guided?.Dispose();
        _grimace?.Dispose();

        if (_recorder.IsRecording)
            _recorder.StopSessionAsync().GetAwaiter().GetResult();
    }
}
