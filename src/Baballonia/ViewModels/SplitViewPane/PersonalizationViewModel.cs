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

    // ---- training -----------------------------------------------------------------------------
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
    [ObservableProperty] private bool _personalModelEnabled;
    [ObservableProperty] private double _personalStrength = 100;
    [ObservableProperty] private bool _showAdvanced;
    [ObservableProperty] private bool _sortByDelta = true;

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
        ILogger<PersonalizationViewModel> logger)
    {
        _recorder = recorder;
        _modelManager = modelManager;
        _trainingService = trainingService;
        _environment = environment;
        _settings = settings;
        _faceEventBus = faceEventBus;
        _logger = logger;

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

        // Offer setup only when it can actually succeed: scripts located and Python available.
        CanSetUpTools = !setup.TrainingTools.Ready
                        && setup.TrainingRoot != null
                        && PersonalizationEnvironment.FindHostPython() != null;

        NeutralCount = setup.DatasetStatus.NeutralSessions;
        SpeechCount = setup.DatasetStatus.SpeechSessions;
        RecordingAdvice = setup.DatasetStatus.NextRecommendation
                          ?? "You have enough recordings to train a good model.";

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

    // =============================================================================================
    // Training
    // =============================================================================================

    [RelayCommand]
    private async Task TrainAsync()
    {
        await RunWorkAsync(
            "Starting...",
            (progress, token) => _trainingService.TrainAsync("a", progress, token));

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

        if (summary.NeutralStockFalseActivation is { } stockNeutral &&
            summary.NeutralPersonalFalseActivation is { } personalNeutral)
        {
            lines.Add($"Expressions firing while your face is at rest: " +
                      $"{stockNeutral:P1} before, {personalNeutral:P1} after.");
        }

        if (summary.MeanStockMae is { } stockMae && summary.MeanPersonalMae is { } personalMae)
            lines.Add($"Average expression error: {stockMae:F3} before, {personalMae:F3} after.");

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
            lines.Add("More recordings usually help, especially Neutral ones taken on different days.");
        }

        lines.Add("Use Stock / Personal below to hear and see the difference for yourself.");
        ResultDetail = string.Join(Environment.NewLine + Environment.NewLine, lines);
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
        _faceEventBus.Unsubscribe(_frameHandler);
        _faceEventBus.Unsubscribe(_rawHandler);
        _faceEventBus.Unsubscribe(_correctedHandler);

        _workCancellation?.Cancel();
        _workCancellation?.Dispose();

        if (_recorder.IsRecording)
            _recorder.StopSessionAsync().GetAwaiter().GetResult();
    }
}
