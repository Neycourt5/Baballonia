using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
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
/// Personalization workspace. In this phase it exists to collect training data and to make the
/// capture path observable: start/stop a session, watch the exact frames being recorded, and see
/// how many unique frames per second the camera is really delivering.
///
/// Guided avatar-driven capture and the stock-vs-personal debug panel land in later phases; the
/// wiring here (recorder, live preview) is what they build on.
/// </summary>
public partial class PersonalizationViewModel : ViewModelBase, IDisposable
{
    private readonly DatasetRecorderService _recorder;
    private readonly PersonalModelManager _modelManager;
    private readonly ILocalSettingsService _settings;
    private readonly IFacePipelineEventBus _faceEventBus;
    private readonly ILogger<PersonalizationViewModel> _logger;

    private readonly Action<FacePipelineEvents.NewTransformedFrameEvent> _frameHandler;
    private readonly Action<FacePipelineEvents.NewRawExpressionsEvent> _rawHandler;
    private readonly Action<FacePipelineEvents.NewCorrectedExpressionsEvent> _correctedHandler;
    private readonly DispatcherTimer _statusTimer;

    private WriteableBitmap? _backingBitmap;

    // Latest values, written on the processing tick and drained by the status timer. Binding at
    // 100 Hz would swamp the UI; 4 Hz is more than enough to watch values move.
    private readonly float[] _latestStock = new float[PersonalizationSchema.ExpressionCount];
    private readonly float[] _latestPersonal = new float[PersonalizationSchema.ExpressionCount];
    private volatile bool _hasCorrectedValues;

    [ObservableProperty] private WriteableBitmap? _preview;
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private string _statusText = "Idle.";
    [ObservableProperty] private string _datasetPath = PersonalizationPaths.DatasetRoot;

    [ObservableProperty] private bool _personalModelEnabled;
    [ObservableProperty] private double _personalStrength = 100;
    [ObservableProperty] private string _modelStatusText = "Personal model not loaded.";
    [ObservableProperty] private string _modelPath = "";
    [ObservableProperty] private bool _sortByDelta = true;

    /// <summary>One row per expression, mutated in place rather than rebuilt.</summary>
    public ObservableCollection<ExpressionComparisonRow> Comparison { get; } = [];

    /// <summary>Neutral and Speech are recordable now; Guided needs the cue engine.</summary>
    public IReadOnlyList<string> SessionTypes { get; } = [nameof(SessionType.Neutral), nameof(SessionType.Speech)];

    [ObservableProperty] private string _selectedSessionType = nameof(SessionType.Neutral);
    [ObservableProperty] private string _notes = "";

    public PersonalizationViewModel(
        DatasetRecorderService recorder,
        PersonalModelManager modelManager,
        ILocalSettingsService settings,
        IFacePipelineEventBus faceEventBus,
        ILogger<PersonalizationViewModel> logger)
    {
        _recorder = recorder;
        _modelManager = modelManager;
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
        _modelPath = _modelManager.ModelPath;

        // Stats and comparison values are polled rather than pushed: the recorder updates its
        // counters on a background writer, and 4 Hz is plenty for a readout.
        _statusTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _statusTimer.Tick += (_, _) => RefreshStatus();
        _statusTimer.Start();

        _ = ReloadModelAsync();
    }

    [RelayCommand]
    private void StartRecording()
    {
        if (_recorder.IsRecording)
            return;

        try
        {
            var type = Enum.Parse<SessionType>(SelectedSessionType);
            var id = _recorder.StartSession(type, camera: null, notes: string.IsNullOrWhiteSpace(Notes) ? null : Notes);
            IsRecording = true;
            StatusText = $"Recording {id}...";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Personalization: failed to start recording");
            StatusText = $"Could not start recording: {ex.Message}";
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

            StatusText = summary == null
                ? "Idle."
                : $"Saved {summary.FrameCount} frames at {summary.EffectiveFps:F1} unique fps" +
                  (summary.DroppedFrames > 0 ? $" ({summary.DroppedFrames} dropped)" : "") +
                  $" to {summary.SessionId}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Personalization: failed to stop recording");
            IsRecording = false;
            StatusText = $"Error finishing session: {ex.Message}";
        }
    }

    [RelayCommand]
    private void OpenDatasetFolder()
    {
        System.IO.Directory.CreateDirectory(PersonalizationPaths.DatasetRoot);
        Utils.OpenUrl(PersonalizationPaths.DatasetRoot);
    }

    [RelayCommand]
    private async Task ReloadModelAsync()
    {
        var result = await _modelManager.ReloadAsync();
        ModelStatusText = result.Success
            ? (_modelManager.IsActive ? $"Active: {result.Message}" : result.Message)
            : $"Not loaded: {result.Message}";
        ModelPath = _modelManager.ModelPath;
    }

    partial void OnPersonalModelEnabledChanged(bool value)
    {
        _settings.SaveSetting(PersonalModelManager.EnabledSetting, value);
        _ = ReloadModelAsync();
    }

    partial void OnPersonalStrengthChanged(double value)
    {
        // Blend is a live evaluation control: apply immediately rather than on reload, so A/B
        // comparison is instant.
        _modelManager.Blend = (float)(value / 100.0);
    }

    private void OnRawExpressions(FacePipelineEvents.NewRawExpressionsEvent e)
    {
        // Runs under the event bus lock on the processing tick: copy and return.
        if (_hasCorrectedValues)
            return;

        // With no corrector installed, personal == stock so the panel shows a passthrough.
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

    private void RefreshStatus()
    {
        if (_recorder.IsRecording)
            StatusText = $"Recording {_recorder.CurrentSessionId} - {_recorder.FramesWritten} frames";

        for (var i = 0; i < Comparison.Count; i++)
            Comparison[i].Update(_latestStock[i], _latestPersonal[i]);

        if (SortByDelta)
            SortComparisonByDelta();
    }

    /// <summary>
    /// Surfaces the expressions the personal model is changing most - the fastest way to see what
    /// it actually learned, and to spot it moving something it should not.
    /// </summary>
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

    /// <summary>
    /// Shows the post-transform frame, i.e. exactly what inference and recording consume, so what
    /// the user sees here is what the model is being trained on.
    /// </summary>
    private void OnTransformedFrame(FacePipelineEvents.NewTransformedFrameEvent e)
    {
        if (e.image is null || e.image.Empty())
            return;

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

        // Same nudge the home page uses: the bitmap instance is unchanged, so the binding needs a
        // null round-trip to repaint.
        Preview = null;
        Preview = _backingBitmap;
    }

    public void Dispose()
    {
        _statusTimer.Stop();
        _faceEventBus.Unsubscribe(_frameHandler);
        _faceEventBus.Unsubscribe(_rawHandler);
        _faceEventBus.Unsubscribe(_correctedHandler);

        if (_recorder.IsRecording)
            _recorder.StopSessionAsync().GetAwaiter().GetResult();
    }
}
