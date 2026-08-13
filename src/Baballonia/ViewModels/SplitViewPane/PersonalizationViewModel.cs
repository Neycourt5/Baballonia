using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
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
    private readonly IFacePipelineEventBus _faceEventBus;
    private readonly ILogger<PersonalizationViewModel> _logger;

    private readonly Action<FacePipelineEvents.NewTransformedFrameEvent> _frameHandler;
    private readonly DispatcherTimer _statusTimer;

    private WriteableBitmap? _backingBitmap;

    [ObservableProperty] private WriteableBitmap? _preview;
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private string _statusText = "Idle.";
    [ObservableProperty] private string _datasetPath = PersonalizationPaths.DatasetRoot;

    /// <summary>Neutral and Speech are recordable now; Guided needs the cue engine.</summary>
    public IReadOnlyList<string> SessionTypes { get; } = [nameof(SessionType.Neutral), nameof(SessionType.Speech)];

    [ObservableProperty] private string _selectedSessionType = nameof(SessionType.Neutral);
    [ObservableProperty] private string _notes = "";

    public PersonalizationViewModel(
        DatasetRecorderService recorder,
        IFacePipelineEventBus faceEventBus,
        ILogger<PersonalizationViewModel> logger)
    {
        _recorder = recorder;
        _faceEventBus = faceEventBus;
        _logger = logger;

        _frameHandler = OnTransformedFrame;
        _faceEventBus.Subscribe(_frameHandler);

        // Recording stats are polled rather than pushed: the recorder updates them on a background
        // writer, and 4 Hz is plenty for a progress readout.
        _statusTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _statusTimer.Tick += (_, _) => RefreshStatus();
        _statusTimer.Start();
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

    private void RefreshStatus()
    {
        if (!_recorder.IsRecording)
            return;

        StatusText = $"Recording {_recorder.CurrentSessionId} - {_recorder.FramesWritten} frames";
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

        if (_recorder.IsRecording)
            _recorder.StopSessionAsync().GetAwaiter().GetResult();
    }
}
