using Avalonia.Threading;
using Baballonia.Models;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Personalization.Eye;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;

namespace Baballonia.ViewModels.SplitViewPane;

/// <summary>
/// Record a guided eye capture, fit a correction from it, and judge the result live.
/// </summary>
/// <remarks>
/// <para>Kept apart from <see cref="PersonalizationViewModel"/> rather than added to it. That class
/// is already 1,900 lines covering the whole face workflow, and the two share no state — only a
/// page.</para>
///
/// <para>The comparison table is the point of the whole screen: you cannot judge a correction
/// from a number in a status line, only by watching the two columns move while you look around. It
/// is fed from the pipeline's own events on the inference thread and drained to the UI a few times
/// a second, because binding at frame rate would swamp the dispatcher for a table nobody can read
/// that fast.</para>
/// </remarks>
public partial class EyePersonalizationViewModel : ViewModelBase, IDisposable
{
    /// <summary>Fast enough to watch values move, slow enough not to flood the UI thread.</summary>
    private static readonly TimeSpan UiRefresh = TimeSpan.FromMilliseconds(250);

    /// <summary>The routine is a fixed sequence; this is just what pumps it.</summary>
    private static readonly TimeSpan CaptureTick = TimeSpan.FromMilliseconds(50);

    private readonly EyePersonalizationManager _manager;
    private readonly EyeGuidedCaptureService _capture;
    private readonly IEyePipelineEventBus _bus;
    private readonly ExpressionComparisonBuffer _values;
    private readonly DispatcherTimer _uiTimer;
    private readonly DispatcherTimer _captureTimer;
    private readonly Action<EyePipelineEvents.NewRawEyeExpressionsEvent> _rawHandler;
    private readonly Action<EyePipelineEvents.NewCorrectedEyeExpressionsEvent> _correctedHandler;

    private bool _disposed;
    private string? _lastSessionId;

    public EyePersonalizationViewModel(
        EyePersonalizationManager manager,
        EyeGuidedCaptureService capture,
        IEyePipelineEventBus bus)
    {
        _manager = manager;
        _capture = capture;
        _bus = bus;
        _values = new ExpressionComparisonBuffer(EyePersonalizationSchema.ExpressionCount);

        Comparison = new ObservableCollection<ExpressionComparisonRow>(
            EyePersonalizationSchema.ExpressionNames
                .Select((name, index) => new ExpressionComparisonRow(index, name)));

        _rawHandler = message => _values.AcceptRaw(message.rawResult, _manager.IsActive);
        _correctedHandler = message => _values.AcceptCorrected(message.rawResult, message.correctedResult);
        _bus.Subscribe(_rawHandler);
        _bus.Subscribe(_correctedHandler);

        Strength = _manager.Blend * 100.0;
        Enabled = _manager.Enabled;
        Status = _manager.Status;
        RoutineDescription =
            $"About {EyeGuidedCaptureService.RoutineSeconds:F0} seconds: follow the dot in nine " +
            "directions, then open wide, squint, close, wink each eye and blink normally.";

        _uiTimer = new DispatcherTimer { Interval = UiRefresh };
        _uiTimer.Tick += RefreshFromPipeline;
        _uiTimer.Start();

        _captureTimer = new DispatcherTimer { Interval = CaptureTick };
        _captureTimer.Tick += AdvanceCapture;
    }

    /// <summary>One row per eye channel, mutated in place so the list does not flicker.</summary>
    public ObservableCollection<ExpressionComparisonRow> Comparison { get; }

    [ObservableProperty] private string _status = "";
    [ObservableProperty] private string _routineDescription = "";
    [ObservableProperty] private bool _isCapturing;
    [ObservableProperty] private double _captureProgress;
    [ObservableProperty] private string _captureInstruction = "";
    [ObservableProperty] private bool _hasSavedFit;
    [ObservableProperty] private string _fitSummary = "";
    [ObservableProperty] private bool _calibrationStale;

    /// <summary>Where the dot is, mirrored on the desktop when there is no headset.</summary>
    [ObservableProperty] private bool _showDesktopDot;

    /// <summary>
    /// Dot position in the desktop canvas, in pixels.
    /// </summary>
    /// <remarks>
    /// The canvas is a fixed size so the mapping is deterministic and the dot cannot drift with the
    /// window. Both are already screen-space (positive Y is down), matching the headset panel, so
    /// no further sign flip happens here.
    /// </remarks>
    [ObservableProperty] private double _dotCanvasLeft = DotCanvasWidth / 2 - DotRadius;
    [ObservableProperty] private double _dotCanvasTop = DotCanvasHeight / 2 - DotRadius;

    public const double DotCanvasWidth = 440;
    public const double DotCanvasHeight = 240;
    private const double DotRadius = 19;

    /// <summary>Percent, because a slider labelled 0..1 reads as a mistake.</summary>
    [ObservableProperty] private double _strength = 100;

    [ObservableProperty] private bool _enabled = true;

    public bool CanRecord => !IsCapturing;
    public bool CanFit => !IsCapturing && _lastSessionId is not null;
    public bool CanClear => !IsCapturing && HasSavedFit;

    partial void OnStrengthChanged(double value) =>
        // Applied immediately: A/B is only useful if it is instant.
        _manager.Blend = (float)(value / 100.0);

    partial void OnEnabledChanged(bool value)
    {
        _manager.Enabled = value;
        RefreshStatus();
    }

    partial void OnIsCapturingChanged(bool value)
    {
        OnPropertyChanged(nameof(CanRecord));
        OnPropertyChanged(nameof(CanFit));
        OnPropertyChanged(nameof(CanClear));
    }

    partial void OnHasSavedFitChanged(bool value) => OnPropertyChanged(nameof(CanClear));

    /// <summary>Blend to 0 — the base model, unmodified.</summary>
    [RelayCommand]
    private void CompareBase() => Strength = 0;

    /// <summary>Blend to 100 — the fitted correction at full strength.</summary>
    [RelayCommand]
    private void ComparePersonal() => Strength = 100;

    [RelayCommand]
    private void StartCapture()
    {
        if (IsCapturing)
            return;

        var result = _capture.Start();
        Status = result.Message;
        if (!result.Started)
            return;

        IsCapturing = true;
        _captureTimer.Start();
    }

    [RelayCommand]
    private Task CancelCapture() => FinishCapture(completed: false);

    [RelayCommand]
    private void RetryPose() => _capture.Retry();

    [RelayCommand]
    private void SkipPose() => _capture.Skip();

    /// <summary>Fits a correction from the session just recorded and applies it.</summary>
    [RelayCommand]
    private void FitFromLastCapture()
    {
        if (_lastSessionId is not { } sessionId)
            return;

        var result = _manager.FitFromSession(sessionId);
        Status = result.Message;
        RefreshStatus();
    }

    [RelayCommand]
    private void ClearFit()
    {
        _manager.Clear();
        RefreshStatus();
        Status = "Personalized eye correction removed. Running the base model.";
    }

    private void AdvanceCapture(object? sender, EventArgs e)
    {
        if (_disposed)
            return;

        var running = _capture.Tick();
        CaptureProgress = _capture.Progress;
        CaptureInstruction = _capture.Instruction;

        var frame = _capture.CurrentFrame;
        ShowDesktopDot = frame is { TargetX: not null, TargetY: not null };
        if (frame is { TargetX: { } x, TargetY: { } y })
        {
            DotCanvasLeft = (x + 1) / 2 * DotCanvasWidth - DotRadius;
            DotCanvasTop = (y + 1) / 2 * DotCanvasHeight - DotRadius;
        }

        if (!running)
            _ = FinishCapture(completed: true);
    }

    private async Task FinishCapture(bool completed)
    {
        _captureTimer.Stop();
        ShowDesktopDot = false;

        var session = await _capture.StopAsync(completed);
        IsCapturing = false;
        CaptureProgress = 0;
        CaptureInstruction = "";

        if (session is null)
        {
            Status = "Capture stopped.";
            return;
        }

        _lastSessionId = session.sessionId;
        OnPropertyChanged(nameof(CanFit));

        Status = completed
            ? $"Recorded {session.frameCount} frames. Fit a correction from it when you are ready."
            : $"Capture cancelled after {session.frameCount} frames; it was still saved.";
    }

    private void RefreshFromPipeline(object? sender, EventArgs e)
    {
        if (_disposed)
            return;

        for (var i = 0; i < Comparison.Count && i < _values.Count; i++)
            Comparison[i].Update(_values.StockAt(i), _values.PersonalAt(i));
    }

    private void RefreshStatus()
    {
        Status = _manager.Status;
        var profile = _manager.SavedProfile;
        HasSavedFit = profile is not null;
        FitSummary = profile is null
            ? "No correction fitted yet."
            : $"Left gain {profile.Left.Ax:F3}/{profile.Left.By:F3}, " +
              $"right gain {profile.Right.Ax:F3}/{profile.Right.By:F3}.";
        CalibrationStale = _manager.IsCalibrationStale;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        _uiTimer.Stop();
        _captureTimer.Stop();
        _bus.Unsubscribe(_rawHandler);
        _bus.Unsubscribe(_correctedHandler);

        // A capture left running would keep recording into a session nobody is watching.
        if (_capture.IsRunning)
            _ = _capture.StopAsync(completed: false);
    }
}
