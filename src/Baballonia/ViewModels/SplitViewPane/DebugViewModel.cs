using Avalonia.Threading;
using Baballonia.Services;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.VideoSources;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.IO;
using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace Baballonia.ViewModels.SplitViewPane;

/// <summary>
/// One camera's lifecycle state, so a stall is visible as a state and not only as a frame-age
/// number that has quietly stopped climbing.
/// </summary>
public partial class CameraStateRow : ObservableObject
{
    [ObservableProperty] private string _name = "";
    [ObservableProperty] private string _state = "";
}

/// <summary>One row in the per-thread CPU table.</summary>
public partial class ThreadCpuRow : ObservableObject
{
    [ObservableProperty] private string _name = "";
    [ObservableProperty] private double _cpuPercent;
}

/// <summary>One camera card (eye left/right or mouth).</summary>
public partial class CameraRow : ObservableObject
{
    [ObservableProperty] private string _name = "";
    [ObservableProperty] private double _deliveredFps;
    [ObservableProperty] private string _frameAge = "No frames yet";
    [ObservableProperty] private string _droppedSummary = "—";
    [ObservableProperty] private double _negotiatedFps;
    [ObservableProperty] private string _resolution = "—";
    [ObservableProperty] private string _format = "—";
}

/// <summary>
/// Live performance page. Rates are monotonic counter deltas measured against a Stopwatch clock and
/// EWMA-smoothed; per-thread CPU comes from the always-on <see cref="ThreadProfiler"/>.
/// </summary>
public partial class DebugViewModel : ViewModelBase, IDisposable
{
    private const int MaxThreadRows = 14;

    // EWMA weight per 500 ms sample: smooths jitter but still reacts within ~1-2 s.
    private const double Smoothing = 0.3;

    private static readonly long DropWindowTicks = 60 * Stopwatch.Frequency;

    private sealed class CamState
    {
        public long PrevFrames;
        public double FpsEwma;
        public readonly Queue<(long Timestamp, long Dropped)> DropWindow = new();
    }

    private readonly PipelineMetrics _metrics;
    private readonly ThreadProfiler _profiler;
    private readonly IReadOnlyList<ICameraSlotHost> _slotHosts;
    private readonly CameraWatchdogService? _watchdog;
    private readonly DispatcherTimer _timer;
    private readonly Dictionary<SingleCameraSource, CamState> _camStates = new();

    private long _prevUi, _prevEye, _prevFace, _prevRender;
    private long _prevTimestamp;

    // Incremented once per compositor frame by the view; surfaces Avalonia's render-thread fps.
    public long RenderTicks;

    [ObservableProperty] private double _renderFps;
    [ObservableProperty] private double _uiLoopFps;
    [ObservableProperty] private double _eyeInferenceFps;
    [ObservableProperty] private double _faceInferenceFps;

    // CPU hotspots.
    [ObservableProperty] private double _processCpuPercent;
    [ObservableProperty] private int _processorCount;
    public ObservableCollection<ThreadCpuRow> Threads { get; } = new();

    // One card per active camera (eye left/right or mouth).
    public ObservableCollection<CameraRow> Cameras { get; } = new();

    // Recovery: what each slot thinks it is doing, and how often the watchdog has had to step in.
    public ObservableCollection<CameraStateRow> CameraStates { get; } = new();
    [ObservableProperty] private long _reconnectCount;
    [ObservableProperty] private long _corruptFrames;

    // Live eye output, after calibration and shaping. Raw sits beside calibrated so a bad
    // calibration reads as a divergence between the two rather than as vaguely wrong tracking.
    [ObservableProperty] private string _eyeShapeSource = "—";
    [ObservableProperty] private string _eyeCorrector = "—";
    [ObservableProperty] private string _eyeSync = "—";
    [ObservableProperty] private string _eyeBlinkGuard = "—";
    [ObservableProperty] private string _leftOpenness = "—";
    [ObservableProperty] private string _rightOpenness = "—";
    [ObservableProperty] private string _leftShapes = "—";
    [ObservableProperty] private string _rightShapes = "—";
    [ObservableProperty] private string _leftGaze = "—";
    [ObservableProperty] private string _rightGaze = "—";

    // Eye pipeline stage timings (ms).
    [ObservableProperty] private double _eyeCaptureMs;
    [ObservableProperty] private double _eyeTransformMs;
    [ObservableProperty] private double _eyeInferenceMs;
    [ObservableProperty] private double _eyePostMs;
    [ObservableProperty] private double _eyeCorrectMs;

    // Face pipeline stage timings (ms).
    [ObservableProperty] private double _faceCaptureMs;
    [ObservableProperty] private double _faceTransformMs;
    [ObservableProperty] private double _faceInferenceMs;
    [ObservableProperty] private double _facePostMs;

    private readonly EyeStageTrace? _eyeTrace;
    [ObservableProperty] private string _eyeTraceStatus = "Eye trace off. Numerical stages only; no images or audio.";

    [RelayCommand]
    private void StartEyeTrace()
    {
        _eyeTrace?.Start();
        EyeTraceStatus = "Eye trace running in a bounded memory ring. Reproduce the issue, then press Mark eye issue.";
    }

    [RelayCommand]
    private void StopEyeTrace()
    {
        _eyeTrace?.Stop();
        EyeTraceStatus = "Eye trace stopped. The ring can still be exported with Mark eye issue.";
    }

    [RelayCommand]
    private async Task MarkEyeIssueAsync()
    {
        if (_eyeTrace == null || _eyeTrace.Count == 0) { EyeTraceStatus = "Start the trace with eye tracking running first."; return; }
        try
        {
            var path = await _eyeTrace.MarkAsync(Path.Combine(Utils.PersistentDataDirectory, "Diagnostics", "Eyes"));
            EyeTraceStatus = "Marked and saved locally: " + path + ". Receiver and avatar stages are not observed.";
        }
        catch (Exception ex) { EyeTraceStatus = "Could not export eye trace: " + ex.Message; }
    }

    public DebugViewModel(
        PipelineMetrics metrics,
        ThreadProfiler profiler,
        IEnumerable<ICameraSlotHost>? slotHosts = null,
        CameraWatchdogService? watchdog = null,
        EyeStageTrace? eyeTrace = null)
    {
        _eyeTrace = eyeTrace;
        _metrics = metrics;
        _profiler = profiler;
        _slotHosts = slotHosts?.ToArray() ?? [];
        _watchdog = watchdog;
        ProcessorCount = profiler.ProcessorCount;
        _prevTimestamp = Stopwatch.GetTimestamp();
        _prevUi = metrics.UiTicks;
        _prevEye = metrics.EyeInferences;
        _prevFace = metrics.FaceInferences;
        _prevRender = RenderTicks;

        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
        _timer.Tick += Sample;
        _timer.Start();
    }

    /// <summary>EWMA that seeds from the first sample (so it converges immediately, not from zero).</summary>
    private static double Smooth(double previous, double sample) =>
        previous <= 0 ? sample : previous + Smoothing * (sample - previous);

    private void Sample(object? sender, EventArgs e)
    {
        var now = Stopwatch.GetTimestamp();
        var dt = (now - _prevTimestamp) / (double)Stopwatch.Frequency;
        if (dt <= 0)
            return;

        var eye = _metrics.EyeInferences;
        RenderFps = Smooth(RenderFps, (RenderTicks - _prevRender) / dt);
        UiLoopFps = Smooth(UiLoopFps, (_metrics.UiTicks - _prevUi) / dt);
        EyeInferenceFps = Smooth(EyeInferenceFps, (eye - _prevEye) / dt);
        FaceInferenceFps = Smooth(FaceInferenceFps, (_metrics.FaceInferences - _prevFace) / dt);

        UpdateCameras(now, dt);

        EyeCaptureMs = _metrics.EyeCaptureMs;
        EyeTransformMs = _metrics.EyeTransformMs;
        EyeInferenceMs = _metrics.EyeInferenceMs;
        EyePostMs = _metrics.EyePostMs;
        EyeCorrectMs = _metrics.EyeCorrectMs;

        FaceCaptureMs = _metrics.FaceCaptureMs;
        FaceTransformMs = _metrics.FaceTransformMs;
        FaceInferenceMs = _metrics.FaceInferenceMs;
        FacePostMs = _metrics.FacePostMs;

        UpdateThreads();
        UpdateRecovery();
        UpdateEyeOutput();

        _prevRender = RenderTicks;
        _prevUi = _metrics.UiTicks;
        _prevEye = eye;
        _prevFace = _metrics.FaceInferences;
        _prevTimestamp = now;
    }

    private void UpdateRecovery()
    {
        ReconnectCount = _watchdog?.ReconnectCount ?? 0;
        CorruptFrames = Interlocked.Read(ref _metrics.EyeCorruptFrames);

        var slots = _slotHosts.SelectMany(host => host.Slots).ToList();

        while (CameraStates.Count < slots.Count) CameraStates.Add(new CameraStateRow());
        while (CameraStates.Count > slots.Count) CameraStates.RemoveAt(CameraStates.Count - 1);

        for (var i = 0; i < slots.Count; i++)
        {
            CameraStates[i].Name = slots[i].Name;
            CameraStates[i].State = slots[i].State.ToString();
        }
    }

    private void UpdateEyeOutput()
    {
        EyeShapeSource = _metrics.EyeWidenSquintDerived
            ? "derived from openness (model has no widen/squint)"
            : "model channels";

        // A mean delta of exactly zero distinguishes "no corrector installed" from "installed and
        // currently agreeing with the base model".
        EyeCorrector = _metrics.EyeCorrectorDelta > 0f
            ? $"active, mean |Δ| {_metrics.EyeCorrectorDelta:F4}, {_metrics.EyeCorrectMs:F2} ms"
            : "not correcting";

        EyeSync = _metrics.EyeSyncReleased ? "released (wink)" : "coupled";

        // Compact by design: the settings panel owns the full picture, this is the at-a-glance one.
        EyeBlinkGuard = !_metrics.EyeBlinkGuardEnabled
            ? "off"
            : (_metrics.EyeBlinkGuardIntervening ? "REACQUIRING" : "good") +
              $"   glitches prevented {_metrics.EyeBlinkGuardGlitches}";

        LeftOpenness = Openness(_metrics.EyeLeftRawOpenness, _metrics.EyeLeftOpenness);
        RightOpenness = Openness(_metrics.EyeRightRawOpenness, _metrics.EyeRightOpenness);
        LeftShapes = Shapes(_metrics.EyeLeftWiden, _metrics.EyeLeftSquint);
        RightShapes = Shapes(_metrics.EyeRightWiden, _metrics.EyeRightSquint);
        LeftGaze = Gaze(_metrics.EyeLeftGazeX, _metrics.EyeLeftGazeY);
        RightGaze = Gaze(_metrics.EyeRightGazeX, _metrics.EyeRightGazeY);

        static string Openness(float raw, float calibrated) =>
            $"{calibrated:F2}  (raw {raw:F2})";

        static string Shapes(float widen, float squint) =>
            $"widen {widen:F2}   squint {squint:F2}";

        static string Gaze(float x, float y) => $"x {x:+0.00;-0.00; 0.00}   y {y:+0.00;-0.00; 0.00}";
    }

    private void UpdateCameras(long now, double dt)
    {
        var current = new List<(string Label, SingleCameraSource Source)>(3);
        if (_metrics.EyeDual)
        {
            if (_metrics.EyeLeftSource is { } el) current.Add(("Eye (left)", el));
            if (_metrics.EyeRightSource is { } er) current.Add(("Eye (right)", er));
        }
        else if (_metrics.EyeLeftSource is { } single)
        {
            current.Add(("Eye", single));
        }
        if (_metrics.FaceSource is { } mouth) current.Add(("Mouth", mouth));

        while (Cameras.Count < current.Count) Cameras.Add(new CameraRow());
        while (Cameras.Count > current.Count) Cameras.RemoveAt(Cameras.Count - 1);

        var active = new HashSet<SingleCameraSource>();
        for (var i = 0; i < current.Count; i++)
        {
            var (label, src) = current[i];
            active.Add(src);
            var cap = src.Capture;

            if (!_camStates.TryGetValue(src, out var st))
            {
                st = new CamState { PrevFrames = cap.FramesProduced };
                st.DropWindow.Enqueue((now, cap.FramesDropped));
                _camStates[src] = st;
            }

            var frames = cap.FramesProduced;
            st.FpsEwma = Smooth(st.FpsEwma, (frames - st.PrevFrames) / dt);
            st.PrevFrames = frames;

            var drop = cap.FramesDropped;
            st.DropWindow.Enqueue((now, drop));
            while (st.DropWindow.Count > 1 && st.DropWindow.Peek().Timestamp < now - DropWindowTicks)
                st.DropWindow.Dequeue();
            var lost = Math.Max(0, drop - st.DropWindow.Peek().Dropped);

            var row = Cameras[i];
            row.Name = label;
            row.DeliveredFps = st.FpsEwma;
            row.FrameAge = cap.TimeSinceLastFrame == TimeSpan.MaxValue
                ? "No frames yet"
                : $"{cap.TimeSinceLastFrame.TotalMilliseconds:F0} ms";
            row.DroppedSummary = $"{lost} frames";
            row.NegotiatedFps = cap.TargetFps;
            row.Resolution = src.CameraSize.Width > 0 ? $"{src.CameraSize.Width} x {src.CameraSize.Height}" : "—";
            row.Format = string.IsNullOrEmpty(cap.PixelFormatName) ? "—" : cap.PixelFormatName;
        }

        // Drop state for cameras that went away (swap/disconnect).
        if (_camStates.Count > active.Count)
        {
            var stale = new List<SingleCameraSource>();
            foreach (var key in _camStates.Keys)
                if (!active.Contains(key)) stale.Add(key);
            foreach (var key in stale) _camStates.Remove(key);
        }
    }

    /// <summary>Reconcile the hottest threads into the bound collection, reusing rows to avoid UI churn.</summary>
    private void UpdateThreads()
    {
        ProcessCpuPercent = _profiler.ProcessCpuPercent;

        var samples = _profiler.Snapshot;
        var show = Math.Min(samples.Count, MaxThreadRows);

        while (Threads.Count < show) Threads.Add(new ThreadCpuRow());
        while (Threads.Count > show) Threads.RemoveAt(Threads.Count - 1);

        for (var i = 0; i < show; i++)
        {
            var s = samples[i];
            Threads[i].Name = s.Count > 1 ? $"{s.Name} (×{s.Count})" : s.Name;
            Threads[i].CpuPercent = s.CpuPercent;
        }
    }

    public void Dispose()
    {
        _timer.Stop();
        _timer.Tick -= Sample;
    }
}
