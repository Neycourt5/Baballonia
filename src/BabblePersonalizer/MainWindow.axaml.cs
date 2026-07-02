using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using BabblePersonalizer.Core.Camera;
using BabblePersonalizer.Core.Calibration;
using BabblePersonalizer.Core.Models;
using BabblePersonalizer.Core.Onnx;
using BabblePersonalizer.Core.Storage;
using OpenCvSharp;
using System.Diagnostics;
using System.Reflection;

namespace BabblePersonalizer;

public partial class MainWindow : Avalonia.Controls.Window
{
    private readonly StockModelInspector _modelInspector = new();
    private readonly CameraCaptureService _camera = new();
    private readonly BaballoniaCompatiblePreprocessor _preprocessor = new();
    private readonly PersonalizerDataPaths _dataPaths = new();
    private ModelContract? _contract;
    private FaceInferenceSession? _inference;
    private GuidedCalibrationSession? _guidedSession;
    private readonly PersonalProfileStore _profileStore;
    private PersonalCalibrationProfile? _activeProfile;
    private PersonalCorrectionEngine? _correction;
    private CameraConfiguration _configuration = new();
    private long _frameSequence;
    private int _processingFrame;
    private int _finishingCalibration;
    private bool _recordFrames;
    private volatile bool _personalizationEnabled = true;
    private volatile int _diagnosticIndex;
    private readonly Queue<(float Raw, float Personal)> _diagnosticHistory = new();
    private readonly object _diagnosticSync = new();
    private Bitmap? _rawBitmap;
    private Bitmap? _inferenceBitmap;

    public MainWindow()
    {
        InitializeComponent();
        _profileStore = new PersonalProfileStore(_dataPaths);
        CameraCombo.ItemsSource = _camera.EnumerateCandidates();
        CameraCombo.DisplayMemberBinding = new Avalonia.Data.Binding(nameof(CameraDevice.DisplayName));
        _camera.FrameAvailable += CameraFrameAvailable;
        _camera.Failed += message => Dispatcher.UIThread.Post(() => SetStatus("Camera error: " + message));
    }

    private async void BrowseModel_Click(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select your existing stock faceModel.onnx", AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("ONNX model") { Patterns = new[] { "*.onnx" } } }
        });
        if (files.Count == 0) return;
        var path = files[0].Path.LocalPath;
        ModelPathText.Text = path; ModelProgress.IsVisible = true; BrowseModelButton.IsEnabled = false;
        SetStatus("Validating model…");
        try
        {
            var result = await _modelInspector.InspectAsync(path);
            _contract = result.Contract?.IsCompatible == true ? result.Contract : null;
            ExpressionList.ItemsSource = result.Contract?.Parameters.Select(x =>
                $"{x.OutputIndex,2}  {x.CanonicalName,-24} {x.Category,-10} {x.CalibrationStrategy}").ToArray();
            DiagnosticExpressionCombo.ItemsSource = result.Contract?.Parameters.Select(x => x.CanonicalName).ToArray();
            DiagnosticExpressionCombo.SelectedIndex = 0;
            ModelDetailsText.Text = FormatResult(result);
            SetStatus(result.IsValid ? "Model validated" : "Model is not compatible");
        }
        catch (Exception ex) { _contract = null; ModelDetailsText.Text = ex.Message; SetStatus("Validation failed"); }
        finally { ModelProgress.IsVisible = false; BrowseModelButton.IsEnabled = true; }
    }

    private async void StartCamera_Click(object? sender, RoutedEventArgs e)
    {
        if (_contract == null) { SetStatus("Select a compatible model first"); return; }
        try
        {
            _configuration = ReadCameraConfiguration();
            if (_activeProfile != null && _activeProfile.CameraConfigurationHash != PersonalProfileGenerator.HashCamera(_configuration))
                SetStatus("Warning: camera/preprocessing settings differ from the loaded profile");
            _inference?.Dispose();
            _inference = new FaceInferenceSession(_contract, _preprocessor);
            await _camera.StartAsync(_configuration);
            SetStatus("Camera and stock inference running");
        }
        catch (Exception ex) { SetStatus("Could not start camera: " + ex.Message); }
    }

    private async void StopCamera_Click(object? sender, RoutedEventArgs e)
    {
        await CancelCalibrationAsync(); await _camera.StopAsync(); SetStatus("Camera stopped");
    }

    private void StartRecording_Click(object? sender, RoutedEventArgs e)
    {
        if (_contract == null || _inference == null || !_camera.IsRunning)
        { SetStatus("Start a validated model and camera before calibration"); return; }
        if (_guidedSession != null) return;
        try
        {
            var sessionId = $"Session-{DateTime.Now:yyyyMMdd-HHmmss}";
            _recordFrames = RecordFramesCheck.IsChecked == true;
            var metadata = new SessionMetadata(2, sessionId, DateTimeOffset.UtcNow,
                Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "unknown",
                _contract, _configuration, _recordFrames, 2);
            var writer = new CalibrationSessionWriter(_dataPaths, metadata);
            var plan = GuidedCalibrationPlan.Create(_contract.Parameters);
            _guidedSession = new GuidedCalibrationSession(plan, writer, _profileStore);
            _finishingCalibration = 0; _frameSequence = 0;
            lock (_diagnosticSync) _diagnosticHistory.Clear();
            RecordingText.Text = $"Running — approximately {plan.TotalDuration.TotalMinutes:0} minutes";
            SetStatus("Guided personal calibration running");
        }
        catch (Exception ex) { SetStatus("Could not start calibration: " + ex.Message); }
    }

    private async void StopRecording_Click(object? sender, RoutedEventArgs e) => await CancelCalibrationAsync();

    private void RetryPose_Click(object? sender, RoutedEventArgs e) => _guidedSession?.RetryCurrentParameter();
    private void SkipPose_Click(object? sender, RoutedEventArgs e) => _guidedSession?.SkipCurrentParameter();

    private void CameraFrameAvailable(Mat frame)
    {
        if (Interlocked.Exchange(ref _processingFrame, 1) != 0) return;
        try
        {
            var inference = _inference; if (inference == null) return;
            using var result = inference.Run(frame, _configuration);
            var sequence = Interlocked.Increment(ref _frameSequence);
            var personalized = _personalizationEnabled && _correction != null
                ? _correction.Apply(result.RawOutput) : (float[])result.RawOutput.Clone();
            var guided = _guidedSession;
            if (guided != null)
            {
                var timestamp = Stopwatch.GetTimestamp();
                guided.RecordFrame(timestamp, sequence, result.RawOutput, personalized, result.TransformedImage);
                if (guided.Engine.IsComplete && Interlocked.Exchange(ref _finishingCalibration, 1) == 0)
                    Dispatcher.UIThread.Post(async () => await FinishCalibrationAsync());
            }
            var diagnosticIndex = Math.Clamp(_diagnosticIndex, 0, result.RawOutput.Length - 1);
            lock (_diagnosticSync)
            {
                _diagnosticHistory.Enqueue((result.RawOutput[diagnosticIndex], personalized[diagnosticIndex]));
                while (_diagnosticHistory.Count > 30) _diagnosticHistory.Dequeue();
            }
            Cv2.ImEncode(".png", frame, out var rawPng);
            Cv2.ImEncode(".png", result.TransformedImage, out var inputPng);
            var outputText = string.Join("  ", _contract!.Parameters.Take(8)
                .Select(x => $"{x.CanonicalName}: {result.RawOutput[x.OutputIndex]:0.000}/{personalized[x.OutputIndex]:0.000}"));
            Dispatcher.UIThread.Post(() => UpdatePreview(rawPng, inputPng, outputText,
                result.RawOutput, personalized, guided));
        }
        catch (Exception ex) { Dispatcher.UIThread.Post(() => SetStatus("Inference error: " + ex.Message)); }
        finally { Volatile.Write(ref _processingFrame, 0); }
    }

    private void UpdatePreview(byte[] rawPng, byte[] inputPng, string outputText,
        float[] rawOutput, float[] personalized, GuidedCalibrationSession? guided)
    {
        var raw = new Bitmap(new MemoryStream(rawPng)); var transformed = new Bitmap(new MemoryStream(inputPng));
        RawPreview.Source = raw; InferencePreview.Source = transformed;
        _rawBitmap?.Dispose(); _inferenceBitmap?.Dispose(); _rawBitmap = raw; _inferenceBitmap = transformed;
        LiveOutputText.Text = outputText;
        DiagnosticsLiveText.Text = string.Join(Environment.NewLine, _contract!.Parameters.Take(12).Select(x =>
            $"{x.OutputIndex,2} {x.CanonicalName,-23} stock {rawOutput[x.OutputIndex],6:0.000}  personal {personalized[x.OutputIndex],6:0.000}"));
        if (guided != null)
        {
            var segment = guided.Engine.Current;
            CalibrationPoseText.Text = segment?.PoseName ?? "Finishing";
            CalibrationInstructionText.Text = segment?.Instruction ?? "Generating robust profile and validation report…";
            CalibrationTargetText.Text = segment == null ? "" :
                $"Repetition {segment.Repetition}   Target {segment.RequestedIntensity:P0}   {segment.RampDirection}";
            SegmentProgress.Value = guided.Engine.SegmentProgress;
            CalibrationProgress.Value = guided.Engine.OverallProgress;
            var latest = guided.LatestSample;
            CalibrationMetricsText.Text = latest == null ? "" :
                $"Stability {latest.Stability:P0}   Noise {latest.Noise:0.0000}   Samples {guided.SampleCount:N0}";
            RecordingText.Text = $"Dropped samples {guided.DroppedSamples}   Dropped images {guided.DroppedFrames}";
        }
        UpdateDiagnosticDetail();
    }

    private CameraConfiguration ReadCameraConfiguration()
    {
        var device = CameraCombo.SelectedItem as CameraDevice ?? new CameraDevice(0, "Camera 0");
        int I(NumericUpDown value, int fallback) => decimal.ToInt32(value.Value ?? fallback);
        double D(NumericUpDown value, double fallback) => decimal.ToDouble(value.Value ?? (decimal)fallback);
        return new CameraConfiguration(device.Index, I(WidthInput, 640), I(HeightInput, 480), D(FpsInput, 30),
            new CropRegion(I(CropXInput, 0), I(CropYInput, 0), I(CropWidthInput, 256), I(CropHeightInput, 256)),
            D(RotationInput, 0), D(GammaInput, 1), RedChannelCheck.IsChecked == true,
            MirrorHorizontalCheck.IsChecked == true, MirrorVerticalCheck.IsChecked == true);
    }

    private async Task CancelCalibrationAsync()
    {
        var session = Interlocked.Exchange(ref _guidedSession, null); if (session == null) return;
        await session.CancelAsync(); await session.DisposeAsync();
        RecordingText.Text = $"Cancelled; partial data retained at {session.SessionDirectory}";
        CalibrationPoseText.Text = "Cancelled"; SetStatus("Calibration cancelled");
    }

    private async Task FinishCalibrationAsync()
    {
        var session = Interlocked.Exchange(ref _guidedSession, null); if (session == null) return;
        try
        {
            SetStatus("Generating robust profile and held-out validation…");
            var result = await session.FinishAsync();
            _activeProfile = result.Profile; _correction = new PersonalCorrectionEngine(result.Profile);
            ProfilePathText.Text = result.ProfilePath;
            RecordingText.Text = $"Profile saved: {result.ProfilePath}";
            ShowProfile(result.Profile); SetStatus(result.Profile.Validation?.Summary ?? "Profile generated");
        }
        catch (Exception ex) { RecordingText.Text = "Profile generation failed: " + ex.Message; SetStatus("Profile generation failed"); }
        finally { await session.DisposeAsync(); }
    }

    private async void LoadProfile_Click(object? sender, RoutedEventArgs e)
    {
        if (_contract == null) { SetStatus("Select the matching stock model before loading a profile"); return; }
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Load personal calibration profile", AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("Personal profile") { Patterns = new[] { "*.json" } } }
        });
        if (files.Count == 0) return;
        var path = files[0].Path.LocalPath;
        var loaded = await _profileStore.LoadAsync(path, _contract, _configuration);
        if (!loaded.Success)
        {
            SetStatus("Profile rejected: " + loaded.Warning); return;
        }
        _activeProfile = loaded.Profile; _correction = new PersonalCorrectionEngine(loaded.Profile!);
        ProfilePathText.Text = path; ShowProfile(loaded.Profile!);
        SetStatus(loaded.CameraMismatch ? "Profile loaded with camera mismatch warning" : "Profile loaded");
    }

    private void PersonalizedToggle_Changed(object? sender, RoutedEventArgs e) =>
        _personalizationEnabled = UsePersonalizedCheck.IsChecked == true;

    private void DiagnosticExpression_Changed(object? sender, SelectionChangedEventArgs e)
    {
        _diagnosticIndex = Math.Max(0, DiagnosticExpressionCombo.SelectedIndex);
        lock (_diagnosticSync) _diagnosticHistory.Clear();
        UpdateDiagnosticDetail();
    }

    private void ShowProfile(PersonalCalibrationProfile profile)
    {
        ConfidenceList.ItemsSource = profile.Parameters.Select(x =>
            $"{x.OriginalOutputIndex,2}  {x.CanonicalName,-24} confidence {x.Confidence,6:P0}  " +
            (x.Enabled ? "enabled" : $"passthrough — {x.PassthroughReason}")).ToArray();
        var validation = profile.Validation;
        ValidationText.Text = validation == null ? "Held-out validation unavailable." :
            $"{validation.Summary}\nStock score: {validation.StockScore:0.000}   " +
            $"Personalized score: {validation.PersonalizedScore:0.000}\n" +
            $"Improved parameters: {validation.Parameters.Count(x => x.Improved)} / {validation.Parameters.Count}";
        UpdateDiagnosticDetail();
    }

    private void UpdateDiagnosticDetail()
    {
        var profile = _activeProfile;
        if (profile == null || profile.Parameters.Count == 0)
        { DiagnosticDetailText.Text = "Generate or load a profile."; return; }
        var index = Math.Clamp(_diagnosticIndex, 0, profile.Parameters.Count - 1);
        var parameter = profile.Parameters[index];
        (float Raw, float Personal)[] history;
        lock (_diagnosticSync) history = _diagnosticHistory.ToArray();
        var curve = string.Join("  ", parameter.ResponseCurve.Select(x => $"{x.Input:0.00}→{x.Output:0.00}"));
        var recent = history.Length == 0 ? "none" : string.Join("  ", history.TakeLast(12).Select(x => $"{x.Raw:0.00}/{x.Personal:0.00}"));
        var cross = profile.CrossActivations.Where(x => x.InstructedExpression == parameter.CanonicalName ||
                                                        x.ActivatedExpression == parameter.CanonicalName).Take(5).ToArray();
        DiagnosticDetailText.Text =
            $"{parameter.CanonicalName} [{parameter.OriginalOutputIndex}]   {parameter.Category}\n" +
            $"Enabled: {parameter.Enabled}   Confidence: {parameter.Confidence:P1}   Stability: {parameter.Stability:P1}\n" +
            $"Neutral: {parameter.NeutralMedian:0.0000}   P05/P95: {parameter.NeutralP05:0.0000}/{parameter.NeutralP95:0.0000}\n" +
            $"Noise MAD: {parameter.NeutralNoiseMad:0.0000}   Dead zone: {parameter.DeadZone:0.0000}\n" +
            $"Reliable range: {parameter.ReliableMinimum:0.000}–{parameter.ReliableMaximum:0.000}   Side scale: {parameter.LeftRightScale:0.000}\n" +
            $"Repetition consistency: {parameter.RepetitionConsistency:P1}   Monotonicity: {parameter.RampMonotonicity:P1}   Hysteresis: {parameter.Hysteresis:P1}\n" +
            $"Accepted/rejected: {parameter.AcceptedSamples}/{parameter.RejectedSamples}\n" +
            $"Curve: {curve}\nRecent stock/personal: {recent}\n" +
            (cross.Length == 0 ? "Cross-activation: none recorded" :
                "Cross-activation: " + string.Join("; ", cross.Select(x =>
                    $"{x.InstructedExpression}→{x.ActivatedExpression} {x.MedianActivation:P0}")));
    }

    private static string FormatResult(ModelValidationResult result)
    {
        if (result.Contract == null) return string.Join(Environment.NewLine, result.Errors);
        var c = result.Contract;
        var lines = new List<string>
        {
            $"Compatibility: {(c.IsCompatible ? "PASS" : "FAIL")}", $"File: {c.Path}",
            $"Size: {c.FileSize:N0} bytes", $"SHA-256: {c.Sha256}", $"ONNX opset: {c.OpsetVersion?.ToString() ?? "unknown"}",
            $"Input: {c.Input.Name}  {c.Input.ElementType}  [{string.Join(", ", c.Input.Dimensions)}]  NCHW",
            $"Output: {c.Output.Name}  {c.Output.ElementType}  [{string.Join(", ", c.Output.Dimensions)}]",
            $"Direct expressions: {c.Parameters.Count}", $"Expression-list SHA-256: {c.ExpressionListHash}",
            $"Metadata keys: {(c.Metadata.Count == 0 ? "none" : string.Join(", ", c.Metadata.Keys))}"
        };
        lines.AddRange(c.Warnings.Select(x => "WARNING: " + x)); lines.AddRange(result.Errors.Select(x => "ERROR: " + x));
        return string.Join(Environment.NewLine, lines);
    }

    private void SetStatus(string message) => StatusText.Text = message;

    protected override async void OnClosed(EventArgs e)
    {
        await CancelCalibrationAsync(); await _camera.DisposeAsync(); _inference?.Dispose();
        _rawBitmap?.Dispose(); _inferenceBitmap?.Dispose(); base.OnClosed(e);
    }
}
