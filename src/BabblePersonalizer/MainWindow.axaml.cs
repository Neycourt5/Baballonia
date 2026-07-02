using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using BabblePersonalizer.Core.Camera;
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
    private CalibrationSessionWriter? _writer;
    private CameraConfiguration _configuration = new();
    private readonly Stopwatch _sessionClock = new();
    private long _frameSequence;
    private int _processingFrame;
    private bool _recordFrames;
    private Bitmap? _rawBitmap;
    private Bitmap? _inferenceBitmap;

    public MainWindow()
    {
        InitializeComponent();
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
            _inference?.Dispose();
            _inference = new FaceInferenceSession(_contract, _preprocessor);
            await _camera.StartAsync(_configuration);
            SetStatus("Camera and stock inference running");
        }
        catch (Exception ex) { SetStatus("Could not start camera: " + ex.Message); }
    }

    private async void StopCamera_Click(object? sender, RoutedEventArgs e)
    {
        await StopRecordingAsync(); await _camera.StopAsync(); SetStatus("Camera stopped");
    }

    private void StartRecording_Click(object? sender, RoutedEventArgs e)
    {
        if (_contract == null || _inference == null || !_camera.IsRunning)
        { SetStatus("Start a validated model and camera before recording"); return; }
        if (_writer != null) return;
        var sessionId = $"Session-{DateTime.Now:yyyyMMdd-HHmmss}";
        _recordFrames = RecordFramesCheck.IsChecked == true;
        var metadata = new SessionMetadata(1, sessionId, DateTimeOffset.UtcNow,
            Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "unknown",
            _contract, _configuration, _recordFrames, 2);
        _writer = new CalibrationSessionWriter(_dataPaths, metadata);
        _sessionClock.Restart(); _frameSequence = 0;
        RecordingText.Text = "Recording full inference stream…"; SetStatus("Recording session");
    }

    private async void StopRecording_Click(object? sender, RoutedEventArgs e) => await StopRecordingAsync();

    private void CameraFrameAvailable(Mat frame)
    {
        if (Interlocked.Exchange(ref _processingFrame, 1) != 0) return;
        try
        {
            var inference = _inference; if (inference == null) return;
            using var result = inference.Run(frame, _configuration);
            var sequence = Interlocked.Increment(ref _frameSequence);
            var writer = _writer;
            if (writer != null)
            {
                writer.TryWrite(new CalibrationSample(Stopwatch.GetTimestamp(), _sessionClock.Elapsed.TotalSeconds,
                    sequence, "Unlabeled", "", 0, 0, "None", "Phase1Capture", 0, 0,
                    (float[])result.RawOutput.Clone()));
                if (_recordFrames && sequence % 15 == 0)
                    writer.TryWriteFrame(result.TransformedImage, sequence);
            }
            Cv2.ImEncode(".png", frame, out var rawPng);
            Cv2.ImEncode(".png", result.TransformedImage, out var inputPng);
            var outputText = string.Join("  ", _contract!.Parameters.Take(8)
                .Select(x => $"{x.CanonicalName}: {result.RawOutput[x.OutputIndex]:0.000}"));
            Dispatcher.UIThread.Post(() => UpdatePreview(rawPng, inputPng, outputText, writer));
        }
        catch (Exception ex) { Dispatcher.UIThread.Post(() => SetStatus("Inference error: " + ex.Message)); }
        finally { Volatile.Write(ref _processingFrame, 0); }
    }

    private void UpdatePreview(byte[] rawPng, byte[] inputPng, string outputText, CalibrationSessionWriter? writer)
    {
        var raw = new Bitmap(new MemoryStream(rawPng)); var transformed = new Bitmap(new MemoryStream(inputPng));
        RawPreview.Source = raw; InferencePreview.Source = transformed;
        _rawBitmap?.Dispose(); _inferenceBitmap?.Dispose(); _rawBitmap = raw; _inferenceBitmap = transformed;
        LiveOutputText.Text = outputText;
        if (writer != null)
            RecordingText.Text = $"Frames: {_frameSequence:N0}   Dropped samples: {writer.DroppedSamples}   Dropped images: {writer.DroppedFrames}";
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

    private async Task StopRecordingAsync()
    {
        var writer = Interlocked.Exchange(ref _writer, null); if (writer == null) return;
        _sessionClock.Stop();
        try
        {
            await writer.CompleteAsync();
            RecordingText.Text = $"Saved: {writer.SessionDirectory}"; SetStatus("Session finalized");
        }
        catch (Exception ex) { RecordingText.Text = "Write failed: " + ex.Message; SetStatus("Session write failed"); }
        finally { await writer.DisposeAsync(); }
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
        await StopRecordingAsync(); await _camera.DisposeAsync(); _inference?.Dispose();
        _rawBitmap?.Dispose(); _inferenceBitmap?.Dispose(); base.OnClosed(e);
    }
}
