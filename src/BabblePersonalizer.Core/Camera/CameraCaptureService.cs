using OpenCvSharp;

namespace BabblePersonalizer.Core.Camera;

public sealed class CameraCaptureService : IAsyncDisposable
{
    private VideoCapture? _capture;
    private CancellationTokenSource? _cancellation;
    private Task? _loop;
    public event Action<Mat>? FrameAvailable;
    public event Action<string>? Failed;
    public bool IsRunning => _loop is { IsCompleted: false };

    public IReadOnlyList<CameraDevice> EnumerateCandidates(int count = 10) =>
        Enumerable.Range(0, count).Select(x => new CameraDevice(x, $"Camera {x}")).ToArray();

    public async Task StartAsync(CameraConfiguration configuration, CancellationToken cancellationToken = default)
    {
        await StopAsync().ConfigureAwait(false);
        var backend = OperatingSystem.IsWindows() ? VideoCaptureAPIs.DSHOW :
            OperatingSystem.IsLinux() ? VideoCaptureAPIs.GSTREAMER :
            OperatingSystem.IsMacOS() ? VideoCaptureAPIs.AVFOUNDATION : VideoCaptureAPIs.ANY;
        _capture = await Task.Run(() => VideoCapture.FromCamera(configuration.DeviceIndex, backend), cancellationToken).ConfigureAwait(false);
        _capture.Set(VideoCaptureProperties.FrameWidth, configuration.Width);
        _capture.Set(VideoCaptureProperties.FrameHeight, configuration.Height);
        _capture.Set(VideoCaptureProperties.Fps, configuration.FramesPerSecond);
        if (!_capture.IsOpened())
        {
            _capture.Dispose(); _capture = null;
            throw new InvalidOperationException($"Camera {configuration.DeviceIndex} could not be opened.");
        }
        _cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _loop = Task.Run(() => CaptureLoop(_capture, _cancellation.Token), _cancellation.Token);
    }

    public async Task StopAsync()
    {
        if (_cancellation != null) await _cancellation.CancelAsync().ConfigureAwait(false);
        if (_loop != null) try { await _loop.ConfigureAwait(false); } catch (OperationCanceledException) { }
        _capture?.Release(); _capture?.Dispose(); _capture = null;
        _cancellation?.Dispose(); _cancellation = null; _loop = null;
    }

    private void CaptureLoop(VideoCapture capture, CancellationToken cancellationToken)
    {
        using var reusable = new Mat();
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                if (!capture.Read(reusable) || reusable.Empty()) { Thread.Yield(); continue; }
                using var frame = reusable.Clone();
                FrameAvailable?.Invoke(frame);
            }
            catch (Exception ex) { Failed?.Invoke(ex.Message); break; }
        }
    }

    public async ValueTask DisposeAsync() => await StopAsync().ConfigureAwait(false);
}
