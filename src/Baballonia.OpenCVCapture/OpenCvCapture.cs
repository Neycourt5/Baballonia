using Microsoft.Extensions.Logging;
using OpenCvSharp;
using Capture = Baballonia.SDK.Capture;

namespace Baballonia.OpenCVCapture;

/// <summary>
/// Wrapper class for OpenCV
/// </summary>
public sealed class OpenCvCapture(string source, ILogger<OpenCvCapture> logger) : Capture(source, logger)
{
    private VideoCapture? _videoCapture;
    private static readonly VideoCaptureAPIs PreferredBackend;

    private Task? _updateTask;
    private readonly CancellationTokenSource _updateTaskCts = new();
    private readonly FrameContentLivenessMonitor _contentLiveness = new();

    public override TimeSpan? TimeSinceLastContentChange =>
        _contentLiveness.TimeSinceLastContentChange;

    static OpenCvCapture()
    {
        // Choose the most appropriate backend based on the detected OS
        // This is needed to handle concurrent camera access
        if (OperatingSystem.IsWindows())
        {
            PreferredBackend = VideoCaptureAPIs.DSHOW;
        }
        else if (OperatingSystem.IsLinux())
        {
            PreferredBackend = VideoCaptureAPIs.GSTREAMER;
        }
        else if (OperatingSystem.IsMacOS())
        {
            PreferredBackend = VideoCaptureAPIs.AVFOUNDATION;
        }
        else
        {
            // Fallback to ANY which lets OpenCV choose
            PreferredBackend = VideoCaptureAPIs.ANY;
        }
    }

    public override async Task<bool> StartCapture()
    {
        // A bare "/dev/videoN" path or numeric index is a local device: open it by index, not via the
        // string ctor. The mini runtime has no V4L2 backend, so CAP_ANY falls back to GStreamer's file
        // source and tries to read the char device as a media file ("unable to start pipeline"). Going
        // through FromCamera makes GStreamer build a proper v4l2src pipeline instead.
        var isLocalDevice = int.TryParse(Source, out var index) || TryGetV4l2Index(Source, out index);
        using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(15)))
        {
            try
            {
                if (isLocalDevice)
                    _videoCapture = await Task.Run(() => VideoCapture.FromCamera(index, PreferredBackend), cts.Token);
                else
                    _videoCapture = await Task.Run(() => new VideoCapture(Source), cts.Token);
            }
            catch (Exception e)
            {
                ReportStartFailure(e, "Could not open OpenCV camera {Source}", Source);
                IsReady = false;
                return false;
            }
        }

        // Handle edge case cameras like the Varjo Aero that send frames in YUV
        // This won't activate the IR illuminators, but it's a good idea to standardize inputs
        _videoCapture.ConvertRgb = true;
        IsReady = _videoCapture.IsOpened();

        // Fail-fast only for local /dev/video* or index cameras: this build's OpenCV ships only the
        // GStreamer backend, which needs the (unbundled) v4l2src plugin to read them, so a false
        // IsOpened() there is terminal — bail with a pointer to the dependency-free "V4L2 Camera"
        // backend instead of leaving the caller to wait out its frame-arrival timeout.
        //
        // Network/URL sources (http MJPEG streams, appsink pipelines) are different: VideoCapture
        // .IsOpened() can read false right after construction yet still deliver frames once the read
        // loop pumps the stream — which is how IP/streaming cameras opened before this fail-fast was
        // added. For those, start the loop and let the caller's frame-arrival timeout be the real gate.
        if (!IsReady && isLocalDevice)
        {
            if (OperatingSystem.IsLinux())
                ReportStartFailure(
                    "Could not open '{Source}' via OpenCV's GStreamer backend. Install the v4l2src GStreamer " +
                    "plugin (gst-plugins-good), or use the 'V4L2 Camera' backend which needs no GStreamer.", Source);
            else
                ReportStartFailure("Could not open '{Source}' via OpenCV.", Source);

            _videoCapture.Dispose();
            _videoCapture = null;
            return false;
        }

        CancellationToken token = _updateTaskCts.Token;
        _updateTask = Task.Run(() => VideoCapture_UpdateLoop(_videoCapture, token), token);

        return true;
    }

    // Parses the index out of a Linux "/dev/videoN" path so it can be opened via FromCamera.
    private static bool TryGetV4l2Index(string source, out int index)
    {
        index = 0;
        const string prefix = "/dev/video";
        return source.StartsWith(prefix) && int.TryParse(source.AsSpan(prefix.Length), out index);
    }

    /// <summary>How long to wait after a failed read before trying again.</summary>
    private static readonly TimeSpan FailedReadBackoff = TimeSpan.FromMilliseconds(10);

    /// <summary>Failed reads before announcing an outage, roughly one second at the backoff.</summary>
    private const int FailuresBeforeReporting = 100;

    /// <summary>Consecutive failed reads, exposed for diagnostics.</summary>
    public long ConsecutiveReadFailures { get; private set; }

    private void VideoCapture_UpdateLoop(VideoCapture capture, CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            var read = false;
            Mat? frame = null;

            try
            {
                // Fresh Mat per frame: SetRawMat hands ownership to the consumer, so reusing
                // one buffer races the next Read against it and stalls the feed.
                frame = new Mat();
                read = capture.Read(frame);
                if (read && !frame.Empty())
                {
                    _contentLiveness.Observe(frame);
                    if (ConsecutiveReadFailures >= FailuresBeforeReporting)
                        logger.LogInformation("Camera {Source} started producing frames again", Source);

                    ConsecutiveReadFailures = 0;
                    SetRawMat(frame);
                    frame = null; // ownership transferred to Capture
                }
                else
                {
                    read = false;
                }
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                if (ConsecutiveReadFailures == 0)
                    logger.LogDebug(e, "Camera read failed for {Source}", Source);
            }
            finally
            {
                frame?.Dispose();
            }

            IsReady = read;

            if (read)
                continue;

            ConsecutiveReadFailures++;

            if (ConsecutiveReadFailures == FailuresBeforeReporting)
            {
                logger.LogInformation(
                    "Camera {Source} stopped producing frames (read failing, last frame {Age:F0} ms ago)",
                    Source, TimeSinceLastFrame.TotalMilliseconds);
            }

            if (ct.WaitHandle.WaitOne(FailedReadBackoff))
                return;
        }
    }

    public override Task<bool> StopCapture()
    {
        if (_videoCapture is null)
            return Task.FromResult(false);

        if (_updateTask != null)
        {
            _updateTaskCts.Cancel();

            // A driver read can block after a physical disconnect. Never wait forever on teardown.
            try
            {
                if (!_updateTask.Wait(StopJoinTimeout))
                    logger.LogDebug("Read loop for {Source} did not stop within the timeout; abandoning it", Source);
            }
            catch (AggregateException e)
            {
                logger.LogDebug(e, "Read loop for {Source} faulted during teardown", Source);
            }

            _updateTask = null;
        }

        IsReady = false;
        try
        {
            _videoCapture.Release();
            _videoCapture.Dispose();
        }
        catch (Exception e)
        {
            logger.LogDebug(e, "Releasing the capture for {Source} threw", Source);
        }
        finally
        {
            _videoCapture = null;
        }

        return Task.FromResult(true);
    }

    private static readonly TimeSpan StopJoinTimeout = TimeSpan.FromSeconds(2);
    private bool _disposed;

    public override void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;

        try
        {
            StopCapture().GetAwaiter().GetResult();
        }
        catch (Exception e)
        {
            logger.LogDebug(e, "Stopping the capture for {Source} during disposal threw", Source);
        }

        _updateTaskCts.Dispose();
        base.Dispose();
        GC.SuppressFinalize(this);
    }
}
