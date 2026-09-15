using System.Threading;
using System;
using Baballonia.SDK;
using Baballonia.Services.Inference.Enums;
using Microsoft.Extensions.Logging;
using OpenCvSharp;

namespace Baballonia.Services.Inference.VideoSources;

public class SingleCameraSource : IVideoSource
{
    private ILogger _logger;
    public Size CameraSize;
    private string _cameraAddress;
    private readonly Capture _capture;

    /// <summary>The underlying capture source (exposes frame-rate / throughput stats).</summary>
    public Capture Capture => _capture;

    /// <summary>The stable address this source was opened with.</summary>
    public string Address => _cameraAddress;

    /// <summary>How long since the underlying capture produced a frame.</summary>
    public TimeSpan TimeSinceLastFrame => _capture.TimeSinceLastFrame;

    /// <summary>
    /// The worse of producer freshness and content freshness. OpenCV uses the latter to expose a
    /// disconnected DirectShow graph that keeps returning one cached frame forever.
    /// </summary>
    public TimeSpan TimeSinceLastHealthyFrame
    {
        get
        {
            var producerAge = _capture.TimeSinceLastFrame;
            var contentAge = _capture.TimeSinceLastContentChange;
            return contentAge is { } age && age > producerAge ? age : producerAge;
        }
    }

    /// <summary>Frames produced by the underlying capture since it opened.</summary>
    public long FramesProduced => _capture.FramesProduced;
    public VideoFrameIdentity? LastFrameIdentity { get; private set; }

    public SingleCameraSource(
        ILogger logger,
        Capture capture,
        string cameraAddress)
    {
        _logger = logger;
        _capture = capture;
        _cameraAddress = cameraAddress;
        CameraSize = new Size(0, 0);
    }

    public bool Start()
    {
        // Surface the backend's real result so a failed open (e.g. GStreamer with no v4l2src) fails
        // fast instead of being mistaken for a slow camera and waiting out the frame timeout.
        return _capture.StartCapture().GetAwaiter().GetResult();
    }

    public WaitHandle[] GetFrameWaitHandles() => [_capture.FrameWaitHandle];

    public bool Stop()
    {
        _capture.StopCapture();
        return true;
    }

    /// <summary>
    /// Captures Image and transforms it to target colorspace
    /// </summary>
    /// <param name="color">colorspace to which captured image would be transformed, uses captured image colorspace by default.</param>
    /// <returns>captured image</returns>
    public Mat? GetFrame(ColorType? color = null)
    {
        // Acquire first. Some backends clear IsReady on the same failed read that follows their
        // final good frame; checking the flag first would strand that owned Mat in the capture.
        var rawMat = _capture.AcquireRawMat(out var sequence, out var receipt);
        if (rawMat == null)
            return null;
        LastFrameIdentity = new(sequence, receipt, sequence, receipt, true);

        CameraSize.Width = rawMat.Width;
        CameraSize.Height = rawMat.Height;

        Mat image;
        if (color == null ||
            color == (rawMat.Channels() == 1 ? ColorType.Gray8 : ColorType.Bgr24))
        {
            image = rawMat;
        }
        else
        {
            var convertedMat = new Mat();
            try
            {
                Cv2.CvtColor(rawMat, convertedMat,
                    (rawMat.Channels() == 1)
                        ? color switch
                        {
                            ColorType.Bgr24 => ColorConversionCodes.GRAY2BGR,
                            ColorType.Rgb24 => ColorConversionCodes.GRAY2RGB,
                            ColorType.Rgba32 => ColorConversionCodes.GRAY2RGBA,
                        }
                        : color switch
                        {
                            ColorType.Gray8 => ColorConversionCodes.BGR2GRAY,
                            ColorType.Rgb24 => ColorConversionCodes.BGR2RGB,
                            ColorType.Rgba32 => ColorConversionCodes.BGR2RGBA,
                        });
            }
            catch
            {
                convertedMat.Dispose();
                throw;
            }
            finally
            {
                rawMat.Dispose();
            }
            image = convertedMat;
        }

        if (image.Empty())
        {
            image.Dispose();
            return null;
        }

        return image;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;

        try
        {
            _capture.StopCapture().GetAwaiter().GetResult();
        }
        catch (Exception e)
        {
            _logger.LogDebug(e, "Stopping the capture for {Address} threw during disposal", _cameraAddress);
        }

        try
        {
            _capture.Dispose();
        }
        catch (Exception e)
        {
            _logger.LogDebug(e, "Disposing the capture for {Address} threw", _cameraAddress);
        }

        GC.SuppressFinalize(this);
    }

    private bool _disposed;
}
