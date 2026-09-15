using Baballonia.Services.Inference.Enums;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace Baballonia.Services.Inference;

public class DualCameraSource : IVideoSource
{
    public static readonly TimeSpan StaleHalfTimeout = TimeSpan.FromMilliseconds(500);

    public IVideoSource? LeftCam;
    public IVideoSource? RightCam;

    private Mat? _lastLeftImage;
    private Mat? _lastRightImage;
    private long _lastLeftAtTicks;
    private long _lastRightAtTicks;
    private VideoFrameIdentity? _leftIdentity;
    private VideoFrameIdentity? _rightIdentity;
    public VideoFrameIdentity? LastFrameIdentity { get; private set; }

    public bool Start()
    {
        var leftStarted = LeftCam?.Start() ?? true;
        var rightStarted = ReferenceEquals(RightCam, LeftCam) || (RightCam?.Start() ?? true);
        return leftStarted && rightStarted;
    }

    public bool Stop()
    {
        var leftStopped = LeftCam?.Stop() ?? true;
        var rightStopped = ReferenceEquals(RightCam, LeftCam) || (RightCam?.Stop() ?? true);
        return leftStopped && rightStopped;
    }

    public WaitHandle[] GetFrameWaitHandles()
    {
        var handles = new List<WaitHandle>(2);
        if (LeftCam != null) handles.AddRange(LeftCam.GetFrameWaitHandles());
        if (RightCam != null) handles.AddRange(RightCam.GetFrameWaitHandles());
        return handles.ToArray();
    }

    public void InvalidateLeftCache()
    {
        _lastLeftImage?.Dispose();
        _lastLeftImage = null;
        _lastLeftAtTicks = 0;
        _leftIdentity = null;
    }

    public void InvalidateRightCache()
    {
        _lastRightImage?.Dispose();
        _lastRightImage = null;
        _lastRightAtTicks = 0;
        _rightIdentity = null;
    }

    // A recently missing half may use its cache to absorb ordinary frame-rate skew; after 500 ms
    // it mirrors the live half instead of freezing an old eye indefinitely.
    public Mat? GetFrame(ColorType? color = null)
    {
        var leftImage = LeftCam?.GetFrame(color);
        var rightImage = RightCam?.GetFrame(color);
        var leftIsFresh = leftImage != null;
        var rightIsFresh = rightImage != null;
        var ownsLeft = leftIsFresh;
        var ownsRight = rightIsFresh;
        var now = Stopwatch.GetTimestamp();

        try
        {
            if (leftImage == null && rightImage == null)
                return null;

            if (!leftIsFresh && CacheExpired(_lastLeftAtTicks, now))
                InvalidateLeftCache();
            if (!rightIsFresh && CacheExpired(_lastRightAtTicks, now))
                InvalidateRightCache();

            leftImage ??= _lastLeftImage;
            rightImage ??= _lastRightImage;
            var leftMirrored = leftImage == null;
            var rightMirrored = rightImage == null;

            if (leftImage == null)
            {
                leftImage = rightImage!.Clone();
                ownsLeft = true;
            }
            else if (rightImage == null)
            {
                rightImage = leftImage.Clone();
                ownsRight = true;
            }

            if (leftImage.Empty() || rightImage.Empty())
                return null;

            var minHeight = Math.Min(leftImage.Rows, rightImage.Rows);
            var minWidth = Math.Min(leftImage.Cols, rightImage.Cols);

            using var resizedLeft = new Mat();
            using var resizedRight = new Mat();
            Cv2.Resize(leftImage, resizedLeft, new Size(minWidth, minHeight));
            Cv2.Resize(rightImage, resizedRight, new Size(minWidth, minHeight));

            Mat? result = null;
            try
            {
                result = new Mat(minHeight, minWidth * 2, resizedLeft.Type(), Scalar.All(0));
                using (var target = new Mat(result, new Rect(0, 0, minWidth, minHeight)))
                    resizedLeft.CopyTo(target);
                using (var target = new Mat(result, new Rect(minWidth, 0, minWidth, minHeight)))
                    resizedRight.CopyTo(target);

                if (leftIsFresh)
                {
                    InvalidateLeftCache();
                    _lastLeftImage = resizedLeft.Clone();
                    _lastLeftAtTicks = now;
                    _leftIdentity = LeftCam?.LastFrameIdentity;
                }

                if (rightIsFresh)
                {
                    InvalidateRightCache();
                    _lastRightImage = resizedRight.Clone();
                    _lastRightAtTicks = now;
                    _rightIdentity = RightCam?.LastFrameIdentity;
                }

                var li = leftMirrored ? _rightIdentity : _leftIdentity;
                var ri = rightMirrored ? _leftIdentity : _rightIdentity;
                LastFrameIdentity = new(li?.LeftSequence, li?.LeftReceiptTimestamp,
                    ri?.LeftSequence, ri?.LeftReceiptTimestamp, false, leftMirrored, rightMirrored);

                return result;
            }
            catch
            {
                result?.Dispose();
                throw;
            }
        }
        finally
        {
            if (ownsLeft)
                leftImage?.Dispose();
            if (ownsRight && !ReferenceEquals(rightImage, leftImage))
                rightImage?.Dispose();
        }
    }

    private static bool CacheExpired(long timestamp, long now) =>
        timestamp == 0 || Stopwatch.GetElapsedTime(timestamp, now) > StaleHalfTimeout;

    public void Dispose()
    {
        LeftCam?.Dispose();
        if (!ReferenceEquals(RightCam, LeftCam))
            RightCam?.Dispose();
        LeftCam = null;
        RightCam = null;
        InvalidateLeftCache();
        InvalidateRightCache();
        GC.SuppressFinalize(this);
    }
}
