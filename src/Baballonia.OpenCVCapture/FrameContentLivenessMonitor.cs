using System.Diagnostics;
using System.Threading;
using OpenCvSharp;

namespace Baballonia.OpenCVCapture;

/// <summary>
/// Samples frame statistics cheaply enough to distinguish a live sensor from a DirectShow graph
/// that keeps returning the exact same cached image after its USB device disappears.
/// </summary>
public sealed class FrameContentLivenessMonitor(TimeSpan? sampleInterval = null)
{
    public static readonly TimeSpan DefaultSampleInterval = TimeSpan.FromMilliseconds(250);

    private readonly long _sampleIntervalTicks = (long)(
        (sampleInterval ?? DefaultSampleInterval).TotalSeconds * Stopwatch.Frequency);
    private FrameFingerprint? _previous;
    private long _nextSampleAtTicks;
    private long _lastContentChangeAtTicks;

    public TimeSpan TimeSinceLastContentChange
    {
        get
        {
            var last = Interlocked.Read(ref _lastContentChangeAtTicks);
            return last == 0 ? TimeSpan.MaxValue : Stopwatch.GetElapsedTime(last);
        }
    }

    public void Observe(Mat frame)
    {
        var now = Stopwatch.GetTimestamp();
        if (now < _nextSampleAtTicks)
            return;
        _nextSampleAtTicks = now + _sampleIntervalTicks;

        Cv2.MeanStdDev(frame, out var mean, out var deviation);
        var current = new FrameFingerprint(
            frame.Rows,
            frame.Cols,
            frame.Type().Value,
            mean.Val0,
            mean.Val1,
            mean.Val2,
            mean.Val3,
            deviation.Val0,
            deviation.Val1,
            deviation.Val2,
            deviation.Val3);

        if (_previous is null || _previous.Value != current)
        {
            _previous = current;
            Interlocked.Exchange(ref _lastContentChangeAtTicks, now);
        }
    }

    private readonly record struct FrameFingerprint(
        int Rows,
        int Columns,
        int Type,
        double Mean0,
        double Mean1,
        double Mean2,
        double Mean3,
        double Deviation0,
        double Deviation1,
        double Deviation2,
        double Deviation3);
}
