using Microsoft.Extensions.Logging;
using OpenCvSharp;
using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Threading;

namespace Baballonia.SDK;

/// <summary>
/// Defines custom camera stream behavior
/// </summary>
public abstract class Capture(string source, ILogger logger) : IDisposable
{
    protected ILogger Logger = logger;
    private Mat? _rawMat;
    private object _rawMatLock = new();

    // Signalled while an unconsumed frame is available; reset once it is acquired. Lets a consumer
    // block until a fresh frame arrives instead of busy-polling AcquireRawMat. Set/Reset happen
    // under _rawMatLock so the signal state always matches "is there a frame waiting".
    private readonly ManualResetEventSlim _frameReady = new(false);

    /// <summary>
    /// A wait handle that becomes signalled when a fresh, unconsumed frame is available and is
    /// reset once <see cref="AcquireRawMat"/> takes it. Consumers can <c>WaitHandle.WaitAny</c>
    /// across several sources to pace themselves to the real capture rate. Thread safe.
    /// </summary>
    public WaitHandle FrameWaitHandle => _frameReady.WaitHandle;

    /// <summary>
    /// Where this Capture source is currently pulling data from
    /// </summary>
    public string Source { get; set; } = source;

    /// <summary>
    /// Represents the incoming frame data for this capture source.
    /// Will be `dimension` in BGR color space. <br/>
    /// Acquiring this value the caller takes ownership of the Mat object and sets the internal reference to null. <br/>
    /// Thread safe
    /// </summary>
    public Mat? AcquireRawMat() => AcquireRawMat(out _, out _);

    /// <summary>Atomically pairs ownership with host receipt identity; not sensor exposure time.</summary>
    public Mat? AcquireRawMat(out long sequence, out long receiptTimestamp)
    {
        Mat? result;
        lock (_rawMatLock)
        {
            sequence = 0;
            receiptTimestamp = 0;
            if (_disposed)
                return null;

            result = _rawMat;
            if (result != null)
            {
                sequence = _framesProduced;
                receiptTimestamp = _lastFrameAtTicks;
            }
            _rawMat = null;
            _frameReady.Reset();
        }
        return result;
    }

    /// <summary>
    /// Sets current Mat object that can be acquired by someone else. <br/>
    /// The caller gives up the responsibility for the object <br/>
    /// It is prohibited to use the value object after calling this method <br/>
    /// Thread safe
    /// </summary>
    /// <param name="value">value</param>
    protected void SetRawMat(Mat value)
    {
        lock (_rawMatLock)
        {
            if (_disposed)
            {
                value.Dispose();
                return;
            }

            // Producer freshness must advance even when a backend reuses one Mat instance. Keep
            // this before the reference check so a healthy producer cannot look stalled merely
            // because the consumer has not drained its previous frame yet.
            Interlocked.Exchange(ref _lastFrameAtTicks, Stopwatch.GetTimestamp());
            Interlocked.Increment(ref _framesProduced);

            if (ReferenceEquals(_rawMat, value)) return;

            if (_rawMat != null)
            {
                // Previous frame was never acquired by the consumer — it's lost.
                _rawMat.Dispose();
                Interlocked.Increment(ref _framesDropped);
            }
            _rawMat = value;
            _frameReady.Set();
        }
    }

    private long _lastFrameAtTicks;
    private long _framesProduced;
    private long _framesDropped;

    /// <summary>The last producer timestamp on the monotonic clock.</summary>
    public long LastFrameAtTicks => Interlocked.Read(ref _lastFrameAtTicks);

    /// <summary>
    /// Total frames this source has produced so far (incremented once per delivered frame).
    /// Sample the delta over time to compute the real capture throughput. Thread safe.
    /// </summary>
    public long FramesProduced => Interlocked.Read(ref _framesProduced);

    /// <summary>
    /// How long since this capture last produced a frame, or <see cref="TimeSpan.MaxValue"/>
    /// before its first frame.
    /// </summary>
    public TimeSpan TimeSinceLastFrame
    {
        get
        {
            var last = LastFrameAtTicks;
            return last == 0 ? TimeSpan.MaxValue : Stopwatch.GetElapsedTime(last);
        }
    }

    /// <summary>
    /// How long since frame contents last changed, when the backend can detect cached-frame loops.
    /// Null means the backend does not provide a separate content-liveness signal.
    /// </summary>
    public virtual TimeSpan? TimeSinceLastContentChange => null;

    /// <summary>Frames overwritten before the consumer acquired them — frames actually lost (not in-flight). Thread safe.</summary>
    public long FramesDropped => Interlocked.Read(ref _framesDropped);

    /// <summary>Negotiated capture frame rate if the backend knows it (0 = unknown).</summary>
    public virtual double TargetFps => 0;

    /// <summary>Human-readable pixel format if the backend knows it (empty = unknown).</summary>
    public virtual string PixelFormatName => "";

    /// <summary>Whether a failure to start is an expected retry and should be logged quietly.</summary>
    public bool QuietFailures { get; set; }

    /// <summary>Reports a start failure at the level appropriate for this capture attempt.</summary>
    protected void ReportStartFailure(Exception error, string message, params object?[] args)
    {
        if (QuietFailures)
            Logger.LogDebug(error, message, args);
        else
            Logger.LogError(error, message, args);
    }

    /// <summary>Reports a start failure that has no exception at the appropriate level.</summary>
    protected void ReportStartFailure(string message, params object?[] args)
    {
        if (QuietFailures)
            Logger.LogDebug(message, args);
        else
            Logger.LogError(message, args);
    }

    /// <summary>
    /// Is this Capture source ready to produce data?
    /// </summary>
    public bool IsReady { get; protected set; } = false;

    /// <summary>
    /// Start Capture on this source
    /// </summary>
    /// <returns></returns>
    public abstract Task<bool> StartCapture();

    /// <summary>
    /// Stop Capture on this source
    /// </summary>
    /// <returns></returns>
    public abstract Task<bool> StopCapture();

    private bool _disposed;

    public virtual void Dispose()
    {
        lock (_rawMatLock)
        {
            if (_disposed)
                return;

            _disposed = true;
            _rawMat?.Dispose();
            _rawMat = null;
            _frameReady.Reset();
        }

        _frameReady.Dispose();
        GC.SuppressFinalize(this);
    }
}
