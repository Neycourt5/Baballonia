using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Baballonia.Services.events;
using Microsoft.Extensions.Logging;
using OpenCvSharp;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Keeps the last few seconds of tracking in memory so a mistake can be saved *after* it happens.
///
/// This exists because of an ordering problem that no amount of recording discipline solves: the
/// user only knows a frame was wrong once they have seen the avatar get it wrong, which is already
/// too late to start recording. Holding a short rolling window means the evidence is still in hand
/// when they press the button.
///
/// Nothing here ever touches the disk. The buffer is written only when the user explicitly flags a
/// moment (see <see cref="HardExampleService"/>), which keeps a feature that continuously observes
/// the face from continuously writing it down.
///
/// <para><b>Cost.</b> Storage is a fixed ring of preallocated frames - at 224x224 gray and the
/// recorder's 30 fps cap, ten seconds is 300 x 50,176 bytes, about 15 MB, allocated once with no
/// steady-state garbage. Per frame the tick pays one ~50 KB memcpy and two 45-float copies, roughly
/// 1.5 MB/s. Raw bytes rather than JPEG is deliberate: encoding belongs at save time, on a
/// background thread, not on the inference tick where it would cost more than the copy it saves.</para>
/// </summary>
public sealed class HardExampleBuffer : IDisposable
{
    /// <summary>Ten seconds at the recorder's frame cap - long enough to cover "wait, what was that?"</summary>
    public const int DefaultCapacityFrames = 300;

    /// <summary>Matches <see cref="DatasetRecorderService.MaxRecordFps"/>; the tick runs far faster.</summary>
    private const int MaxCaptureFps = DatasetRecorderService.MaxRecordFps;

    private readonly ILogger _logger;
    private readonly int _capacity;
    private readonly long _minTicksBetweenFrames = Stopwatch.Frequency / MaxCaptureFps;
    private readonly object _gate = new();

    private readonly Action<FacePipelineEvents.NewRawExpressionsEvent> _rawHandler;
    private readonly Action<FacePipelineEvents.NewCorrectedExpressionsEvent> _correctedHandler;
    private readonly IFacePipelineEventBus? _eventBus;

    private Entry[]? _entries;
    private int _pixelCount;
    private int _width;
    private int _height;

    private int _next;
    private int _count;
    private long _lastAcceptedTimestamp;
    private ulong _lastChecksum;
    private volatile bool _enabled;

    /// <summary>
    /// Whether the raw frame published this tick was kept. The corrected event fires immediately
    /// after the raw one on the same tick, so "attach to the newest entry" is only correct when that
    /// entry is in fact from this tick; if the frame was dropped as a duplicate or over-rate, the
    /// newest entry is an older frame and attaching would pair a prediction with the wrong picture.
    /// </summary>
    private bool _lastOfferAccepted;

    public HardExampleBuffer(ILogger logger, IFacePipelineEventBus? eventBus = null,
                             int capacityFrames = DefaultCapacityFrames, bool enabled = true)
    {
        _logger = logger;
        _capacity = Math.Max(1, capacityFrames);
        _eventBus = eventBus;
        _enabled = enabled;

        _rawHandler = OnRawExpressions;
        _correctedHandler = OnCorrectedExpressions;

        eventBus?.Subscribe(_rawHandler);
        eventBus?.Subscribe(_correctedHandler);
    }

    /// <summary>Frames currently held. Rises to capacity and then stays there.</summary>
    public int Count
    {
        get { lock (_gate) return _count; }
    }

    /// <summary>Bytes actually allocated, for reporting the cost honestly in diagnostics.</summary>
    public long AllocatedBytes
    {
        get
        {
            lock (_gate)
                return _entries is null ? 0 : (long)_entries.Length * (_pixelCount + 2 * 45 * sizeof(float));
        }
    }

    /// <summary>
    /// Turns capture on or off. Off makes the event handlers return before checksums or copies and
    /// releases the preallocated ring. Enabling again starts with an empty ring on the next frame.
    /// </summary>
    public bool Enabled
    {
        get => _enabled;
        set
        {
            if (value == _enabled)
                return;

            _enabled = value;
            if (!value)
                Reset(releaseMemory: true);
        }
    }

    /// <summary>One captured moment, detached from the ring so persistence can run without the lock.</summary>
    public sealed record Snapshot(
        IReadOnlyList<SnapshotFrame> Frames,
        int Width,
        int Height);

    public sealed record SnapshotFrame(
        byte[] Pixels,
        long TimestampTicks,
        float[] Stock,
        float[]? Personal);

    private sealed class Entry
    {
        public byte[] Pixels = [];
        public long TimestampTicks;
        public readonly float[] Stock = new float[PersonalizationSchema.ExpressionCount];
        public readonly float[] Personal = new float[PersonalizationSchema.ExpressionCount];
        public bool HasPersonal;
    }

    private void OnRawExpressions(FacePipelineEvents.NewRawExpressionsEvent e)
    {
        _lastOfferAccepted = false;

        if (!Enabled)
            return;

        try
        {
            var frame = e.transformedFrame;
            if (frame is null || frame.Empty())
                return;

            var now = Stopwatch.GetTimestamp();
            if (_lastAcceptedTimestamp != 0 && now - _lastAcceptedTimestamp < _minTicksBetweenFrames)
                return;

            var checksum = FrameChecksum.Sparse(frame);
            if (checksum == _lastChecksum)
                return;

            lock (_gate)
            {
                // The UI can disable capture after the inexpensive early check but before this
                // lock. Rechecking here prevents one final frame from repopulating a cleared ring.
                if (!Enabled)
                    return;

                if (!EnsureCapacity(frame))
                    return;

                var entry = _entries![_next];

                if (!CopyPixels(frame, entry.Pixels))
                    return;

                entry.TimestampTicks = e.timestampTicks;
                entry.HasPersonal = false;

                var stock = e.rawResult;
                var limit = Math.Min(stock.Length, entry.Stock.Length);
                Array.Copy(stock, entry.Stock, limit);

                _next = (_next + 1) % _entries.Length;
                if (_count < _entries.Length)
                    _count++;

                _lastChecksum = checksum;
                _lastAcceptedTimestamp = now;
                _lastOfferAccepted = true;
            }
        }
        catch (Exception ex)
        {
            // A diagnostic aid must never take tracking down with it.
            _logger.LogWarning(ex, "Hard-example buffer: dropping frame after an error");
            Enabled = false;
        }
    }

    private void OnCorrectedExpressions(FacePipelineEvents.NewCorrectedExpressionsEvent e)
    {
        if (!Enabled || !_lastOfferAccepted)
            return;

        try
        {
            lock (_gate)
            {
                if (_entries is null || _count == 0)
                    return;

                // The entry just written by this tick's raw event.
                var index = (_next - 1 + _entries.Length) % _entries.Length;
                var entry = _entries[index];

                var corrected = e.correctedResult;
                var limit = Math.Min(corrected.Length, entry.Personal.Length);
                Array.Copy(corrected, entry.Personal, limit);
                entry.HasPersonal = true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Hard-example buffer: could not attach corrected values");
        }
    }

    /// <summary>Allocates on first frame, and reallocates if the geometry ever changes.</summary>
    private bool EnsureCapacity(Mat frame)
    {
        var pixels = (int)((long)frame.Total() * frame.ElemSize());
        if (pixels <= 0)
            return false;

        if (_entries is not null && pixels == _pixelCount)
            return true;

        // A geometry change invalidates everything held: the frames would no longer be comparable,
        // and a saved mixture would be unloadable. Starting over is the only honest option.
        _entries = new Entry[_capacity];
        for (var i = 0; i < _entries.Length; i++)
            _entries[i] = new Entry { Pixels = new byte[pixels] };

        _pixelCount = pixels;
        _width = frame.Width;
        _height = frame.Height;
        _next = 0;
        _count = 0;

        _logger.LogInformation(
            "Hard-example buffer: holding {Frames} frames of {Width}x{Height} ({Megabytes:F1} MB)",
            _capacity, _width, _height, AllocatedBytesUnlocked() / (1024.0 * 1024.0));

        return true;
    }

    private long AllocatedBytesUnlocked() =>
        _entries is null ? 0 : (long)_entries.Length * (_pixelCount + 2 * 45 * sizeof(float));

    private static bool CopyPixels(Mat frame, byte[] destination)
    {
        // A non-continuous Mat has row padding, so a flat copy would interleave garbage. The
        // pipeline's transformed frames are always continuous; refusing rather than copying
        // regardless keeps a future change from silently corrupting saved evidence.
        if (!frame.IsContinuous())
            return false;

        var bytes = (int)((long)frame.Total() * frame.ElemSize());
        if (bytes != destination.Length)
            return false;

        Marshal.Copy(frame.Data, destination, 0, bytes);
        return true;
    }

    /// <summary>
    /// Copies out everything captured within <paramref name="window"/> of now, oldest first.
    /// </summary>
    /// <remarks>
    /// Deep-copies under the lock so persistence - JPEG encoding and file writes - runs without
    /// holding up the tick. That is a ~15 MB memcpy in the worst case, on a user-initiated action
    /// that happens a handful of times per session.
    /// </remarks>
    public Snapshot Take(TimeSpan window)
    {
        var cutoff = DateTime.UtcNow.Ticks - window.Ticks;
        var frames = new List<SnapshotFrame>();

        lock (_gate)
        {
            if (_entries is null || _count == 0)
                return new Snapshot(frames, _width, _height);

            var start = (_next - _count + _entries.Length) % _entries.Length;
            for (var i = 0; i < _count; i++)
            {
                var entry = _entries[(start + i) % _entries.Length];
                if (entry.TimestampTicks < cutoff)
                    continue;

                frames.Add(new SnapshotFrame(
                    Pixels: (byte[])entry.Pixels.Clone(),
                    TimestampTicks: entry.TimestampTicks,
                    Stock: (float[])entry.Stock.Clone(),
                    Personal: entry.HasPersonal ? (float[])entry.Personal.Clone() : null));
            }

            return new Snapshot(frames, _width, _height);
        }
    }

    /// <summary>Drops everything held. Used when personalization is reconfigured under the buffer.</summary>
    public void Clear()
    {
        Reset(releaseMemory: false);
    }

    private void Reset(bool releaseMemory)
    {
        lock (_gate)
        {
            if (releaseMemory)
            {
                _entries = null;
                _pixelCount = 0;
                _width = 0;
                _height = 0;
            }

            _next = 0;
            _count = 0;
            _lastChecksum = 0;
            _lastAcceptedTimestamp = 0;
            _lastOfferAccepted = false;
        }
    }

    public void Dispose()
    {
        _eventBus?.Unsubscribe(_rawHandler);
        _eventBus?.Unsubscribe(_correctedHandler);

        lock (_gate)
        {
            _entries = null;
            _count = 0;
        }
    }
}
