using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Baballonia.Services.events;
using Microsoft.Extensions.Logging;
using OpenCvSharp;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Records synchronized training data: the exact 224x224 frame the stock model consumed, the raw
/// pre-filter prediction it produced, and (during guided calibration) the commanded target that the
/// user was imitating.
///
/// Recording the post-transform frame rather than the raw camera image is deliberate. It is exactly
/// the model input, so the trainer never has to re-implement ImageTransformer's ROI crop, gamma and
/// fused rotate/flip/resize in Python - a classic source of silent train/serve skew.
///
/// Performance: the pipeline event fires on the 10 ms processing tick while the event bus holds its
/// lock, so the handler only clones the Mat, copies floats and enqueues. All JPEG encoding and disk
/// I/O happen on a background consumer. The queue is bounded and drops oldest, so a slow disk
/// degrades into dropped frames rather than stalled face tracking.
/// </summary>
public sealed class DatasetRecorderService : IDisposable
{
    /// <summary>
    /// The processing tick (~100 Hz) re-serves the same camera frame when the camera is slower, so
    /// frames are deduplicated and capped. 30 fps is plenty for expression dynamics and keeps a
    /// session's disk footprint sane (~23 MB/min).
    /// </summary>
    public const int MaxRecordFps = 30;

    private const int JpegQuality = 95;
    private const int QueueCapacity = 256;

    private readonly IFacePipelineEventBus _eventBus;
    private readonly ILogger<DatasetRecorderService> _logger;
    private readonly IReadOnlyCueStateSource? _cueState;
    private readonly TrainingCaptureGate? _captureGate;

    private readonly Action<FacePipelineEvents.NewRawExpressionsEvent> _handler;
    private readonly object _sessionLock = new();

    private RecordingSession? _session;
    private IDisposable? _captureLease;
    private long _lastSourceFrameTimestamp;

    /// <summary>
    /// A positive readiness signal from the same post-transform/inference event that recordings
    /// consume. Camera selection or a running capture object alone is insufficient: guided labels
    /// must not start until a real, usable face frame has reached the recorder tap.
    /// </summary>
    public bool HasRecentSourceFrame
    {
        get
        {
            var last = Interlocked.Read(ref _lastSourceFrameTimestamp);
            return last != 0 &&
                   Stopwatch.GetElapsedTime(last, Stopwatch.GetTimestamp()) <= TimeSpan.FromSeconds(2);
        }
    }

    public DatasetRecorderService(
        IFacePipelineEventBus eventBus,
        ILogger<DatasetRecorderService> logger,
        IReadOnlyCueStateSource? cueState = null,
        TrainingCaptureGate? captureGate = null)
    {
        _eventBus = eventBus;
        _logger = logger;
        _cueState = cueState;
        _captureGate = captureGate;

        _handler = OnRawExpressions;
        _eventBus.Subscribe(_handler);
    }

    public bool IsRecording
    {
        get { lock (_sessionLock) return _session != null; }
    }

    public string? CurrentSessionId
    {
        get { lock (_sessionLock) return _session?.SessionId; }
    }

    public int FramesWritten
    {
        get { lock (_sessionLock) return _session?.FramesWritten ?? 0; }
    }

    /// <summary>
    /// Starts a session and returns its id. Metadata is written immediately so a crashed session
    /// still leaves an interpretable directory.
    /// </summary>
    public string StartSession(
        SessionType type,
        SessionMetadata.CameraGeometry? camera = null,
        string? notes = null,
        bool requireRecentSourceFrame = false,
        string? datasetRoot = null,
        C2.C2CaptureContext? c2 = null,
        IReadOnlyCueStateSource? cueSource = null)
    {
        lock (_sessionLock)
        {
            if (_session != null)
                throw new InvalidOperationException($"Session '{_session.SessionId}' is already recording.");

            // UI enablement is advisory and can become stale between frames. Callers that are
            // creating user-facing recordings opt into this backend check so a stopped camera can
            // never create a zero-frame session directory merely because a button stayed enabled.
            if (requireRecentSourceFrame && !HasRecentSourceFrame)
            {
                throw new InvalidOperationException(
                    "No fresh face-camera inference frame is available. Start the face camera, " +
                    "wait for tracking to move, then try again.");
            }

            IDisposable? captureLease = null;
            if (_captureGate != null &&
                !_captureGate.TryEnter($"a {type} training recording", out captureLease, out var message))
            {
                throw new InvalidOperationException(message);
            }

            try
            {
                var startedUtc = DateTime.UtcNow;
                var sessionId = PersonalizationPaths.NewSessionId(type, startedUtc);

                var directory = Path.Combine(datasetRoot ?? PersonalizationPaths.DatasetRoot, sessionId);
                Directory.CreateDirectory(Path.Combine(directory, "frames"));

                var metadata = new SessionMetadata
                {
                    SessionId = sessionId,
                    SessionType = type.ToString(),
                    StartedUtc = startedUtc.ToString("o"),
                    AppVersion = typeof(DatasetRecorderService).Assembly.GetName().Version?.ToString() ?? "unknown",
                    ImageWidth = 0,
                    ImageHeight = 0,
                    JpegQuality = JpegQuality,
                    Camera = camera,
                    Notes = notes,
                    C2 = c2
                };

                // Leave enough provenance to recover an interrupted recording. The finalizer
                // rewrites this file with geometry and counts, but a crash before StopSessionAsync
                // must not strand otherwise usable frames in an anonymous directory.
                WriteSessionMetadata(directory, metadata);

                _session = new RecordingSession(sessionId, directory, metadata, _logger, cueSource);
                _session.Start();
                _captureLease = captureLease;

                _logger.LogInformation("Personalization: recording session {SessionId} to {Path}",
                    sessionId, PersonalizationPaths.SessionDirectory(sessionId));

                return sessionId;
            }
            catch
            {
                captureLease?.Dispose();
                throw;
            }
        }
    }

    /// <summary>
    /// Stops recording, drains anything still queued, and finalizes session.json with the frame
    /// count and measured unique-frame rate.
    /// </summary>
    public async Task<SessionSummary?> StopSessionAsync()
    {
        RecordingSession? session;
        IDisposable? captureLease;
        lock (_sessionLock)
        {
            session = _session;
            _session = null;
            captureLease = _captureLease;
            _captureLease = null;
        }

        if (session == null)
        {
            captureLease?.Dispose();
            return null;
        }

        // Dispose can synchronously wait for this method while running on the Avalonia UI thread.
        // Do not capture that context: the writer must be able to drain and finalize independently.
        SessionSummary summary;
        try { summary = await session.CompleteAsync().ConfigureAwait(false); }
        finally { captureLease?.Dispose(); }

        _logger.LogInformation(
            "Personalization: session {SessionId} finished - {Frames} frames, {Fps:F1} unique fps, {Dropped} dropped",
            summary.SessionId, summary.FrameCount, summary.EffectiveFps, summary.DroppedFrames);

        if (summary.DroppedFrames > 0)
        {
            _logger.LogWarning(
                "Personalization: {Dropped} frames were dropped; the writer could not keep up with capture.",
                summary.DroppedFrames);
        }

        return summary;
    }

    /// <summary>
    /// Runs on the processing tick under the event bus lock. Must stay cheap: clone, copy, enqueue.
    /// </summary>
    private void OnRawExpressions(FacePipelineEvents.NewRawExpressionsEvent e)
    {
        if (e.transformedFrame is { } frame && !frame.Empty() &&
            e.rawResult.Length >= PersonalizationSchema.ExpressionCount)
            Interlocked.Exchange(ref _lastSourceFrameTimestamp, Stopwatch.GetTimestamp());

        RecordingSession? session;
        lock (_sessionLock) session = _session;

        if (session == null)
            return;

        try
        {
            session.Offer(e, _cueState?.CurrentCue());
        }
        catch (Exception ex)
        {
            // Never let recording break tracking.
            _logger.LogError(ex, "Personalization: failed to enqueue frame; recording continues.");
        }
    }

    public void Dispose()
    {
        _eventBus.Unsubscribe(_handler);

        RecordingSession? session;
        IDisposable? captureLease;
        lock (_sessionLock)
        {
            session = _session;
            _session = null;
            captureLease = _captureLease;
            _captureLease = null;
        }

        try { session?.CompleteAsync().GetAwaiter().GetResult(); }
        finally { captureLease?.Dispose(); }
    }

    private static void WriteSessionMetadata(string directory, SessionMetadata metadata)
    {
        var path = Path.Combine(directory, "session.json");
        var temporaryPath = $"{path}.{Guid.NewGuid():N}.tmp";
        try
        {
            File.WriteAllText(
                temporaryPath,
                JsonSerializer.Serialize(metadata, PersonalizationPaths.IndentedJson),
                PersonalizationPaths.Utf8NoBom);
            File.Move(temporaryPath, path, overwrite: true);
        }
        finally
        {
            try
            {
                if (File.Exists(temporaryPath))
                    File.Delete(temporaryPath);
            }
            catch
            {
                // Keep the original write/move exception; a same-directory temp file is harmless
                // and can be diagnosed without concealing why session.json was not committed.
            }
        }
    }

    /// <summary>Outcome of a recording, useful for UI display and for tests.</summary>
    public sealed record SessionSummary(
        string SessionId,
        string Directory,
        int FrameCount,
        int DroppedFrames,
        double EffectiveFps);

    /// <summary>One in-flight recording: bounded queue, background writer, running statistics.</summary>
    private sealed class RecordingSession
    {
        private readonly Channel<CapturedFrame> _channel;
        private readonly SessionMetadata _metadata;
        private readonly ILogger _logger;
        private readonly string _directory;
        private readonly IReadOnlyCueStateSource? _cueSource;
        private readonly long _minTicksBetweenFrames = Stopwatch.Frequency / MaxRecordFps;

        public string SessionId { get; }

        private Task? _writerTask;
        private int _framesOffered;
        private int _framesWritten;
        private int _droppedFrames;
        private long _lastAcceptedTimestamp;
        private ulong _lastChecksum;
        private long _firstFrameTimestamp;
        private long _lastFrameTimestamp;
        private int _imageWidth;
        private int _imageHeight;

        public RecordingSession(string sessionId, string directory, SessionMetadata metadata, ILogger logger,
            IReadOnlyCueStateSource? cueSource)
        {
            SessionId = sessionId;
            _metadata = metadata;
            _logger = logger;
            _directory = directory;
            _cueSource = cueSource;

            _channel = Channel.CreateBounded<CapturedFrame>(
                new BoundedChannelOptions(QueueCapacity)
                {
                    // Prefer losing the oldest frames over blocking the processing tick: stalled
                    // face tracking is far worse than a gap in a training recording.
                    FullMode = BoundedChannelFullMode.DropOldest,
                    SingleReader = true,
                    SingleWriter = true
                },
                dropped =>
                {
                    // Reclaim the clone that would otherwise leak native memory.
                    dropped.Image.Dispose();
                    Interlocked.Increment(ref _droppedFrames);
                });
        }

        public int FramesWritten => Volatile.Read(ref _framesWritten);

        public void Start() => _writerTask = Task.Run(WriteLoopAsync);

        /// <summary>
        /// Called on the processing tick. Rejects duplicate and over-rate frames before doing any
        /// copying, so the common "camera slower than tick" case costs almost nothing.
        /// </summary>
        public void Offer(FacePipelineEvents.NewRawExpressionsEvent e, FrameLabel.CueLabel? cue)
        {
            if (_metadata.C2 != null && (e.embedding == null ||
                e.embedding.Length != _metadata.C2.EmbeddingDim || e.inferenceTimestamp <= 0))
                return; // C2 never silently backfills a missing live feature or invents capture time.
            var now = Stopwatch.GetTimestamp();

            if (_lastAcceptedTimestamp != 0 && now - _lastAcceptedTimestamp < _minTicksBetweenFrames)
                return;

            // The tick re-serves the latest camera Mat, so identical frames arrive repeatedly.
            // A sparse checksum is enough to spot them and costs far less than encoding.
            var checksum = FrameChecksum.Sparse(e.transformedFrame);
            if (checksum == _lastChecksum)
                return;

            _lastChecksum = checksum;
            _lastAcceptedTimestamp = now;

            if (_firstFrameTimestamp == 0)
                _firstFrameTimestamp = now;
            _lastFrameTimestamp = now;

            var index = _framesOffered++;

            var captured = new CapturedFrame(
                Index: index,
                Image: e.transformedFrame.Clone(),
                Stock: (float[])e.rawResult.Clone(),
                TimestampTicks: e.timestampTicks,
                Cue: _cueSource != null ? _cueSource.CurrentCue() : cue,
                Embedding: _metadata.C2 != null ? e.embedding!.ToArray() : null,
                InferenceTimestamp: _metadata.C2 != null ? e.inferenceTimestamp : 0,
                SourceFrame: _metadata.C2 != null ? e.sourceFrame : null);

            if (!_channel.Writer.TryWrite(captured))
            {
                captured.Image.Dispose();
                Interlocked.Increment(ref _droppedFrames);
            }
        }

        private async Task WriteLoopAsync()
        {
            var labelsPath = Path.Combine(_directory, "labels.jsonl");
            var framesDir = Path.Combine(_directory, "frames");
            var encodeParams = new[] { new ImageEncodingParam(ImwriteFlags.JpegQuality, JpegQuality) };

            await using var labels = new StreamWriter(labelsPath, append: true, PersonalizationPaths.Utf8NoBom);
            using var embeddings = _metadata.C2 != null
                ? new BinaryWriter(File.Open(Path.Combine(_directory, "embeddings.f32"), FileMode.CreateNew)) : null;
            var embeddingRow = 0;

            await foreach (var frame in _channel.Reader.ReadAllAsync())
            {
                try
                {
                    if (_imageWidth == 0)
                    {
                        _imageWidth = frame.Image.Width;
                        _imageHeight = frame.Image.Height;
                    }

                    var imagePath = Path.Combine(framesDir, $"{frame.Index:D6}.jpg");
                    frame.Image.SaveImage(imagePath, encodeParams);

                    int? row = null;
                    if (frame.Embedding != null && embeddings != null)
                    {
                        row = embeddingRow++;
                        foreach (var value in frame.Embedding) embeddings.Write(value);
                    }
                    var label = new FrameLabel
                    {
                        Index = frame.Index,
                        TimestampTicks = frame.TimestampTicks,
                        Stock = frame.Stock,
                        Cue = frame.Cue,
                        InferenceTimestamp = frame.InferenceTimestamp,
                        EmbeddingRow = row,
                        SourceFrame = frame.SourceFrame
                    };

                    await labels.WriteLineAsync(JsonSerializer.Serialize(label, PersonalizationPaths.Json));
                    Interlocked.Increment(ref _framesWritten);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Personalization: failed writing frame {Index}", frame.Index);
                    // A partial feature row can shift all following rows. Retain incomplete
                    // recovery metadata instead of blessing damaged C2 pairs as complete.
                    if (_metadata.C2 != null) throw;
                }
                finally
                {
                    frame.Image.Dispose();
                }
            }

            await labels.FlushAsync();
        }

        public async Task<SessionSummary> CompleteAsync()
        {
            _channel.Writer.TryComplete();

            try
            {
                if (_writerTask != null)
                    await _writerTask.ConfigureAwait(false);
            }
            finally
            {
                while (_channel.Reader.TryRead(out var pending)) pending.Image.Dispose();
            }

            var elapsedSeconds = _firstFrameTimestamp == 0
                ? 0d
                : (_lastFrameTimestamp - _firstFrameTimestamp) / (double)Stopwatch.Frequency;

            var written = FramesWritten;
            var fps = elapsedSeconds > 0 ? (written - 1) / elapsedSeconds : 0d;

            // Image dimensions are only known once a frame has been seen, so session.json is
            // rewritten here with the observed geometry alongside the final counts.
            var finalized = new SessionMetadata
            {
                SchemaVersion = _metadata.SchemaVersion,
                SessionId = _metadata.SessionId,
                SessionType = _metadata.SessionType,
                StartedUtc = _metadata.StartedUtc,
                EndedUtc = DateTime.UtcNow.ToString("o"),
                AppVersion = _metadata.AppVersion,
                ImageWidth = _imageWidth,
                ImageHeight = _imageHeight,
                JpegQuality = _metadata.JpegQuality,
                Camera = _metadata.Camera,
                Notes = _metadata.Notes,
                FrameCount = written,
                EffectiveFps = Math.Round(fps, 2),
                C2 = _metadata.C2
            };

            // Preserve the immediately written recovery document until the finalized replacement
            // is complete. A process interruption during shutdown must not truncate session.json.
            WriteSessionMetadata(_directory, finalized);

            return new SessionSummary(
                SessionId,
                _directory,
                written,
                Volatile.Read(ref _droppedFrames),
                fps);
        }

    }

    private sealed record CapturedFrame(
        int Index,
        Mat Image,
        float[] Stock,
        long TimestampTicks,
        FrameLabel.CueLabel? Cue,
        float[]? Embedding,
        long InferenceTimestamp,
        Inference.VideoFrameIdentity? SourceFrame);
}

/// <summary>
/// Read-only view of the current calibration cue, so the recorder can stamp supervision metadata
/// without depending on the cue engine itself. Null outside guided sessions.
/// </summary>
public interface IReadOnlyCueStateSource
{
    FrameLabel.CueLabel? CurrentCue();
}
