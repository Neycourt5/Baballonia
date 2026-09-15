using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Baballonia.Services.events;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Baballonia.Services.Personalization.Eye;

/// <summary>
/// Writes the guided eye-capture dataset: one jsonl line per frame, plus a session header.
/// </summary>
/// <remarks>
/// <para>Taps <see cref="EyePipelineEvents.NewRawEyeExpressionsEvent"/>, which is the model's own
/// output before correction, filtering and geometry — the same space the corrector will run in, so
/// what is trained on is what will be seen at inference time.</para>
///
/// <para>That event is published on the inference thread while the bus holds its lock, so the
/// handler does the minimum: copy twelve floats and hand them to a bounded channel. Everything else
/// — JSON, file I/O — happens on a background writer. The channel drops the oldest frame when full
/// rather than blocking, because losing a frame is survivable and stalling eye tracking is not.</para>
///
/// <para>UTF-8 without a BOM, deliberately: <c>Encoding.UTF8</c>'s preamble breaks Python's
/// <c>json.loads</c> on the first line, which the face dataset learned the hard way.</para>
/// </remarks>
public sealed class EyeDatasetRecorder : IDisposable
{
    /// <summary>Enough for several seconds of backlog at 90 Hz; far more than the writer needs.</summary>
    public const int QueueCapacity = 512;

    private static readonly UTF8Encoding Utf8NoBom = new(encoderShouldEmitUTF8Identifier: false);

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        // Matches the record property names exactly, which are already the short wire names.
        WriteIndented = false,
    };

    private readonly IEyePipelineEventBus _bus;
    private readonly ILogger _logger;
    private readonly Action<EyePipelineEvents.NewRawEyeExpressionsEvent> _handler;
    private readonly object _gate = new();

    private RecordingSession? _session;

    public EyeDatasetRecorder(IEyePipelineEventBus bus, ILogger<EyeDatasetRecorder>? logger = null)
    {
        _bus = bus;
        _logger = logger ?? NullLogger<EyeDatasetRecorder>.Instance;
        _handler = OnRawEyeExpressions;
        _bus.Subscribe(_handler);
    }

    public bool IsRecording
    {
        get { lock (_gate) return _session is not null; }
    }

    public string? SessionId
    {
        get { lock (_gate) return _session?.Id; }
    }

    public int FrameCount
    {
        get { lock (_gate) return _session?.Written ?? 0; }
    }

    public int DroppedFrames
    {
        get { lock (_gate) return _session?.Dropped ?? 0; }
    }

    /// <summary>
    /// The cue to stamp onto incoming frames, or null between poses. Set by the capture service as
    /// the routine changes phase; read on the inference thread, so it is swapped whole rather than
    /// mutated.
    /// </summary>
    public EyeCueLabel? CurrentCue { get; set; }

    /// <summary>Begins a session and writes its header immediately.</summary>
    /// <remarks>
    /// The header is written up front, not just at the end, so a session interrupted by a crash
    /// still leaves an interpretable directory rather than orphaned frames.
    /// </remarks>
    public string Start(EyeSessionMetadata metadata)
    {
        lock (_gate)
        {
            if (_session is not null)
                throw new InvalidOperationException("An eye capture session is already recording.");

            Directory.CreateDirectory(EyeDatasetPaths.SessionDirectory(metadata.sessionId));
            WriteMetadata(metadata);

            _session = new RecordingSession(metadata, _logger);
            return metadata.sessionId;
        }
    }

    /// <summary>Flushes, rewrites the header with the final counts, and writes the quality sidecar.</summary>
    public async Task<EyeSessionMetadata?> StopAsync(EyeSessionQuality quality)
    {
        RecordingSession? session;
        lock (_gate)
        {
            session = _session;
            _session = null;
            CurrentCue = null;
        }

        if (session is null)
            return null;

        var finalized = await session.CompleteAsync().ConfigureAwait(false);
        WriteMetadata(finalized);

        // Written after the frames are safely on disk: a writer failure must not be able to turn a
        // cancelled attempt back into trusted training data.
        File.WriteAllText(
            EyeDatasetPaths.QualityPath(finalized.sessionId),
            JsonSerializer.Serialize(quality, JsonOptions),
            Utf8NoBom);

        _logger.LogInformation(
            "Eye capture {Session} finished: {Frames} frames, {Dropped} dropped",
            finalized.sessionId, finalized.frameCount, session.Dropped);

        return finalized;
    }

    private static void WriteMetadata(EyeSessionMetadata metadata)
    {
        var path = EyeDatasetPaths.MetadataPath(metadata.sessionId);
        var temporary = path + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(metadata, JsonOptions), Utf8NoBom);
        File.Move(temporary, path, overwrite: true);
    }

    private void OnRawEyeExpressions(EyePipelineEvents.NewRawEyeExpressionsEvent message)
    {
        RecordingSession? session;
        lock (_gate)
            session = _session;

        if (session is null)
            return;

        // The payload is the pipeline's reusable scratch buffer; copying is not optional.
        session.Offer(message.rawResult, message.timestampTicks, CurrentCue);
    }

    public void Dispose()
    {
        _bus.Unsubscribe(_handler);

        RecordingSession? session;
        lock (_gate)
        {
            session = _session;
            _session = null;
        }

        session?.Abandon();
    }

    private sealed class RecordingSession
    {
        private readonly Channel<EyeFrameLabel> _channel;
        private readonly Task _writer;
        private readonly StreamWriter _output;
        private readonly EyeSessionMetadata _metadata;
        private readonly long _startedTicks = DateTime.UtcNow.Ticks;

        private int _offered;
        private int _written;
        private int _dropped;

        public RecordingSession(EyeSessionMetadata metadata, ILogger logger)
        {
            _metadata = metadata;
            Id = metadata.sessionId;

            _channel = Channel.CreateBounded<EyeFrameLabel>(
                new BoundedChannelOptions(QueueCapacity)
                {
                    FullMode = BoundedChannelFullMode.DropOldest,
                    SingleReader = true,
                    SingleWriter = true,
                },
                _ => Interlocked.Increment(ref _dropped));

            _output = new StreamWriter(
                new FileStream(EyeDatasetPaths.LabelsPath(Id), FileMode.Create, FileAccess.Write,
                    FileShare.Read),
                Utf8NoBom);

            _writer = Task.Run(async () =>
            {
                try
                {
                    await foreach (var frame in _channel.Reader.ReadAllAsync().ConfigureAwait(false))
                    {
                        await _output.WriteLineAsync(JsonSerializer.Serialize(frame, JsonOptions))
                            .ConfigureAwait(false);
                        Interlocked.Increment(ref _written);
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Eye capture writer failed for session {Session}", Id);
                }
            });
        }

        public string Id { get; }
        public int Written => Volatile.Read(ref _written);
        public int Dropped => Volatile.Read(ref _dropped);

        public void Offer(float[] stock, long timestampTicks, EyeCueLabel? cue)
        {
            var copy = new float[stock.Length];
            stock.AsSpan().CopyTo(copy);
            var index = Interlocked.Increment(ref _offered) - 1;
            _channel.Writer.TryWrite(new EyeFrameLabel(index, timestampTicks, copy, cue));
        }

        public async Task<EyeSessionMetadata> CompleteAsync()
        {
            _channel.Writer.TryComplete();
            await _writer.ConfigureAwait(false);
            await _output.FlushAsync().ConfigureAwait(false);
            await _output.DisposeAsync().ConfigureAwait(false);

            var seconds = Math.Max((DateTime.UtcNow.Ticks - _startedTicks) / (double)TimeSpan.TicksPerSecond, 1e-3);
            return _metadata with
            {
                endedUtc = DateTime.UtcNow.ToString("o"),
                frameCount = Written,
                effectiveFps = Written / seconds,
            };
        }

        public void Abandon()
        {
            _channel.Writer.TryComplete();
            try { _writer.Wait(TimeSpan.FromSeconds(2)); } catch { /* shutting down */ }
            try { _output.Dispose(); } catch { /* shutting down */ }
        }
    }
}
