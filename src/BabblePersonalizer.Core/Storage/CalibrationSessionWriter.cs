using BabblePersonalizer.Core.Camera;
using BabblePersonalizer.Core.Models;
using OpenCvSharp;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;

namespace BabblePersonalizer.Core.Storage;

public sealed record SessionMetadata(
    int FormatVersion, string SessionId, DateTimeOffset StartedUtc,
    string ApplicationVersion, ModelContract Model, CameraConfiguration Camera,
    bool FrameRecordingEnabled, double FrameRecordingRate);

public sealed record CalibrationSample(
    long MonotonicTimestamp, double SessionSeconds, long FrameSequence,
    string PoseName, string CanonicalTarget, int Repetition, float RequestedIntensity,
    string RampDirection, string StepType, float Stability, float Noise,
    float[] RawOutput, float[]? CorrectedOutput = null);

public sealed class CalibrationSessionWriter : IAsyncDisposable
{
    private abstract record Item;
    private sealed record SampleItem(CalibrationSample Value) : Item;
    private sealed record FrameItem(Mat Image, string FileName) : Item;

    private readonly Channel<Item> _channel;
    private readonly string _directory;
    private readonly string[] _expressionNames;
    private readonly CancellationTokenSource _cancellation = new();
    private readonly Task _worker;
    private int _completed;
    public long DroppedSamples { get; private set; }
    public long DroppedFrames { get; private set; }
    public string SessionDirectory => _directory;

    public CalibrationSessionWriter(PersonalizerDataPaths paths, SessionMetadata metadata, int queueCapacity = 512)
    {
        paths.EnsureCreated();
        _directory = Path.Combine(paths.Sessions, $"Session-{DateTime.Now:yyyyMMdd-HHmmss}");
        Directory.CreateDirectory(_directory);
        if (metadata.FrameRecordingEnabled) Directory.CreateDirectory(Path.Combine(_directory, "frames"));
        _expressionNames = metadata.Model.Parameters.Select(x => x.CanonicalName).ToArray();
        File.WriteAllText(Path.Combine(_directory, "metadata.json"),
            JsonSerializer.Serialize(metadata, JsonOptions()));
        File.WriteAllText(Path.Combine(_directory, "manual-labels.json"), "[]");
        _channel = Channel.CreateBounded<Item>(new BoundedChannelOptions(queueCapacity)
        {
            FullMode = BoundedChannelFullMode.Wait, SingleReader = true, SingleWriter = false
        });
        _worker = Task.Run(() => WriteLoop(_cancellation.Token));
    }

    public bool TryWrite(CalibrationSample sample)
    {
        if (Volatile.Read(ref _completed) != 0) return false;
        if (_channel.Writer.TryWrite(new SampleItem(sample))) return true;
        DroppedSamples++; return false;
    }

    public bool TryWriteFrame(Mat exactInferenceImage, long sequence)
    {
        if (Volatile.Read(ref _completed) != 0) return false;
        var clone = exactInferenceImage.Clone();
        if (_channel.Writer.TryWrite(new FrameItem(clone, $"frame-{sequence:000000000}.png"))) return true;
        clone.Dispose(); DroppedFrames++; return false;
    }

    public async Task CompleteAsync(CancellationToken cancellationToken = default)
    {
        if (Interlocked.Exchange(ref _completed, 1) != 0) return;
        _channel.Writer.TryComplete();
        await _worker.WaitAsync(cancellationToken).ConfigureAwait(false);
        await File.WriteAllTextAsync(Path.Combine(_directory, "diagnostics.json"), JsonSerializer.Serialize(new
        {
            formatVersion = 1, completedUtc = DateTimeOffset.UtcNow, DroppedSamples, DroppedFrames
        }, JsonOptions()), cancellationToken).ConfigureAwait(false);
    }

    public async Task CancelAsync()
    {
        if (Interlocked.Exchange(ref _completed, 1) != 0) return;
        _channel.Writer.TryComplete();
        await _worker.ConfigureAwait(false);
        await File.WriteAllTextAsync(Path.Combine(_directory, "diagnostics.json"), JsonSerializer.Serialize(new
        {
            formatVersion = 1, cancelledUtc = DateTimeOffset.UtcNow, DroppedSamples, DroppedFrames
        }, JsonOptions())).ConfigureAwait(false);
    }

    private async Task WriteLoop(CancellationToken cancellationToken)
    {
        await using var writer = new StreamWriter(Path.Combine(_directory, "samples.csv"), false, new UTF8Encoding(false));
        await writer.WriteLineAsync(Header()).ConfigureAwait(false);
        await foreach (var item in _channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            switch (item)
            {
                case SampleItem sample: await writer.WriteLineAsync(ToCsv(sample.Value)).ConfigureAwait(false); break;
                case FrameItem frame:
                    try { Cv2.ImWrite(Path.Combine(_directory, "frames", frame.FileName), frame.Image); }
                    finally { frame.Image.Dispose(); }
                    break;
            }
        }
    }

    private string Header() => string.Join(',', new[]
    {
        "monotonic_timestamp", "session_seconds", "frame_sequence", "pose", "canonical_target",
        "repetition", "requested_intensity", "ramp_direction", "step_type", "stability", "noise"
    }.Concat(_expressionNames.Select(x => "raw_" + x)).Concat(_expressionNames.Select(x => "corrected_" + x)));

    private static string ToCsv(CalibrationSample sample)
    {
        static string F(float x) => x.ToString("R", CultureInfo.InvariantCulture);
        static string E(string x) => '"' + x.Replace("\"", "\"\"") + '"';
        var fields = new[]
        {
            sample.MonotonicTimestamp.ToString(CultureInfo.InvariantCulture),
            sample.SessionSeconds.ToString("R", CultureInfo.InvariantCulture),
            sample.FrameSequence.ToString(CultureInfo.InvariantCulture), E(sample.PoseName), E(sample.CanonicalTarget),
            sample.Repetition.ToString(CultureInfo.InvariantCulture), F(sample.RequestedIntensity),
            E(sample.RampDirection), E(sample.StepType), F(sample.Stability), F(sample.Noise)
        };
        var corrected = sample.CorrectedOutput?.Select(F) ??
                        Enumerable.Repeat(string.Empty, sample.RawOutput.Length);
        return string.Join(',', fields.Concat(sample.RawOutput.Select(F)).Concat(corrected));
    }

    private static JsonSerializerOptions JsonOptions() => new() { WriteIndented = true };

    public async ValueTask DisposeAsync()
    {
        if (Volatile.Read(ref _completed) == 0) await CancelAsync().ConfigureAwait(false);
        _cancellation.Dispose();
    }
}
