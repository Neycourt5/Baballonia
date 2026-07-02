using BabblePersonalizer.Core.Storage;
using OpenCvSharp;
using System.Diagnostics;
using System.Text.Json;

namespace BabblePersonalizer.Core.Calibration;

public sealed class GuidedCalibrationSession : IAsyncDisposable
{
    private readonly GuidedCalibrationEngine _engine;
    private readonly CalibrationSessionWriter _writer;
    private readonly PersonalProfileStore _profileStore;
    private readonly List<CalibrationSample> _samples = new();
    private readonly Queue<float> _recentTargetValues = new();
    private readonly IReadOnlyDictionary<string, int> _parameterIndexes;
    private readonly Dictionary<string, int> _attempts = new(StringComparer.Ordinal);
    private readonly object _sync = new();
    private string _lastSegmentKey = "";
    private long _lastSavedFrameTimestamp;
    private int _finished;

    public GuidedCalibrationEngine Engine => _engine;
    public string SessionDirectory => _writer.SessionDirectory;
    public long DroppedSamples => _writer.DroppedSamples;
    public long DroppedFrames => _writer.DroppedFrames;
    public int SampleCount { get { lock (_sync) return _samples.Count; } }
    public CalibrationSample? LatestSample { get { lock (_sync) return _samples.Count == 0 ? null : _samples[^1]; } }
    public IReadOnlyList<CalibrationSample> Samples { get { lock (_sync) return _samples.ToArray(); } }

    public GuidedCalibrationSession(
        GuidedCalibrationPlan plan, CalibrationSessionWriter writer, PersonalProfileStore profileStore)
    {
        _engine = new GuidedCalibrationEngine(plan); _writer = writer; _profileStore = profileStore;
        _parameterIndexes = writer.Metadata.Model.Parameters.ToDictionary(x => x.CanonicalName, x => x.OutputIndex);
        _engine.Start(Stopwatch.GetTimestamp());
    }

    public bool RecordFrame(long timestamp, long sequence, float[] raw, float[]? corrected, Mat exactInferenceImage)
    {
        if (Volatile.Read(ref _finished) != 0) return false;
        _engine.Update(timestamp);
        var segment = _engine.Current;
        if (segment == null) return false;
        var targetIndex = _parameterIndexes.GetValueOrDefault(segment.CanonicalTarget, -1);
        var stability = 0f; var noise = 0f;
        lock (_sync)
        {
            var segmentKey = $"{segment.CanonicalTarget}:{segment.Repetition}:{segment.RequestedIntensity}:{segment.RampDirection}:{segment.IsValidation}";
            if (_lastSegmentKey != segmentKey) { _recentTargetValues.Clear(); _lastSegmentKey = segmentKey; }
            if (targetIndex >= 0 && targetIndex < raw.Length)
            {
                _recentTargetValues.Enqueue(raw[targetIndex]);
                while (_recentTargetValues.Count > 15) _recentTargetValues.Dequeue();
                stability = RobustStatistics.Stability(_recentTargetValues);
                noise = RobustStatistics.MedianAbsoluteDeviation(_recentTargetValues) * 1.4826f;
            }
            var sample = new CalibrationSample(timestamp,
                _samples.Count == 0 ? 0 : (timestamp - _samples[0].MonotonicTimestamp) / (double)Stopwatch.Frequency,
                sequence, segment.PoseName, segment.CanonicalTarget, segment.Repetition,
                segment.RequestedIntensity, segment.RampDirection, segment.StepType,
                stability, noise, (float[])raw.Clone(), corrected == null ? null : (float[])corrected.Clone(),
                segment.IsValidation, _attempts.GetValueOrDefault(AttemptKey(segment)));
            _samples.Add(sample); _writer.TryWrite(sample);
        }
        if (_writer.Metadata.FrameRecordingEnabled &&
            (timestamp - Interlocked.Read(ref _lastSavedFrameTimestamp)) / (double)Stopwatch.Frequency >=
            1 / Math.Max(.1, _writer.Metadata.FrameRecordingRate))
        {
            Interlocked.Exchange(ref _lastSavedFrameTimestamp, timestamp);
            _writer.TryWriteFrame(exactInferenceImage, sequence);
        }
        return true;
    }

    public void RetryCurrentParameter()
    {
        lock (_sync)
        {
            var segment = _engine.Current; if (segment == null || string.IsNullOrEmpty(segment.CanonicalTarget)) return;
            var key = AttemptKey(segment); _attempts[key] = _attempts.GetValueOrDefault(key) + 1;
            _engine.RetryCurrentParameter();
        }
    }

    public void SkipCurrentParameter() { lock (_sync) _engine.SkipCurrentParameter(); }

    public async Task<GuidedCalibrationResult> FinishAsync(CancellationToken cancellationToken = default)
    {
        if (Interlocked.Exchange(ref _finished, 1) != 0)
            throw new InvalidOperationException("The calibration session has already ended.");
        await _writer.CompleteAsync(cancellationToken);
        CalibrationSample[] samples; lock (_sync) samples = _samples.ToArray();
        var profile = await Task.Run(() => PersonalProfileGenerator.Generate(samples, _writer.Metadata),
            cancellationToken).ConfigureAwait(false);
        var profilePath = await _profileStore.SaveAsync(profile, cancellationToken);
        var diagnosticsPath = Path.Combine(_writer.SessionDirectory, "diagnostics.json");
        await File.WriteAllTextAsync(diagnosticsPath, JsonSerializer.Serialize(new
        {
            formatVersion = 2, completedUtc = DateTimeOffset.UtcNow, profile.ProfileId, profilePath,
            sampleCount = samples.Length, droppedSamples = _writer.DroppedSamples, droppedFrames = _writer.DroppedFrames,
            confidence = profile.Parameters.ToDictionary(x => x.CanonicalName, x => x.Confidence),
            rejected = profile.Parameters.ToDictionary(x => x.CanonicalName, x => new { x.RejectedSamples, x.RejectionReasons }),
            profile.CrossActivations, profile.Validation
        }, PersonalProfileStore.Options()), cancellationToken);
        return new GuidedCalibrationResult(profile, profilePath, _writer.SessionDirectory);
    }

    public async Task CancelAsync()
    {
        if (Interlocked.Exchange(ref _finished, 1) != 0) return;
        _engine.Cancel(); await _writer.CancelAsync();
    }

    public async ValueTask DisposeAsync()
    {
        if (Volatile.Read(ref _finished) == 0) await CancelAsync();
        await _writer.DisposeAsync();
    }

    private static string AttemptKey(CalibrationSegment segment) =>
        $"{segment.CanonicalTarget}:{segment.IsValidation}";
}

public sealed record GuidedCalibrationResult(
    PersonalCalibrationProfile Profile, string ProfilePath, string SessionDirectory);
