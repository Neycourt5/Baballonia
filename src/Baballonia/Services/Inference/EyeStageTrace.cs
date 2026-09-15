using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace Baballonia.Services.Inference;

/// <summary>Opt-in numerical diagnostics only. Copy to a bounded ring; disk I/O only on a marker.</summary>
public sealed class EyeStageTrace(int capacity = 8192)
{
    private readonly object _gate = new();
    private readonly TraceRow?[] _rows = new TraceRow[Math.Max(16, capacity)];
    private int _next, _count;
    private volatile bool _enabled;
    public bool Enabled => _enabled;
    public int Count { get { lock (_gate) return _count; } }
    public void Start() { lock (_gate) { _next = _count = 0; _enabled = true; } }
    public void Stop() => _enabled = false;

    public void Add(long frame, string stage, OrderedFloatMap values, VideoFrameIdentity? source = null,
        string? state = null)
    {
        if (!_enabled) return;
        var row = new TraceRow(frame, Stopwatch.GetTimestamp(), stage, source,
            values.ToDictionary(p => p.Key, p => float.IsFinite(p.Value) ? (float?)p.Value : null), state);
        lock (_gate)
        {
            if (!_enabled) return;
            _rows[_next] = row;
            _next = (_next + 1) % _rows.Length;
            _count = Math.Min(_count + 1, _rows.Length);
        }
    }

    public TraceRow[] Snapshot()
    {
        lock (_gate)
        {
            var first = _count < _rows.Length ? 0 : _next;
            return Enumerable.Range(0, _count).Select(i => _rows[(first + i) % _rows.Length]!).ToArray();
        }
    }

    public Task<string> MarkAsync(string directory)
    {
        var rows = Snapshot();
        var markedAt = Stopwatch.GetTimestamp();
        return Task.Run(() =>
        {
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, $"eye-issue-{DateTime.UtcNow:yyyyMMdd-HHmmss}-{Guid.NewGuid():N}.json");
            var data = new { Version = 1, MarkedAt = markedAt, StopwatchFrequency = Stopwatch.Frequency,
                Clock = "Host Stopwatch; source receipt is not exposure time",
                Receiver = "Not observed", LookInLookOut = "Downstream mapping not observed",
                Confidence = "Not supplied by the active Alpha contract; never inferred from synchronized lids",
                SourceIdentity = "Camera half order; inspect SwapSplitEyes and collector reversal before anatomical interpretation",
                Rows = rows };
            using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write);
            JsonSerializer.Serialize(stream, data, new JsonSerializerOptions { WriteIndented = true });
            return path;
        });
    }

    public sealed record TraceRow(long Frame, long Timestamp, string Stage, VideoFrameIdentity? Source,
        Dictionary<string, float?> Values, string? State);
}
