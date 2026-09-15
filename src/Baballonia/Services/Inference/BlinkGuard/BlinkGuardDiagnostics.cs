using System;
using System.Globalization;
using System.Text;

namespace Baballonia.Services.Inference.BlinkGuard;

/// <summary>One frame of both eyes, as recorded during a diagnostic capture.</summary>
public readonly record struct BlinkGuardRecord(
    double TimeSeconds,
    float LeftRawX, float LeftRawY, float LeftOutX, float LeftOutY,
    float LeftClosedness, bool LeftValid, float LeftConfidence,
    BlinkGuardState LeftState, BlinkGuardRejection LeftRejection,
    int LeftCandidates, float LeftDegreesFromRaw, float LeftDegreesFromOutput,
    float RightRawX, float RightRawY, float RightOutX, float RightOutY,
    float RightClosedness, bool RightValid, float RightConfidence,
    BlinkGuardState RightState, BlinkGuardRejection RightRejection,
    int RightCandidates, float RightDegreesFromRaw, float RightDegreesFromOutput);

/// <summary>Running totals, for answering "is this actually helping".</summary>
public sealed class BlinkGuardCounters
{
    public long BlinksDetected;
    public long SamplesRejected;
    public long ReacquisitionCount;
    public long TimeoutCount;
    public long GlitchesPrevented;
    public double ReacquisitionTotalSeconds;
    public double ReacquisitionMaxSeconds;

    public double ReacquisitionAverageSeconds =>
        ReacquisitionCount > 0 ? ReacquisitionTotalSeconds / ReacquisitionCount : 0d;

    public void Reset()
    {
        BlinksDetected = SamplesRejected = ReacquisitionCount = TimeoutCount = GlitchesPrevented = 0;
        ReacquisitionTotalSeconds = ReacquisitionMaxSeconds = 0d;
    }
}

/// <summary>
/// A bounded recorder. Counters always run; frame records only while a capture is active.
/// </summary>
/// <remarks>
/// Bounded and preallocated on purpose. This sits in a 90 Hz path, so an unbounded log would be
/// both an allocation source and a way to fill a disk during a long session. The ring simply
/// overwrites, and a capture is something the user starts and stops deliberately.
/// </remarks>
public sealed class BlinkGuardDiagnostics(int capacity = 8192)
{
    private readonly BlinkGuardRecord[] _records = new BlinkGuardRecord[Math.Max(16, capacity)];
    private readonly object _gate = new();
    private int _next;
    private int _count;

    public BlinkGuardCounters Counters { get; } = new();

    public bool IsCapturing { get; private set; }

    public int RecordCount
    {
        get { lock (_gate) return _count; }
    }

    public int Capacity => _records.Length;

    public void StartCapture()
    {
        lock (_gate)
        {
            _next = 0;
            _count = 0;
            IsCapturing = true;
        }
    }

    public void StopCapture()
    {
        lock (_gate) IsCapturing = false;
    }

    public void Add(in BlinkGuardRecord record)
    {
        lock (_gate)
        {
            if (!IsCapturing)
                return;

            _records[_next] = record;
            _next = (_next + 1) % _records.Length;
            if (_count < _records.Length)
                _count++;
        }
    }

    /// <summary>Oldest-first CSV of everything currently held.</summary>
    public string ToCsv()
    {
        var builder = new StringBuilder();
        builder.AppendLine(
            "t,l_raw_x,l_raw_y,l_out_x,l_out_y,l_closed,l_valid,l_conf,l_state,l_reject," +
            "l_candidates,l_deg_from_raw,l_deg_from_out," +
            "r_raw_x,r_raw_y,r_out_x,r_out_y,r_closed,r_valid,r_conf,r_state,r_reject," +
            "r_candidates,r_deg_from_raw,r_deg_from_out");

        lock (_gate)
        {
            var start = _count < _records.Length ? 0 : _next;
            for (var i = 0; i < _count; i++)
            {
                var r = _records[(start + i) % _records.Length];
                builder.Append(F(r.TimeSeconds)).Append(',')
                    .Append(F(r.LeftRawX)).Append(',').Append(F(r.LeftRawY)).Append(',')
                    .Append(F(r.LeftOutX)).Append(',').Append(F(r.LeftOutY)).Append(',')
                    .Append(F(r.LeftClosedness)).Append(',').Append(r.LeftValid ? 1 : 0).Append(',')
                    .Append(F(r.LeftConfidence)).Append(',')
                    .Append(r.LeftState).Append(',').Append(r.LeftRejection).Append(',')
                    .Append(r.LeftCandidates).Append(',')
                    .Append(F(r.LeftDegreesFromRaw)).Append(',').Append(F(r.LeftDegreesFromOutput)).Append(',')
                    .Append(F(r.RightRawX)).Append(',').Append(F(r.RightRawY)).Append(',')
                    .Append(F(r.RightOutX)).Append(',').Append(F(r.RightOutY)).Append(',')
                    .Append(F(r.RightClosedness)).Append(',').Append(r.RightValid ? 1 : 0).Append(',')
                    .Append(F(r.RightConfidence)).Append(',')
                    .Append(r.RightState).Append(',').Append(r.RightRejection).Append(',')
                    .Append(r.RightCandidates).Append(',')
                    .Append(F(r.RightDegreesFromRaw)).Append(',').Append(F(r.RightDegreesFromOutput));
                builder.AppendLine();
            }
        }

        return builder.ToString();
    }

    /// <summary>Invariant culture throughout: a capture may be read on a machine with a comma decimal separator.</summary>
    private static string F(double value) => value.ToString("G9", CultureInfo.InvariantCulture);
}
