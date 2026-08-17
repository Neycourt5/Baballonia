using System;

namespace Baballonia.Services.EyeV2;

/// <summary>
/// V2-A personal affine gaze map plus lid-anchor/temporal interpretation.
/// The upstream model exposes only one lid scalar per eye. Time can distinguish short closures
/// from sustained partial closures, but it cannot perfectly distinguish a tensed squint from a
/// half blink at the same aperture. That information ceiling is intentional and belongs to V2-B.
/// </summary>
public sealed class EyeV2Mapper : IEyeStateMapper
{
    private readonly EyeTemporalState _left = new();
    private readonly EyeTemporalState _right = new();
    private readonly Action<EyeV2Diagnostics>? _diagnostics;
    private long _lastDiagnosticsTicks;

    public EyeV2Calibration Calibration { get; }

    public EyeV2Mapper(EyeV2Calibration calibration, Action<EyeV2Diagnostics>? diagnostics = null)
    {
        if (!calibration.IsValid())
            throw new ArgumentException("Eye V2 calibration is invalid.", nameof(calibration));
        Calibration = calibration;
        _diagnostics = diagnostics;
    }

    public float[] Map(float[] stockState, float[] filteredRawState, long timestampTicks)
    {
        if (stockState.Length < EyeStateLayout.LegacyCount ||
            filteredRawState.Length < EyeStateLayout.LegacyCount)
            throw new ArgumentException("Eye V2 requires six stock and raw values.");

        var leftRaw = EyeRawState.FromModel(filteredRawState, EyeSide.Left);
        var rightRaw = EyeRawState.FromModel(filteredRawState, EyeSide.Right);
        var left = MapEye(leftRaw, Calibration.Left, _left, timestampTicks);
        var right = MapEye(rightRaw, Calibration.Right, _right, timestampTicks);

        var output = new float[EyeStateLayout.V2Count];
        output[EyeStateLayout.LeftX] = left.MappedX;
        output[EyeStateLayout.LeftY] = left.MappedY;
        output[EyeStateLayout.LeftLid] = left.NormalizedOpenness;
        output[EyeStateLayout.RightX] = right.MappedX;
        output[EyeStateLayout.RightY] = right.MappedY;
        output[EyeStateLayout.RightLid] = right.NormalizedOpenness;
        output[EyeStateLayout.LeftWide] = left.Wide;
        output[EyeStateLayout.LeftSquint] = left.Squint;
        output[EyeStateLayout.RightWide] = right.Wide;
        output[EyeStateLayout.RightSquint] = right.Squint;

        // Debug updates are capped at 10 Hz. The mapper itself remains allocation-light except for
        // its required output array; normal users never bind at the processing tick's full rate.
        if (_diagnostics != null && timestampTicks - _lastDiagnosticsTicks >= TimeSpan.TicksPerMillisecond * 100)
        {
            _lastDiagnosticsTicks = timestampTicks;
            _diagnostics(new EyeV2Diagnostics(timestampTicks, left, right, Calibration));
        }

        return output;
    }

    public void Reset()
    {
        _left.Reset();
        _right.Reset();
        _lastDiagnosticsTicks = 0;
    }

    private static EyeV2PerEyeDiagnostics MapEye(
        EyeRawState raw,
        EyeV2PerEyeCalibration calibration,
        EyeTemporalState temporal,
        long ticks)
    {
        var (mappedX, mappedY) = calibration.Gaze.Map(raw.X, raw.Y);
        mappedX = Math.Clamp(mappedX, -1f, 1f);
        mappedY = Math.Clamp(mappedY, -1f, 1f);

        var anchors = calibration.Lid;
        var normalized = Math.Clamp(
            (raw.Openness - anchors.Closed) / Math.Max(anchors.Neutral - anchors.Closed, 0.05f),
            0f, 1f);
        var wide = Math.Clamp(
            (raw.Openness - anchors.NeutralHigh) /
            Math.Max(anchors.Wide - anchors.NeutralHigh, 0.02f),
            0f, 1f);

        var fullClosure = anchors.Closed + (anchors.Neutral - anchors.Closed) * 0.22f;
        var recovery = anchors.NeutralLow + (anchors.NeutralHigh - anchors.NeutralLow) * 0.25f;
        var partial = raw.Openness < anchors.NeutralLow;
        var elapsedMs = temporal.LastTicks == 0
            ? 0f
            : Math.Max(0f, (ticks - temporal.LastTicks) / (float)TimeSpan.TicksPerMillisecond);
        temporal.LastTicks = ticks;

        if (partial)
        {
            temporal.ClosureStartedTicks = temporal.ClosureStartedTicks == 0
                ? ticks
                : temporal.ClosureStartedTicks;
            if (raw.Openness <= fullClosure)
                temporal.Blink = true;
        }
        else if (raw.Openness >= recovery)
        {
            temporal.ClosureStartedTicks = 0;
            temporal.Blink = false;
        }

        var dwellMs = Math.Clamp(anchors.TypicalBlinkMilliseconds * 0.75f, 300f, 700f);
        var heldMs = temporal.ClosureStartedTicks == 0
            ? 0f
            : (ticks - temporal.ClosureStartedTicks) / (float)TimeSpan.TicksPerMillisecond;
        var apertureSquint = Math.Clamp(
            (anchors.NeutralLow - raw.Openness) /
            Math.Max(anchors.NeutralLow - anchors.Squint, 0.025f),
            0f, 1f);

        // Any closure that reaches the personal closed region is a blink until recovery. A partial
        // closure must outlast the calibrated blink transient before it can become Squint.
        var targetSquint = !temporal.Blink && partial && heldMs >= dwellMs ? apertureSquint : 0f;
        if (targetSquint >= temporal.Squint)
        {
            temporal.Squint = targetSquint;
        }
        else
        {
            var decay = elapsedMs <= 0f ? 1f : MathF.Exp(-elapsedMs / 120f);
            temporal.Squint *= decay;
            if (temporal.Squint < 0.005f) temporal.Squint = 0f;
        }

        var jitter = temporal.UpdateJitter(mappedX, mappedY);
        return new EyeV2PerEyeDiagnostics(
            raw.X, raw.Y, mappedX, mappedY, raw.Openness, normalized,
            temporal.Squint, wide, temporal.Blink, jitter);
    }

    private sealed class EyeTemporalState
    {
        public long ClosureStartedTicks;
        public long LastTicks;
        public bool Blink;
        public float Squint;

        private bool _hasGaze;
        private float _meanX;
        private float _meanY;
        private float _variance;

        public float UpdateJitter(float x, float y)
        {
            if (!_hasGaze)
            {
                _hasGaze = true;
                _meanX = x;
                _meanY = y;
                return 0f;
            }

            const float alpha = 0.05f;
            var dx = x - _meanX;
            var dy = y - _meanY;
            _meanX += alpha * dx;
            _meanY += alpha * dy;
            _variance = (1f - alpha) * (_variance + alpha * (dx * dx + dy * dy));
            return MathF.Sqrt(Math.Max(0f, _variance));
        }

        public void Reset()
        {
            ClosureStartedTicks = 0;
            LastTicks = 0;
            Blink = false;
            Squint = 0f;
            _hasGaze = false;
            _meanX = 0f;
            _meanY = 0f;
            _variance = 0f;
        }
    }
}
