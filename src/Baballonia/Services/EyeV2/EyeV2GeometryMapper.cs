using OpenCvSharp;
using System;

namespace Baballonia.Services.EyeV2;

/// <summary>
/// V2-B geometry hybrid. V2-A remains the complete fallback/output path; confident direct lid and
/// pupil visibility measurements only add evidence for Wide and Squint. Missing, stale, or failed
/// geometry therefore produces byte-for-byte V2-A output for that tick.
/// </summary>
public sealed class EyeV2GeometryMapper : IEyeStateMapper, IEyeFrameAwareMapper, IDisposable
{
    private readonly EyeV2Mapper _v2A;
    private readonly IEyeGeometryExtractor _extractor;
    private readonly Action<EyeV2Diagnostics>? _diagnostics;
    private readonly Action<EyeGeometryFrame>? _geometryDiagnostics;
    private readonly GeometryBaseline _left = new();
    private readonly GeometryBaseline _right = new();
    private EyeGeometryFrame? _latest;
    private EyeV2Diagnostics? _baseDiagnostics;
    private long _lastPublishedDiagnostics;

    public EyeV2GeometryMapper(
        EyeV2Calibration calibration,
        IEyeGeometryExtractor extractor,
        Action<EyeV2Diagnostics>? diagnostics = null,
        Action<EyeGeometryFrame>? geometryDiagnostics = null)
    {
        _extractor = extractor ?? throw new ArgumentNullException(nameof(extractor));
        _diagnostics = diagnostics;
        _geometryDiagnostics = geometryDiagnostics;
        _v2A = new EyeV2Mapper(calibration, value => _baseDiagnostics = value);
    }

    public void ObserveFrame(Mat transformedEyeFrame, long timestampTicks)
    {
        try
        {
            var geometry = _extractor.Extract(transformedEyeFrame, timestampTicks);
            _latest = geometry;
            if (geometry.DebugImage != null) _geometryDiagnostics?.Invoke(geometry);
        }
        catch
        {
            // Extraction is experimental. A failed frame is missing evidence, not failed tracking.
            _latest = null;
        }
    }

    public float[] Map(float[] stockState, float[] filteredRawState, long timestampTicks)
    {
        var output = _v2A.Map(stockState, filteredRawState, timestampTicks);
        var geometry = _latest;
        var baseline = _baseDiagnostics;
        if (geometry == null || baseline == null ||
            timestampTicks - geometry.TimestampTicks > TimeSpan.TicksPerMillisecond * 200)
        {
            PublishDiagnostics(baseline, output);
            return output;
        }

        ApplyEye(output, EyeStateLayout.LeftWide, EyeStateLayout.LeftSquint,
            geometry.Left, baseline.Left, _left);
        ApplyEye(output, EyeStateLayout.RightWide, EyeStateLayout.RightSquint,
            geometry.Right, baseline.Right, _right);

        PublishDiagnostics(baseline, output);
        return output;
    }

    private void PublishDiagnostics(EyeV2Diagnostics? baseline, float[] output)
    {
        if (_diagnostics != null && baseline != null &&
            baseline.TimestampTicks != _lastPublishedDiagnostics)
        {
            _lastPublishedDiagnostics = baseline.TimestampTicks;
            _diagnostics(baseline with
            {
                Left = baseline.Left with
                {
                    Wide = output[EyeStateLayout.LeftWide],
                    Squint = output[EyeStateLayout.LeftSquint]
                },
                Right = baseline.Right with
                {
                    Wide = output[EyeStateLayout.RightWide],
                    Squint = output[EyeStateLayout.RightSquint]
                }
            });
        }
    }

    private static void ApplyEye(
        float[] output,
        int wideIndex,
        int squintIndex,
        EyeGeometry geometry,
        EyeV2PerEyeDiagnostics v2A,
        GeometryBaseline baseline)
    {
        if (!geometry.Valid || geometry.Confidence < 0.35f)
            return;

        // Learn only a neutral *geometry scale*, never expression labels. V2-A decides that this
        // tick is relaxed/open; its persisted personal anchors remain authoritative.
        if (!v2A.Blink && v2A.Squint < 0.05f && v2A.Wide < 0.05f &&
            v2A.NormalizedOpenness is >= 0.75f and <= 1.10f)
            baseline.ObserveNeutral(geometry.NormalizedAperture);

        if (!baseline.Ready) return;
        var ratio = geometry.NormalizedAperture / Math.Max(baseline.NeutralAperture, 0.01f);
        var reliable = geometry.Confidence * geometry.PupilVisibility;

        // A closed lid hides the pupil, while a deliberate squint usually retains a visible pupil.
        // That is the extra information V2-A's single lid scalar fundamentally does not contain.
        var geometrySquint = reliable < 0.25f
            ? 0f
            : Math.Clamp((0.88f - ratio) / 0.24f, 0f, 1f) * reliable;
        var geometryWide = Math.Clamp((ratio - 1.12f) / 0.24f, 0f, 1f) * geometry.Confidence;

        output[squintIndex] = Math.Max(output[squintIndex], geometrySquint);
        output[wideIndex] = Math.Max(output[wideIndex], geometryWide);
    }

    public void Reset()
    {
        _v2A.Reset();
        _extractor.Reset();
        _left.Reset();
        _right.Reset();
        _latest = null;
        _baseDiagnostics = null;
        _lastPublishedDiagnostics = 0;
    }

    public void Dispose() => _extractor.Dispose();

    private sealed class GeometryBaseline
    {
        private int _samples;
        public float NeutralAperture { get; private set; }
        public bool Ready => _samples >= 12;

        public void ObserveNeutral(float aperture)
        {
            if (!float.IsFinite(aperture) || aperture <= 0) return;
            NeutralAperture = _samples == 0
                ? aperture
                : NeutralAperture * 0.95f + aperture * 0.05f;
            _samples++;
        }

        public void Reset()
        {
            _samples = 0;
            NeutralAperture = 0;
        }
    }
}
