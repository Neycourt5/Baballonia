using System;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.EyeV2;

/// <summary>
/// Robust, deterministic V2-A fitting. Five gaze medians fit a two-dimensional affine map: two
/// gains, two cross-axis terms and two offsets. Five points do not justify a quadratic, while the
/// cross-axis terms correct the common case where the camera is slightly rotated after a reseat.
/// </summary>
public static class EyeV2CalibrationFitter
{
    private const int MinimumSamplesPerAnchor = 12;
    private const int MinimumSamplesPerGazePoint = 8;

    public static EyeV2Calibration Fit(EyeV2CalibrationCapture capture, string appVersion = "unknown")
    {
        Require(capture.Relax.Count >= MinimumSamplesPerAnchor, "Relax capture was too short.");
        Require(capture.Blinks.Count >= MinimumSamplesPerAnchor, "Blink capture was too short.");
        Require(capture.Squint.Count >= MinimumSamplesPerAnchor, "Squint capture was too short.");
        Require(capture.Wide.Count >= MinimumSamplesPerAnchor, "Wide capture was too short.");

        foreach (var target in EyeV2GazeTarget.FivePoint)
            Require(capture.Gaze.TryGetValue(target.Name, out var samples) &&
                    samples.Count >= MinimumSamplesPerGazePoint,
                $"Gaze point {target.Name} was too short.");

        var calibration = new EyeV2Calibration
        {
            Left = FitEye(capture, EyeSide.Left),
            Right = FitEye(capture, EyeSide.Right),
            Capture = new EyeV2CaptureMetadata
            {
                AppVersion = appVersion,
                RelaxSamples = capture.Relax.Count,
                BlinkSamples = capture.Blinks.Count,
                SquintSamples = capture.Squint.Count,
                WideSamples = capture.Wide.Count,
                GazeSamples = capture.Gaze.Values.Sum(x => x.Count),
            },
        };

        Require(calibration.IsValid(),
            "The captured anchors overlap or the gaze points do not span both axes. Please retry calibration.");
        return calibration;
    }

    public static EyeV2Calibration Recenter(
        EyeV2Calibration calibration,
        IReadOnlyList<EyeV2Sample> centerSamples)
    {
        Require(calibration.IsValid(), "The stored Eye V2 calibration is invalid.");
        Require(centerSamples.Count >= MinimumSamplesPerGazePoint, "Recenter capture was too short.");

        var left = MedianState(centerSamples, EyeSide.Left);
        var right = MedianState(centerSamples, EyeSide.Right);
        var updated = calibration with
        {
            Left = calibration.Left with { Gaze = calibration.Left.Gaze.Recenter(left.X, left.Y) },
            Right = calibration.Right with { Gaze = calibration.Right.Gaze.Recenter(right.X, right.Y) },
            Capture = calibration.Capture with { CreatedUtc = DateTime.UtcNow },
        };

        Require(updated.IsValid(), "Recenter produced an invalid gaze map.");
        return updated;
    }

    public static EyeV2ValidityResult AssessValidity(
        EyeV2Calibration calibration,
        IReadOnlyList<EyeV2Sample> relaxedSamples)
    {
        Require(calibration.IsValid(), "The stored Eye V2 calibration is invalid.");
        Require(relaxedSamples.Count >= MinimumSamplesPerGazePoint, "Validity capture was too short.");

        var left = MedianState(relaxedSamples, EyeSide.Left);
        var right = MedianState(relaxedSamples, EyeSide.Right);
        var (leftMappedX, leftMappedY) = calibration.Left.Gaze.Map(left.X, left.Y);
        var (rightMappedX, rightMappedY) = calibration.Right.Gaze.Map(right.X, right.Y);
        var leftGaze = MathF.Sqrt(leftMappedX * leftMappedX + leftMappedY * leftMappedY);
        var rightGaze = MathF.Sqrt(rightMappedX * rightMappedX + rightMappedY * rightMappedY);

        var leftLid = NormalizedLidShift(left.Openness, calibration.Left.Lid);
        var rightLid = NormalizedLidShift(right.Openness, calibration.Right.Lid);
        var maxLid = Math.Max(leftLid, rightLid);
        var maxGaze = Math.Max(leftGaze, rightGaze);

        if (maxLid > 0.35f)
            return new EyeV2ValidityResult(
                EyeV2ValidityKind.FullCalibrationRecommended,
                "Eye geometry changed substantially; run full Eye V2 calibration.",
                leftGaze, rightGaze, leftLid, rightLid);

        if (maxGaze > 0.18f)
            return new EyeV2ValidityResult(
                EyeV2ValidityKind.RecenterRecommended,
                "Gaze center shifted but lid anchors still look plausible; use Recenter Eyes.",
                leftGaze, rightGaze, leftLid, rightLid);

        return new EyeV2ValidityResult(
            EyeV2ValidityKind.Valid,
            "Eye V2 calibration matches the current headset position.",
            leftGaze, rightGaze, leftLid, rightLid);
    }

    private static EyeV2PerEyeCalibration FitEye(EyeV2CalibrationCapture capture, EyeSide side)
    {
        var relax = capture.Relax.Select(x => x.Eye(side)).ToArray();
        var blinks = capture.Blinks.Select(x => x.Eye(side)).ToArray();
        var squint = capture.Squint.Select(x => x.Eye(side)).ToArray();
        var wide = capture.Wide.Select(x => x.Eye(side)).ToArray();
        var gazeStates = EyeV2GazeTarget.FivePoint
            .Select(target => (target, state: MedianState(capture.Gaze[target.Name], side)))
            .ToArray();

        var neutral = Median(relax.Select(x => x.Openness));
        var spread = Math.Max(0.008f,
            1.4826f * Median(relax.Select(x => Math.Abs(x.Openness - neutral))));
        var gazeOpen = capture.Gaze.Values.SelectMany(x => x).Select(x => x.Eye(side).Openness).ToArray();

        var closed = Quantile(blinks.Select(x => x.Openness), 0.04);
        var neutralLow = Math.Min(neutral - 2.5f * spread, Quantile(gazeOpen, 0.03));
        var neutralHigh = Math.Max(neutral + 2.5f * spread, Quantile(gazeOpen, 0.97));
        var squintAnchor = Median(squint.Select(x => x.Openness));
        var wideAnchor = Median(wide.Select(x => x.Openness));
        var blinkMs = EstimateBlinkDurationMilliseconds(capture.Blinks, side, closed, neutral);

        return new EyeV2PerEyeCalibration
        {
            Gaze = FitAffine(gazeStates),
            Lid = new EyeLidAnchors
            {
                Closed = closed,
                Neutral = neutral,
                NeutralLow = neutralLow,
                NeutralHigh = neutralHigh,
                Squint = squintAnchor,
                Wide = wideAnchor,
                TypicalBlinkMilliseconds = blinkMs,
            },
        };
    }

    private static EyeGazeMap FitAffine(
        IReadOnlyList<(EyeV2GazeTarget target, EyeRawState state)> points)
    {
        // Normal equations for [rawX rawY 1] * beta = target. Inputs are robust medians, so OLS
        // here is both stable and deliberately smaller than a barely-constrained polynomial.
        var normal = new double[3, 3];
        var targetX = new double[3];
        var targetY = new double[3];

        foreach (var (target, state) in points)
        {
            var row = new[] { (double)state.X, state.Y, 1d };
            for (var i = 0; i < 3; i++)
            {
                targetX[i] += row[i] * target.X;
                targetY[i] += row[i] * target.Y;
                for (var j = 0; j < 3; j++)
                    normal[i, j] += row[i] * row[j];
            }
        }

        var bx = Solve3(normal, targetX);
        var by = Solve3(normal, targetY);
        return new EyeGazeMap
        {
            XX = (float)bx[0],
            XY = (float)bx[1],
            OffsetX = (float)bx[2],
            YX = (float)by[0],
            YY = (float)by[1],
            OffsetY = (float)by[2],
        };
    }

    private static double[] Solve3(double[,] source, double[] target)
    {
        var augmented = new double[3, 4];
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 3; c++) augmented[r, c] = source[r, c];
            augmented[r, 3] = target[r];
        }

        for (var pivot = 0; pivot < 3; pivot++)
        {
            var best = pivot;
            for (var r = pivot + 1; r < 3; r++)
                if (Math.Abs(augmented[r, pivot]) > Math.Abs(augmented[best, pivot])) best = r;
            Require(Math.Abs(augmented[best, pivot]) > 1e-8, "Gaze points were degenerate.");

            if (best != pivot)
                for (var c = pivot; c < 4; c++)
                    (augmented[pivot, c], augmented[best, c]) = (augmented[best, c], augmented[pivot, c]);

            var divisor = augmented[pivot, pivot];
            for (var c = pivot; c < 4; c++) augmented[pivot, c] /= divisor;
            for (var r = 0; r < 3; r++)
            {
                if (r == pivot) continue;
                var factor = augmented[r, pivot];
                for (var c = pivot; c < 4; c++) augmented[r, c] -= factor * augmented[pivot, c];
            }
        }

        return [augmented[0, 3], augmented[1, 3], augmented[2, 3]];
    }

    private static float EstimateBlinkDurationMilliseconds(
        IReadOnlyList<EyeV2Sample> samples,
        EyeSide side,
        float closed,
        float neutral)
    {
        var threshold = closed + (neutral - closed) * 0.55f;
        var durations = new List<float>();
        long? start = null;
        foreach (var sample in samples.OrderBy(x => x.TimestampTicks))
        {
            if (sample.Eye(side).Openness < threshold)
            {
                start ??= sample.TimestampTicks;
            }
            else if (start.HasValue)
            {
                durations.Add((sample.TimestampTicks - start.Value) / (float)TimeSpan.TicksPerMillisecond);
                start = null;
            }
        }
        if (start.HasValue && samples.Count > 0)
            durations.Add((samples[^1].TimestampTicks - start.Value) / (float)TimeSpan.TicksPerMillisecond);

        var plausible = durations.Where(x => x is >= 100f and <= 2500f).ToArray();
        return plausible.Length == 0 ? 450f : Median(plausible);
    }

    private static float NormalizedLidShift(float openness, EyeLidAnchors anchors) =>
        Math.Abs(openness - anchors.Neutral) / Math.Max(anchors.Neutral - anchors.Closed, 0.05f);

    private static EyeRawState MedianState(IReadOnlyList<EyeV2Sample> samples, EyeSide side)
    {
        var states = samples.Select(x => x.Eye(side)).ToArray();
        return new EyeRawState(
            Median(states.Select(x => x.X)),
            Median(states.Select(x => x.Y)),
            Median(states.Select(x => x.Openness)));
    }

    private static float Median(IEnumerable<float> values) => Quantile(values, 0.5);

    private static float Quantile(IEnumerable<float> values, double quantile)
    {
        var sorted = values.Where(float.IsFinite).OrderBy(x => x).ToArray();
        Require(sorted.Length > 0, "Capture contained no finite samples.");
        var position = Math.Clamp(quantile, 0d, 1d) * (sorted.Length - 1);
        var lower = (int)Math.Floor(position);
        var upper = (int)Math.Ceiling(position);
        if (lower == upper) return sorted[lower];
        var fraction = (float)(position - lower);
        return sorted[lower] * (1f - fraction) + sorted[upper] * fraction;
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
}
