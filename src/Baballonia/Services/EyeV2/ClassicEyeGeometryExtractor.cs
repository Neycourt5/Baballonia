using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Baballonia.Services.EyeV2;

/// <summary>
/// License-safe first V2-B extractor: dark-pupil ellipse plus vertical-gradient eyelid lines.
/// The interface deliberately does not expose OpenCV details, so a learned IR landmark extractor
/// can replace this class without changing calibration, mapping, persistence, or UI code.
/// </summary>
public sealed class ClassicEyeGeometryExtractor : IEyeGeometryExtractor
{
    private long _lastDebugTicks;
    public string Name => "Classic pupil/eyelid geometry";

    public EyeGeometryFrame Extract(Mat transformedEyeFrame, long timestampTicks)
    {
        if (transformedEyeFrame.Empty() || transformedEyeFrame.Channels() != 2)
            return new EyeGeometryFrame(timestampTicks, EyeGeometry.Missing, EyeGeometry.Missing);

        var channels = transformedEyeFrame.Split();
        try
        {
            var left = ExtractEye(channels[0]);
            var right = ExtractEye(channels[1]);
            EyeGeometryDebugImage? debug = null;
            if (timestampTicks - _lastDebugTicks >= TimeSpan.TicksPerMillisecond * 100)
            {
                _lastDebugTicks = timestampTicks;
                debug = BuildDebug(channels[0], channels[1], left, right);
            }

            return new EyeGeometryFrame(timestampTicks, left, right, debug);
        }
        finally
        {
            foreach (var channel in channels) channel.Dispose();
        }
    }

    private static EyeGeometry ExtractEye(Mat gray)
    {
        using var blurred = new Mat();
        using var dark = new Mat();
        Cv2.GaussianBlur(gray, blurred, new Size(7, 7), 0);
        Cv2.Threshold(blurred, dark, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
        // A wider opening removes thin lid/eyelash strokes before contour fitting while retaining
        // the substantially larger pupil/iris blob. Otherwise a touching lid line can turn the
        // pupil into one very wide contour and defeat ellipse scoring.
        Cv2.MorphologyEx(dark, dark, MorphTypes.Open,
            Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(7, 7)));

        Cv2.FindContours(dark, out Point[][] contours, out _, RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        RotatedRect? pupil = null;
        var bestScore = 0d;
        var maxArea = gray.Width * gray.Height * 0.28;
        foreach (var contour in contours)
        {
            var area = Cv2.ContourArea(contour);
            if (area < 25 || area > maxArea || contour.Length < 5) continue;
            var perimeter = Cv2.ArcLength(contour, true);
            if (perimeter <= 0) continue;
            var ellipse = Cv2.FitEllipse(contour);
            var aspect = Math.Min(ellipse.Size.Width, ellipse.Size.Height) /
                         Math.Max(ellipse.Size.Width, ellipse.Size.Height);
            var circularity = Math.Clamp(4 * Math.PI * area / (perimeter * perimeter), 0, 1);
            var nx = ellipse.Center.X / gray.Width;
            var ny = ellipse.Center.Y / gray.Height;
            if (nx is < 0.08f or > 0.92f || ny is < 0.08f or > 0.92f) continue;
            var centerPrior = 1d - Math.Min(1d, Math.Abs(nx - 0.5) + Math.Abs(ny - 0.5));
            var score = area * (0.35 + 0.65 * aspect) * (0.4 + 0.6 * circularity) *
                        (0.65 + 0.35 * centerPrior);
            if (score <= bestScore) continue;
            bestScore = score;
            pupil = ellipse;
        }

        if (pupil == null) return EyeGeometry.Missing;
        var p = pupil.Value;
        var radius = (p.Size.Width + p.Size.Height) * 0.25f;

        using var gradient = new Mat();
        Cv2.Sobel(blurred, gradient, MatType.CV_32F, 0, 1, 3);
        var upperPoints = new List<Point2f>();
        var lowerPoints = new List<Point2f>();
        var xRadius = Math.Clamp((int)(radius * 1.8f), 8, gray.Width / 3);
        var minX = Math.Max(1, (int)p.Center.X - xRadius);
        var maxX = Math.Min(gray.Width - 2, (int)p.Center.X + xRadius);
        var upperTop = Math.Max(1, (int)(p.Center.Y - radius * 3f));
        var upperBottom = Math.Max(upperTop + 1, (int)(p.Center.Y - radius * 0.55f));
        var lowerTop = Math.Min(gray.Height - 2, (int)(p.Center.Y + radius * 0.55f));
        var lowerBottom = Math.Min(gray.Height - 1, (int)(p.Center.Y + radius * 3f));

        for (var x = minX; x <= maxX; x += 3)
        {
            var upper = PeakY(gradient, x, upperTop, upperBottom);
            var lower = PeakY(gradient, x, lowerTop, lowerBottom);
            if (upper >= 0) upperPoints.Add(new Point2f(x, upper));
            if (lower >= 0) lowerPoints.Add(new Point2f(x, lower));
        }

        if (upperPoints.Count < 3 || lowerPoints.Count < 3)
            return EyeGeometry.Missing;

        var upperY = Median(upperPoints.Select(point => point.Y));
        var lowerY = Median(lowerPoints.Select(point => point.Y));
        if (lowerY <= upperY + 2) return EyeGeometry.Missing;

        var aperture = (lowerY - upperY) / gray.Height;
        var minDiameter = Math.Min(p.Size.Width, p.Size.Height);
        var maxDiameter = Math.Max(p.Size.Width, p.Size.Height);
        var ellipseQuality = Math.Clamp(minDiameter / Math.Max(maxDiameter, 1f), 0f, 1f);
        var visibility = Math.Clamp((lowerY - upperY) / Math.Max(minDiameter, 1f), 0f, 1f);
        var lidCoverage = Math.Min(upperPoints.Count, lowerPoints.Count) /
                          (float)Math.Max(1, (maxX - minX) / 3);
        var confidence = Math.Clamp(MathF.Sqrt(ellipseQuality * Math.Clamp(lidCoverage, 0, 1)), 0, 1);

        return new EyeGeometry(
            true,
            p.Center.X / gray.Width,
            p.Center.Y / gray.Height,
            radius / Math.Min(gray.Width, gray.Height),
            new EyeGeometryLine(minX / (float)gray.Width, upperY / gray.Height,
                maxX / (float)gray.Width, upperY / gray.Height),
            new EyeGeometryLine(minX / (float)gray.Width, lowerY / gray.Height,
                maxX / (float)gray.Width, lowerY / gray.Height),
            aperture,
            visibility,
            confidence);
    }

    private static int PeakY(Mat gradient, int x, int start, int end)
    {
        if (end <= start) return -1;
        var bestY = -1;
        var best = 0f;
        for (var y = start; y <= end; y++)
        {
            var value = Math.Abs(gradient.At<float>(y, x));
            if (value <= best) continue;
            best = value;
            bestY = y;
        }
        return bestY;
    }

    private static float Median(IEnumerable<float> values)
    {
        var sorted = values.OrderBy(value => value).ToArray();
        return sorted.Length == 0 ? 0 : sorted[sorted.Length / 2];
    }

    private static EyeGeometryDebugImage BuildDebug(
        Mat leftGray, Mat rightGray, EyeGeometry left, EyeGeometry right)
    {
        using var combined = new Mat();
        using var bgr = new Mat();
        Cv2.HConcat([leftGray, rightGray], combined);
        Cv2.CvtColor(combined, bgr, ColorConversionCodes.GRAY2BGR);
        DrawEye(bgr, left, 0, leftGray.Width, leftGray.Height);
        DrawEye(bgr, right, leftGray.Width, rightGray.Width, rightGray.Height);
        var continuous = bgr.IsContinuous() ? bgr : bgr.Clone();
        try
        {
            var bytes = new byte[continuous.Rows * continuous.Cols * continuous.ElemSize()];
            Marshal.Copy(continuous.Data, bytes, 0, bytes.Length);
            return new EyeGeometryDebugImage(continuous.Width, continuous.Height, bytes);
        }
        finally
        {
            if (!ReferenceEquals(continuous, bgr)) continuous.Dispose();
        }
    }

    private static void DrawEye(Mat image, EyeGeometry eye, int offsetX, int width, int height)
    {
        if (!eye.Valid) return;
        var center = new Point(offsetX + eye.PupilX * width, eye.PupilY * height);
        Cv2.Circle(image, center, Math.Max(2, (int)(eye.PupilRadius * Math.Min(width, height))),
            new Scalar(0, 255, 255), 1);
        Cv2.DrawMarker(image, center, new Scalar(0, 255, 255), MarkerTypes.Cross, 8, 1);
        DrawLine(image, eye.UpperLid, offsetX, width, height, new Scalar(255, 160, 0));
        DrawLine(image, eye.LowerLid, offsetX, width, height, new Scalar(0, 200, 255));
    }

    private static void DrawLine(Mat image, EyeGeometryLine line, int offsetX, int width, int height, Scalar color) =>
        Cv2.Line(image,
            new Point(offsetX + line.X1 * width, line.Y1 * height),
            new Point(offsetX + line.X2 * width, line.Y2 * height), color, 1);

    public void Reset() => _lastDebugTicks = 0;
    public void Dispose() { }
}
