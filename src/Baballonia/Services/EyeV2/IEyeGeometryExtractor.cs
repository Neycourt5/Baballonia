using OpenCvSharp;
using System;

namespace Baballonia.Services.EyeV2;

/// <summary>
/// Replaceable visual eye-state seam. Implementations may use classic CV, a learned landmark
/// network, or another method, but must return the same normalized geometry and confidence.
/// </summary>
public interface IEyeGeometryExtractor : IDisposable
{
    string Name { get; }
    EyeGeometryFrame Extract(Mat transformedEyeFrame, long timestampTicks);
    void Reset();
}

/// <summary>Optional extension implemented by a mapper that consumes the real 128x128 eye crops.</summary>
public interface IEyeFrameAwareMapper
{
    void ObserveFrame(Mat transformedEyeFrame, long timestampTicks);
}

public readonly record struct EyeGeometryLine(float X1, float Y1, float X2, float Y2);

public sealed record EyeGeometry(
    bool Valid,
    float PupilX,
    float PupilY,
    float PupilRadius,
    EyeGeometryLine UpperLid,
    EyeGeometryLine LowerLid,
    float NormalizedAperture,
    float PupilVisibility,
    float Confidence)
{
    public static EyeGeometry Missing { get; } = new(
        false, 0.5f, 0.5f, 0f, new(), new(), 0f, 0f, 0f);
}

/// <summary>BGR pixels already annotated by the extractor; safe for the UI to retain.</summary>
public sealed record EyeGeometryDebugImage(int Width, int Height, byte[] BgrPixels);

public sealed record EyeGeometryFrame(
    long TimestampTicks,
    EyeGeometry Left,
    EyeGeometry Right,
    EyeGeometryDebugImage? DebugImage = null);
