using System;
using SkiaSharp;
using Valve.VR;

namespace Baballonia.Desktop.Calibration;

/// <summary>A persistent GPU texture owned by one calibration overlay session.</summary>
/// <remarks>
/// The presenter serializes access. Upload a complete frame before first submission, and clear or
/// destroy the OpenVR overlay before disposing this resource. Pixels use premultiplied RGBA alpha.
/// </remarks>
public interface ICalibrationOverlayTexture : IDisposable
{
    void Upload(SKBitmap bitmap);
    Texture_t Texture { get; }
}
