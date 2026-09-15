using System;
using System.Runtime.InteropServices;
using Baballonia.Desktop.Calibration;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SkiaSharp;
using Valve.VR;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;
using static Vortice.Direct3D11.D3D11;

namespace Baballonia.Tests.Services.Calibration;

/// <summary>Exercises actual D3D11 sharing through Windows WARP; no headset or physical GPU is used.</summary>
[TestClass]
public class D3D11OverlayTextureTest
{
    [TestInitialize]
    public void RequireWindows()
    {
        if (!OperatingSystem.IsWindows())
            Assert.Inconclusive("D3D11 software-rendering checks require Windows.");
    }

    [TestMethod]
    public void SharedHandlePublishesCompleteFramesAfterCpuPixelsAreReused()
    {
        const int width = 3, height = 2, rowBytes = 20;
        using var uploader = D3D11OverlayTexture.CreateWarp(width, height);
        var texture = uploader.Texture;
        Assert.AreNotEqual(IntPtr.Zero, texture.handle);
        Assert.AreEqual(ETextureType.DXGISharedHandle, texture.eType);
        Assert.AreEqual(EColorSpace.Gamma, texture.eColorSpace);
        using var reader = new SharedTextureReader(texture.handle, width, height);

        for (var frame = 0; frame < 40; frame++)
        {
            var expected = new byte[width * height * 4];
            using (var bitmap = new SKBitmap(
                       new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Premul), rowBytes))
            {
                var padded = new byte[rowBytes * height];
                Array.Fill(padded, (byte)0xee);
                for (var y = 0; y < height; y++)
                for (var x = 0; x < width; x++)
                {
                    var index = (y * width + x) * 4;
                    expected[index] = (byte)(20 + frame + x);
                    expected[index + 1] = (byte)(40 + frame + y);
                    expected[index + 2] = (byte)(60 + frame + x + y);
                    expected[index + 3] = (byte)(128 + frame);
                    Buffer.BlockCopy(expected, index, padded, y * rowBytes + x * 4, 4);
                }
                Marshal.Copy(padded, 0, bitmap.GetPixels(), padded.Length);
                uploader.Upload(bitmap);
                // Both mutation and disposal happen before the other device reads the texture.
                // This catches uploads that retain the caller's CPU buffer after returning.
                bitmap.Erase(SKColors.Transparent);
            }

            Assert.AreEqual(texture.handle, uploader.Texture.handle, "The compositor resource must stay stable.");
            CollectionAssert.AreEqual(expected, reader.Read(), $"Incorrect shared pixels in frame {frame}.");
        }
    }

    [TestMethod]
    public void ProductionSizeRasterArrivesCompleteOnSecondDevice()
    {
        const int width = 1024, height = 768;
        using var uploader = D3D11OverlayTexture.CreateWarp(width, height);
        using var bitmap = new SKBitmap(new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Premul));
        using (var canvas = new SKCanvas(bitmap))
        using (var paint = new SKPaint { Color = new SKColor(210, 70, 30, 128), IsAntialias = true })
        {
            canvas.Clear(SKColors.Transparent);
            canvas.DrawRoundRect(new SKRect(8, 8, width - 8, 352), 22, 22, paint);
            paint.Color = new SKColor(40, 220, 100, 230);
            canvas.DrawCircle(700, 500, 113, paint);
            paint.Color = new SKColor(50, 90, 240, 190);
            canvas.DrawRect(new SKRect(0, height - 37, width, height), paint);
        }
        var expected = new byte[width * height * 4];
        Marshal.Copy(bitmap.GetPixels(), expected, 0, expected.Length);
        uploader.Upload(bitmap);
        bitmap.Erase(SKColors.Transparent);
        using var reader = new SharedTextureReader(uploader.Texture.handle, width, height);
        CollectionAssert.AreEqual(expected, reader.Read(), "Every pixel of the production-size raster must arrive.");
    }

    [TestMethod]
    public void InvalidRasterDoesNotReplaceTheLastCompleteFrame()
    {
        using var uploader = D3D11OverlayTexture.CreateWarp(3, 2);
        using var valid = new SKBitmap(new SKImageInfo(3, 2, SKColorType.Rgba8888, SKAlphaType.Premul));
        valid.Erase(new SKColor(80, 100, 120, 128));
        uploader.Upload(valid);
        using var reader = new SharedTextureReader(uploader.Texture.handle, 3, 2);
        var before = reader.Read();

        using var wrongSize = new SKBitmap(new SKImageInfo(4, 2, SKColorType.Rgba8888, SKAlphaType.Premul));
        using var wrongChannels = new SKBitmap(new SKImageInfo(3, 2, SKColorType.Bgra8888, SKAlphaType.Premul));
        using var wrongAlpha = new SKBitmap(new SKImageInfo(3, 2, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        Assert.ThrowsExactly<ArgumentException>(() => uploader.Upload(wrongSize));
        Assert.ThrowsExactly<ArgumentException>(() => uploader.Upload(wrongChannels));
        Assert.ThrowsExactly<ArgumentException>(() => uploader.Upload(wrongAlpha));
        Assert.ThrowsExactly<ArgumentNullException>(() => uploader.Upload(null!));
        CollectionAssert.AreEqual(before, reader.Read());
    }

    [TestMethod]
    public void DisposedTextureRejectsUploadAndHandleAccess()
    {
        var uploader = D3D11OverlayTexture.CreateWarp(3, 2);
        using var bitmap = new SKBitmap(new SKImageInfo(3, 2, SKColorType.Rgba8888, SKAlphaType.Premul));
        uploader.Dispose();
        uploader.Dispose();
        Assert.ThrowsExactly<ObjectDisposedException>(() => uploader.Upload(bitmap));
        Assert.ThrowsExactly<ObjectDisposedException>(() => { _ = uploader.Texture; });
    }

    [TestMethod]
    public void MissingSteamVrAdapterCannotFallBackToAnotherDevice()
    {
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new D3D11OverlayTexture(-1, 3, 2));
        Assert.ThrowsExactly<InvalidOperationException>(() => new D3D11OverlayTexture(int.MaxValue, 3, 2));
    }

    private sealed class SharedTextureReader : IDisposable
    {
        private readonly ID3D11Device _device;
        private readonly ID3D11DeviceContext _context;
        private readonly ID3D11Texture2D _shared;
        private readonly ID3D11Texture2D _staging;
        private readonly int _width;
        private readonly int _height;

        public SharedTextureReader(IntPtr handle, int width, int height)
        {
            _width = width;
            _height = height;
            _device = D3D11CreateDevice(DriverType.Warp, DeviceCreationFlags.None, FeatureLevel.Level_11_0);
            _context = _device.ImmediateContext;
            _shared = _device.OpenSharedResource<ID3D11Texture2D>(handle);
            _staging = _device.CreateTexture2D(new Texture2DDescription(
                Format.R8G8B8A8_UNorm, (uint)width, (uint)height,
                mipLevels: 1, arraySize: 1, bindFlags: BindFlags.None,
                usage: ResourceUsage.Staging, cpuAccessFlags: CpuAccessFlags.Read));
        }

        public byte[] Read()
        {
            _context.CopyResource(_staging, _shared);
            _context.Map(_staging, 0, MapMode.Read, Vortice.Direct3D11.MapFlags.None, out var mapped).CheckError();
            try
            {
                var result = new byte[_width * _height * 4];
                for (var y = 0; y < _height; y++)
                    Marshal.Copy(IntPtr.Add(mapped.DataPointer, checked(y * (int)mapped.RowPitch)),
                        result, y * _width * 4, _width * 4);
                return result;
            }
            finally
            {
                _context.Unmap(_staging, 0);
            }
        }

        public void Dispose()
        {
            _staging.Dispose();
            _shared.Dispose();
            _context.Dispose();
            _device.Dispose();
        }
    }
}
