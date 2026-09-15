using System;
using System.Diagnostics;
using System.Threading;
using SkiaSharp;
using Valve.VR;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;
using static Vortice.Direct3D11.D3D11;
using static Vortice.DXGI.DXGI;

namespace Baballonia.Desktop.Calibration;

/// <summary>Publishes complete raster frames through one persistent D3D11 shared texture.</summary>
/// <remarks>
/// SetOverlayRaw reloads an image each time. A DXGISharedHandle instead lets SteamVR retain the
/// same resource. OpenVR requires atomic CopyResource/resolve writes to this shared texture, so CPU
/// pixels first enter a private texture and only the finished image is copied into the shared one.
/// No window, swap chain, OpenGL context, or SteamVR initialization is needed here.
/// </remarks>
public sealed class D3D11OverlayTexture : ICalibrationOverlayTexture
{
    private static readonly TimeSpan UploadCompletionTimeout = TimeSpan.FromMilliseconds(500);
    private readonly ID3D11Query _completionQuery;
    private readonly int _width;
    private readonly int _height;
    private readonly ID3D11Device _device;
    private readonly ID3D11DeviceContext _context;
    private readonly ID3D11Texture2D _uploadTexture;
    private readonly ID3D11Texture2D _sharedTexture;
    private readonly Texture_t _texture;
    private bool _disposed;

    /// <param name="adapterIndex">The exact adapter returned by IVRSystem.GetDXGIOutputInfo.</param>
    public D3D11OverlayTexture(int adapterIndex, int width, int height)
        : this(width, height, () => CreateHardwareDevice(adapterIndex))
    {
    }

    private D3D11OverlayTexture(int width, int height, Func<ID3D11Device> createDevice)
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("D3D11 calibration textures require Windows.");
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(width, 16384);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(height, 16384);
        _width = width;
        _height = height;

        ID3D11Device? device = null;
        ID3D11DeviceContext? context = null;
        ID3D11Texture2D? uploadTexture = null;
        ID3D11Texture2D? sharedTexture = null;
        ID3D11Query? completionQuery = null;
        try
        {
            device = createDevice();
            context = device.ImmediateContext;
            completionQuery = device.CreateQuery(new QueryDescription(QueryType.Event));
            var description = new Texture2DDescription(
                Format.R8G8B8A8_UNorm, (uint)width, (uint)height,
                mipLevels: 1, arraySize: 1,
                bindFlags: BindFlags.ShaderResource | BindFlags.RenderTarget,
                usage: ResourceUsage.Default,
                cpuAccessFlags: CpuAccessFlags.None,
                sampleCount: 1, sampleQuality: 0);
            uploadTexture = device.CreateTexture2D(description);
            description.MiscFlags = ResourceOptionFlags.Shared;
            sharedTexture = device.CreateTexture2D(description);
            using var resource = sharedTexture.QueryInterface<IDXGIResource>();
            var handle = resource.SharedHandle;
            if (handle == IntPtr.Zero)
                throw new InvalidOperationException("D3D11 did not create a shareable calibration texture.");

            _texture = new Texture_t
            {
                handle = handle,
                eType = ETextureType.DXGISharedHandle,
                eColorSpace = EColorSpace.Gamma,
            };
            _completionQuery = completionQuery;
            _device = device;
            _context = context;
            _uploadTexture = uploadTexture;
            _sharedTexture = sharedTexture;
        }
        catch
        {
            completionQuery?.Dispose();
            sharedTexture?.Dispose();
            uploadTexture?.Dispose();
            context?.Dispose();
            device?.Dispose();
            throw;
        }
    }

    /// <summary>Uses Windows software rendering for tests without selecting a physical GPU.</summary>
    internal static D3D11OverlayTexture CreateWarp(int width, int height) =>
        new(width, height, () => D3D11CreateDevice(
            DriverType.Warp, DeviceCreationFlags.None, FeatureLevel.Level_11_0));

    private static ID3D11Device CreateHardwareDevice(int adapterIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(adapterIndex);
        using var factory = CreateDXGIFactory1<IDXGIFactory1>();
        var result = factory.EnumAdapters1((uint)adapterIndex, out var adapter);
        if (result.Failure || adapter is null)
        {
            adapter?.Dispose();
            throw new InvalidOperationException(
                "The graphics adapter selected by SteamVR is unavailable. Restart SteamVR and retry calibration.");
        }

        using (adapter)
        {
            // An explicit adapter requires Unknown, not Hardware. Never silently fall back to the
            // default adapter or WARP: SteamVR must be able to open this resource on its own GPU.
            result = D3D11CreateDevice(adapter, DriverType.Unknown, DeviceCreationFlags.None,
                [FeatureLevel.Level_11_0], out ID3D11Device? device);
            if (result.Failure)
            {
                device?.Dispose();
                result.CheckError();
            }
            return device ?? throw new InvalidOperationException("Could not create the SteamVR graphics device.");
        }
    }

    public Texture_t Texture
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _texture;
        }
    }

    public void Upload(SKBitmap bitmap)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(bitmap);
        if (bitmap.Width != _width || bitmap.Height != _height ||
            bitmap.ColorType != SKColorType.Rgba8888 || bitmap.AlphaType != SKAlphaType.Premul ||
            bitmap.RowBytes < _width * 4 || bitmap.GetPixels() == IntPtr.Zero)
        {
            throw new ArgumentException(
                "The overlay frame must match the texture size and contain premultiplied RGBA8888 pixels.",
                nameof(bitmap));
        }

        _device.DeviceRemovedReason.CheckError();
        // UpdateSubresource snapshots CPU bytes before returning, so the caller can immediately
        // redraw or dispose its bitmap. Respect RowBytes because a raster can have padded rows.
        _context.UpdateSubresource(_uploadTexture, 0, null, bitmap.GetPixels(), (uint)bitmap.RowBytes, 0);
        _context.CopyResource(_sharedTexture, _uploadTexture);
        _context.End(_completionQuery);
        // Required for updates consumed through another D3D11 device, such as SteamVR's compositor.
        _context.Flush();
        WaitForUploadCompletion();
        _device.DeviceRemovedReason.CheckError();
    }

    private void WaitForUploadCompletion()
    {
        // Flush submits work but does not wait for it. A second device can otherwise still see the
        // previous cue when Upload returns. The event query covers the complete shared copy.
        var started = Stopwatch.GetTimestamp();
        while (true)
        {
            var result = _context.GetData(_completionQuery, IntPtr.Zero, 0, AsyncGetDataFlags.DoNotFlush);
            result.CheckError();
            if (result == SharpGen.Runtime.Result.Ok) return;
            _device.DeviceRemovedReason.CheckError();
            if (Stopwatch.GetElapsedTime(started) >= UploadCompletionTimeout)
                throw new TimeoutException("The calibration texture did not finish uploading within 500 ms.");
            Thread.Sleep(1);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _completionQuery.Dispose();
        _sharedTexture.Dispose();
        _uploadTexture.Dispose();
        _context.Dispose();
        _device.Dispose();
        // IDXGIResource.GetSharedHandle returns a legacy DXGI handle, not an NT handle. Its
        // lifetime follows the resource; CloseHandle must not be called on it.
    }
}
