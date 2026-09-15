using Baballonia.Contracts;
using Baballonia.SDK;
using Baballonia.Services.Inference.Platforms;
using Baballonia.Services.Inference.VideoSources;
using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using System.IO.Ports;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.Services.Inference;

public class SingleCameraSourceFactory
{
    private readonly ILogger<SingleCameraSourceFactory> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private readonly IDeviceEnumerator _deviceEnumerator;
    private readonly IPlatformConnector _platformConnector;

    public SingleCameraSourceFactory(ILogger<SingleCameraSourceFactory> logger, ILoggerFactory loggerFactory, IDeviceEnumerator deviceEnumerator, IPlatformConnector platformConnector)
    {
        _logger = logger;
        _loggerFactory = loggerFactory;
        _deviceEnumerator = deviceEnumerator;
        _platformConnector = platformConnector;
    }

    public SingleCameraSource? Create(string address, string providerName) =>
        Create(address, providerName, quiet: false);

    public SingleCameraSource? Create(string address, string providerName, bool quiet)
    {
        ICaptureFactory? provider;
        if (!string.IsNullOrEmpty(providerName))
        {
            provider = _platformConnector.GetCaptureFactories()
                .FirstOrDefault(factory => factory.GetProviderName() == providerName && factory.CanConnect(address));
            if(provider == null)
                throw new ArgumentNullException($"No provider \"{provider}\" is not compatible with \"{address}\"");

        }
        else
        {
            provider = _platformConnector.GetCaptureFactories().First(factory => factory.CanConnect(address));
            if(provider == null)
                throw new ArgumentNullException($"No suitable provider for {address} found");
        }

        var capture = provider.Create(address);
        capture.QuietFailures = quiet;

        return new SingleCameraSource(_loggerFactory.CreateLogger<SingleCameraSource>(), capture, address);
    }

    /// <summary>How long to wait for a first frame when a person pressed Start.</summary>
    public static readonly TimeSpan DefaultFirstFrameTimeout = TimeSpan.FromSeconds(13);

    public Task<SingleCameraSource?> CreateStart(string address) =>
        CreateStart(address, "", DefaultFirstFrameTimeout, CancellationToken.None, quiet: false);

    public Task<SingleCameraSource?> CreateStart(string address, string providerName) =>
        CreateStart(address, providerName, DefaultFirstFrameTimeout, CancellationToken.None, quiet: false);

    public Task<SingleCameraSource?> CreateStart(
        string address,
        TimeSpan firstFrameTimeout,
        CancellationToken cancellationToken,
        bool quiet) =>
        CreateStart(address, "", firstFrameTimeout, cancellationToken, quiet);

    /// <summary>
    /// Creates a source and waits for its first frame. Recovery supplies a short timeout and marks
    /// failed attempts quiet; an explicit Start keeps the longer timeout and visible errors.
    /// </summary>
    public Task<SingleCameraSource?> CreateStart(
        string address,
        string providerName,
        TimeSpan firstFrameTimeout,
        CancellationToken cancellationToken,
        bool quiet = false)
    {
        var camera = address;
        RefreshCameras();
        string? mappedAddress = null;
        _deviceEnumerator.Cameras?.TryGetValue(camera, out mappedAddress);
        if (mappedAddress != null)
            camera = mappedAddress;

        // A scheme-less host like "openiristracker.local" or "192.168.0.42:81/mjpeg" matches no
        // backend as typed (each one wants a local device or a fully-schemed URL). When the resolved
        // address looks like a bare network endpoint, assume an http MJPEG stream so the IP/OpenCV
        // backends can claim it.
        camera = NormalizeNetworkAddress(camera);

        return Task.Run(
            () => StartWithFallback(
                address, camera, providerName, firstFrameTimeout, cancellationToken, quiet),
            cancellationToken);
    }

    /// <summary>Re-reads the device list so friendly-name to ordinal mappings cannot go stale.</summary>
    private void RefreshCameras() => _deviceEnumerator.Cameras = _deviceEnumerator.UpdateCameras();

    /// <summary>Whether the saved device address is present in the latest enumeration.</summary>
    public bool IsDevicePresent(string address)
    {
        if (string.IsNullOrWhiteSpace(address))
            return false;

        if (LooksLikeASerialPort(address))
            return SerialPortIsPresent(address);

        if (!LooksLikeAFriendlyName(address))
            return true;

        RefreshCameras();
        return _deviceEnumerator.Cameras is { Count: > 0 } cameras && cameras.ContainsKey(address);
    }

    private static bool SerialPortIsPresent(string address)
    {
        try
        {
            return SerialPort.GetPortNames()
                .Any(port => string.Equals(port, address, StringComparison.OrdinalIgnoreCase));
        }
        catch (Exception)
        {
            // Enumeration failure is not evidence that the device itself is absent.
            return true;
        }
    }

    private static bool LooksLikeASerialPort(string address) =>
        address.StartsWith("COM", StringComparison.OrdinalIgnoreCase) ||
        address.StartsWith("/dev/tty", StringComparison.Ordinal) ||
        address.StartsWith("/dev/cu", StringComparison.Ordinal);

    private static bool LooksLikeAFriendlyName(string address) =>
        !address.Contains("://", StringComparison.Ordinal) &&
        !LooksLikeASerialPort(address) &&
        !address.StartsWith("/dev/", StringComparison.Ordinal);

    // Turns a bare network endpoint into an http URL the capture backends can open. Leaves untouched:
    // already-schemed URLs (http/https/rtsp/rtmp/... — handled by OpenCV), unix device paths,
    // friendly names (which contain whitespace) and GStreamer pipeline strings, and camera indices.
    // Anything else that looks like a host — has a '.' (domain/IP) or ':' (port) — is assumed to be an
    // http MJPEG stream, by far the most common scheme-less case (e.g. OpenIris/wireless trackers).
    private static string NormalizeNetworkAddress(string address)
    {
        if (string.IsNullOrWhiteSpace(address)) return address;
        if (address.Contains("://")) return address;            // already has a scheme
        if (address.StartsWith('/')) return address;            // unix device path
        if (address.Any(char.IsWhiteSpace)) return address;     // friendly name / pipeline string
        if (int.TryParse(address, out _)) return address;       // camera index
        if (address.Contains('.') || address.Contains(':'))
            return "http://" + address;
        return address;
    }

    // Tries each compatible backend in preference order until one delivers a frame. A specific
    // providerName is tried first; the rest stay as fallbacks. This lets a device drop from e.g. the
    // OpenCV/GStreamer "Normal Camera" backend (which needs an unbundled v4l2src plugin) to the
    // dependency-free "V4L2 Camera" one instead of failing outright.
    private SingleCameraSource? StartWithFallback(
        string address,
        string camera,
        string providerName,
        TimeSpan firstFrameTimeout,
        CancellationToken cancellationToken,
        bool quiet)
    {
        // The Vive Facial Tracker backend fires a native USB tracker-enable (enableViveFacialTracker)
        // *before* it ever opens the camera. On a name that isn't a real, present tracker — e.g. a
        // stale saved camera that has since been unplugged — that native call wedges a thread Windows
        // can't reap, so the process lingers as a ghost after the window closes. Only offer VFT for a
        // device the current enumeration actually knows about. Other backends open lazily and fail
        // gracefully, so they don't need this guard. (Linux's VFT only matches /dev/video* paths, which
        // don't exist when unplugged, so present setups there are unaffected.)
        var deviceIsPresent = IsEnumeratedDevice(address) || IsEnumeratedDevice(camera);
        var candidates = _platformConnector.GetCaptureFactories()
            .Where(factory => factory.CanConnect(camera))
            .Where(factory => deviceIsPresent || !IsViveFacialTracker(factory))
            .ToList();

        if (!string.IsNullOrEmpty(providerName))
            candidates = candidates.OrderBy(factory => factory.GetProviderName() == providerName ? 0 : 1).ToList();
        else if (_deviceEnumerator.IsViveFacialTracker(camera) || _deviceEnumerator.IsViveFacialTracker(address))
        {
            // Positively-identified Vive Facial Tracker (USB VID 0x0BB4/PID 0x0321 or "HTC Boot"):
            // the generic "Normal Camera"/"V4L2 Camera" backends can also open it but skip the USB
            // activation and YUYV decode, producing a recognizable-but-broken image. Force the VFT
            // backend first unless the user explicitly picked a different one.
            candidates = candidates.OrderBy(factory => IsViveFacialTracker(factory) ? 0 : 1).ToList();
            _logger.LogInformation("{} is a Vive Facial Tracker; preferring the VFT capture backend", address);
        }

        if (candidates.Count == 0)
        {
            Report(quiet, "No capture backend can open {Address}", address);
            return null;
        }

        foreach (var factory in candidates)
        {
            var providerLabel = factory.GetProviderName();

            // Create() can throw before any frame is attempted — e.g. a backend whose native deps are
            // missing runs a failing static ctor (the PSVR2 module throws TypeInitializationException
            // from Create()). Treat any such failure as "this backend can't open the device" and fall
            // through to the next candidate instead of letting it abort the whole fallback chain.
            SingleCameraSource source;
            try
            {
                var capture = factory.Create(camera);
                capture.QuietFailures = quiet;
                source = new SingleCameraSource(_loggerFactory.CreateLogger<SingleCameraSource>(), capture, camera);
            }
            catch (Exception e)
            {
                Report(quiet,
                    "{Provider} threw while creating a capture for {Address}: {Reason}; trying the next backend",
                    providerLabel, address, e.Message);
                continue;
            }

            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                var openedAt = Stopwatch.StartNew();

                if (!source.Start())
                {
                    Report(quiet,
                        "{Provider} could not open {Address} after {Ms:F0} ms; trying the next backend",
                        providerLabel, address, openedAt.Elapsed.TotalMilliseconds);
                    source.Dispose();
                    continue;
                }

                var waitHandles = source.GetFrameWaitHandles();
                var sw = Stopwatch.StartNew();
                while (sw.Elapsed < firstFrameTimeout)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Block until a frame is signalled rather than busy-polling.
                    WaitHandle.WaitAny(waitHandles, TimeSpan.FromMilliseconds(250));
                    using var frame = source.GetFrame();
                    if (frame != null)
                    {
                        _logger.LogInformation("Opened {} with {}", address, providerLabel);
                        return source;
                    }
                }

                Report(quiet,
                    "No data from {Address} via {Provider} after {Ms:F0} ms; trying the next backend",
                    address, providerLabel, openedAt.Elapsed.TotalMilliseconds);
                source.Dispose();
            }
            catch (OperationCanceledException)
            {
                source.Dispose();
                throw;
            }
            catch (Exception e)
            {
                Report(quiet,
                    "{Provider} failed while starting {Address}: {Reason}; trying the next backend",
                    providerLabel, address, e.Message);
                source.Dispose();
            }
        }

        Report(quiet,
            "No data was received from {Address} on any backend, closing... Maybe the camera is opened somewhere else?",
            address);
        return null;
    }

    private void Report(bool quiet, string message, params object?[] args)
    {
        if (quiet)
            _logger.LogDebug(message, args);
        else
            _logger.LogWarning(message, args);
    }

    // A device is "present" when the current enumeration knows it — either as a friendly-name key or
    // as a resolved value (a camera index like "0", a "/dev/videoN" path, or a COM port). Used to keep
    // the wedge-prone VFT backend off stale/absent device names.
    private bool IsEnumeratedDevice(string address)
    {
        var cameras = _deviceEnumerator.Cameras;
        return cameras != null && (cameras.ContainsKey(address) || cameras.Values.Contains(address));
    }

    // Matches VFTCaptureFactory.GetProviderName().
    private static bool IsViveFacialTracker(ICaptureFactory factory) =>
        string.Equals(factory.GetProviderName(), "Vive Facial Tracker", StringComparison.Ordinal);
}
