using Baballonia.Services;
using Baballonia.Services.Calibration;
using Microsoft.Extensions.Logging;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using Valve.VR;

namespace Baballonia.Desktop.Calibration;

/// <summary>
/// A compact, process-local OpenVR overlay for behavioral calibration instructions. It is attached
/// to the HMD rather than mirroring the Avalonia window, so targets and text remain visible while
/// VRChat is the scene application. Calibration logic remains in core services.
/// </summary>
public sealed class OpenVrCalibrationPresenter : IVrCalibrationPresenter
{
    internal const int TextureWidth = 1024;
    internal const int TextureHeight = 768;
    private const string OverlayKey = "projectbabble.calibration.presenter.v1";
    private const string OverlayName = "Baballonia calibration";

    /// <summary>
    /// Consecutive upload failures tolerated before the surface is torn down. SteamVR can refuse a
    /// single frame across a compositor hiccup or scene-application switch; destroying the overlay
    /// on the first of those makes the panel vanish mid-calibration for a fault that would have
    /// cleared on its own. A genuine loss fails every attempt, so it still surfaces immediately.
    /// </summary>
    private const int MaxConsecutiveFailures = 3;

    private readonly OpenVRService _openVr;
    private readonly ILogger<OpenVrCalibrationPresenter> _logger;
    private readonly object _sync = new();
    private CVROverlay? _overlay;
    private ulong _handle;
    private string _status = "SteamVR presenter has not been started.";
    private bool _healthy;
    private VrCalibrationAction _pendingAction;
    private VrCalibrationFrame? _lastFrame;
    private VrCalibrationFrame? _lastRendered;
    private Timer? _completionTimer;
    private bool _showingCompletion;
    private int _consecutiveFailures;

    // The drawing surface outlives individual frames. Besides saving an allocation and a clear per
    // update, this keeps the pixel buffer handed to SetOverlayRaw alive for as long as the overlay
    // exists, rather than freeing it the instant the call returns.
    private SKBitmap? _surface;
    private SKCanvas? _canvas;

    // Font lookup is the expensive part of drawing text, and the old code paid it per string - eight
    // or more times per frame, twenty times a second.
    private static readonly Lazy<SKTypeface> RegularTypeface =
        new(() => SKTypeface.FromFamilyName(null, SKFontStyle.Normal) ?? SKTypeface.Default);
    private static readonly Lazy<SKTypeface> BoldTypeface =
        new(() => SKTypeface.FromFamilyName(null, SKFontStyle.Bold) ?? SKTypeface.Default);

    public OpenVrCalibrationPresenter(
        OpenVRService openVr,
        ILogger<OpenVrCalibrationPresenter> logger)
    {
        _openVr = openVr;
        _logger = logger;
    }

    public bool IsAvailable
    {
        get
        {
            if (!OperatingSystem.IsWindows()) return false;
            try
            {
                return OpenVR.IsRuntimeInstalled() && OpenVR.IsHmdPresent();
            }
            catch
            {
                return false;
            }
        }
    }

    public bool IsPresenting
    {
        get { lock (_sync) return _handle != 0; }
    }

    public bool IsHealthy
    {
        get { lock (_sync) return _handle != 0 && _healthy; }
    }

    public string Status
    {
        get { lock (_sync) return _status; }
    }

    public VrPresenterStartResult Begin(string sessionTitle)
    {
        lock (_sync)
        {
            // Eye and guided calibration share this singleton presenter. Replacing an active
            // overlay here would leave the first calibration sampling against a display owned by
            // the second one, so contention must fail closed rather than "helpfully" stealing it.
            //
            // A finished session lingering on its completion message is not contention: nothing is
            // sampling against it. Starting the next calibration within those two seconds used to be
            // refused as if a routine were still running, so take the surface over instead.
            if (_handle != 0)
            {
                if (!_showingCompletion)
                {
                    return VrPresenterStartResult.Failure(
                        "Another in-headset calibration is already active. Finish or cancel it first.");
                }

                CloseOverlayLocked(preserveStatus: true);
            }

            _completionTimer?.Dispose();
            _completionTimer = null;
            _showingCompletion = false;
            _healthy = false;
            _consecutiveFailures = 0;

            if (!OperatingSystem.IsWindows())
                return VrPresenterStartResult.Failure(
                    _status = "The built-in true VR presenter currently requires Windows SteamVR/OpenVR.");

            _overlay = _openVr.TryGetOverlay(out var runtimeStatus);
            if (_overlay == null)
                return VrPresenterStartResult.Failure(_status = runtimeStatus);

            try
            {
                // Recover only our own stable key if a prior process exited before teardown.
                ulong staleHandle = 0;
                if (_overlay.FindOverlay(OverlayKey, ref staleHandle) == EVROverlayError.None && staleHandle != 0)
                    _overlay.DestroyOverlay(staleHandle);

                var error = _overlay.CreateOverlay(OverlayKey, OverlayName, ref _handle);
                if (error != EVROverlayError.None || _handle == 0)
                    return FailAndCloseLocked($"SteamVR could not create the calibration overlay: {error}.");

                ThrowIfError(_overlay.SetOverlayWidthInMeters(_handle, 1.35f), "set overlay size");
                ThrowIfError(_overlay.SetOverlayAlpha(_handle, 0.98f), "set overlay opacity");
                ThrowIfError(_overlay.SetOverlayInputMethod(_handle, VROverlayInputMethod.Mouse),
                    "enable controller pointer input");

                var mouseScale = new HmdVector2_t { v0 = TextureWidth, v1 = TextureHeight };
                ThrowIfError(_overlay.SetOverlayMouseScale(_handle, ref mouseScale), "set pointer scale");

                // OpenVR HMD coordinates use -Z in front of the viewer. Keeping the panel just
                // below center lets the user still see their avatar mirror around it.
                var transform = new HmdMatrix34_t
                {
                    m0 = 1, m1 = 0, m2 = 0, m3 = 0,
                    m4 = 0, m5 = 1, m6 = 0, m7 = -0.05f,
                    m8 = 0, m9 = 0, m10 = 1, m11 = -1.25f,
                };
                ThrowIfError(_overlay.SetOverlayTransformTrackedDeviceRelative(
                    _handle, OpenVR.k_unTrackedDeviceIndex_Hmd, ref transform), "attach overlay to headset");

                _pendingAction = VrCalibrationAction.None;
                CreateSurfaceLocked();
                PresentLocked(new VrCalibrationFrame(
                    sessionTitle, "Get ready. Calibration will begin in the headset.",
                    VrCalibrationPhase.Preparing, 0, AllowCancel: true));
                ThrowIfError(_overlay.ShowOverlay(_handle), "show overlay");

                _healthy = true;
                _status = "True SteamVR headset presenter active.";
                return VrPresenterStartResult.Success(_status);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Could not start the OpenVR calibration presenter");
                return FailAndCloseLocked($"SteamVR presenter failed: {ex.Message}");
            }
        }
    }

    public void Present(VrCalibrationFrame frame)
    {
        lock (_sync)
        {
            if (_handle == 0 || _overlay == null) return;
            try
            {
                PresentLocked(frame.Clamp());
                _consecutiveFailures = 0;
            }
            catch (Exception ex)
            {
                _consecutiveFailures++;
                _status = $"SteamVR presenter update failed: {ex.Message}";

                if (_consecutiveFailures < MaxConsecutiveFailures && IsCosmeticUpdate(frame))
                {
                    // The instruction on screen is still the right one - only a progress bar or a
                    // countdown digit failed to land - so leaving the previous frame up is correct
                    // and the next tick can retry. A failed update that carried NEW instructions is
                    // never tolerated: the headset would say one thing while the recorder labelled
                    // another, which is exactly how a session gets confidently mislabelled.
                    _logger.LogDebug(ex,
                        "Cosmetic overlay update {Failure} of {Limit} failed; retrying next tick",
                        _consecutiveFailures, MaxConsecutiveFailures);
                    return;
                }

                _logger.LogError(ex, "Could not update the OpenVR calibration presenter");
                // A stale instruction is worse than no instruction: close the surface and expose
                // unhealthy state so calibration services abort before publishing more samples.
                _healthy = false;
                CloseOverlayLocked(preserveStatus: true);
            }
        }
    }

    public VrCalibrationAction ConsumeAction()
    {
        lock (_sync)
        {
            if (_handle == 0 || _overlay == null) return VrCalibrationAction.None;

            try
            {
                var vrEvent = new VREvent_t();
                var size = (uint)Marshal.SizeOf<VREvent_t>();
                while (_overlay.PollNextOverlayEvent(_handle, ref vrEvent, size))
                {
                    if ((EVREventType)vrEvent.eventType != EVREventType.VREvent_MouseButtonDown)
                        continue;

                    var frame = _lastFrame;
                    if (frame != null)
                    {
                        var action = HitTestControllerButton(
                            frame, vrEvent.data.mouse.x, vrEvent.data.mouse.y);
                        if (action != VrCalibrationAction.None)
                            _pendingAction = action;
                    }
                }
            }
            catch (Exception ex)
            {
                _status = $"SteamVR controller input failed: {ex.Message}";
                _logger.LogWarning(ex, "Could not read calibration overlay input");
                _healthy = false;
                CloseOverlayLocked(preserveStatus: true);
            }

            var result = _pendingAction;
            _pendingAction = VrCalibrationAction.None;
            return result;
        }
    }

    /// <summary>
    /// Resolves a controller-pointer click against the rendered button strip. OpenVR bindings have
    /// historically exposed overlay mouse Y using either the texture's top-left convention or its
    /// reflected bottom-left convention, depending on compositor/binding version; accepting the
    /// exact band in both coordinate systems keeps the hit target narrow without making HOME users
    /// guess which invisible vertical strip is clickable.
    /// </summary>
    public static VrCalibrationAction HitTestControllerButton(
        VrCalibrationFrame frame,
        float x,
        float y)
    {
        const float buttonTop = 680;
        const float buttonBottom = 738;
        var reflectedY = TextureHeight - y;
        var inButtonBand = y is >= buttonTop and <= buttonBottom ||
                           reflectedY is >= buttonTop and <= buttonBottom;
        if (!inButtonBand)
            return VrCalibrationAction.None;

        if (frame.AllowRetry && x is >= 64 and <= 320)
            return VrCalibrationAction.Retry;
        if (frame.AllowSkip && x is >= 384 and <= 640)
            return VrCalibrationAction.Skip;
        if (frame.AllowCancel && x is >= 704 and <= 960)
            return VrCalibrationAction.Cancel;
        return VrCalibrationAction.None;
    }

    public void End(string? completionMessage = null)
    {
        lock (_sync)
        {
            if (_handle == 0) return;

            if (!string.IsNullOrWhiteSpace(completionMessage))
            {
                try
                {
                    PresentLocked(new VrCalibrationFrame(
                        "CALIBRATION COMPLETE", completionMessage, VrCalibrationPhase.Complete,
                        1, 1, AllowCancel: false));
                    _status = completionMessage;
                    _showingCompletion = true;
                    _completionTimer?.Dispose();
                    _completionTimer = new Timer(_ =>
                    {
                        lock (_sync)
                        {
                            // Only tear down if this is still the completion frame; a calibration
                            // started in the meantime now owns the surface.
                            if (_showingCompletion) CloseOverlayLocked();
                        }
                    }, null, TimeSpan.FromSeconds(2), Timeout.InfiniteTimeSpan);
                }
                catch (Exception ex)
                {
                    _status = $"SteamVR presenter update failed: {ex.Message}";
                    _logger.LogDebug(ex, "Could not show the calibration completion frame");
                    _healthy = false;
                    CloseOverlayLocked(preserveStatus: true);
                }
                return;
            }

            CloseOverlayLocked();
        }
    }

    /// <summary>
    /// True when <paramref name="frame"/> says the same thing to the user as the frame currently on
    /// screen, differing only in progress, countdown or intensity readouts. Determines whether a
    /// failed upload can be retried or has to fail closed.
    /// </summary>
    public static bool IsCosmeticUpdate(VrCalibrationFrame frame, VrCalibrationFrame? onScreen) =>
        onScreen is not null &&
        onScreen.Title == frame.Title &&
        onScreen.Instruction == frame.Instruction &&
        onScreen.Phase == frame.Phase &&
        onScreen.Repetition == frame.Repetition &&
        onScreen.AllowRetry == frame.AllowRetry &&
        onScreen.AllowSkip == frame.AllowSkip &&
        onScreen.AllowCancel == frame.AllowCancel &&
        onScreen.TargetX.Equals(frame.TargetX) &&
        onScreen.TargetY.Equals(frame.TargetY);

    private bool IsCosmeticUpdate(VrCalibrationFrame frame) => IsCosmeticUpdate(frame, _lastRendered);

    /// <summary>Allocates the reusable drawing surface. Called once per overlay session.</summary>
    private void CreateSurfaceLocked()
    {
        DisposeSurfaceLocked();
        _surface = new SKBitmap(new SKImageInfo(
            TextureWidth, TextureHeight, SKColorType.Rgba8888, SKAlphaType.Premul));
        _canvas = new SKCanvas(_surface);
        _lastRendered = null;
    }

    private void DisposeSurfaceLocked()
    {
        _canvas?.Dispose();
        _surface?.Dispose();
        _canvas = null;
        _surface = null;
        _lastRendered = null;
    }

    private void PresentLocked(VrCalibrationFrame frame)
    {
        if (_overlay == null || _handle == 0) return;
        if (_surface == null || _canvas == null) CreateSurfaceLocked();

        // Skip identical content before drawing anything. Consumers publish far faster than the
        // panel actually changes, and redrawing plus re-uploading a megabytes-wide texture for a
        // pixel-identical result is what made the overlay visibly unstable.
        frame = frame.Quantize();
        _lastFrame = frame;
        if (_lastRendered == frame) return;

        var bitmap = _surface!;
        var canvas = _canvas!;
        // Fully opaque: a translucent panel shows every upload seam against the scene behind it.
        canvas.Clear(new SKColor(7, 10, 18, 255));

        DrawText(canvas, frame.Title.ToUpperInvariant(), 54, 70, 46, SKColors.White, true);

        var phaseColor = frame.Phase switch
        {
            VrCalibrationPhase.Hold or VrCalibrationPhase.Sampling or VrCalibrationPhase.Target
                => new SKColor(61, 214, 140),
            VrCalibrationPhase.Settling => new SKColor(255, 201, 71),
            VrCalibrationPhase.Relax => new SKColor(94, 190, 255),
            VrCalibrationPhase.Error => new SKColor(255, 105, 105),
            VrCalibrationPhase.Complete => new SKColor(88, 230, 150),
            _ => new SKColor(255, 201, 71),
        };
        var phaseLabel = frame.Phase switch
        {
            VrCalibrationPhase.Preparing => frame.CountdownSeconds is { } countdown
                ? $"GET READY  {Math.Max(1, (int)Math.Ceiling(countdown))}" : "GET READY",
            VrCalibrationPhase.Sampling => "SAMPLING",
            VrCalibrationPhase.Settling => "SETTLE",
            VrCalibrationPhase.Hold => "HOLD",
            VrCalibrationPhase.Relax => "RELAX",
            VrCalibrationPhase.Target => "LOOK AT THE DOT",
            VrCalibrationPhase.Error => "ERROR",
            VrCalibrationPhase.Complete => "DONE",
            _ => frame.Phase.ToString().ToUpperInvariant(),
        };
        DrawText(canvas, phaseLabel, 54, 137, 50, phaseColor, true);

        DrawWrappedText(canvas, frame.Instruction, 54, 190, TextureWidth - 108, 34,
            new SKColor(230, 234, 243), 44);

        if (frame.TargetX is { } targetX && frame.TargetY is { } targetY)
        {
            var x = TextureWidth / 2f + targetX * 340f;
            var y = 410f + targetY * 180f;
            using var halo = new SKPaint { Color = new SKColor(255, 255, 255, 60), IsAntialias = true };
            using var dot = new SKPaint { Color = new SKColor(255, 224, 72), IsAntialias = true };
            canvas.DrawCircle(x, y, 43, halo);
            canvas.DrawCircle(x, y, 19, dot);
        }

        var detailParts = new List<string>();
        if (frame.Intensity is { } intensity) detailParts.Add($"Intensity {(int)Math.Round(intensity * 100)}%");
        if (frame.RepetitionCount > 0)
            detailParts.Add($"Repetition {Math.Clamp(frame.Repetition, 1, frame.RepetitionCount)} of {frame.RepetitionCount}");
        if (detailParts.Count > 0)
            DrawText(canvas, string.Join("   •   ", detailParts), 54, 568, 27, new SKColor(188, 197, 214));

        DrawProgress(canvas, 54, 600, TextureWidth - 108, 18, frame.PhaseProgress, phaseColor);
        DrawProgress(canvas, 54, 638, TextureWidth - 108, 12, frame.OverallProgress,
            new SKColor(94, 190, 255));

        if (frame.AllowRetry) DrawButton(canvas, "RETRY", 64, 680, 256, phaseColor);
        if (frame.AllowSkip) DrawButton(canvas, "SKIP", 384, 680, 256, new SKColor(130, 145, 170));
        if (frame.AllowCancel) DrawButton(canvas, "CANCEL", 704, 680, 256, new SKColor(222, 91, 91));

        var error = _overlay.SetOverlayRaw(_handle, bitmap.GetPixels(), TextureWidth, TextureHeight, 4);
        ThrowIfError(error, "upload overlay pixels");

        // Only after the upload succeeds, so a failed frame leaves this describing what is still on
        // screen. Present() relies on that to tell a retryable cosmetic update from a lost
        // instruction change.
        _lastRendered = frame;
    }

    private static void DrawProgress(
        SKCanvas canvas, float x, float y, float width, float height, double fraction, SKColor color)
    {
        using var background = new SKPaint { Color = new SKColor(255, 255, 255, 35), IsAntialias = true };
        using var foreground = new SKPaint { Color = color, IsAntialias = true };
        canvas.DrawRoundRect(new SKRect(x, y, x + width, y + height), height / 2, height / 2, background);
        var fill = (float)(width * Math.Clamp(fraction, 0, 1));
        if (fill > 0)
            canvas.DrawRoundRect(new SKRect(x, y, x + fill, y + height), height / 2, height / 2, foreground);
    }

    private static void DrawButton(
        SKCanvas canvas, string label, float x, float y, float width, SKColor color)
    {
        using var fill = new SKPaint { Color = color.WithAlpha(55), IsAntialias = true };
        using var stroke = new SKPaint
        {
            Color = color, IsAntialias = true, Style = SKPaintStyle.Stroke, StrokeWidth = 2,
        };
        var rect = new SKRect(x, y, x + width, y + 58);
        canvas.DrawRoundRect(rect, 10, 10, fill);
        canvas.DrawRoundRect(rect, 10, 10, stroke);
        DrawCenteredText(canvas, label, x + width / 2, y + 39, 25, color, true);
    }

    private static void DrawWrappedText(
        SKCanvas canvas, string text, float x, float y, float maxWidth, float size,
        SKColor color, float lineHeight)
    {
        using var paint = TextPaint(size, color, false);
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var line = "";
        foreach (var word in words)
        {
            var candidate = line.Length == 0 ? word : $"{line} {word}";
            if (paint.MeasureText(candidate) <= maxWidth)
            {
                line = candidate;
                continue;
            }

            canvas.DrawText(line, x, y, paint);
            y += lineHeight;
            line = word;
        }
        if (line.Length > 0) canvas.DrawText(line, x, y, paint);
    }

    private static void DrawText(
        SKCanvas canvas, string text, float x, float y, float size, SKColor color, bool bold = false)
    {
        using var paint = TextPaint(size, color, bold);
        canvas.DrawText(text, x, y, paint);
    }

    private static void DrawCenteredText(
        SKCanvas canvas, string text, float x, float y, float size, SKColor color, bool bold = false)
    {
        using var paint = TextPaint(size, color, bold);
        paint.TextAlign = SKTextAlign.Center;
        canvas.DrawText(text, x, y, paint);
    }

    private static SKPaint TextPaint(float size, SKColor color, bool bold) => new()
    {
        Color = color,
        TextSize = size,
        IsAntialias = true,
        // Resolved once per process. Looking a family up per string was the dominant cost of
        // drawing a frame, and every frame draws eight or more of them.
        Typeface = bold ? BoldTypeface.Value : RegularTypeface.Value,
    };

    private static void ThrowIfError(EVROverlayError error, string operation)
    {
        if (error != EVROverlayError.None)
            throw new InvalidOperationException($"Could not {operation}: {error}");
    }

    private VrPresenterStartResult FailAndCloseLocked(string message)
    {
        _status = message;
        CloseOverlayLocked(preserveStatus: true);
        return VrPresenterStartResult.Failure(message);
    }

    private void CloseOverlayLocked(bool preserveStatus = false)
    {
        if (_handle != 0 && _overlay != null)
        {
            try
            {
                _overlay.HideOverlay(_handle);
                _overlay.DestroyOverlay(_handle);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "OpenVR presenter teardown was incomplete");
            }
        }

        DisposeSurfaceLocked();
        _handle = 0;
        _overlay = null;
        _healthy = false;
        _showingCompletion = false;
        _consecutiveFailures = 0;
        _lastFrame = null;
        _pendingAction = VrCalibrationAction.None;
        if (!preserveStatus) _status = "SteamVR presenter stopped.";
    }

    public void Dispose()
    {
        lock (_sync)
        {
            _completionTimer?.Dispose();
            _completionTimer = null;
            CloseOverlayLocked();
        }
    }
}
