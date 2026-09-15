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

    /// <summary>
    /// How the panel is placed, which differs completely between the two things it is used for.
    /// </summary>
    /// <remarks>
    /// <para><see cref="Instructions"/> has to be read <em>while looking somewhere else</em> — at
    /// your avatar in a mirror, copying its expression. A panel big enough to cover the mirror
    /// defeats the exercise, so it is small, translucent, and parked below the line of sight where
    /// it can be glanced at.</para>
    ///
    /// <para><see cref="GazeTarget"/> is the opposite: the dot <em>is</em> what you look at, so it
    /// needs the full panel to reach the outer targets, centred and unobstructed.</para>
    /// </remarks>
    private enum OverlayLayout
    {
        Instructions,
        GazeTarget,
    }

    // Instruction panel: about +/-17 degrees wide, sitting from roughly 13 to 25 degrees below eye
    // level. That leaves the whole central view - where the mirror is - clear.
    private const float InstructionWidthMeters = 0.75f;
    private const float InstructionHeightFraction = 360f / TextureHeight;
    private const float InstructionCentreY = -0.42f;
    private const float InstructionDistance = -1.20f;
    private const float InstructionAlpha = 0.80f;

    // Gaze panel: wide enough for the +/-18 degree dot grid, centred.
    private const float TargetWidthMeters = 1.35f;
    private const float TargetCentreY = -0.05f;
    private const float TargetDistance = -1.25f;
    private const float TargetAlpha = 0.95f;

    private OverlayLayout? _layout;

    internal delegate IOpenVrOverlay? AcquireOverlay(out string status);
    private readonly AcquireOverlay _acquireOverlay;
    private readonly Func<int, int, ICalibrationOverlayTexture> _createTexture;
    private readonly TimeProvider _timeProvider;
    private ICalibrationOverlayTexture? _gpuTexture;
    private readonly ILogger<OpenVrCalibrationPresenter> _logger;
    private readonly object _sync = new();
    private IOpenVrOverlay? _overlay;
    private ulong _handle;
    private string _status = "SteamVR presenter has not been started.";
    private bool _healthy;
    private VrCalibrationAction _pendingAction;
    private VrCalibrationFrame? _lastFrame;
    private VrCalibrationFrame? _lastRendered;
    private ITimer? _completionTimer;
    private long _completionGeneration;
    private bool _showingCompletion;
    private int _consecutiveFailures;

    /// <summary>
    /// Shortest gap between uploads that only move a progress bar or a countdown digit.
    /// </summary>
    /// <remarks>
    /// Progress quantised to 2% crosses a boundary roughly every 60 ms during a phase, so the
    /// overlay was re-uploading a three-megabyte texture about sixteen times a second to redraw a
    /// bar a few pixels longer. Capping these cosmetic uploads avoids repeated CPU/GPU work
    /// while leaving instruction changes immediate.
    ///
    /// Only cosmetic updates are held back. Anything that changes what the overlay *says* -
    /// instruction, phase, the dot, the buttons - still goes up immediately, because a headset
    /// showing the previous instruction is how a session gets mislabelled.
    /// </remarks>
    private static readonly TimeSpan MinimumCosmeticInterval = TimeSpan.FromMilliseconds(100);

    private long _lastUploadTimestamp;

    // Raster buffers stay local to Skia. SteamVR sees a persistent GPU texture, not
    // a succession of raw images that may disappear while the replacement is loaded.
    private SKBitmap? _surface;
    private SKBitmap? _backSurface;
    private SKCanvas? _backCanvas;
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
        : this((out string status) =>
        {
            var overlay = openVr.TryGetOverlay(out status);
            return overlay == null ? null : new OpenVrOverlay(overlay);
        }, CreateGpuTexture, logger)
    {
    }

    internal OpenVrCalibrationPresenter(
        AcquireOverlay acquireOverlay,
        Func<int, int, ICalibrationOverlayTexture> createTexture,
        ILogger<OpenVrCalibrationPresenter> logger,
        TimeProvider? timeProvider = null)
    {
        _acquireOverlay = acquireOverlay;
        _createTexture = createTexture;
        _logger = logger;
        _timeProvider = timeProvider ?? TimeProvider.System;
    }

    private static ICalibrationOverlayTexture CreateGpuTexture(int width, int height)
    {
        var system = OpenVR.System ?? throw new InvalidOperationException("SteamVR system interface is unavailable.");
        var adapter = -1;
        system.GetDXGIOutputInfo(ref adapter);
        return new D3D11OverlayTexture(adapter, width, height);
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

            _overlay = _acquireOverlay(out var runtimeStatus);
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

                ThrowIfError(_overlay.SetOverlayInputMethod(_handle, VROverlayInputMethod.Mouse),
                    "enable controller pointer input");

                var mouseScale = new HmdVector2_t { v0 = TextureWidth, v1 = TextureHeight };
                ThrowIfError(_overlay.SetOverlayMouseScale(_handle, ref mouseScale), "set pointer scale");
                // Skia rasterizes premultiplied RGBA; tell the compositor to blend it that way.
                ThrowIfError(_overlay.SetOverlayFlag(_handle, VROverlayFlags.IsPremultiplied, true),
                    "set premultiplied alpha");

                _layout = null;
                ApplyLayoutLocked(OverlayLayout.Instructions);

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
        var target = UsesGazeTargetLayout(frame);

        if (frame.AllowRetry && Hit(ButtonRect.Retry(target), x, y, target))
            return VrCalibrationAction.Retry;
        if (frame.AllowSkip && Hit(ButtonRect.Skip(target), x, y, target))
            return VrCalibrationAction.Skip;
        if (frame.AllowCancel && Hit(ButtonRect.Cancel(target), x, y, target))
            return VrCalibrationAction.Cancel;
        return VrCalibrationAction.None;
    }

    /// <summary>Which layout a frame will be drawn with. One rule, used by drawing and hit-testing.</summary>
    private static bool UsesGazeTargetLayout(VrCalibrationFrame frame) =>
        frame.Phase == VrCalibrationPhase.Target && frame is { TargetX: not null, TargetY: not null };

    /// <summary>
    /// Where a button is drawn, in texture pixels. The single source both the renderer and the
    /// pointer hit-test read, so a moved button cannot become an invisible one that still works.
    /// </summary>
    private readonly record struct ButtonRect(float Left, float Right, float Top, float Bottom)
    {
        // The gaze layout shows the whole texture and keeps the original strip along the bottom.
        // The instruction layout is cropped and compact, so its buttons sit inside the visible part.
        public static ButtonRect Retry(bool target) =>
            target ? new(64, 320, 680, 738) : new(44, 244, 236, 294);
        public static ButtonRect Skip(bool target) =>
            target ? new(384, 640, 680, 738) : new(260, 420, 236, 294);
        public static ButtonRect Cancel(bool target) =>
            target ? new(704, 960, 680, 738) : new(436, 636, 236, 294);
    }

    /// <summary>
    /// Whether a pointer landed on a button, tolerating every coordinate convention OpenVR might
    /// hand us.
    /// </summary>
    /// <remarks>
    /// Two independent ambiguities, and guessing wrong makes a visible button silently unclickable.
    /// Bindings have historically reported overlay mouse Y using either the texture's top-left
    /// origin or its reflected bottom-left one. And the instruction layout crops the texture, so Y
    /// may arrive scaled across the visible strip rather than the whole texture. Accepting all four
    /// readings keeps the target narrow in X - where there is no ambiguity - while not making the
    /// user guess which invisible band works.
    /// </remarks>
    private static bool Hit(ButtonRect rect, float x, float y, bool target)
    {
        if (x < rect.Left || x > rect.Right)
            return false;

        if (InBand(y) || InBand(TextureHeight - y))
            return true;

        if (target)
            return false;

        // The cropped layout may report Y across the visible strip instead of the full texture.
        var scale = InstructionHeightFraction;
        return InBand(y * scale) || InBand((TextureHeight - y) * scale);

        bool InBand(float value) => value >= rect.Top && value <= rect.Bottom;
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
                    var generation = ++_completionGeneration;
                    _completionTimer = _timeProvider.CreateTimer(_ =>
                    {
                        lock (_sync)
                        {
                            // Only tear down if this is still the completion frame; a calibration
                            // started in the meantime now owns the surface.
                            if (_showingCompletion && generation == _completionGeneration)
                                CloseOverlayLocked();
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
        onScreen.RepetitionCount == frame.RepetitionCount &&
        onScreen.AllowRetry == frame.AllowRetry &&
        onScreen.AllowSkip == frame.AllowSkip &&
        onScreen.AllowCancel == frame.AllowCancel &&
        onScreen.TargetX.Equals(frame.TargetX) &&
        onScreen.TargetY.Equals(frame.TargetY);

    private bool IsCosmeticUpdate(VrCalibrationFrame frame) => IsCosmeticUpdate(frame, _lastRendered);

    /// <summary>
    /// Places and sizes the panel for what it is currently being used for.
    /// </summary>
    /// <remarks>
    /// Only applied when the mode actually changes, which happens at most once per pose — calling
    /// these every frame would fight the compositor for no benefit.
    /// </remarks>
    private void ApplyLayoutLocked(OverlayLayout layout)
    {
        if (_layout == layout || _overlay == null || _handle == 0)
            return;

        var instructions = layout == OverlayLayout.Instructions;

        // Crop the instruction layout to the part of the texture it actually draws in, so the panel
        // is short rather than a mostly-empty rectangle hanging in the view.
        var bounds = new VRTextureBounds_t
        {
            uMin = 0,
            vMin = 0,
            uMax = 1,
            vMax = instructions ? InstructionHeightFraction : 1f,
        };
        ThrowIfError(_overlay.SetOverlayTextureBounds(_handle, ref bounds), "set overlay texture bounds");

        ThrowIfError(_overlay.SetOverlayWidthInMeters(
            _handle, instructions ? InstructionWidthMeters : TargetWidthMeters), "set overlay size");
        ThrowIfError(_overlay.SetOverlayAlpha(
            _handle, instructions ? InstructionAlpha : TargetAlpha), "set overlay opacity");

        // OpenVR HMD coordinates use -Z in front of the viewer.
        var transform = new HmdMatrix34_t
        {
            m0 = 1, m1 = 0, m2 = 0, m3 = 0,
            m4 = 0, m5 = 1, m6 = 0, m7 = instructions ? InstructionCentreY : TargetCentreY,
            m8 = 0, m9 = 0, m10 = 1, m11 = instructions ? InstructionDistance : TargetDistance,
        };
        ThrowIfError(_overlay.SetOverlayTransformTrackedDeviceRelative(
            _handle, OpenVR.k_unTrackedDeviceIndex_Hmd, ref transform), "attach overlay to headset");

        _layout = layout;

        // The panel moved, so nothing that was on it is still valid to compare against.
        _lastRendered = null;
    }

    /// <summary>Allocates the reusable drawing surface. Called once per overlay session.</summary>
    private void CreateSurfaceLocked()
    {
        DisposeSurfaceLocked();
        var info = new SKImageInfo(
            TextureWidth, TextureHeight, SKColorType.Rgba8888, SKAlphaType.Premul);
        _surface = new SKBitmap(info);
        _canvas = new SKCanvas(_surface);
        _backSurface = new SKBitmap(info);
        _backCanvas = new SKCanvas(_backSurface);
        _gpuTexture = _createTexture(TextureWidth, TextureHeight);
        _lastRendered = null;
        _lastUploadTimestamp = 0;
    }

    private void DisposeSurfaceLocked()
    {
        _gpuTexture?.Dispose();
        _gpuTexture = null;
        _canvas?.Dispose();
        _surface?.Dispose();
        _backCanvas?.Dispose();
        _backSurface?.Dispose();
        _canvas = null;
        _surface = null;
        _backCanvas = null;
        _backSurface = null;
        _lastRendered = null;
        _lastUploadTimestamp = 0;
    }

    /// <summary>
    /// Makes the buffer just uploaded the front one, so the next frame is drawn into the other.
    /// </summary>
    private void SwapSurfacesLocked()
    {
        (_surface, _backSurface) = (_backSurface, _surface);
        (_canvas, _backCanvas) = (_backCanvas, _canvas);
    }

    private void PresentLocked(VrCalibrationFrame frame)
    {
        if (_overlay == null || _handle == 0) return;
        if (_surface == null || _canvas == null) CreateSurfaceLocked();

        // Skip identical content before drawing anything. Consumers publish far faster than the
        // panel actually changes; identical content needs neither rasterization nor a GPU copy.
        frame = frame.Quantize();

        if (_lastRendered == frame) return;

        // Hold back updates that only move a bar. The instruction on screen is already correct, so
        // waiting a few tens of milliseconds costs the user nothing and spares the compositor a
        // three-megabyte upload it cannot show them anyway.
        if (IsCosmeticUpdate(frame) && _lastUploadTimestamp != 0 &&
            _timeProvider.GetElapsedTime(_lastUploadTimestamp) < MinimumCosmeticInterval)
        {
            return;
        }

        // Finish rasterizing locally before copying a complete image into the GPU texture.
        var bitmap = _backSurface!;
        var canvas = _backCanvas!;

        var wantsTarget = frame.Phase == VrCalibrationPhase.Target &&
                          frame is { TargetX: not null, TargetY: not null };
        ApplyLayoutLocked(wantsTarget ? OverlayLayout.GazeTarget : OverlayLayout.Instructions);

        canvas.Clear(SKColors.Transparent);

        // A gaze target is the one frame where the chrome actively hurts: any text on the panel is
        // something to read, and reading it means looking away from the dot whose position is being
        // recorded as ground truth. So the Target phase draws the dot, a thin progress bar and a
        // cancel affordance, and nothing else.
        if (wantsTarget)
        {
            DrawGazeTarget(canvas, frame.TargetX!.Value, frame.TargetY!.Value);
            DrawProgress(canvas, 54, 700, TextureWidth - 108, 10, frame.PhaseProgress,
                new SKColor(61, 214, 140));
            if (frame.AllowCancel)
                DrawButton(canvas, "CANCEL", ButtonRect.Cancel(true), new SKColor(222, 91, 91));

            UploadLocked(bitmap, frame);
            return;
        }

        DrawInstructionPanel(canvas, frame);
        UploadLocked(bitmap, frame);
    }

    /// <summary>
    /// The instruction layout, drawn compactly in the top of the texture.
    /// </summary>
    /// <remarks>
    /// Everything lives above <see cref="InstructionHeightFraction"/> of the texture, because that
    /// is the part the panel actually shows. The whole panel is small and parked below the line of
    /// sight, so this is written to be <em>glanceable</em>: one large line saying what to do and how
    /// far through you are, and nothing that needs studying.
    /// </remarks>
    private static void DrawInstructionPanel(SKCanvas canvas, VrCalibrationFrame frame)
    {
        // A rounded translucent slab rather than a full-bleed rectangle: the corners let the scene
        // through and stop it reading as a box bolted to your face.
        using (var background = new SKPaint { Color = new SKColor(9, 12, 20, 214), IsAntialias = true })
            canvas.DrawRoundRect(new SKRect(8, 8, TextureWidth - 8, 352), 22, 22, background);

        var phaseColor = PhaseColor(frame.Phase);

        DrawText(canvas, PhaseLabel(frame).ToUpperInvariant(), 44, 96, 62, phaseColor, true);

        var heading = frame.Title.ToUpperInvariant();
        if (!string.IsNullOrWhiteSpace(heading))
            DrawText(canvas, heading, 44, 152, 34, new SKColor(178, 188, 206));

        DrawWrappedText(canvas, frame.Instruction, 44, 206, TextureWidth - 88, 34,
            new SKColor(232, 236, 245), 42, maxLines: 2);

        DrawProgress(canvas, 44, 284, TextureWidth - 88, 12, frame.PhaseProgress, phaseColor);
        DrawProgress(canvas, 44, 306, TextureWidth - 88, 8, frame.OverallProgress,
            new SKColor(94, 190, 255));

        if (frame.AllowRetry) DrawButton(canvas, "RETRY", ButtonRect.Retry(false), phaseColor);
        if (frame.AllowSkip) DrawButton(canvas, "SKIP", ButtonRect.Skip(false), new SKColor(130, 145, 170));
        if (frame.AllowCancel) DrawButton(canvas, "CANCEL", ButtonRect.Cancel(false), new SKColor(222, 91, 91));
    }

    private static SKColor PhaseColor(VrCalibrationPhase phase) => phase switch
    {
        VrCalibrationPhase.Hold or VrCalibrationPhase.Sampling or VrCalibrationPhase.Target
            => new SKColor(61, 214, 140),
        VrCalibrationPhase.Settling => new SKColor(255, 201, 71),
        VrCalibrationPhase.Relax => new SKColor(94, 190, 255),
        VrCalibrationPhase.Error => new SKColor(255, 105, 105),
        VrCalibrationPhase.Complete => new SKColor(88, 230, 150),
        _ => new SKColor(255, 201, 71),
    };

    private static string PhaseLabel(VrCalibrationFrame frame) => frame.Phase switch
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

    private void UploadLocked(SKBitmap bitmap, VrCalibrationFrame frame)
    {
        _gpuTexture!.Upload(bitmap);
        var texture = _gpuTexture.Texture;
        ThrowIfError(_overlay!.SetOverlayTexture(_handle, ref texture), "submit overlay texture");

        // Only after the upload succeeds, so a failed frame leaves this describing what is still on
        // screen. Present() relies on that to tell a retryable cosmetic update from a lost
        // instruction change.
        SwapSurfacesLocked();
        _lastUploadTimestamp = _timeProvider.GetTimestamp();
        _lastRendered = frame;
        _lastFrame = frame;
        _consecutiveFailures = 0;
    }

    /// <summary>
    /// The fixation dot. Centred on the panel and scaled to its reach, which is roughly +/-20
    /// degrees horizontally and a narrower band vertically.
    /// </summary>
    private static void DrawGazeTarget(SKCanvas canvas, float targetX, float targetY)
    {
        var x = TextureWidth / 2f + targetX * 340f;
        var y = 410f + targetY * 180f;
        using var halo = new SKPaint { Color = new SKColor(255, 255, 255, 60), IsAntialias = true };
        using var dot = new SKPaint { Color = new SKColor(255, 224, 72), IsAntialias = true };
        canvas.DrawCircle(x, y, 43, halo);
        canvas.DrawCircle(x, y, 19, dot);
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

    private static void DrawButton(SKCanvas canvas, string label, ButtonRect rect, SKColor color)
    {
        DrawButton(canvas, label, rect.Left, rect.Top, rect.Right - rect.Left, color,
            rect.Bottom - rect.Top);
    }

    private static void DrawButton(
        SKCanvas canvas, string label, float x, float y, float width, SKColor color,
        float height = 58)
    {
        using var fill = new SKPaint { Color = color.WithAlpha(55), IsAntialias = true };
        using var stroke = new SKPaint
        {
            Color = color, IsAntialias = true, Style = SKPaintStyle.Stroke, StrokeWidth = 2,
        };
        var rect = new SKRect(x, y, x + width, y + height);
        canvas.DrawRoundRect(rect, 10, 10, fill);
        canvas.DrawRoundRect(rect, 10, 10, stroke);
        DrawCenteredText(canvas, label, x + width / 2, y + height / 2 + 9, 25, color, true);
    }

    private static void DrawWrappedText(
        SKCanvas canvas, string text, float x, float y, float maxWidth, float size,
        SKColor color, float lineHeight, int maxLines = int.MaxValue)
    {
        using var paint = TextPaint(size, color, false);
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var line = "";
        var drawn = 0;
        foreach (var word in words)
        {
            var candidate = line.Length == 0 ? word : $"{line} {word}";
            if (paint.MeasureText(candidate) <= maxWidth)
            {
                line = candidate;
                continue;
            }

            // The compact panel is cropped, so an over-long instruction would otherwise be drawn
            // into the part of the texture nobody can see - silently truncated with no ellipsis to
            // say so.
            if (drawn + 1 >= maxLines)
            {
                canvas.DrawText(Ellipsize(line + " ...", paint, maxWidth), x, y, paint);
                return;
            }

            canvas.DrawText(line, x, y, paint);
            drawn++;
            y += lineHeight;
            line = word;
        }
        if (line.Length > 0) canvas.DrawText(line, x, y, paint);
    }

    private static string Ellipsize(string text, SKPaint paint, float maxWidth)
    {
        while (text.Length > 4 && paint.MeasureText(text) > maxWidth)
            text = text[..(text.Length - 5)] + " ...";
        return text;
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
        // Disposed timers may already have queued a callback. A generation prevents an
        // earlier completion screen from closing a later session's overlay.
        _completionGeneration++;
        _completionTimer?.Dispose();
        _completionTimer = null;
        if (_handle != 0 && _overlay != null)
        {
            // Keep the GPU texture alive until SteamVR has released its reference. Each cleanup
            // is attempted even if a disconnected runtime throws during an earlier operation.
            Teardown(() => _overlay.HideOverlay(_handle));
            Teardown(() => _overlay.ClearOverlayTexture(_handle));
            Teardown(() => _overlay.DestroyOverlay(_handle));
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

    private void Teardown(Func<EVROverlayError> operation)
    {
        try { operation(); }
        catch (Exception ex) { _logger.LogDebug(ex, "OpenVR presenter teardown was incomplete"); }
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
