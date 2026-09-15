using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Desktop.Calibration;
using Baballonia.Services.Calibration;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SkiaSharp;
using Valve.VR;

namespace Baballonia.Tests.Services.Calibration;

[TestClass]
public class OverlayPresentationTest
{
    private static VrCalibrationFrame Hold() => new(
        "SMILE", "Smile naturally.", VrCalibrationPhase.Hold, 0.4, 0.5,
        AllowRetry: true, AllowSkip: true);

    [TestMethod]
    public void UploadsFinishedTextureBeforeShowing_AndReusesItAcrossUpdates()
    {
        using var rig = new Rig();
        Assert.IsTrue(rig.Presenter.Begin("TEST").Started);
        Assert.IsTrue(rig.Api.Premultiplied);
        Assert.IsTrue(rig.Calls.IndexOf("gpu.upload") < rig.Calls.IndexOf("texture.submit"));
        Assert.IsTrue(rig.Calls.IndexOf("texture.submit") < rig.Calls.IndexOf("show"));
        for (var i = 0; i < 5; i++)
            rig.Presenter.Present(Hold() with { Title = "STEP " + i });
        Assert.AreEqual(1, rig.Textures.Count, "A session must keep the same GPU allocation.");
        Assert.AreEqual(1, rig.Api.SubmittedHandles.Distinct().Count());
        Assert.AreEqual(1, rig.Calls.Count(x => x == "show"));
        Assert.IsFalse(rig.Calls.Contains("hide"), "Updates must not blank the visible panel.");
        Assert.IsFalse(rig.Calls.Contains("texture.clear"));
        Assert.IsTrue(rig.Presenter.IsHealthy);
    }

    [TestMethod]
    public void IdenticalAndCosmeticFramesSkipUploads_InstructionChangesAreImmediate()
    {
        using var rig = new Rig();
        rig.Presenter.Begin("TEST");
        rig.Presenter.Present(Hold());
        var texture = rig.Textures.Single();
        Assert.AreEqual(2, texture.Uploads);
        rig.Presenter.Present(Hold());
        rig.Clock.Advance(TimeSpan.FromMilliseconds(99));
        rig.Presenter.Present(Hold() with { PhaseProgress = 0.8 });
        Assert.AreEqual(2, texture.Uploads);
        rig.Clock.Advance(TimeSpan.FromMilliseconds(1));
        rig.Presenter.Present(Hold() with { PhaseProgress = 0.8 });
        Assert.AreEqual(3, texture.Uploads);
        rig.Presenter.Present(Hold() with { Instruction = "Now relax." });
        rig.Presenter.Present(Hold() with { AllowRetry = false });
        rig.Presenter.Present(Hold() with { RepetitionCount = 3 });
        Assert.AreEqual(6, texture.Uploads, "Actionable changes bypass the cosmetic interval.");
    }

    [TestMethod]
    public void GazeMovesUploadImmediately_EvenInsideCosmeticInterval()
    {
        using var rig = new Rig();
        rig.Presenter.Begin("TEST");
        var target = Hold() with { Phase = VrCalibrationPhase.Target, TargetX = 0, TargetY = 0 };
        rig.Presenter.Present(target);
        rig.Presenter.Present(target with { TargetX = 0.4f });
        rig.Presenter.Present(target with { TargetY = -0.3f });
        Assert.AreEqual(4, rig.Textures.Single().Uploads);
        Assert.IsTrue(rig.Presenter.IsHealthy);
    }

    [TestMethod]
    public void CosmeticUploadFailureKeepsPanel_RealSuccessResetsFailureCount()
    {
        using var rig = new Rig();
        rig.Presenter.Begin("TEST");
        rig.Presenter.Present(Hold());
        rig.Clock.Advance(TimeSpan.FromMilliseconds(110));
        rig.Api.SubmissionError = EVROverlayError.InvalidTexture;
        rig.Presenter.Present(Hold() with { PhaseProgress = 0.7 });
        Assert.IsTrue(rig.Presenter.IsHealthy);
        Assert.IsFalse(rig.Calls.Contains("hide"));
        rig.Api.SubmissionError = EVROverlayError.None;
        rig.Presenter.Present(Hold() with { PhaseProgress = 0.7 });
        rig.Clock.Advance(TimeSpan.FromMilliseconds(110));
        rig.Api.SubmissionError = EVROverlayError.InvalidTexture;
        rig.Presenter.Present(Hold() with { PhaseProgress = 0.8 });
        rig.Presenter.Present(Hold() with { PhaseProgress = 0.9 });
        Assert.IsTrue(rig.Presenter.IsHealthy, "A successful upload must reset the failure streak.");
        rig.Presenter.Present(Hold() with { PhaseProgress = 1 });
        Assert.IsFalse(rig.Presenter.IsHealthy);
        Assert.IsFalse(rig.Presenter.IsPresenting);
    }

    [TestMethod]
    public void SkippedFramesDoNotEraseFailedSubmissionStreak()
    {
        using var rig = new Rig();
        rig.Presenter.Begin("TEST");
        rig.Presenter.Present(Hold());
        rig.Clock.Advance(TimeSpan.FromMilliseconds(110));
        rig.Api.SubmissionError = EVROverlayError.InvalidTexture;
        for (var i = 0; i < 3; i++)
        {
            rig.Presenter.Present(Hold() with { PhaseProgress = 0.7 });
            rig.Presenter.Present(Hold()); // identical to the last successful frame
        }
        Assert.IsFalse(rig.Presenter.IsHealthy);
        Assert.AreEqual(1, rig.Calls.Count(x => x == "hide"));
    }

    [TestMethod]
    public void FailedInstructionStopsPresentationImmediately()
    {
        using var rig = new Rig();
        rig.Presenter.Begin("TEST");
        rig.Presenter.Present(Hold());
        rig.Api.SubmissionError = EVROverlayError.InvalidTexture;
        rig.Presenter.Present(Hold() with { Instruction = "Relax your face." });
        Assert.IsFalse(rig.Presenter.IsHealthy);
        Assert.IsFalse(rig.Presenter.IsPresenting);
        Assert.IsTrue(rig.Presenter.Status.Contains("submit overlay texture"));
        Assert.IsTrue(rig.Textures.Single().Disposed);
    }

    [TestMethod]
    public void DetachesTextureBeforeReleasingGpuResources_EvenIfHideThrows()
    {
        using var rig = new Rig();
        rig.Presenter.Begin("TEST");
        rig.Api.ThrowOnHide = true;
        rig.Presenter.End();
        var clear = rig.Calls.IndexOf("texture.clear");
        var destroy = rig.Calls.IndexOf("destroy");
        var dispose = rig.Calls.IndexOf("gpu.dispose");
        Assert.IsTrue(clear >= 0 && clear < destroy && destroy < dispose);
        rig.Presenter.Dispose();
        Assert.AreEqual(1, rig.Calls.Count(x => x == "gpu.dispose"));
    }

    [TestMethod]
    public void OldQueuedCompletionCannotCloseNewSessionOrItsCompletion()
    {
        using var rig = new Rig();
        rig.Presenter.Begin("FIRST");
        rig.Presenter.End("First done.");
        var oldTimer = rig.Clock.Timers.Single();
        Assert.IsTrue(rig.Presenter.Begin("SECOND").Started);
        rig.Presenter.End("Second done.");
        oldTimer.FireEvenIfDisposed();
        Assert.IsTrue(rig.Presenter.IsPresenting, "An already queued callback belongs to its old session.");
        Assert.IsFalse(rig.Textures[1].Disposed);
        rig.Clock.Timers[1].FireEvenIfDisposed();
        Assert.IsFalse(rig.Presenter.IsPresenting);
        Assert.IsTrue(rig.Textures[1].Disposed);
    }

    [TestMethod]
    public void ConcurrentBeginDoesNotStealAnActiveOverlay()
    {
        using var rig = new Rig();
        Assert.IsTrue(rig.Presenter.Begin("FIRST").Started);
        Assert.IsFalse(rig.Presenter.Begin("SECOND").Started);
        Assert.AreEqual(1, rig.Textures.Count);
        Assert.IsFalse(rig.Calls.Contains("destroy"));
    }

    [TestMethod]
    public void GpuInitializationFailureNeverShowsABlankOverlay()
    {
        var calls = new List<string>();
        var api = new FakeOverlay(calls);
        using var presenter = new OpenVrCalibrationPresenter(
            (out string status) => { status = "test"; return api; },
            (_, _) => throw new InvalidOperationException("GPU creation failed"),
            NullLogger<OpenVrCalibrationPresenter>.Instance);
        Assert.IsFalse(presenter.Begin("TEST").Started);
        Assert.IsFalse(calls.Contains("show"));
        Assert.IsTrue(calls.Contains("destroy"));
        Assert.IsFalse(presenter.IsHealthy);
    }

    private sealed class Rig : IDisposable
    {
        public readonly List<string> Calls = new();
        public readonly List<FakeTexture> Textures = new();
        public readonly ManualClock Clock = new();
        public readonly FakeOverlay Api;
        public readonly OpenVrCalibrationPresenter Presenter;
        public Rig()
        {
            Api = new FakeOverlay(Calls);
            Presenter = new OpenVrCalibrationPresenter(
                (out string status) => { status = "fake runtime"; return Api; },
                (width, height) =>
                {
                    var texture = new FakeTexture(Calls, new IntPtr(Textures.Count + 42));
                    Textures.Add(texture);
                    return texture;
                }, NullLogger<OpenVrCalibrationPresenter>.Instance, Clock);
        }
        public void Dispose() => Presenter.Dispose();
    }

    private sealed class FakeTexture(List<string> calls, IntPtr handle) : ICalibrationOverlayTexture
    {
        public int Uploads;
        public bool Disposed;
        public Texture_t Texture => new() { handle = handle, eType = ETextureType.DXGISharedHandle, eColorSpace = EColorSpace.Gamma };
        public void Upload(SKBitmap bitmap)
        {
            Assert.IsFalse(Disposed);
            Assert.AreEqual(SKColorType.Rgba8888, bitmap.ColorType);
            Assert.IsTrue(bitmap.GetPixels() != IntPtr.Zero);
            Uploads++;
            calls.Add("gpu.upload");
        }
        public void Dispose()
        {
            Assert.IsFalse(Disposed, "The texture must be released only once.");
            Disposed = true;
            calls.Add("gpu.dispose");
        }
    }

    private sealed class FakeOverlay(List<string> calls) : IOpenVrOverlay
    {
        private ulong _next = 1;
        public bool Premultiplied;
        public bool ThrowOnHide;
        public EVROverlayError SubmissionError;
        public readonly List<IntPtr> SubmittedHandles = new();
        public EVROverlayError FindOverlay(string key, ref ulong handle) => EVROverlayError.UnknownOverlay;
        public EVROverlayError CreateOverlay(string key, string name, ref ulong handle) { handle = _next++; calls.Add("create"); return EVROverlayError.None; }
        public EVROverlayError DestroyOverlay(ulong handle) { calls.Add("destroy"); return EVROverlayError.None; }
        public EVROverlayError SetOverlayInputMethod(ulong handle, VROverlayInputMethod method) => EVROverlayError.None;
        public EVROverlayError SetOverlayMouseScale(ulong handle, ref HmdVector2_t scale) => EVROverlayError.None;
        public EVROverlayError SetOverlayFlag(ulong handle, VROverlayFlags flag, bool enabled) { if (flag == VROverlayFlags.IsPremultiplied) Premultiplied = enabled; return EVROverlayError.None; }
        public EVROverlayError ShowOverlay(ulong handle) { calls.Add("show"); return EVROverlayError.None; }
        public EVROverlayError HideOverlay(ulong handle) { calls.Add("hide"); if (ThrowOnHide) throw new InvalidOperationException("runtime disconnected"); return EVROverlayError.None; }
        public bool PollNextOverlayEvent(ulong handle, ref VREvent_t vrEvent, uint size) => false;
        public EVROverlayError SetOverlayTextureBounds(ulong handle, ref VRTextureBounds_t bounds) => EVROverlayError.None;
        public EVROverlayError SetOverlayWidthInMeters(ulong handle, float width) => EVROverlayError.None;
        public EVROverlayError SetOverlayAlpha(ulong handle, float alpha) => EVROverlayError.None;
        public EVROverlayError SetOverlayTransformTrackedDeviceRelative(ulong handle, uint device, ref HmdMatrix34_t transform) => EVROverlayError.None;
        public EVROverlayError SetOverlayTexture(ulong handle, ref Texture_t texture) { calls.Add("texture.submit"); SubmittedHandles.Add(texture.handle); return SubmissionError; }
        public EVROverlayError ClearOverlayTexture(ulong handle) { calls.Add("texture.clear"); return EVROverlayError.None; }
    }

    private sealed class ManualClock : TimeProvider
    {
        private long _ticks = TimeSpan.TicksPerSecond;
        public readonly List<ManualTimer> Timers = new();
        public override long TimestampFrequency => TimeSpan.TicksPerSecond;
        public override long GetTimestamp() => _ticks;
        public void Advance(TimeSpan time) => _ticks += time.Ticks;
        public override ITimer CreateTimer(TimerCallback callback, object? state, TimeSpan dueTime, TimeSpan period)
        {
            var timer = new ManualTimer(callback, state);
            Timers.Add(timer);
            return timer;
        }
    }

    private sealed class ManualTimer(TimerCallback callback, object? state) : ITimer
    {
        public void FireEvenIfDisposed() => callback(state);
        public bool Change(TimeSpan dueTime, TimeSpan period) => true;
        public void Dispose() { }
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
