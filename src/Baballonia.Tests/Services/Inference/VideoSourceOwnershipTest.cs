using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.SDK;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Enums;
using Baballonia.Services.Inference.VideoSources;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Inference;

[TestClass]
public class VideoSourceOwnershipTest
{
    [TestMethod]
    public void SingleSourceDisposesTheRawFrameAfterColorConversion()
    {
        var capture = new FakeCapture();
        var raw = new Mat(8, 8, MatType.CV_8UC3, Scalar.All(42));
        capture.Offer(raw);
        using var source = new SingleCameraSource(
            NullLogger<SingleCameraSource>.Instance, capture, "fake");

        using var converted = source.GetFrame(ColorType.Gray8);

        Assert.IsNotNull(converted);
        Assert.AreEqual(1, converted.Channels());
        Assert.IsTrue(raw.IsDisposed);
    }

    [TestMethod]
    public void SingleSourceTransfersAnUnconvertedFrameToTheCaller()
    {
        var capture = new FakeCapture();
        var raw = new Mat(8, 8, MatType.CV_8UC1, Scalar.All(42));
        capture.Offer(raw);
        using var source = new SingleCameraSource(
            NullLogger<SingleCameraSource>.Instance, capture, "fake");

        var returned = source.GetFrame(ColorType.Gray8);

        Assert.AreSame(raw, returned);
        Assert.IsFalse(raw.IsDisposed);
        returned!.Dispose();
        Assert.IsTrue(raw.IsDisposed);
    }

    [TestMethod]
    public void DualSourceReleasesBothFreshChildFrames()
    {
        var left = new FakeVideoSource();
        var right = new FakeVideoSource();
        left.Enqueue(10);
        right.Enqueue(20);
        using var source = new DualCameraSource { LeftCam = left, RightCam = right };

        using var combined = source.GetFrame(ColorType.Gray8);

        Assert.IsNotNull(combined);
        Assert.AreEqual(16, combined.Width);
        Assert.IsTrue(left.Issued[0].IsDisposed);
        Assert.IsTrue(right.Issued[0].IsDisposed);
    }

    [TestMethod]
    public void DualSourceStartAndStopDelegateToTheMatchingOperation()
    {
        var left = new FakeVideoSource();
        var right = new FakeVideoSource();
        using var source = new DualCameraSource { LeftCam = left, RightCam = right };

        Assert.IsTrue(source.Start());
        Assert.IsTrue(source.Stop());

        Assert.AreEqual(1, left.StartCalls);
        Assert.AreEqual(1, right.StartCalls);
        Assert.AreEqual(1, left.StopCalls);
        Assert.AreEqual(1, right.StopCalls);
    }

    [TestMethod]
    public void AStaleMissingHalfMirrorsTheLiveHalfInsteadOfFreezing()
    {
        var left = new FakeVideoSource();
        var right = new FakeVideoSource();
        left.Enqueue(10);
        right.Enqueue(20);
        using var source = new DualCameraSource { LeftCam = left, RightCam = right };
        using var initial = source.GetFrame(ColorType.Gray8);

        Thread.Sleep(DualCameraSource.StaleHalfTimeout + TimeSpan.FromMilliseconds(60));
        left.Enqueue(30);
        using var afterOutage = source.GetFrame(ColorType.Gray8);

        Assert.IsNotNull(afterOutage);
        Assert.AreEqual((byte)30, afterOutage.At<byte>(0, 0));
        Assert.AreEqual((byte)30, afterOutage.At<byte>(0, 8));
    }

    [TestMethod]
    public void InvalidatingOneHalfMirrorsImmediately()
    {
        var left = new FakeVideoSource();
        var right = new FakeVideoSource();
        left.Enqueue(10);
        right.Enqueue(20);
        using var source = new DualCameraSource { LeftCam = left, RightCam = right };
        using var initial = source.GetFrame(ColorType.Gray8);

        source.InvalidateRightCache();
        left.Enqueue(40);
        using var afterInvalidation = source.GetFrame(ColorType.Gray8);

        Assert.IsNotNull(afterInvalidation);
        Assert.AreEqual((byte)40, afterInvalidation.At<byte>(0, 0));
        Assert.AreEqual((byte)40, afterInvalidation.At<byte>(0, 8));
    }

    [TestMethod]
    public void ResetDropsTheTemporalSequence()
    {
        using var collector = new ImageCollector();

        for (var i = 0; i < 4; i++)
            using (var input = SplitEyeFrame((byte)i))
                Assert.IsNull(collector.Apply(input));

        collector.Reset();

        for (var i = 0; i < 4; i++)
            using (var input = SplitEyeFrame((byte)(10 + i)))
                Assert.IsNull(collector.Apply(input));

        using var fifth = SplitEyeFrame(20);
        using var output = collector.Apply(fifth);
        Assert.IsNotNull(output);
        Assert.AreEqual(8, output.Channels());
    }

    private static Mat SplitEyeFrame(byte value) =>
        new(8, 8, MatType.CV_8UC2, Scalar.All(value));

    private sealed class FakeCapture() : Capture("fake", NullLogger<FakeCapture>.Instance)
    {
        public void Offer(Mat frame)
        {
            SetRawMat(frame);
            IsReady = true;
        }

        public override Task<bool> StartCapture()
        {
            IsReady = true;
            return Task.FromResult(true);
        }

        public override Task<bool> StopCapture()
        {
            IsReady = false;
            return Task.FromResult(true);
        }
    }

    private sealed class FakeVideoSource : IVideoSource
    {
        private readonly Queue<byte> _frames = new();
        public List<Mat> Issued { get; } = [];
        public int StartCalls { get; private set; }
        public int StopCalls { get; private set; }

        public void Enqueue(byte value) => _frames.Enqueue(value);

        public bool Start() { StartCalls++; return true; }
        public bool Stop() { StopCalls++; return true; }

        public Mat? GetFrame(ColorType? color = null)
        {
            if (_frames.Count == 0)
                return null;
            var frame = new Mat(8, 8, MatType.CV_8UC1, Scalar.All(_frames.Dequeue()));
            Issued.Add(frame);
            return frame;
        }

        public WaitHandle[] GetFrameWaitHandles() => [];
        public void Dispose() { }
    }
}
