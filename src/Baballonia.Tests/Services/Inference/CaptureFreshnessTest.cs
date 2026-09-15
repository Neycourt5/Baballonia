using System;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.SDK;
using Baballonia.Services.Inference.VideoSources;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Inference;

[TestClass]
public class CaptureFreshnessTest
{
    private sealed class FakeCapture(string source) : Capture(source, NullLogger<FakeCapture>.Instance)
    {
        public int StopCount { get; private set; }
        public int DisposeCount { get; private set; }
        public TimeSpan? ContentAge { get; set; }
        public override TimeSpan? TimeSinceLastContentChange => ContentAge;

        public void Publish(Mat frame) => SetRawMat(frame);
        public void MarkNotReady() => IsReady = false;

        public override Task<bool> StartCapture()
        {
            IsReady = true;
            return Task.FromResult(true);
        }

        public override Task<bool> StopCapture()
        {
            StopCount++;
            IsReady = false;
            return Task.FromResult(true);
        }

        public override void Dispose()
        {
            DisposeCount++;
            base.Dispose();
        }
    }

    [TestMethod]
    public void AcquiredReceiptIdentityBelongsToTheOwnedFrameAndDoesNotAdvanceInCache()
    {
        using var capture = new FakeCapture("test");
        capture.Publish(new Mat(4,4,MatType.CV_8UC1,new Scalar(10)));
        using var first = capture.AcquireRawMat(out var firstSequence,out var firstReceipt);
        Assert.AreEqual(1L,firstSequence);
        Assert.IsTrue(firstReceipt > 0);
        capture.Publish(new Mat(4,4,MatType.CV_8UC1,new Scalar(20)));
        using var second = capture.AcquireRawMat(out var secondSequence,out var secondReceipt);
        Assert.AreEqual(2L,secondSequence);
        Assert.IsTrue(secondReceipt >= firstReceipt);
        Assert.AreEqual((byte)10,first!.At<byte>(0,0));
        Assert.AreEqual((byte)20,second!.At<byte>(0,0));
        Assert.IsNull(capture.AcquireRawMat(out var emptySequence,out var emptyReceipt));
        Assert.AreEqual(0L,emptySequence);
        Assert.AreEqual(0L,emptyReceipt);
    }

    [TestMethod]
    public void ThereIsNoFrameAgeBeforeTheFirstFrame()
    {
        using var capture = new FakeCapture("test");

        Assert.AreEqual(TimeSpan.MaxValue, capture.TimeSinceLastFrame);
        Assert.AreEqual(0, capture.FramesProduced);
    }

    [TestMethod]
    public void PublishingTheSameMatTwiceRefreshesTheProducerTimestamp()
    {
        var capture = new FakeCapture("test");
        var reused = new Mat(4, 4, MatType.CV_8UC1);

        capture.Publish(reused);
        var firstTimestamp = capture.LastFrameAtTicks;
        Thread.Sleep(20);
        capture.Publish(reused);

        Assert.AreEqual(2, capture.FramesProduced);
        Assert.IsTrue(capture.LastFrameAtTicks > firstTimestamp);
        Assert.IsTrue(capture.TimeSinceLastFrame < TimeSpan.FromMilliseconds(20));

        capture.AcquireRawMat()?.Dispose();
    }

    [TestMethod]
    public void FrameAgeGrowsWhileTheProducerIsSilent()
    {
        var capture = new FakeCapture("test");
        capture.Publish(new Mat(4, 4, MatType.CV_8UC1));

        var immediately = capture.TimeSinceLastFrame;
        Thread.Sleep(50);
        var later = capture.TimeSinceLastFrame;

        Assert.IsTrue(later > immediately);
        Assert.IsTrue(later >= TimeSpan.FromMilliseconds(30));

        capture.AcquireRawMat()?.Dispose();
    }

    [TestMethod]
    public void HealthyFrameAgeIncludesAStuckContentSignal()
    {
        var capture = new FakeCapture("test");
        using var source = new SingleCameraSource(
            NullLogger<SingleCameraSource>.Instance, capture, "test");
        capture.Publish(new Mat(4, 4, MatType.CV_8UC1));
        capture.ContentAge = TimeSpan.FromSeconds(3);

        Assert.IsTrue(source.TimeSinceLastHealthyFrame >= TimeSpan.FromSeconds(3));
    }

    [TestMethod]
    public void SingleCameraSourceDrainsTheFinalFrameAfterBackendStopsBeingReady()
    {
        var capture = new FakeCapture("test");
        using var source = new SingleCameraSource(
            NullLogger<SingleCameraSource>.Instance, capture, "test");

        capture.Publish(new Mat(4, 4, MatType.CV_8UC1));
        capture.MarkNotReady();

        using var frame = source.GetFrame();
        Assert.IsNotNull(frame);
        Assert.AreEqual(4, frame.Width);
    }

    [TestMethod]
    public void SingleCameraSourceDisposesItsCaptureExactlyOnce()
    {
        var capture = new FakeCapture("test");
        var source = new SingleCameraSource(
            NullLogger<SingleCameraSource>.Instance, capture, "test");

        source.Dispose();
        source.Dispose();

        Assert.AreEqual(1, capture.StopCount);
        Assert.AreEqual(1, capture.DisposeCount);
    }
}
