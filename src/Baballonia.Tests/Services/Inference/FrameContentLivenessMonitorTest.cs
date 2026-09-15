using System;
using System.Threading;
using Baballonia.OpenCVCapture;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Inference;

[TestClass]
public class FrameContentLivenessMonitorTest
{
    [TestMethod]
    public void RepeatedCachedFrameAgesUntilPixelsChange()
    {
        var monitor = new FrameContentLivenessMonitor(TimeSpan.Zero);
        using var frame = new Mat(32, 32, MatType.CV_8UC1, Scalar.All(20));

        monitor.Observe(frame);
        Thread.Sleep(30);
        monitor.Observe(frame);

        Assert.IsTrue(monitor.TimeSinceLastContentChange >= TimeSpan.FromMilliseconds(20));

        frame.SetTo(Scalar.All(21));
        monitor.Observe(frame);

        Assert.IsTrue(monitor.TimeSinceLastContentChange < TimeSpan.FromMilliseconds(20));
    }

    [TestMethod]
    public void LiveSensorNoiseRefreshesContentLiveness()
    {
        var monitor = new FrameContentLivenessMonitor(TimeSpan.Zero);
        using var frame = new Mat(32, 32, MatType.CV_8UC3, Scalar.All(20));

        monitor.Observe(frame);
        Thread.Sleep(30);
        frame.Set<Vec3b>(10, 10, new Vec3b(20, 21, 20));
        monitor.Observe(frame);

        Assert.IsTrue(monitor.TimeSinceLastContentChange < TimeSpan.FromMilliseconds(20));
    }
}
