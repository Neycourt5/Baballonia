using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The rolling buffer that makes "save what just went wrong" possible.
///
/// Two properties matter more than the rest and are tested hardest.
///
/// The first is that a saved frame is paired with the prediction that was actually made from it. The
/// pipeline publishes the raw event and then the corrected event on the same tick, so "attach to the
/// newest entry" is only right when that entry came from this tick - and it does not, whenever the
/// frame was rejected as a duplicate or over-rate. Getting this wrong produces evidence that looks
/// perfectly valid and blames the wrong picture.
///
/// The second is that this thing runs on the inference tick, so it must be cheap and must never
/// throw into the pipeline.
/// </summary>
[TestClass]
public class HardExampleBufferTest
{
    private const int Size = 224;
    private static readonly int Count = PersonalizationSchema.ExpressionCount;

    private FacePipelineEventBus _bus = null!;
    private HardExampleBuffer _buffer = null!;

    [TestInitialize]
    public void Initialize()
    {
        _bus = new FacePipelineEventBus();
        _buffer = new HardExampleBuffer(NullLogger.Instance, _bus);
    }

    [TestCleanup]
    public void Cleanup() => _buffer.Dispose();

    /// <summary>A frame whose pixels differ per call, so the duplicate filter does not reject it.</summary>
    private static Mat Frame(byte fill) => new(Size, Size, MatType.CV_8UC1, new Scalar(fill));

    private static float[] Vector(float jawOpen)
    {
        var values = new float[Count];
        values[PersonalizationSchema.IndexOf("JawOpen")] = jawOpen;
        return values;
    }

    /// <summary>Publishes one tick's worth of events the way FaceProcessingPipeline does.</summary>
    private void Tick(byte fill, float stockJaw, float? correctedJaw = null)
    {
        using var frame = Frame(fill);
        var stock = Vector(stockJaw);

        _bus.Publish(new FacePipelineEvents.NewRawExpressionsEvent(frame, stock, DateTime.UtcNow.Ticks));

        if (correctedJaw is { } corrected)
            _bus.Publish(new FacePipelineEvents.NewCorrectedExpressionsEvent(stock, Vector(corrected)));
    }

    /// <summary>The 30 fps cap rejects frames offered faster than that.</summary>
    private static void WaitForNextFrameSlot() => Thread.Sleep(40);

    [TestMethod]
    public void CapturesFramesFromThePipeline()
    {
        Tick(10, 0.5f);
        WaitForNextFrameSlot();
        Tick(20, 0.6f);

        Assert.AreEqual(2, _buffer.Count);
    }

    [TestMethod]
    public void IdenticalFramesAreNotStoredTwice()
    {
        // The 10 ms tick re-serves the same camera Mat when the camera is slower.
        Tick(10, 0.5f);
        WaitForNextFrameSlot();
        Tick(10, 0.5f);

        Assert.AreEqual(1, _buffer.Count, "a repeated frame is not new evidence");
    }

    [TestMethod]
    public void FramesOfferedFasterThanTheCapAreDropped()
    {
        Tick(10, 0.5f);
        Tick(20, 0.6f);  // immediately after: inside the 33 ms window
        Tick(30, 0.7f);

        Assert.AreEqual(1, _buffer.Count);
    }

    [TestMethod]
    public void CorrectedValuesAttachToTheFrameTheyCameFrom()
    {
        Tick(10, 0.8f, correctedJaw: 0.05f);

        var snapshot = _buffer.Take(TimeSpan.FromSeconds(10));
        var frame = snapshot.Frames.Single();
        var jaw = PersonalizationSchema.IndexOf("JawOpen");

        Assert.AreEqual(0.8f, frame.Stock[jaw], 1e-6, "stock value must be the one recorded");
        Assert.IsNotNull(frame.Personal);
        Assert.AreEqual(0.05f, frame.Personal[jaw], 1e-6);
    }

    [TestMethod]
    public void ADroppedFrameDoesNotMisattributeItsCorrectionToTheStoredOne()
    {
        // The failure this guards: tick 1 is stored, tick 2's frame is rejected as a duplicate, and
        // tick 2's corrected event then overwrites tick 1's - pairing a prediction with a picture it
        // was not computed from. The saved evidence would look valid and blame the wrong frame.
        Tick(10, 0.8f, correctedJaw: 0.05f);
        WaitForNextFrameSlot();
        Tick(10, 0.9f, correctedJaw: 0.99f);  // same pixels: rejected as duplicate

        var frame = _buffer.Take(TimeSpan.FromSeconds(10)).Frames.Single();
        var jaw = PersonalizationSchema.IndexOf("JawOpen");

        Assert.AreEqual(0.05f, frame.Personal![jaw], 1e-6,
            "the rejected tick's correction must not overwrite the stored frame's");
    }

    [TestMethod]
    public void FramesWithNoCorrectorHaveNoPersonalValues()
    {
        // Personalization off: the corrected event never fires. Null is the honest answer here -
        // zeros would read as "the model output nothing", which is a different claim.
        Tick(10, 0.8f);

        var frame = _buffer.Take(TimeSpan.FromSeconds(10)).Frames.Single();

        Assert.IsNull(frame.Personal);
    }

    [TestMethod]
    public void TakeReturnsOnlyFramesInsideTheWindow()
    {
        Tick(10, 0.5f);
        WaitForNextFrameSlot();
        Tick(20, 0.6f);

        var everything = _buffer.Take(TimeSpan.FromSeconds(10));
        var nothing = _buffer.Take(TimeSpan.Zero);

        Assert.AreEqual(2, everything.Frames.Count);
        Assert.AreEqual(0, nothing.Frames.Count, "a zero-length window covers no frames");
    }

    [TestMethod]
    public void TakeReturnsFramesOldestFirst()
    {
        Tick(10, 0.1f);
        WaitForNextFrameSlot();
        Tick(20, 0.2f);
        WaitForNextFrameSlot();
        Tick(30, 0.3f);

        var jaw = PersonalizationSchema.IndexOf("JawOpen");
        var order = _buffer.Take(TimeSpan.FromSeconds(10)).Frames.Select(f => f.Stock[jaw]).ToList();

        CollectionAssert.AreEqual(new[] { 0.1f, 0.2f, 0.3f }, order,
            "frames must come back in the order they happened");
    }

    [TestMethod]
    public void TheRingWrapsAndKeepsTheMostRecentFrames()
    {
        var bus = new FacePipelineEventBus();
        using var small = new HardExampleBuffer(NullLogger.Instance, bus, capacityFrames: 3);
        var jaw = PersonalizationSchema.IndexOf("JawOpen");

        for (var i = 1; i <= 5; i++)
        {
            using var frame = Frame((byte)(i * 10));
            var stock = Vector(i / 10f);
            bus.Publish(new FacePipelineEvents.NewRawExpressionsEvent(frame, stock, DateTime.UtcNow.Ticks));
            WaitForNextFrameSlot();
        }

        var frames = small.Take(TimeSpan.FromSeconds(30)).Frames;

        Assert.AreEqual(3, frames.Count);
        CollectionAssert.AreEqual(new[] { 0.3f, 0.4f, 0.5f }, frames.Select(f => f.Stock[jaw]).ToList(),
            "a full ring keeps the newest frames and discards the oldest");
    }

    [TestMethod]
    public void SnapshotsAreDetachedCopies()
    {
        Tick(10, 0.5f);
        var snapshot = _buffer.Take(TimeSpan.FromSeconds(10));

        // Overwrite the ring several times over.
        for (var i = 0; i < 5; i++)
        {
            WaitForNextFrameSlot();
            Tick((byte)(100 + i), 0.9f);
        }

        var jaw = PersonalizationSchema.IndexOf("JawOpen");
        Assert.AreEqual(0.5f, snapshot.Frames[0].Stock[jaw], 1e-6,
            "a snapshot must not change under the caller while it is being written out");
    }

    [TestMethod]
    public void DisabledBufferCapturesNothing()
    {
        _buffer.Enabled = false;

        Tick(10, 0.5f);
        WaitForNextFrameSlot();
        Tick(20, 0.6f);

        Assert.AreEqual(0, _buffer.Count);
    }

    [TestMethod]
    public void DisablingReleasesMemoryAndEnablingStartsFresh()
    {
        Tick(10, 0.5f);
        Assert.IsGreaterThan(0L, _buffer.AllocatedBytes);

        _buffer.Enabled = false;

        Assert.AreEqual(0, _buffer.Count);
        Assert.AreEqual(0L, _buffer.AllocatedBytes,
            "turning quick correction off must release its preallocated frame ring");

        _buffer.Enabled = true;
        WaitForNextFrameSlot();
        Tick(20, 0.6f);

        Assert.AreEqual(1, _buffer.Count);
        Assert.IsGreaterThan(0L, _buffer.AllocatedBytes);
    }

    [TestMethod]
    public void ClearDropsEverythingHeld()
    {
        Tick(10, 0.5f);
        WaitForNextFrameSlot();
        Tick(20, 0.6f);

        _buffer.Clear();

        Assert.AreEqual(0, _buffer.Count);
    }

    [TestMethod]
    public void MemoryUseIsBoundedAndPredictable()
    {
        Tick(10, 0.5f);

        // 300 frames x (224*224 + 2*45*4 bytes) ~= 15 MB, allocated once.
        var expected = (long)HardExampleBuffer.DefaultCapacityFrames * (Size * Size + 2 * 45 * sizeof(float));

        Assert.AreEqual(expected, _buffer.AllocatedBytes);
        Assert.IsTrue(_buffer.AllocatedBytes < 20L * 1024 * 1024,
            "the buffer must stay small enough to leave on by default");
    }

    [TestMethod]
    public void CaptureCostIsNegligibleOnTheProcessingTick()
    {
        // Budget is the 10 ms tick, already carrying stock face and eye inference. A ~50 KB copy
        // should not be visible in it; this asserts the order of magnitude, not a precise figure.
        using var frame = Frame(42);
        var stock = Vector(0.5f);

        // Warm up JIT and force the first allocation.
        _bus.Publish(new FacePipelineEvents.NewRawExpressionsEvent(frame, stock, DateTime.UtcNow.Ticks));

        const int iterations = 200;
        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < iterations; i++)
        {
            // Vary the pixels so the duplicate filter does not short-circuit the copy under test.
            using var varying = Frame((byte)(i % 251));
            _buffer.Clear();
            _bus.Publish(new FacePipelineEvents.NewRawExpressionsEvent(varying, stock, DateTime.UtcNow.Ticks));
        }
        stopwatch.Stop();

        var perFrameMs = stopwatch.Elapsed.TotalMilliseconds / iterations;
        Assert.IsTrue(perFrameMs < 1.0,
            $"capture cost {perFrameMs:F3} ms per frame; budget is a 10 ms tick");
    }

    [TestMethod]
    public void AnEmptyBufferSnapshotsCleanly()
    {
        var snapshot = _buffer.Take(TimeSpan.FromSeconds(10));

        Assert.AreEqual(0, snapshot.Frames.Count);
    }
}
