using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// End-to-end checks on dataset recording: real files on disk, real JPEG encoding, real JSONL.
///
/// The important guarantees are that frames and labels can never drift out of correspondence, that
/// the processing tick is not made to wait on disk, and that the duplicate frames produced by the
/// ~100 Hz tick re-serving a 30 fps camera do not inflate the dataset.
///
/// Sessions are written under the real dataset root, so each test cleans up the directory it made.
/// </summary>
[TestClass]
[TestSubject(typeof(DatasetRecorderService))]
public class DatasetRecorderServiceTest
{
    private const int N = PersonalizationSchema.ExpressionCount;

    private FacePipelineEventBus _bus = null!;
    private DatasetRecorderService _recorder = null!;
    private readonly List<string> _createdSessions = [];

    [TestInitialize]
    public void Initialize()
    {
        _bus = new FacePipelineEventBus();
        _recorder = new DatasetRecorderService(_bus, NullLogger<DatasetRecorderService>.Instance);
    }

    [TestCleanup]
    public void Cleanup()
    {
        _recorder.Dispose();

        foreach (var id in _createdSessions)
        {
            var dir = PersonalizationPaths.SessionDirectory(id);
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }

    private string StartSession(SessionType type = SessionType.Neutral)
    {
        var id = _recorder.StartSession(type);
        _createdSessions.Add(id);
        return id;
    }

    /// <summary>Publishes a frame whose pixel value varies, so it is not deduplicated.</summary>
    private void PublishFrame(int seed, float[]? stock = null)
    {
        using var mat = new Mat(224, 224, MatType.CV_8UC1, new Scalar(seed % 250));
        _bus.Publish(new FacePipelineEvents.NewRawExpressionsEvent(
            mat,
            stock ?? Enumerable.Range(0, N).Select(i => (i + seed) / 100f).ToArray(),
            DateTime.UtcNow.Ticks));
    }

    /// <summary>Frames arrive faster than the record cap, so pace them past the rate limiter.</summary>
    private void PublishFrames(int count)
    {
        for (var i = 0; i < count; i++)
        {
            PublishFrame(i + 1);
            Thread.Sleep(1000 / DatasetRecorderService.MaxRecordFps + 5);
        }
    }

    [TestMethod]
    public void NotRecording_IgnoresFramesEntirely()
    {
        PublishFrame(1);

        Assert.IsFalse(_recorder.IsRecording);
        Assert.AreEqual(0, _recorder.FramesWritten);
    }

    [TestMethod]
    public async Task Session_WritesFramesLabelsAndMetadata()
    {
        var id = StartSession(SessionType.Neutral);
        PublishFrames(5);
        var summary = await _recorder.StopSessionAsync();

        Assert.IsNotNull(summary);
        Assert.AreEqual(5, summary.FrameCount);
        Assert.AreEqual(0, summary.DroppedFrames);

        var frames = Directory.GetFiles(PersonalizationPaths.FramesDirectory(id), "*.jpg");
        Assert.AreEqual(5, frames.Length, "One image per recorded frame.");

        var labels = File.ReadAllLines(PersonalizationPaths.LabelsPath(id));
        Assert.AreEqual(5, labels.Length, "One label line per recorded frame.");

        Assert.IsTrue(File.Exists(PersonalizationPaths.SessionMetadataPath(id)));
    }

    [TestMethod]
    public async Task FrameIndicesMatchImageFilenames_SoLabelsCannotDriftFromImages()
    {
        var id = StartSession();
        PublishFrames(4);
        await _recorder.StopSessionAsync();

        var labels = File.ReadAllLines(PersonalizationPaths.LabelsPath(id))
            .Select(l => JsonSerializer.Deserialize<FrameLabel>(l)!)
            .ToList();

        CollectionAssert.AreEqual(new[] { 0, 1, 2, 3 }, labels.Select(l => l.Index).ToArray(),
            "Indices must be contiguous and ordered.");

        foreach (var label in labels)
        {
            var expected = Path.Combine(PersonalizationPaths.FramesDirectory(id), $"{label.Index:D6}.jpg");
            Assert.IsTrue(File.Exists(expected), $"Label {label.Index} has no matching image at {expected}.");
        }
    }

    [TestMethod]
    public async Task StockVectorsArePreservedExactly()
    {
        var id = StartSession();
        var sent = Enumerable.Range(0, N).Select(i => i / (float)(N - 1)).ToArray();

        PublishFrame(7, sent);
        await _recorder.StopSessionAsync();

        var label = JsonSerializer.Deserialize<FrameLabel>(
            File.ReadAllLines(PersonalizationPaths.LabelsPath(id))[0])!;

        Assert.AreEqual(N, label.Stock.Length);
        for (var i = 0; i < N; i++)
            Assert.AreEqual(sent[i], label.Stock[i], 1e-6, $"Stock value {i} was altered.");
    }

    [TestMethod]
    public async Task DuplicateFrames_AreNotRecordedTwice()
    {
        var id = StartSession();

        // The processing tick re-serves the same camera Mat when the camera is slower than 100 Hz.
        var stock = new float[N];
        for (var i = 0; i < 10; i++)
        {
            using var identical = new Mat(224, 224, MatType.CV_8UC1, new Scalar(42));
            _bus.Publish(new FacePipelineEvents.NewRawExpressionsEvent(identical, stock, DateTime.UtcNow.Ticks));
            Thread.Sleep(1000 / DatasetRecorderService.MaxRecordFps + 5);
        }

        var summary = await _recorder.StopSessionAsync();

        Assert.AreEqual(1, summary!.FrameCount,
            "Identical consecutive frames must be deduplicated rather than inflating the dataset.");
    }

    [TestMethod]
    public async Task CaptureRateIsCappedEvenWhenFramesArriveFast()
    {
        var id = StartSession();

        // Burst 200 distinct frames with no pacing, as a fast camera plus fast tick would.
        for (var i = 0; i < 200; i++)
            PublishFrame(i + 1);

        var summary = await _recorder.StopSessionAsync();

        Assert.IsTrue(summary!.FrameCount < 200,
            $"Rate cap should reject most of a fast burst, recorded {summary.FrameCount}.");
        Assert.IsTrue(summary.FrameCount >= 1, "At least the first frame should be kept.");
    }

    [TestMethod]
    public void PublishingIsCheap_DoesNotBlockOnDiskWork()
    {
        StartSession();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (var i = 0; i < 50; i++)
            PublishFrame(i + 1);
        sw.Stop();

        // 50 publishes only clone+enqueue (most are rejected by the rate cap). Encoding 50 JPEGs
        // synchronously would take far longer than this bound.
        Assert.IsTrue(sw.ElapsedMilliseconds < 500,
            $"Event handler must not do disk work on the processing tick; took {sw.ElapsedMilliseconds} ms.");
    }

    [TestMethod]
    public async Task Metadata_RecordsSchemaGeometryAndMeasuredFps()
    {
        var id = StartSession(SessionType.Speech);
        PublishFrames(4);
        await _recorder.StopSessionAsync();

        var metadata = JsonSerializer.Deserialize<SessionMetadata>(
            File.ReadAllText(PersonalizationPaths.SessionMetadataPath(id)))!;

        Assert.AreEqual("Speech", metadata.SessionType);
        Assert.AreEqual(PersonalizationSchema.Sha256, metadata.ExpressionSchemaSha256,
            "Trainer relies on this to refuse data recorded against a different schema.");
        CollectionAssert.AreEqual(
            PersonalizationSchema.ExpressionNames.ToArray(), metadata.ExpressionNames.ToArray());
        Assert.AreEqual(224, metadata.ImageWidth);
        Assert.AreEqual(224, metadata.ImageHeight);
        Assert.AreEqual(4, metadata.FrameCount);
        Assert.IsTrue(metadata.EffectiveFps is > 0, "Unique-fps must be measured for diagnostics.");
        Assert.IsFalse(string.IsNullOrWhiteSpace(metadata.EndedUtc));
    }

    [TestMethod]
    public async Task RecordedImagesAreReadableGrayscaleFrames()
    {
        var id = StartSession();
        PublishFrames(2);
        await _recorder.StopSessionAsync();

        var first = Directory.GetFiles(PersonalizationPaths.FramesDirectory(id), "*.jpg").Order().First();
        using var decoded = Cv2.ImRead(first, ImreadModes.Grayscale);

        Assert.IsFalse(decoded.Empty(), "Recorded frame must decode.");
        Assert.AreEqual(224, decoded.Width);
        Assert.AreEqual(224, decoded.Height);
    }

    [TestMethod]
    public void StartingTwice_IsRejected()
    {
        StartSession();
        Assert.ThrowsExactly<InvalidOperationException>(() => _recorder.StartSession(SessionType.Neutral));
    }

    [TestMethod]
    public async Task StopWithoutStart_ReturnsNull()
    {
        Assert.IsNull(await _recorder.StopSessionAsync());
    }

    [TestMethod]
    public async Task CueMetadata_IsStampedWhenGuided()
    {
        var cue = new FrameLabel.CueLabel
        {
            Id = "SmileHold50",
            Phase = "hold",
            Dims = [19, 20],
            Target = BuildTarget(0.5f, 19, 20),
            Level = 0.5f,
            Repetition = 1,
            Source = "avatar"
        };

        var recorder = new DatasetRecorderService(
            _bus, NullLogger<DatasetRecorderService>.Instance, new StubCueSource(cue));

        var id = recorder.StartSession(SessionType.Guided);
        _createdSessions.Add(id);

        PublishFrames(2);
        await recorder.StopSessionAsync();
        recorder.Dispose();

        var label = JsonSerializer.Deserialize<FrameLabel>(
            File.ReadAllLines(PersonalizationPaths.LabelsPath(id))[0])!;

        Assert.IsNotNull(label.Cue, "Guided frames must carry the commanded target as supervision.");
        Assert.AreEqual("SmileHold50", label.Cue.Id);
        Assert.AreEqual("avatar", label.Cue.Source);
        CollectionAssert.AreEqual(new[] { 19, 20 }, label.Cue.Dims.ToArray());
        Assert.AreEqual(0.5f, label.Cue.Target[19], 1e-6);
        Assert.AreEqual(0f, label.Cue.Target[4], 1e-6);
    }

    private static float[] BuildTarget(float value, params int[] dims)
    {
        var v = new float[N];
        foreach (var d in dims) v[d] = value;
        return v;
    }

    private sealed class StubCueSource(FrameLabel.CueLabel cue) : IReadOnlyCueStateSource
    {
        public FrameLabel.CueLabel? CurrentCue() => cue;
    }
}
