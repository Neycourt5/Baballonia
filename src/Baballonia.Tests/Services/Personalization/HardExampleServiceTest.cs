using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.events;
using Baballonia.Services.Inference;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Turning a flagged moment into a training session on disk.
///
/// The assertion that carries the most weight here is <see cref="OnlyTheCorrectedDimensionIsLabelled"/>.
/// The user pressed a button that says "my mouth was closed" - that is a claim about the jaw and
/// nothing else. They may well have been smiling, or talking, or pulling a face at someone. Writing
/// a neutral label for the other forty-four expressions would manufacture supervision the user never
/// gave, at the highest weight in the system, on frames chosen precisely because the model was
/// already confused. That is how a well-meaning feature teaches a model to deaden a face.
/// </summary>
[TestClass]
public class HardExampleServiceTest
{
    private const int Size = 224;

    private FacePipelineEventBus _bus = null!;
    private HardExampleBuffer _buffer = null!;
    private HardExampleService _service = null!;
    private readonly List<string> _createdSessions = [];

    [TestInitialize]
    public void Initialize()
    {
        _bus = new FacePipelineEventBus();
        _buffer = new HardExampleBuffer(NullLogger.Instance, _bus);
        _service = new HardExampleService(_buffer, BuildModelManager(),
            NullLogger<HardExampleService>.Instance);
    }

    [TestCleanup]
    public void Cleanup()
    {
        _buffer.Dispose();

        foreach (var id in _createdSessions)
        {
            var directory = PersonalizationPaths.SessionDirectory(id);
            if (Directory.Exists(directory))
                Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// A manager with no model loaded. Building the real FacePipelineManager would construct the
    /// whole inference graph, and nothing here needs it - the service only reads its metadata.
    /// </summary>
    private static PersonalModelManager BuildModelManager()
    {
        var pipelineManager = (FacePipelineManager)RuntimeHelpers
            .GetUninitializedObject(typeof(FacePipelineManager));

        typeof(FacePipelineManager)
            .GetField("_pipeline", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(pipelineManager, new FaceProcessingPipeline(new FacePipelineEventBus()));

        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<float>(PersonalModelManager.BlendSetting, It.IsAny<float>(), It.IsAny<bool>()))
            .Returns(1f);

        return new PersonalModelManager(pipelineManager, settings.Object,
            NullLogger<PersonalModelManager>.Instance);
    }

    private static float[] Vector(float jawOpen)
    {
        var values = new float[PersonalizationSchema.ExpressionCount];
        values[PersonalizationSchema.IndexOf("JawOpen")] = jawOpen;
        return values;
    }

    /// <summary>Fills the buffer with frames the way the pipeline would.</summary>
    private void Capture(int frames, float stockJaw = 0.7f, float personalJaw = 0.6f)
    {
        for (var i = 0; i < frames; i++)
        {
            using var frame = new Mat(Size, Size, MatType.CV_8UC1, new Scalar((byte)(i * 7 % 251)));
            var stock = Vector(stockJaw);

            _bus.Publish(new FacePipelineEvents.NewRawExpressionsEvent(frame, stock, DateTime.UtcNow.Ticks));
            _bus.Publish(new FacePipelineEvents.NewCorrectedExpressionsEvent(stock, Vector(personalJaw)));

            Thread.Sleep(40);  // clear the 30 fps capture cap
        }
    }

    private async Task<HardExampleService.FlagResult> FlagAsync(TimeSpan? window = null)
    {
        var result = await _service.FlagAsync(CorrectionKind.MouthClosed, window ?? TimeSpan.FromSeconds(10));
        if (result.SessionId is { } id)
            _createdSessions.Add(id);
        return result;
    }

    private static JsonDocument ReadJson(string path) =>
        JsonDocument.Parse(File.ReadAllText(path));

    [TestMethod]
    public async Task WritesASessionTheTrainerCanDiscover()
    {
        Capture(4);

        var result = await FlagAsync();

        Assert.IsTrue(result.Success, result.Message);
        var directory = PersonalizationPaths.SessionDirectory(result.SessionId!);

        Assert.IsTrue(File.Exists(Path.Combine(directory, "session.json")));
        Assert.IsTrue(File.Exists(Path.Combine(directory, "labels.jsonl")));
        Assert.IsTrue(File.Exists(Path.Combine(directory, "correction.json")));
        Assert.IsTrue(Directory.Exists(Path.Combine(directory, "frames")));
        Assert.AreEqual(result.Frames,
            Directory.EnumerateFiles(Path.Combine(directory, "frames"), "*.jpg").Count());
    }

    [TestMethod]
    public async Task OnlyTheCorrectedDimensionIsLabelled()
    {
        Capture(3);

        var result = await FlagAsync();
        using var document = ReadJson(PersonalizationPaths.CorrectionPath(result.SessionId!));
        var root = document.RootElement;

        var dims = root.GetProperty("CorrectedDims").EnumerateArray().Select(e => e.GetInt32()).ToList();

        CollectionAssert.AreEqual(new[] { PersonalizationSchema.IndexOf("JawOpen") }, dims,
            "the user asserted one thing; the file must not claim more");
        Assert.AreEqual(0f, root.GetProperty("Target").GetSingle());
    }

    [TestMethod]
    public async Task RecordsWhichModelGotItWrong()
    {
        // Without provenance a correction is unattributable: it may describe a failure that a later
        // model already fixed, and retraining on it would chase a ghost.
        Capture(3);

        var result = await FlagAsync();
        using var document = ReadJson(PersonalizationPaths.CorrectionPath(result.SessionId!));
        var root = document.RootElement;

        Assert.AreEqual("mouth_closed", root.GetProperty("Kind").GetString());
        Assert.AreEqual("hard_example", root.GetProperty("Source").GetString());
        Assert.IsTrue(root.TryGetProperty("Model", out _), "model provenance must be recorded");
        Assert.IsTrue(root.TryGetProperty("FlaggedUtc", out _));
    }

    [TestMethod]
    public async Task LabelsCarryBothTheStockAndThePersonalValue()
    {
        // The personal value is the mistake itself. Keeping it makes it possible to ask later
        // whether a retrained model still makes it.
        Capture(3, stockJaw: 0.7f, personalJaw: 0.55f);

        var result = await FlagAsync();
        var lines = File.ReadAllLines(PersonalizationPaths.LabelsPath(result.SessionId!))
            .Where(l => l.Length > 0).ToList();

        Assert.AreEqual(result.Frames, lines.Count);

        using var first = JsonDocument.Parse(lines[0]);
        var jaw = PersonalizationSchema.IndexOf("JawOpen");

        var stock = first.RootElement.GetProperty("stock").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        var personal = first.RootElement.GetProperty("personal").EnumerateArray().Select(e => e.GetSingle()).ToArray();

        Assert.AreEqual(0.7f, stock[jaw], 1e-5);
        Assert.AreEqual(0.55f, personal[jaw], 1e-5);
    }

    [TestMethod]
    public async Task SessionIsTypedAsACorrection()
    {
        Capture(3);

        var result = await FlagAsync();
        using var document = ReadJson(PersonalizationPaths.SessionMetadataPath(result.SessionId!));

        Assert.AreEqual("Correction", document.RootElement.GetProperty("SessionType").GetString());
        Assert.IsTrue(result.SessionId!.EndsWith("_correction", StringComparison.Ordinal),
            "the folder name encodes the type, which is how the environment scan counts it");
    }

    [TestMethod]
    public async Task WrittenFilesHaveNoByteOrderMark()
    {
        // A BOM in labels.jsonl is what broke the very first real training run.
        Capture(3);

        var result = await FlagAsync();
        foreach (var name in new[] { "session.json", "labels.jsonl", "correction.json" })
        {
            var bytes = File.ReadAllBytes(
                Path.Combine(PersonalizationPaths.SessionDirectory(result.SessionId!), name));

            var hasBom = bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF;
            Assert.IsFalse(hasBom, $"{name} starts with a UTF-8 BOM");
        }
    }

    [TestMethod]
    public async Task ShorterWindowSavesFewerFrames()
    {
        Capture(8);  // ~0.32 s of frames at the 40 ms spacing above

        var everything = await FlagAsync(TimeSpan.FromSeconds(10));
        await Task.Delay(3100);  // clear the debounce
        var nothing = await FlagAsync(TimeSpan.Zero);

        Assert.IsTrue(everything.Success);
        Assert.IsFalse(nothing.Success, "a zero-length window covers nothing, so there is nothing to save");
    }

    [TestMethod]
    public async Task FlaggingWithAnEmptyBufferFailsKindly()
    {
        var result = await FlagAsync();

        Assert.IsFalse(result.Success);
        StringAssert.Contains(result.Message, "camera");
    }

    [TestMethod]
    public async Task RepeatedPressesAreDebounced()
    {
        // One mistake noticed twice is one example, not two.
        Capture(4);

        var first = await FlagAsync();
        var second = await FlagAsync();

        Assert.IsTrue(first.Success);
        Assert.IsFalse(second.Success);
        StringAssert.Contains(second.Message, "few seconds");
    }

    [TestMethod]
    public async Task DisabledBufferRefusesRatherThanSavingNothing()
    {
        Capture(4);
        _buffer.Enabled = false;

        var result = await FlagAsync();

        Assert.IsFalse(result.Success);
        StringAssert.Contains(result.Message, "turned off");
    }

    [TestMethod]
    public async Task SavedCorrectionsAreCounted()
    {
        var before = HardExampleService.CountSaved();
        Capture(3);

        await FlagAsync();

        Assert.AreEqual(before + 1, HardExampleService.CountSaved());
    }

    [TestMethod]
    public async Task SavedFramesAreReadableGrayscaleImages()
    {
        Capture(3);

        var result = await FlagAsync();
        var first = Path.Combine(PersonalizationPaths.FramesDirectory(result.SessionId!), "000000.jpg");

        using var image = Cv2.ImRead(first, ImreadModes.Grayscale);

        Assert.IsFalse(image.Empty(), "the saved frame must be a readable image");
        Assert.AreEqual(Size, image.Width);
        Assert.AreEqual(Size, image.Height);
    }
}
