using BabblePersonalizer.Core.Camera;
using BabblePersonalizer.Core.Inventory;
using BabblePersonalizer.Core.Models;
using BabblePersonalizer.Core.Storage;
using System.Diagnostics;

namespace BabblePersonalizer.Tests;

[TestClass]
public sealed class CalibrationSessionWriterTests
{
    [TestMethod]
    public async Task BoundedWriterNeverBlocksProducerAndCancellationIsSafe()
    {
        var root = Path.Combine(Path.GetTempPath(), "BabblePersonalizerWriterTests", Guid.NewGuid().ToString("N"));
        try
        {
            var model = new ModelContract
            {
                Path = "synthetic.onnx", FileSize = 1, Sha256 = "HASH", OpsetVersion = 17,
                Input = new TensorContract("input", typeof(float).FullName!, new long[] { 1, 1, 224, 224 }),
                Output = new TensorContract("output", typeof(float).FullName!, new long[] { -1, 45 }),
                Metadata = new Dictionary<string, string>(), Parameters = LegacyBaballoniaFaceCatalog.CreateDefinitions(),
                ExpressionListHash = LegacyBaballoniaFaceCatalog.ExpressionListHash,
                IsCompatible = true, Warnings = Array.Empty<string>()
            };
            var metadata = new SessionMetadata(2, "queue-test", DateTimeOffset.UtcNow, "test", model,
                new CameraConfiguration(), false, 0);
            await using var writer = new CalibrationSessionWriter(new PersonalizerDataPaths(root), metadata, 1);
            var values = new float[45]; var accepted = 0; var timer = Stopwatch.StartNew();
            for (var i = 0; i < 50_000; i++)
                if (writer.TryWrite(new CalibrationSample(i, i / 30d, i, "pose", "JawOpen", 1,
                        .5f, "Increasing", "TrainingRamp", 1, 0, values))) accepted++;
            timer.Stop();
            Assert.IsTrue(timer.Elapsed < TimeSpan.FromSeconds(5));
            Assert.AreEqual(50_000L, accepted + writer.DroppedSamples);
            Assert.IsTrue(writer.DroppedSamples > 0);
            await writer.CancelAsync();
            Assert.IsFalse(writer.TryWrite(new CalibrationSample(0, 0, 0, "", "", 0, 0,
                "", "", 0, 0, values)));
            Assert.IsTrue(File.Exists(Path.Combine(writer.SessionDirectory, "diagnostics.json")));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
