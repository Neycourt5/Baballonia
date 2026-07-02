using BabblePersonalizer.Core.Calibration;
using BabblePersonalizer.Core.Camera;
using BabblePersonalizer.Core.Inventory;
using BabblePersonalizer.Core.Models;
using BabblePersonalizer.Core.Storage;
using Microsoft.ML.OnnxRuntime;

namespace BabblePersonalizer.Tests;

[TestClass]
public sealed class PersonalProfileTests
{
    [TestMethod]
    public void SyntheticSessionProducesRobustCurvesSideScalingAndHeldOutValidation()
    {
        var (profile, _, _) = GenerateProfile();
        var jaw = profile.Parameters.Single(x => x.CanonicalName == "JawOpen");
        Assert.AreEqual(.05f, jaw.NeutralMedian, .002f);
        Assert.IsTrue(jaw.Confidence > .7f);
        Assert.IsTrue(jaw.Enabled);
        Assert.IsTrue(jaw.ResponseCurve.Zip(jaw.ResponseCurve.Skip(1), (a, b) => b.Output >= a.Output).All(x => x));
        var left = profile.Parameters.Single(x => x.CanonicalName == "MouthSmileLeft");
        var right = profile.Parameters.Single(x => x.CanonicalName == "MouthSmileRight");
        Assert.IsTrue(left.LeftRightScale > 1);
        Assert.IsTrue(right.LeftRightScale < 1);
        Assert.IsNotNull(profile.Validation);
        Assert.IsTrue(profile.Validation.ImprovementSupported, profile.Validation.Summary);
        Assert.IsTrue(profile.Validation.PersonalizedScore > profile.Validation.StockScore);
    }

    [TestMethod]
    public void CorrectionCanBeAppliedAndDisabledWithoutReordering()
    {
        var (profile, _, _) = GenerateProfile();
        var engine = new PersonalCorrectionEngine(profile);
        var raw = new float[45]; raw[4] = .05f + .8f * .75f;
        var corrected = engine.Apply(raw);
        Assert.AreEqual(45, corrected.Length);
        Assert.IsTrue(Math.Abs(corrected[4] - .75f) < Math.Abs(raw[4] - .75f));
        profile.Parameters[4].Enabled = false;
        corrected = engine.Apply(raw);
        Assert.AreEqual(raw[4], corrected[4]);
        CollectionAssert.AreEqual(raw.Take(4).ToArray(), corrected.Take(4).ToArray());
    }

    [TestMethod]
    public async Task ProfileRoundTripsAndRejectsWrongModelOrCorruptJson()
    {
        var (profile, metadata, camera) = GenerateProfile();
        var root = Path.Combine(Path.GetTempPath(), "BabblePersonalizerTests", Guid.NewGuid().ToString("N"));
        try
        {
            var store = new PersonalProfileStore(new PersonalizerDataPaths(root));
            var path = await store.SaveAsync(profile);
            var loaded = await store.LoadAsync(path, metadata.Model, camera);
            Assert.IsTrue(loaded.Success, loaded.Warning);
            var mismatchContract = CreateContract("DIFFERENT");
            var mismatch = await store.LoadAsync(path, mismatchContract, camera);
            Assert.IsFalse(mismatch.Success);
            var changedCamera = camera with { HorizontalMirror = true };
            var cameraWarning = await store.LoadAsync(path, metadata.Model, changedCamera);
            Assert.IsTrue(cameraWarning.Success); Assert.IsTrue(cameraWarning.CameraMismatch);
            var legacy = Path.Combine(root, "legacy.json");
            var json = await File.ReadAllTextAsync(path);
            await File.WriteAllTextAsync(legacy, json.Replace("\"SchemaVersion\": 2", "\"SchemaVersion\": 1"));
            var legacyResult = await store.LoadAsync(legacy, metadata.Model);
            Assert.IsFalse(legacyResult.Success);
            var corrupt = Path.Combine(root, "broken.json"); await File.WriteAllTextAsync(corrupt, "{not json");
            var corruptResult = await store.LoadAsync(corrupt, metadata.Model);
            Assert.IsFalse(corruptResult.Success);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    private static (PersonalCalibrationProfile Profile, SessionMetadata Metadata, CameraConfiguration Camera) GenerateProfile()
    {
        var contract = CreateContract("MODEL-HASH");
        var camera = new CameraConfiguration(Crop: new CropRegion(0, 0, 256, 256));
        var metadata = new SessionMetadata(2, "synthetic", DateTimeOffset.UtcNow, "test", contract, camera, false, 0);
        var samples = new List<CalibrationSample>(); long sequence = 0;
        for (var i = 0; i < 60; i++)
        {
            var raw = new float[45]; Array.Fill(raw, .05f + (i % 3 - 1) * .0005f);
            samples.Add(Sample(++sequence, "Neutral", "", 0, 0, "Hold", "NeutralHold", raw, false));
        }
        var targets = new Dictionary<string, (int Index, float Scale)>
        {
            ["JawOpen"] = (4, .8f), ["MouthFunnel"] = (10, .75f), ["MouthPucker"] = (11, .7f),
            ["MouthSmileLeft"] = (19, .5f), ["MouthSmileRight"] = (20, .9f)
        };
        var levels = new[] { .25f, .5f, .75f, 1f, .75f, .5f, .25f };
        foreach (var target in targets)
        {
            for (var repetition = 1; repetition <= 4; repetition++)
            for (var levelIndex = 0; levelIndex < levels.Length; levelIndex++)
            for (var frame = 0; frame < 8; frame++)
            {
                var raw = new float[45]; Array.Fill(raw, .05f);
                raw[target.Value.Index] = .05f + target.Value.Scale * levels[levelIndex] + (frame % 3 - 1) * .0005f;
                samples.Add(Sample(++sequence, target.Key, target.Key, repetition, levels[levelIndex],
                    levelIndex < 4 ? "Increasing" : "Decreasing",
                    repetition == 4 ? "ValidationRamp" : "TrainingRamp", raw, repetition == 4));
            }
        }
        var profile = PersonalProfileGenerator.Generate(samples, metadata);
        return (profile, metadata, camera);
    }

    private static CalibrationSample Sample(long sequence, string pose, string target, int repetition,
        float intensity, string direction, string step, float[] raw, bool validation) =>
        new(sequence, sequence / 30d, sequence, pose, target, repetition, intensity, direction,
            step, .98f, .001f, raw, null, validation);

    private static ModelContract CreateContract(string hash) => new()
    {
        Path = "synthetic.onnx", FileSize = 1, Sha256 = hash, OpsetVersion = 17,
        Input = new TensorContract("input", typeof(float).FullName!, new long[] { 1, 1, 224, 224 }),
        Output = new TensorContract("output", typeof(float).FullName!, new long[] { -1, 45 }),
        Metadata = new Dictionary<string, string>(), Parameters = LegacyBaballoniaFaceCatalog.CreateDefinitions(),
        ExpressionListHash = LegacyBaballoniaFaceCatalog.ExpressionListHash, IsCompatible = true,
        Warnings = Array.Empty<string>()
    };
}
