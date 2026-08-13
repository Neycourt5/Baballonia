using System;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services.Inference;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Covers the rules that decide whether a personal model is allowed to run.
///
/// Every rejection path must end in plain stock behavior with an explanation - never a crash, and
/// never a model quietly applying corrections to the wrong expressions. The schema check is the
/// one that matters most: the failure it prevents looks like erratic tracking, not like an error.
/// </summary>
[TestClass]
[TestSubject(typeof(PersonalModelManager))]
public class PersonalModelValidationTest
{
    private static string AssetPath(string name) =>
        Path.Combine(AppContext.BaseDirectory, "Assets", "PersonalModels", name);

    /// <summary>
    /// Builds a manager over an uninitialized FacePipelineManager: its constructor would otherwise
    /// build the whole inference graph, and these tests only need SetCorrector to be callable.
    /// </summary>
    private static (PersonalModelManager manager, FaceProcessingPipeline pipeline) BuildManager(
        string? modelPath, bool enabled = true, float blend = 1f)
    {
        var pipelineManager = (FacePipelineManager)RuntimeHelpers
            .GetUninitializedObject(typeof(FacePipelineManager));

        var pipeline = new FaceProcessingPipeline(new Baballonia.Services.FacePipelineEventBus());
        typeof(FacePipelineManager)
            .GetField("_pipeline", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(pipelineManager, pipeline);

        var settings = new Mock<ILocalSettingsService>();
        settings.Setup(s => s.ReadSetting<bool>(PersonalModelManager.EnabledSetting, It.IsAny<bool>(), It.IsAny<bool>()))
            .Returns(enabled);
        settings.Setup(s => s.ReadSetting<string>(PersonalModelManager.PathSetting, It.IsAny<string>(), It.IsAny<bool>()))
            .Returns(modelPath ?? "");
        settings.Setup(s => s.ReadSetting<float>(PersonalModelManager.BlendSetting, It.IsAny<float>(), It.IsAny<bool>()))
            .Returns(blend);

        var manager = new PersonalModelManager(
            pipelineManager, settings.Object, NullLogger<PersonalModelManager>.Instance);

        return (manager, pipeline);
    }

    [TestMethod]
    public async Task ValidModel_IsInstalledOnThePipeline()
    {
        var (manager, pipeline) = BuildManager(AssetPath("validAdapter.onnx"));
        using (manager)
        {
            var result = await manager.ReloadAsync();

            Assert.IsTrue(result.Success, result.Message);
            Assert.IsTrue(manager.IsActive);
            Assert.IsNotNull(pipeline.Corrector, "A valid model must actually be installed.");
        }
    }

    [TestMethod]
    public async Task SchemaMismatch_IsRejectedAndLeavesStockBehavior()
    {
        var (manager, pipeline) = BuildManager(AssetPath("schemaMismatchAdapter.onnx"));
        using (manager)
        {
            var result = await manager.ReloadAsync();

            Assert.IsFalse(result.Success);
            StringAssert.Contains(result.Message, "different expression schema");
            Assert.IsNull(pipeline.Corrector,
                "A model trained on another schema would map corrections onto the wrong expressions.");
        }
    }

    [TestMethod]
    public async Task MissingMetadata_IsRejected()
    {
        var (manager, pipeline) = BuildManager(AssetPath("noMetadataAdapter.onnx"));
        using (manager)
        {
            var result = await manager.ReloadAsync();

            Assert.IsFalse(result.Success);
            StringAssert.Contains(result.Message, "expression_schema_sha256");
            Assert.IsNull(pipeline.Corrector);
        }
    }

    [TestMethod]
    public async Task MissingFile_FallsBackToStockWithoutThrowing()
    {
        var (manager, pipeline) = BuildManager(Path.Combine(Path.GetTempPath(), "definitely-not-here.onnx"));
        using (manager)
        {
            var result = await manager.ReloadAsync();

            Assert.IsFalse(result.Success);
            StringAssert.Contains(result.Message, "No personal model");
            Assert.IsNull(pipeline.Corrector);
        }
    }

    [TestMethod]
    public async Task CorruptFile_FallsBackToStockWithoutThrowing()
    {
        var corrupt = Path.Combine(Path.GetTempPath(), $"corrupt-{Guid.NewGuid():N}.onnx");
        await File.WriteAllTextAsync(corrupt, "this is not a protobuf");

        try
        {
            var (manager, pipeline) = BuildManager(corrupt);
            using (manager)
            {
                var result = await manager.ReloadAsync();

                Assert.IsFalse(result.Success);
                Assert.IsNull(pipeline.Corrector);
            }
        }
        finally
        {
            File.Delete(corrupt);
        }
    }

    [TestMethod]
    public async Task Disabled_DoesNotLoadAnything()
    {
        var (manager, pipeline) = BuildManager(AssetPath("validAdapter.onnx"), enabled: false);
        using (manager)
        {
            var result = await manager.ReloadAsync();

            Assert.IsTrue(result.Success, "Disabling is a normal state, not an error.");
            Assert.IsNull(pipeline.Corrector);
            Assert.IsFalse(manager.IsActive);
        }
    }

    [TestMethod]
    public async Task DisablingAfterLoad_RemovesTheCorrectorImmediately()
    {
        var (manager, pipeline) = BuildManager(AssetPath("validAdapter.onnx"));
        using (manager)
        {
            await manager.ReloadAsync();
            Assert.IsNotNull(pipeline.Corrector);

            manager.Uninstall();

            Assert.IsNull(pipeline.Corrector,
                "Turning personalization off must restore stock behavior on the next frame.");
        }
    }

    [TestMethod]
    public async Task ReloadTwice_DoesNotLeakOrDoubleInstall()
    {
        var (manager, pipeline) = BuildManager(AssetPath("validAdapter.onnx"));
        using (manager)
        {
            await manager.ReloadAsync();
            var first = pipeline.Corrector;

            await manager.ReloadAsync();
            var second = pipeline.Corrector;

            Assert.IsNotNull(second);
            Assert.AreNotSame(first, second, "Reload should install a fresh session.");
        }
    }

    [TestMethod]
    public async Task BlendSettingIsAppliedToTheLoadedModel()
    {
        var (manager, pipeline) = BuildManager(AssetPath("validAdapter.onnx"), blend: 0.25f);
        using (manager)
        {
            await manager.ReloadAsync();

            Assert.IsNotNull(pipeline.Corrector);
            Assert.AreEqual(0.25f, pipeline.Corrector.Blend, 1e-6);
        }
    }
}
