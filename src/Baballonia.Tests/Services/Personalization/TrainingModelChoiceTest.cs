using System;
using System.Linq;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The dropdown -> trainer flag -> adapter type -> label chain.
///
/// This is worth testing precisely because breaking it is silent: if picking "B" trained A anyway,
/// the run would succeed, the results screen would look normal, and the only symptom would be a
/// user who cannot tell which model they are running. Each link is asserted, then the whole round
/// trip.
/// </summary>
[TestClass]
[TestSubject(typeof(TrainingModelChoice))]
public class TrainingModelChoiceTest
{
    [TestMethod]
    public void DefaultIndexIsModelA_SoTheExistingFlowIsUnchanged()
    {
        Assert.AreEqual(0, TrainingModelChoice.OutputOnlyIndex,
            "Model A must stay the zero/default selection.");
        Assert.AreEqual("a", TrainingModelChoice.KindForIndex(default),
            "An unset dropdown must train the output-only baseline.");
    }

    [TestMethod]
    public void EachIndexMapsToItsOwnTrainerFlag()
    {
        Assert.AreEqual("a", TrainingModelChoice.KindForIndex(TrainingModelChoice.OutputOnlyIndex));
        Assert.AreEqual("b", TrainingModelChoice.KindForIndex(TrainingModelChoice.ImageConditionedIndex));
        Assert.AreEqual("c", TrainingModelChoice.KindForIndex(TrainingModelChoice.SharedFeaturesIndex));
        CollectionAssert.AreEquivalent(
            new[] { "a", "b", "c" },
            TrainingModelChoice.Options.Select(option => option.Kind).ToArray(),
            "Every selector item must send a distinct trainer flag.");
    }

    [TestMethod]
    public void AdapterTypesMatchThePythonExporter()
    {
        // These strings are written by training/babble_personal/models.py (adapter_type) and read
        // back out of the ONNX metadata. They are a cross-language contract.
        Assert.AreEqual("output_mlp_v1", TrainingModelChoice.AdapterTypeForIndex(TrainingModelChoice.OutputOnlyIndex));
        Assert.AreEqual("image_residual_v1", TrainingModelChoice.AdapterTypeForIndex(TrainingModelChoice.ImageConditionedIndex));
        Assert.AreEqual("embedding_head_v1", TrainingModelChoice.AdapterTypeForIndex(TrainingModelChoice.SharedFeaturesIndex));
    }

    [TestMethod]
    public void RoundTrip_SelectionSurvivesAsFarAsTheLabel()
    {
        foreach (var index in new[]
                 {
                     TrainingModelChoice.OutputOnlyIndex,
                     TrainingModelChoice.ImageConditionedIndex,
                     TrainingModelChoice.SharedFeaturesIndex
                 })
        {
            var adapterType = TrainingModelChoice.AdapterTypeForIndex(index);
            var label = TrainingModelChoice.DisplayName(adapterType);

            var expected = $"Model {(char)('A' + index)}";
            StringAssert.StartsWith(label, expected,
                $"Index {index} should surface as {expected}, got '{label}'.");
            Assert.AreNotEqual("unknown model", label);
        }
    }

    [TestMethod]
    public void SelectorLabelsDescribeTheCurrentABaselineAndExperimentalCHonestly()
    {
        Assert.AreEqual(3, TrainingModelChoice.Options.Count);
        Assert.AreEqual("A — Expressions only (simple)", TrainingModelChoice.Options[0].Label);
        Assert.IsFalse(TrainingModelChoice.Options[0].Label.Contains("recommended"));
        Assert.AreEqual("B — Expressions + camera image (current best)", TrainingModelChoice.Options[1].Label);
        Assert.AreEqual("C — Shared visual features (experimental)", TrainingModelChoice.Options[2].Label);
        StringAssert.Contains(TrainingModelChoice.Options[2].Description, "instead of running a second camera CNN");
    }

    [TestMethod]
    public void TrainCommandGuard_PreservesAButBlocksCWithoutEmbeddings()
    {
        Assert.IsNull(TrainingModelChoice.TrainingUnavailableReason("a", true, false));
        Assert.IsNull(TrainingModelChoice.TrainingUnavailableReason("b", true, false));

        var blocked = TrainingModelChoice.TrainingUnavailableReason("c", true, false);
        Assert.IsNotNull(blocked);
        StringAssert.Contains(blocked, "Prepare Model C");

        Assert.IsNull(TrainingModelChoice.TrainingUnavailableReason("c", true, true));
        Assert.IsNotNull(TrainingModelChoice.TrainingUnavailableReason("future", true, true));
    }

    [TestMethod]
    public void TrainedArchitecturesHaveIndependentPersistentSlots()
    {
        var paths = TrainingModelChoice.Options
            .Select(option => PersonalizationPaths.PersonalModelPath(option.Kind))
            .ToArray();

        Assert.AreEqual(3, paths.Distinct(StringComparer.OrdinalIgnoreCase).Count(),
            "Training B or C must not overwrite the available A model.");
        StringAssert.EndsWith(paths[0], "personalFaceModel-a.onnx");
        StringAssert.EndsWith(paths[1], "personalFaceModel-b.onnx");
        StringAssert.EndsWith(paths[2], "personalFaceModel-c.onnx");
    }

    [TestMethod]
    public void UnrecognizedAdapterTypeIsReportedHonestly()
    {
        Assert.AreEqual("unknown model", TrainingModelChoice.DisplayName(null));
        Assert.AreEqual("unknown model", TrainingModelChoice.DisplayName(""));
        Assert.AreEqual("unknown model", TrainingModelChoice.DisplayName("unknown"));

        // A future adapter should show its raw type rather than being mislabelled as A or B.
        Assert.AreEqual("future_adapter_v9", TrainingModelChoice.DisplayName("future_adapter_v9"));
    }
}
