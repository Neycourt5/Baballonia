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
        Assert.AreNotEqual(
            TrainingModelChoice.KindForIndex(TrainingModelChoice.OutputOnlyIndex),
            TrainingModelChoice.KindForIndex(TrainingModelChoice.ImageConditionedIndex),
            "Selecting B must not train A.");
    }

    [TestMethod]
    public void AdapterTypesMatchThePythonExporter()
    {
        // These strings are written by training/babble_personal/models.py (adapter_type) and read
        // back out of the ONNX metadata. They are a cross-language contract.
        Assert.AreEqual("output_mlp_v1", TrainingModelChoice.AdapterTypeForIndex(TrainingModelChoice.OutputOnlyIndex));
        Assert.AreEqual("image_residual_v1", TrainingModelChoice.AdapterTypeForIndex(TrainingModelChoice.ImageConditionedIndex));
    }

    [TestMethod]
    public void RoundTrip_SelectionSurvivesAsFarAsTheLabel()
    {
        foreach (var index in new[] { TrainingModelChoice.OutputOnlyIndex, TrainingModelChoice.ImageConditionedIndex })
        {
            var adapterType = TrainingModelChoice.AdapterTypeForIndex(index);
            var label = TrainingModelChoice.DisplayName(adapterType);

            var expected = index == TrainingModelChoice.ImageConditionedIndex ? "Model B" : "Model A";
            StringAssert.StartsWith(label, expected,
                $"Index {index} should surface as {expected}, got '{label}'.");
            Assert.AreNotEqual("unknown model", label);
        }
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
