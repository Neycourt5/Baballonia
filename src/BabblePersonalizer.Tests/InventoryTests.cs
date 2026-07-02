using BabblePersonalizer.Core.Inventory;
using BabblePersonalizer.Core.Models;

namespace BabblePersonalizer.Tests;

[TestClass]
public sealed class InventoryTests
{
    [TestMethod]
    public void LegacyContractHasUniqueStableIdentityForEveryDirectOutput()
    {
        var parameters = LegacyBaballoniaFaceCatalog.CreateDefinitions();
        Assert.AreEqual(45, parameters.Count);
        CollectionAssert.AreEqual(Enumerable.Range(0, 45).ToArray(), parameters.Select(x => x.OutputIndex).ToArray());
        Assert.AreEqual(45, parameters.Select(x => x.CanonicalName).Distinct().Count());
        Assert.IsTrue(parameters.All(x => x.DirectModelOutput && x.Minimum == 0 && x.Maximum == 1));
        Assert.IsTrue(parameters.All(x => !string.IsNullOrWhiteSpace(x.SenderDestination)));
        Assert.IsTrue(parameters.All(x => x.CalibrationStrategy != CalibrationStrategy.UnsupportedPassthrough ||
                                          !string.IsNullOrWhiteSpace(x.PassthroughReason)));
    }

    [TestMethod]
    public void LeftRightPairsAreBidirectionalAndNeverEyeParameters()
    {
        var parameters = LegacyBaballoniaFaceCatalog.CreateDefinitions();
        var pairs = LegacyBaballoniaFaceCatalog.LeftRightPairs(parameters);
        Assert.IsTrue(pairs.Count > 0);
        foreach (var pair in pairs)
        {
            Assert.AreEqual(parameters[pair.Right].CanonicalName, parameters[pair.Left].Counterpart);
            Assert.AreEqual(parameters[pair.Left].CanonicalName, parameters[pair.Right].Counterpart);
            Assert.IsFalse(parameters[pair.Left].CanonicalName.Contains("Eye"));
        }
    }

    [TestMethod]
    public void AmbiguousTongueShapesRequireManualReview()
    {
        var byName = LegacyBaballoniaFaceCatalog.CreateDefinitions().ToDictionary(x => x.CanonicalName);
        foreach (var name in new[] { "TongueSquish", "TongueFlat", "TongueTwistLeft", "TongueTwistRight" })
        {
            Assert.AreEqual(CalibrationStrategy.ManualReviewOnly, byName[name].CalibrationStrategy);
            Assert.IsNotNull(byName[name].PassthroughReason);
        }
    }

    [TestMethod]
    public void UnknownMetadataExpressionIsNondestructivePassthrough()
    {
        var parameter = LegacyBaballoniaFaceCatalog.CreateDefinitions(new[] { "FutureFaceShape" }).Single();
        Assert.AreEqual(FaceParameterType.Passthrough, parameter.ParameterType);
        Assert.AreEqual(CalibrationStrategy.UnsupportedPassthrough, parameter.CalibrationStrategy);
        Assert.IsNull(parameter.SenderDestination);
        Assert.IsNotNull(parameter.PassthroughReason);
    }
}
