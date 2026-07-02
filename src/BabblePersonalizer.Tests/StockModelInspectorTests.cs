using BabblePersonalizer.Core.Onnx;

namespace BabblePersonalizer.Tests;

[TestClass]
public sealed class StockModelInspectorTests
{
    [TestMethod]
    public async Task RepositoryStockModelContractIsResolvedWithoutBundlingIt()
    {
        var root = FindRepositoryRoot();
        if (root == null) Assert.Inconclusive("Repository root is unavailable in this test environment.");
        var path = Path.Combine(root!, "src", "Baballonia", "faceModel.onnx");
        var result = await new StockModelInspector().InspectAsync(path);
        Assert.IsTrue(result.IsValid, string.Join(Environment.NewLine, result.Errors));
        Assert.IsNotNull(result.Contract);
        Assert.AreEqual("x.1", result.Contract.Input.Name);
        CollectionAssert.AreEqual(new long[] { 1, 1, 224, 224 }, result.Contract.Input.Dimensions.ToArray());
        Assert.AreEqual("1210", result.Contract.Output.Name);
        Assert.AreEqual(45L, result.Contract.Output.Dimensions[^1]);
        Assert.AreEqual(17L, result.Contract.OpsetVersion);
        Assert.AreEqual(45, result.Contract.Parameters.Count);
        Assert.AreEqual("14A907B116C885B1A383F4297569D305E752CE7499B18A6040FD5126A7859719", result.Contract.Sha256);
        Assert.IsTrue(result.Contract.Warnings.Any(x => x.Contains("no blendshape_names", StringComparison.Ordinal)));
    }

    private static string? FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory != null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Baballonia.sln"))) return directory.FullName;
            directory = directory.Parent;
        }
        return null;
    }
}
