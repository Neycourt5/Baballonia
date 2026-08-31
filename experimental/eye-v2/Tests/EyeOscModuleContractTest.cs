using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Xml.Linq;
using Baballonia.Services;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using VRCFaceTracking.Baballonia;

namespace Baballonia.Tests.Services;

[TestClass]
public class EyeOscModuleContractTest
{
    private static readonly (string Address, ExpressionMapping Destination)[] ExpectedRoutes =
    [
        ("/LeftEyeX", ExpressionMapping.EyeLeftX),
        ("/LeftEyeY", ExpressionMapping.EyeLeftY),
        ("/LeftEyeLid", ExpressionMapping.EyeLeftLid),
        ("/RightEyeX", ExpressionMapping.EyeRightX),
        ("/RightEyeY", ExpressionMapping.EyeRightY),
        ("/RightEyeLid", ExpressionMapping.EyeRightLid),
        ("/LeftEyeWiden", ExpressionMapping.EyeLeftWiden),
        ("/LeftEyeSquint", ExpressionMapping.EyeLeftSquint),
        ("/RightEyeWiden", ExpressionMapping.EyeRightWiden),
        ("/RightEyeSquint", ExpressionMapping.EyeRightSquint),
    ];

    [TestMethod]
    public void SenderAndModuleShareAllTenExactEyeAddressesInFinalVectorOrder()
    {
        Assert.AreEqual(ExpectedRoutes.Length, ParameterSenderService.EyeExpressionOrder.Count);
        Assert.AreEqual(ExpectedRoutes.Length, EyeExpressionRouter.Routes.Count);

        for (var i = 0; i < ExpectedRoutes.Length; i++)
        {
            Assert.AreEqual(ExpectedRoutes[i].Address, ParameterSenderService.EyeExpressionOrder[i].Address);
            Assert.AreEqual(ExpectedRoutes[i].Address, EyeExpressionRouter.Routes[i].Address);
            Assert.AreEqual(ExpectedRoutes[i].Destination, EyeExpressionRouter.Routes[i].Destination);
        }
    }

    [TestMethod]
    public void ModuleRoutesEveryEyeChannelToOnlyItsExactStorageSlot()
    {
        for (var i = 0; i < ExpectedRoutes.Length; i++)
        {
            var destination = Enumerable.Repeat(-1f, 12).ToArray();
            var value = (i + 1) / 20f;

            Assert.IsTrue(EyeExpressionRouter.TryApply(
                ExpectedRoutes[i].Address,
                value,
                destination));

            for (var slot = 0; slot < destination.Length; slot++)
            {
                var expected = slot == (int)ExpectedRoutes[i].Destination ? value : -1f;
                Assert.AreEqual(expected, destination[slot], 1e-6,
                    $"{ExpectedRoutes[i].Address} wrote the wrong module slot {slot}");
            }
        }
    }

    [TestMethod]
    public void ModulePreservesInstalledCaseInsensitiveEyeAddressCompatibility()
    {
        var destination = new float[12];

        Assert.IsTrue(EyeExpressionRouter.TryApply("/lefteyesquint", 0.42f, destination));
        Assert.AreEqual(0.42f, destination[(int)ExpressionMapping.EyeLeftSquint], 1e-6);
    }

    [TestMethod]
    public void PackagedMetadataIdentifiesTheCurrentLocalForkAndEnablesEyes()
    {
        var packageDirectory = Path.Combine(AppContext.BaseDirectory, "ModulePackage");
        using var manifest = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(packageDirectory, "module.json")));
        using var config = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(packageDirectory, "BabbleConfig.json")));

        var root = manifest.RootElement;
        Assert.AreEqual("3.2.1", root.GetProperty("Version").GetString());
        Assert.IsTrue(root.GetProperty("IsLocal").GetBoolean());
        Assert.IsTrue(root.GetProperty("ModuleName").GetString()!.Contains("HOME Fork Local"));
        Assert.AreEqual("VRCFaceTracking.Baballonia.dll", root.GetProperty("DllFileName").GetString());

        Assert.AreEqual(8888, config.RootElement.GetProperty("Port").GetInt32());
        Assert.IsTrue(config.RootElement.GetProperty("IsEyeSupported").GetBoolean());

        var project = XDocument.Load(Path.Combine(
            packageDirectory,
            "VRCFaceTracking.Baballonia.csproj"));
        static string Property(XDocument document, string name) => document
            .Descendants()
            .First(element => element.Name.LocalName == name)
            .Value;

        Assert.AreEqual("3.2.1", Property(project, "Version"));
        Assert.AreEqual(
            "3.2.1-local-baballonia-home-fork",
            Property(project, "InformationalVersion"));
        Assert.AreEqual(
            "VRCFaceTracking.Baballonia-3.2.1-local.zip",
            Property(project, "ModulePackageName"));
    }

    [TestMethod]
    public void InstalledModuleInspectorReadsManifestAndHashesExactDll()
    {
        var directory = Path.Combine(Path.GetTempPath(), "BaballoniaModuleIdentityTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            File.WriteAllText(Path.Combine(directory, "module.json"),
                "{\"Version\":\"3.2.1\",\"IsLocal\":true,\"ModuleName\":\"HOME local\"}");
            File.WriteAllBytes(Path.Combine(directory, VrcftBaballoniaModuleInspector.DllName),
                [1, 2, 3, 4]);

            var identity = VrcftBaballoniaModuleInspector.Inspect(directory);

            Assert.IsTrue(identity.Found);
            Assert.AreEqual("3.2.1", identity.Version);
            Assert.AreEqual(true, identity.IsLocal);
            Assert.AreEqual("HOME local", identity.ModuleName);
            Assert.AreEqual(64, identity.Sha256!.Length);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
