using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.EyeV2;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

namespace Baballonia.Tests.Services.EyeV2;

[TestClass]
public class EyeV2StoreAndModeTest
{
    private string _directory = null!;
    private string _path = null!;
    private Mock<ILocalSettingsService> _settings = null!;
    private Dictionary<string, object> _values = null!;

    [TestInitialize]
    public void Initialize()
    {
        _directory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "BaballoniaEyeV2Tests", Guid.NewGuid().ToString("N"));
        _path = System.IO.Path.Combine(_directory, EyeV2CalibrationStore.FileName);
        _values = new Dictionary<string, object>();
        _settings = new Mock<ILocalSettingsService>();
        _settings.Setup(x => x.ReadSetting(
                It.IsAny<string>(), It.IsAny<EyeTrackingMode>(), It.IsAny<bool>()))
            .Returns((string key, EyeTrackingMode fallback, bool _) =>
                _values.TryGetValue(key, out var value) ? (EyeTrackingMode)value : fallback);
        _settings.Setup(x => x.SaveSetting(
                It.IsAny<string>(), It.IsAny<EyeTrackingMode>(), It.IsAny<bool>()))
            .Callback((string key, EyeTrackingMode value, bool _) => _values[key] = value);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    [TestMethod]
    public async Task Store_RoundTripsVersionedCalibrationSeparately()
    {
        var store = Store();
        var original = EyeV2CalibrationAndMapperTest.IdentityCalibration();

        await store.SaveAsync(original);

        Assert.IsTrue(store.TryLoad(out var loaded, out var error), error);
        Assert.AreEqual(EyeV2Calibration.CurrentSchemaVersion, loaded!.SchemaVersion);
        Assert.AreEqual(original.Left.Lid, loaded.Left.Lid);
        StringAssert.EndsWith(store.Path, EyeV2CalibrationStore.FileName);
    }

    [TestMethod]
    public void Store_RejectsStaleSchemaAndMalformedFiles()
    {
        Directory.CreateDirectory(_directory);
        File.WriteAllText(_path, JsonSerializer.Serialize(
            EyeV2CalibrationAndMapperTest.IdentityCalibration() with { SchemaVersion = 99 }));
        var store = Store();

        Assert.IsFalse(store.TryLoad(out _, out var stale));
        StringAssert.Contains(stale, "schema 99");

        File.WriteAllText(_path, "not json");
        Assert.IsFalse(store.TryLoad(out _, out var malformed));
        StringAssert.Contains(malformed, "Could not load");
    }

    [TestMethod]
    public void MissingCalibration_FailsSafeToDefault()
    {
        IEyeStateMapper? installed = new EyeV2Mapper(EyeV2CalibrationAndMapperTest.IdentityCalibration());
        var manager = Manager(Store(), mapper => installed = mapper);

        var activated = manager.TrySetMode(EyeTrackingMode.ExperimentalV2);

        Assert.IsFalse(activated);
        Assert.AreEqual(EyeTrackingMode.DefaultBaballonia, manager.Mode);
        Assert.IsNull(installed);
        Assert.AreEqual(EyeTrackingMode.DefaultBaballonia, _values[EyeV2Manager.ModeSetting]);
    }

    [TestMethod]
    public async Task ModeSwitch_HotSwapsV2AndRestoresDefaultWithoutRetraining()
    {
        var store = Store();
        await store.SaveAsync(EyeV2CalibrationAndMapperTest.IdentityCalibration());
        IEyeStateMapper? installed = null;
        var manager = Manager(store, mapper => installed = mapper);

        Assert.IsTrue(manager.TrySetMode(EyeTrackingMode.ExperimentalV2));
        Assert.IsInstanceOfType<EyeV2Mapper>(installed);
        Assert.AreEqual(EyeTrackingMode.ExperimentalV2, manager.Mode);

        Assert.IsTrue(manager.TrySetMode(EyeTrackingMode.DefaultBaballonia));
        Assert.IsNull(installed);
        Assert.AreEqual(EyeTrackingMode.DefaultBaballonia, manager.Mode);

        Assert.IsTrue(manager.TrySetMode(EyeTrackingMode.ExperimentalV2));
        Assert.IsInstanceOfType<EyeV2Mapper>(installed);
    }

    [TestMethod]
    public async Task V2Mode_OnlyWritesEyeV2SettingKeys()
    {
        var store = Store();
        await store.SaveAsync(EyeV2CalibrationAndMapperTest.IdentityCalibration());
        var manager = Manager(store, _ => { });

        manager.TrySetMode(EyeTrackingMode.ExperimentalV2);
        manager.TrySetMode(EyeTrackingMode.DefaultBaballonia);

        CollectionAssert.AreEquivalent(new[] { EyeV2Manager.ModeSetting }, new List<string>(_values.Keys));
        _settings.Verify(x => x.SaveSetting("EyeHome_EyeModel", It.IsAny<string>(), It.IsAny<bool>()), Times.Never);
        _settings.Verify(x => x.SaveSetting("CalibrationParams", It.IsAny<object>(), It.IsAny<bool>()), Times.Never);
    }

    [TestMethod]
    public async Task GeometryHybrid_IsSeparatelySelectableAndKeepsV2AAvailable()
    {
        var store = Store();
        await store.SaveAsync(EyeV2CalibrationAndMapperTest.IdentityCalibration());
        IEyeStateMapper? installed = null;
        var manager = new EyeV2Manager(null!, _settings.Object, store,
            Mock.Of<ILogger<EyeV2Manager>>(), mapper => installed = mapper,
            () => new StubGeometryExtractor());

        Assert.IsTrue(manager.TrySetMode(EyeTrackingMode.GeometryHybridV2B));
        Assert.IsInstanceOfType<EyeV2GeometryMapper>(installed);
        Assert.AreEqual(EyeTrackingMode.GeometryHybridV2B, manager.Mode);

        Assert.IsTrue(manager.TrySetMode(EyeTrackingMode.ExperimentalV2));
        Assert.IsInstanceOfType<EyeV2Mapper>(installed,
            "V2-A must remain its own selectable, behaviorally unchanged mapper.");
    }

    [TestMethod]
    public async Task GeometryInitializationFailure_FallsBackToV2A()
    {
        var store = Store();
        await store.SaveAsync(EyeV2CalibrationAndMapperTest.IdentityCalibration());
        IEyeStateMapper? installed = null;
        var manager = new EyeV2Manager(null!, _settings.Object, store,
            Mock.Of<ILogger<EyeV2Manager>>(), mapper => installed = mapper,
            () => throw new InvalidOperationException("no geometry runtime"));

        Assert.IsFalse(manager.TrySetMode(EyeTrackingMode.GeometryHybridV2B));
        Assert.AreEqual(EyeTrackingMode.ExperimentalV2, manager.Mode);
        Assert.IsInstanceOfType<EyeV2Mapper>(installed);
        StringAssert.Contains(manager.Status, "V2-A");
    }

    private EyeV2CalibrationStore Store() => new(
        Mock.Of<ILogger<EyeV2CalibrationStore>>(), _path);

    private EyeV2Manager Manager(EyeV2CalibrationStore store, Action<IEyeStateMapper?> setter) =>
        new(null!, _settings.Object, store, Mock.Of<ILogger<EyeV2Manager>>(), setter);

    private sealed class StubGeometryExtractor : IEyeGeometryExtractor
    {
        public string Name => "stub";
        public EyeGeometryFrame Extract(Mat transformedEyeFrame, long timestampTicks) =>
            new(timestampTicks, EyeGeometry.Missing, EyeGeometry.Missing);
        public void Reset() { }
        public void Dispose() { }
    }
}
