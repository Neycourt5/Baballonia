using System;
using System.IO;
using System.Text.Json;
using Baballonia.Services.Personalization;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

[TestClass]
public class CorrectionSessionStoreTest
{
    private string _root = null!;

    [TestInitialize]
    public void Initialize()
    {
        _root = Path.Combine(Path.GetTempPath(), $"baballonia-correction-delete-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_root);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [TestMethod]
    public void DeleteAll_RemovesOnlyVerifiedCorrectionSessions()
    {
        var correction = MakeSession("real_correction", SessionType.Correction, includeMarker: true);
        var ordinary = MakeSession("neutral_recording", SessionType.Neutral, includeMarker: false);
        var misleading = MakeSession("neutral_with_stray_marker", SessionType.Neutral, includeMarker: true);
        var incomplete = Path.Combine(_root, "incomplete_correction");
        Directory.CreateDirectory(incomplete);
        File.WriteAllText(Path.Combine(incomplete, "correction.json"), "{}");

        var store = new CorrectionSessionStore(_root);
        Assert.AreEqual(1, store.Count());

        var result = store.DeleteAll();

        Assert.IsTrue(result.Success);
        Assert.AreEqual(1, result.Deleted);
        Assert.IsFalse(Directory.Exists(correction));
        Assert.IsTrue(Directory.Exists(ordinary), "ordinary recordings must never be deleted");
        Assert.IsTrue(Directory.Exists(misleading), "a stray marker is not sufficient authority to delete");
        Assert.IsTrue(Directory.Exists(incomplete), "ambiguous/incomplete data must be preserved");
        Assert.AreEqual(0, store.Count());
    }

    private string MakeSession(string name, SessionType type, bool includeMarker)
    {
        var directory = Path.Combine(_root, name);
        Directory.CreateDirectory(directory);
        var metadata = new SessionMetadata { SessionId = name, SessionType = type.ToString() };
        File.WriteAllText(Path.Combine(directory, "session.json"),
            JsonSerializer.Serialize(metadata, PersonalizationPaths.Json));
        if (includeMarker) File.WriteAllText(Path.Combine(directory, "correction.json"), "{}");
        File.WriteAllText(Path.Combine(directory, "keep.txt"), "sentinel");
        return directory;
    }
}
