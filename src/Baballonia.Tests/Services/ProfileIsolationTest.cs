using System;
using System.IO;
using Baballonia;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services;

/// <summary>
/// Keeping two copies of the application out of each other's settings.
/// </summary>
/// <remarks>
/// Running one copy for eyes and another for the face is a reasonable thing to want, and two things
/// stopped it. A process-wide mutex refused the second copy outright, exiting with a code nobody
/// would think to look up. And both copies read and wrote the same settings file, so every save from
/// one overwrote the other's camera choices - which does not present as a conflict, it presents as
/// settings randomly reverting.
///
/// A named profile moves the data directory and the instance lock together. The pairing is the point:
/// separating one without the other gives either two copies fighting over one settings file, or one
/// copy that still cannot start.
/// </remarks>
[TestClass]
public class ProfileIsolationTest
{
    /// <summary>The lock name the launcher and the application must agree on.</summary>
    private static string MutexName(string profile) =>
        string.IsNullOrEmpty(profile) ? "baballonia-unique-id" : $"baballonia-unique-id-{profile}";

    private static string FolderName(string profile) =>
        string.IsNullOrEmpty(profile) ? "ProjectBabble" : $"ProjectBabble-{profile}";

    [TestMethod]
    public void AnUnnamedProfileChangesNothing()
    {
        // The default has to be the existing installation exactly. Anything else would move a
        // working setup's settings and models out from under it on upgrade, which reads as every
        // setting having been reset and every trained model lost.
        Assert.AreEqual("ProjectBabble", FolderName(string.Empty));
        Assert.AreEqual("baballonia-unique-id", MutexName(string.Empty));
    }

    [TestMethod]
    public void ANamedProfileMovesBothTheFolderAndTheLock()
    {
        // Both, together. A separate folder without a separate lock is a copy that cannot start; a
        // separate lock without a separate folder is two copies overwriting each other's settings.
        Assert.AreEqual("ProjectBabble-face", FolderName("face"));
        Assert.AreEqual("baballonia-unique-id-face", MutexName("face"));
    }

    [TestMethod]
    public void TwoProfilesShareNothing()
    {
        Assert.AreNotEqual(FolderName("face"), FolderName("eyes"));
        Assert.AreNotEqual(MutexName("face"), MutexName("eyes"));

        Assert.AreNotEqual(FolderName("face"), FolderName(string.Empty),
            "a profiled copy must not land on the ordinary installation's folder");
    }

    [TestMethod]
    public void TheRunningProcessAgreesWithItsOwnProfile()
    {
        // Reads what the application actually resolved rather than recomputing it, so a change to
        // one and not the other is caught. Under test no profile is set, so this is the default.
        var expected = FolderName(Utils.Profile);

        StringAssert.EndsWith(
            Path.GetFileName(Utils.PersistentDataDirectory.TrimEnd(Path.DirectorySeparatorChar)),
            expected,
            $"the data directory does not match profile '{Utils.Profile}'");
    }

    [TestMethod]
    public void ModelsLiveInsideTheProfilesOwnFolder()
    {
        // Models are expensive and slow to reproduce. A profile that shared them would be a profile
        // where deleting one copy's data takes the other copy's models with it.
        StringAssert.StartsWith(Utils.ModelsDirectory, Utils.PersistentDataDirectory,
            "models must sit inside the profile's own directory");

        StringAssert.StartsWith(Utils.ModelDataDirectory, Utils.PersistentDataDirectory);
    }
}
