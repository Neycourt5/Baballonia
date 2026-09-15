using System;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace Baballonia.Tests.Services;

[TestClass]
public class DataUploaderServiceTest
{
    private static DataUploadConfiguration Configuration(string? accessKey, string? secret) =>
        DataUploadConfiguration.FromEnvironment(name =>
            name == DataUploadConfiguration.AccessKeyEnvironmentVariable ? accessKey : secret);

    [TestMethod]
    public async Task MissingConfigurationCannotReadOrUploadEvenWithSavedConsent()
    {
        foreach (var config in new[] { Configuration(null, null), Configuration("fixture-key", null),
                     Configuration(null, "fixture-secret"), Configuration(" ", "fixture-secret") })
        {
            var identity = new Mock<IIdentityService>(MockBehavior.Strict);
            Assert.IsFalse(config.IsConfigured);
            using var service = new DataUploaderService(identity.Object, config);
            Assert.IsFalse(service.IsConfigured);
            // A nonexistent path proves the disabled path returns before opening a recording.
            await service.UploadDataAsync("missing-recording-" + Guid.NewGuid() + ".bin", hasConsent: true);
            identity.VerifyNoOtherCalls();
        }
    }

    [TestMethod]
    public async Task ConfiguredUploaderStillRequiresExplicitConsentBeforeOpeningARecording()
    {
        var identity = new Mock<IIdentityService>(MockBehavior.Strict);
        using var service = new DataUploaderService(identity.Object,
            Configuration("fixture-access-key", "fixture-secret-not-a-real-credential"));
        Assert.IsTrue(service.IsConfigured);
        await service.UploadDataAsync("missing-recording-" + Guid.NewGuid() + ".bin", hasConsent: false);
        identity.VerifyNoOtherCalls();
    }

    [TestMethod]
    public void ConfigurationDoesNotExposeCredentialsInDiagnostics()
    {
        var config = Configuration("fixture-access-key", "fixture-secret-not-a-real-credential");
        Assert.IsTrue(config.IsConfigured);
        Assert.IsFalse(config.ToString()!.Contains("fixture-access-key", StringComparison.Ordinal));
        Assert.IsFalse(config.ToString()!.Contains("fixture-secret", StringComparison.Ordinal));
    }
}
