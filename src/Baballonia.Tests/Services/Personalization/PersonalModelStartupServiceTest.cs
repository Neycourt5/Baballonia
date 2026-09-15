using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

[TestClass]
public class PersonalModelStartupServiceTest
{
    [TestMethod]
    public async Task HostStartupWaitsForFaceInferenceThenRestoresPersistedModel()
    {
        var inferenceReady = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var reloadCalled = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var order = new List<string>();

        var startup = new PersonalModelStartupService(
            async () =>
            {
                await inferenceReady.Task;
                order.Add("inference");
            },
            () =>
            {
                order.Add("personal-model");
                reloadCalled.SetResult();
                return Task.CompletedTask;
            },
            NullLogger<PersonalModelStartupService>.Instance);

        var start = startup.StartAsync(CancellationToken.None);
        Assert.IsFalse(reloadCalled.Task.IsCompleted,
            "embedding model validation must not race the stock inference runner");

        inferenceReady.SetResult();
        await start.WaitAsync(TimeSpan.FromSeconds(5));

        Assert.IsTrue(reloadCalled.Task.IsCompleted,
            "the persisted model must load without opening the Personalization page");
        CollectionAssert.AreEqual(
            new[] { "inference", "personal-model" },
            order.ToArray());
    }
}
