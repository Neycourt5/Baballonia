using System;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Services.Inference;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Restores the persisted personal model at host startup. Loading cannot depend on visiting the
/// Personalization page: tracking starts from the Home page and should honor the saved selection.
/// </summary>
public sealed class PersonalModelStartupService : IHostedService
{
    private readonly Func<Task> _waitForInitialInference;
    private readonly Func<Task> _reloadPersonalModel;
    private readonly ILogger<PersonalModelStartupService> _logger;

    public PersonalModelStartupService(
        FacePipelineManager facePipelineManager,
        PersonalModelManager personalModelManager,
        ILogger<PersonalModelStartupService> logger)
        : this(
            () => facePipelineManager.InitialInferenceLoad,
            async () => { await personalModelManager.ReloadAsync().ConfigureAwait(false); },
            logger)
    {
    }

    /// <summary>Deterministic lifecycle seam used by regression tests.</summary>
    public PersonalModelStartupService(
        Func<Task> waitForInitialInference,
        Func<Task> reloadPersonalModel,
        ILogger<PersonalModelStartupService> logger)
    {
        _waitForInitialInference = waitForInitialInference ??
                                   throw new ArgumentNullException(nameof(waitForInitialInference));
        _reloadPersonalModel = reloadPersonalModel ??
                               throw new ArgumentNullException(nameof(reloadPersonalModel));
        _logger = logger;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        try
        {
            // Model C validates the embedding capability exposed by the face runner. Waiting here
            // also avoids launching a redundant second ONNX load merely to establish ordering.
            await _waitForInitialInference().ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            await _reloadPersonalModel().ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            // Personalization is optional. A bad persisted artifact must not prevent the other
            // hosted services from starting; PersonalModelManager already falls back to stock for
            // expected validation failures, and this is the final guard for unexpected ones.
            _logger.LogWarning(ex,
                "Could not restore the persisted personal model at application startup; using stock");
        }
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
