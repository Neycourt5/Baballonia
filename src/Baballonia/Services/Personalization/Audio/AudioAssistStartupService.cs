using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;

namespace Baballonia.Services.Personalization.Audio;

/// <summary>Wait seam used to test the always-on reconnect loop without wall-clock sleeps.</summary>
public interface IAudioAssistRetryScheduler
{
    ValueTask WaitForNextAttemptAsync(CancellationToken cancellationToken);
}

/// <summary>
/// Applies persisted audio settings at host startup and keeps retrying an enabled-but-unavailable
/// microphone for the lifetime of the app. Recovery therefore does not depend on opening the
/// Personalization page or keeping one of its UI timers alive.
/// </summary>
public sealed class AudioAssistStartupService : BackgroundService
{
    public static readonly TimeSpan RetryInterval = TimeSpan.FromSeconds(5);

    private readonly AudioAssistService _audioAssist;
    private readonly IAudioAssistRetryScheduler _scheduler;

    /// <summary>Production constructor selected by DI.</summary>
    public AudioAssistStartupService(AudioAssistService audioAssist)
        : this(audioAssist, new PeriodicRetryScheduler(RetryInterval))
    {
    }

    /// <summary>Deterministic test seam; production uses the five-second scheduler above.</summary>
    public AudioAssistStartupService(
        AudioAssistService audioAssist,
        IAudioAssistRetryScheduler scheduler)
    {
        _audioAssist = audioAssist;
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
    }

    public override Task StartAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _audioAssist.Apply();
        return base.StartAsync(cancellationToken);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                await _scheduler.WaitForNextAttemptAsync(stoppingToken).ConfigureAwait(false);
                _audioAssist.TryRecoverAudioCapture();
            }
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Normal host shutdown.
        }
    }

    private sealed class PeriodicRetryScheduler(TimeSpan interval) : IAudioAssistRetryScheduler
    {
        public async ValueTask WaitForNextAttemptAsync(CancellationToken cancellationToken) =>
            await Task.Delay(interval, cancellationToken).ConfigureAwait(false);
    }
}
