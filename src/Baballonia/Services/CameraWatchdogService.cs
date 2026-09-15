using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Services.Inference;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Baballonia.Services;

/// <summary>Detects stalled targeted camera slots and asks their owning managers to recover them.</summary>
public sealed class CameraWatchdogService(
    IEnumerable<ICameraSlotHost> hosts,
    ILogger<CameraWatchdogService> logger) : IHostedService
{
    public static readonly TimeSpan StallTimeout = TimeSpan.FromSeconds(2);
    public static readonly TimeSpan StartGrace = TimeSpan.FromSeconds(3);
    public static readonly TimeSpan TickInterval = TimeSpan.FromMilliseconds(250);
    public static readonly TimeSpan PresencePollInterval = TimeSpan.FromSeconds(1);
    public static readonly TimeSpan FirstFrameTimeout = TimeSpan.FromSeconds(3);
    public static readonly TimeSpan[] Backoff =
    [
        TimeSpan.FromSeconds(1),
        TimeSpan.FromSeconds(2),
        TimeSpan.FromSeconds(4),
        TimeSpan.FromSeconds(8),
        TimeSpan.FromSeconds(10),
    ];

    private sealed class SlotBookkeeping
    {
        public int Attempt;
        public DateTime NextAttemptUtc = DateTime.MinValue;
        public DateTime? LostAtUtc;
        public TimeSpan AbsentFor;
        public DateTime? AbsentSinceUtc;
        public DateTime NextPresenceCheckUtc = DateTime.MinValue;
        public bool LastKnownPresent = true;
    }

    private readonly ICameraSlotHost[] _hosts = hosts.ToArray();
    private readonly Dictionary<IRecoverableCameraSlot, SlotBookkeeping> _bookkeeping = new();
    private CancellationTokenSource? _cancellation;
    private Task? _loop;
    private long _reconnectCount;

    public long ReconnectCount => Interlocked.Read(ref _reconnectCount);

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _loop = Task.Run(() => RunAsync(_cancellation.Token), CancellationToken.None);
        logger.LogDebug("Camera watchdog started for {Count} camera host(s)", _hosts.Length);
        return Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        if (_cancellation is null)
            return;

        await _cancellation.CancelAsync().ConfigureAwait(false);
        if (_loop is not null)
        {
            try
            {
                await _loop.WaitAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (Exception exception) when (exception is OperationCanceledException or TimeoutException)
            {
                logger.LogDebug("Camera watchdog did not stop before shutdown completed");
            }
        }

        _cancellation.Dispose();
        _cancellation = null;
        _loop = null;
    }

    private async Task RunAsync(CancellationToken cancellationToken)
    {
        using var timer = new PeriodicTimer(TickInterval);
        try
        {
            while (await timer.WaitForNextTickAsync(cancellationToken).ConfigureAwait(false))
                await TickAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Normal shutdown.
        }
        catch (Exception exception)
        {
            logger.LogError(exception, "Camera watchdog stopped unexpectedly");
        }
    }

    /// <summary>Runs one observation pass. Public to permit deterministic recovery tests.</summary>
    public async Task TickAsync(CancellationToken cancellationToken = default)
    {
        var slots = _hosts.SelectMany(host => host.Slots).Distinct().ToArray();
        var active = slots.ToHashSet();
        foreach (var removed in _bookkeeping.Keys.Where(slot => !active.Contains(slot)).ToArray())
            _bookkeeping.Remove(removed);

        foreach (var slot in slots)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (slot.Target is null)
            {
                _bookkeeping.Remove(slot);
                continue;
            }

            var state = GetBookkeeping(slot);
            RefreshPresenceIfDue(slot, state);
            switch (slot.State)
            {
                case CameraState.Running when !state.LastKnownPresent:
                    logger.LogWarning(
                        "{Camera} camera disappeared from device enumeration",
                        slot.Name);
                    state.LostAtUtc = DateTime.UtcNow;
                    state.Attempt = 0;
                    state.NextAttemptUtc = DateTime.UtcNow;
                    await AttemptAsync(slot, state, cancellationToken).ConfigureAwait(false);
                    break;

                case CameraState.Running when IsStalled(slot):
                    logger.LogWarning(
                        "{Camera} camera stalled: no frame for {Ms:F0} ms",
                        slot.Name, slot.TimeSinceLastFrame?.TotalMilliseconds ?? -1);
                    state.LostAtUtc = DateTime.UtcNow;
                    state.Attempt = 0;
                    state.NextAttemptUtc = DateTime.UtcNow;
                    await AttemptAsync(slot, state, cancellationToken).ConfigureAwait(false);
                    break;

                case CameraState.Reconnecting when DateTime.UtcNow >= state.NextAttemptUtc:
                    state.LostAtUtc ??= DateTime.UtcNow;
                    await AttemptAsync(slot, state, cancellationToken).ConfigureAwait(false);
                    break;
            }
        }
    }

    private SlotBookkeeping GetBookkeeping(IRecoverableCameraSlot slot)
    {
        if (_bookkeeping.TryGetValue(slot, out var state))
            return state;

        state = new SlotBookkeeping();
        _bookkeeping.Add(slot, state);
        return state;
    }

    private static void RefreshPresenceIfDue(
        IRecoverableCameraSlot slot,
        SlotBookkeeping state)
    {
        if (slot.State != CameraState.Running || DateTime.UtcNow < state.NextPresenceCheckUtc)
            return;

        var target = slot.Target;
        state.LastKnownPresent = target is null || IsPresent(slot, target.Address);
        state.NextPresenceCheckUtc = DateTime.UtcNow + PresencePollInterval;
    }

    private static bool IsStalled(IRecoverableCameraSlot slot)
    {
        if (slot.SourceInstalledAtUtc is not { } installed || DateTime.UtcNow - installed < StartGrace)
            return false;

        var age = slot.TimeSinceLastFrame;
        return age is null || age > StallTimeout;
    }

    private async Task AttemptAsync(
        IRecoverableCameraSlot slot,
        SlotBookkeeping state,
        CancellationToken cancellationToken)
    {
        var attempt = state.Attempt + 1;
        logger.LogDebug("{Camera} camera reconnect attempt {Attempt}", slot.Name, attempt);

        var target = slot.Target;
        if (target is null)
        {
            _bookkeeping.Remove(slot);
            return;
        }

        var present = IsPresent(slot, target.Address);
        if (!present)
            state.AbsentSinceUtc ??= DateTime.UtcNow;
        else if (state.AbsentSinceUtc is { } absentSince)
        {
            state.AbsentFor += DateTime.UtcNow - absentSince;
            state.AbsentSinceUtc = null;
        }

        bool recovered;
        try
        {
            recovered = await slot.RecoverAsync(FirstFrameTimeout, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception)
        {
            logger.LogDebug(exception,
                "{Camera} camera reconnect attempt {Attempt} threw", slot.Name, attempt);
            recovered = false;
        }

        if (recovered)
        {
            var now = DateTime.UtcNow;
            if (state.AbsentSinceUtc is { } absentSince)
                state.AbsentFor += now - absentSince;
            var outage = state.LostAtUtc is { } lost ? now - lost : TimeSpan.Zero;
            var absent = state.AbsentFor;
            logger.LogInformation(
                "{Camera} camera recovered after {Seconds:F1} s ({Absent:F1} s absent, {Reopening:F1} s reopening) over {Attempts} attempt(s)",
                slot.Name, outage.TotalSeconds, absent.TotalSeconds,
                Math.Max(0, (outage - absent).TotalSeconds), attempt);
            Interlocked.Increment(ref _reconnectCount);
            _bookkeeping.Remove(slot);
            return;
        }

        state.Attempt = attempt;
        state.NextAttemptUtc = DateTime.UtcNow + DelayFor(attempt);
        logger.LogDebug("{Camera} camera reconnect attempt {Attempt} did not succeed", slot.Name, attempt);
    }

    private static bool IsPresent(IRecoverableCameraSlot slot, string address)
    {
        try
        {
            return slot.IsTargetPresent(address);
        }
        catch
        {
            // Enumeration failure is not evidence that the camera is absent.
            return true;
        }
    }

    public static TimeSpan DelayFor(int attempt)
    {
        if (attempt < 1)
            return Backoff[0];
        return Backoff[Math.Min(attempt - 1, Backoff.Length - 1)];
    }
}
