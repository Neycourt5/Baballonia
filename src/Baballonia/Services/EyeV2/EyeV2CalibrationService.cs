using Baballonia.Services.events;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.Services.EyeV2;

/// <summary>
/// One-button, mapping-only eye calibration. The raw P3-3 event is the sole sample source; no
/// second inference path and no model training are introduced.
/// </summary>
public sealed class EyeV2CalibrationService : IDisposable
{
    private const int PrepMilliseconds = 750;
    private const string Relax = "relax";
    private const string Blinks = "blinks";
    private const string Squint = "squint";
    private const string Wide = "wide";
    private const string Validity = "validity";
    private const string Recenter = "recenter";

    private readonly IEyePipelineEventBus _events;
    private readonly EyeV2CalibrationStore _store;
    private readonly EyeV2Manager _manager;
    private readonly ILogger<EyeV2CalibrationService> _logger;
    private readonly SemaphoreSlim _operation = new(1, 1);
    private readonly object _sync = new();
    private readonly Dictionary<string, List<EyeV2Sample>> _samples = new();
    private string? _activePhase;

    public bool IsBusy => _operation.CurrentCount == 0;

    public EyeV2CalibrationService(
        IEyePipelineEventBus events,
        EyeV2CalibrationStore store,
        EyeV2Manager manager,
        ILogger<EyeV2CalibrationService> logger)
    {
        _events = events;
        _store = store;
        _manager = manager;
        _logger = logger;
        _events.Subscribe<EyePipelineEvents.NewRawExpressionsEvent>(OnRawEyeState);
    }

    public async Task<EyeV2Calibration> CalibrateAsync(
        IProgress<EyeV2CalibrationProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        await _operation.WaitAsync(cancellationToken);
        try
        {
            lock (_sync) _samples.Clear();
            const double total = 42.75;
            double done = 0;

            await CaptureAsync(Relax, "Relax your eyes and look straight ahead.", 5, done, total, progress, cancellationToken);
            done += 5.75;
            await CaptureAsync(Blinks, "Blink slowly three times, opening fully between blinks.", 8, done, total, progress, cancellationToken);
            done += 8.75;
            await CaptureAsync(Squint, "Hold a comfortable deliberate squint.", 5, done, total, progress, cancellationToken);
            done += 5.75;
            await CaptureAsync(Wide, "Open both eyes wide and hold.", 5, done, total, progress, cancellationToken);
            done += 5.75;

            foreach (var target in EyeV2GazeTarget.FivePoint)
            {
                await CaptureAsync(
                    GazeKey(target),
                    $"Keep your head still and look {target.Name.ToLowerInvariant()}.",
                    2, done, total, progress, cancellationToken, target.X, target.Y);
                done += 2.75;
            }

            var capture = BuildCapture();
            var version = Assembly.GetEntryAssembly()?.GetName().Version?.ToString() ?? "unknown";
            var calibration = EyeV2CalibrationFitter.Fit(capture, version);
            await _store.SaveAsync(calibration, cancellationToken);
            _manager.Activate(calibration);
            progress?.Report(new EyeV2CalibrationProgress("Eye V2 calibration complete.", 1));
            return calibration;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex, "Eye V2 calibration failed. The current eye path remains active.");
            _manager.ReportStatus($"Eye V2 calibration failed: {ex.Message}");
            throw;
        }
        finally
        {
            lock (_sync) _activePhase = null;
            _operation.Release();
        }
    }

    public async Task<EyeV2Calibration> RecenterAsync(
        IProgress<EyeV2CalibrationProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        await _operation.WaitAsync(cancellationToken);
        try
        {
            var current = _manager.Calibration;
            if (current == null || !current.IsValid())
                throw new InvalidOperationException("Calibrate Eye V2 before recentering.");

            lock (_sync) _samples.Remove(Recenter);
            await CaptureAsync(Recenter, "Look straight ahead and relax for a quick recenter.", 2,
                0, 2.75, progress, cancellationToken);
            var updated = EyeV2CalibrationFitter.Recenter(current, Snapshot(Recenter));
            await _store.SaveAsync(updated, cancellationToken);
            _manager.Activate(updated);
            progress?.Report(new EyeV2CalibrationProgress("Eyes recentered.", 1));
            return updated;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex, "Eye V2 recenter failed. The current eye path remains active.");
            _manager.ReportStatus($"Recenter failed: {ex.Message}");
            throw;
        }
        finally
        {
            lock (_sync) _activePhase = null;
            _operation.Release();
        }
    }

    public async Task<EyeV2ValidityResult> CheckValidityAsync(
        IProgress<EyeV2CalibrationProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        await _operation.WaitAsync(cancellationToken);
        try
        {
            var current = _manager.Calibration;
            if (current == null || !current.IsValid())
                throw new InvalidOperationException("No active Eye V2 calibration to check.");

            lock (_sync) _samples.Remove(Validity);
            await CaptureAsync(Validity, "Relax and look straight ahead while Eye V2 checks the headset position.",
                2, 0, 2.75, progress, cancellationToken);
            var result = EyeV2CalibrationFitter.AssessValidity(current, Snapshot(Validity));
            _manager.ReportStatus(result.Message);
            progress?.Report(new EyeV2CalibrationProgress(result.Message, 1));
            return result;
        }
        finally
        {
            lock (_sync) _activePhase = null;
            _operation.Release();
        }
    }

    private async Task CaptureAsync(
        string key,
        string instruction,
        int seconds,
        double completedSeconds,
        double totalSeconds,
        IProgress<EyeV2CalibrationProgress>? progress,
        CancellationToken cancellationToken,
        float? targetX = null,
        float? targetY = null)
    {
        progress?.Report(new EyeV2CalibrationProgress(instruction,
            Math.Clamp(completedSeconds / totalSeconds, 0, 1), targetX, targetY));
        await Task.Delay(PrepMilliseconds, cancellationToken);

        lock (_sync)
        {
            _samples[key] = [];
            _activePhase = key;
        }

        for (var i = 0; i < seconds * 4; i++)
        {
            await Task.Delay(250, cancellationToken);
            progress?.Report(new EyeV2CalibrationProgress(instruction,
                Math.Clamp((completedSeconds + 0.75 + (i + 1) * 0.25) / totalSeconds, 0, 1),
                targetX, targetY));
        }

        lock (_sync) _activePhase = null;
    }

    private void OnRawEyeState(EyePipelineEvents.NewRawExpressionsEvent e)
    {
        lock (_sync)
        {
            if (_activePhase == null || e.rawResult.Length < EyeStateLayout.LegacyCount) return;
            if (!_samples.TryGetValue(_activePhase, out var destination)) return;
            // Calibration is the only time this allocation occurs. The event handler still only
            // copies six floats and returns; fitting and persistence happen after capture.
            destination.Add(new EyeV2Sample(e.timestampTicks, (float[])e.rawResult.Clone()));
        }
    }

    private EyeV2CalibrationCapture BuildCapture() => new()
    {
        Relax = Snapshot(Relax),
        Blinks = Snapshot(Blinks),
        Squint = Snapshot(Squint),
        Wide = Snapshot(Wide),
        Gaze = EyeV2GazeTarget.FivePoint.ToDictionary(
            x => x.Name,
            x => (IReadOnlyList<EyeV2Sample>)Snapshot(GazeKey(x))),
    };

    private IReadOnlyList<EyeV2Sample> Snapshot(string key)
    {
        lock (_sync)
            return _samples.TryGetValue(key, out var samples) ? samples.ToArray() : [];
    }

    private static string GazeKey(EyeV2GazeTarget target) => $"gaze:{target.Name}";

    public void Dispose()
    {
        _events.Unsubscribe<EyePipelineEvents.NewRawExpressionsEvent>(OnRawEyeState);
        _operation.Dispose();
    }
}
