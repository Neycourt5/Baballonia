using Baballonia.Services.Calibration;
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
    private const int PrepMilliseconds = 3000;
    private const int UpdateMilliseconds = 250;
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
    private readonly IVrCalibrationPresenter? _presenter;
    private readonly Func<int, CancellationToken, Task> _delay;
    private readonly SemaphoreSlim _operation = new(1, 1);
    private readonly object _sync = new();
    private readonly Dictionary<string, List<EyeV2Sample>> _samples = new();
    private string? _activePhase;

    public bool IsBusy => _operation.CurrentCount == 0;

    public EyeV2CalibrationService(
        IEyePipelineEventBus events,
        EyeV2CalibrationStore store,
        EyeV2Manager manager,
        ILogger<EyeV2CalibrationService> logger,
        IVrCalibrationPresenter? presenter = null,
        Func<int, CancellationToken, Task>? delay = null)
    {
        _events = events;
        _store = store;
        _manager = manager;
        _logger = logger;
        _presenter = presenter;
        _delay = delay ?? Task.Delay;
        _events.Subscribe<EyePipelineEvents.NewRawExpressionsEvent>(OnRawEyeState);
    }

    public async Task<EyeV2Calibration> CalibrateAsync(
        IProgress<EyeV2CalibrationProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        await _operation.WaitAsync(cancellationToken);
        var presenterStarted = false;
        var completed = false;
        try
        {
            presenterStarted = BeginPresenter("EYE V2-A MAPPING CALIBRATION");
            lock (_sync) _samples.Clear();
            // Each phase has a three-second visible preparation interval. 4 behavioral captures
            // take 35 s total and the five gaze targets take 25 s: the actual routine is 60 s.
            const double total = 60;
            double done = 0;

            await CaptureAsync(Relax, "Relax your eyes and look straight ahead.", 5, done, total, progress, cancellationToken);
            done += 8;
            await CaptureAsync(Blinks, "Blink slowly three times, opening fully between blinks.", 8, done, total, progress, cancellationToken);
            done += 11;
            await CaptureAsync(Squint, "Hold a comfortable deliberate squint.", 5, done, total, progress, cancellationToken);
            done += 8;
            await CaptureAsync(Wide, "Open both eyes wide and hold.", 5, done, total, progress, cancellationToken);
            done += 8;

            foreach (var target in EyeV2GazeTarget.FivePoint)
            {
                await CaptureAsync(
                    GazeKey(target),
                    $"Keep your head still and look {target.Name.ToLowerInvariant()}.",
                    2, done, total, progress, cancellationToken, target.X, target.Y);
                done += 5;
            }

            var capture = BuildCapture();
            var version = Assembly.GetEntryAssembly()?.GetName().Version?.ToString() ?? "unknown";
            var calibration = EyeV2CalibrationFitter.Fit(capture, version);
            await CommitCalibrationAsync(calibration, cancellationToken);
            progress?.Report(new EyeV2CalibrationProgress("Eye V2 calibration complete.", 1));
            completed = true;
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
            if (presenterStarted)
                _presenter?.End(completed
                    ? "Eye V2-A mapping was saved and activated."
                    : null);
            _operation.Release();
        }
    }

    public async Task<EyeV2Calibration> RecenterAsync(
        IProgress<EyeV2CalibrationProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        await _operation.WaitAsync(cancellationToken);
        var presenterStarted = false;
        var completed = false;
        try
        {
            var current = _manager.Calibration;
            if (current == null || !current.IsValid())
                throw new InvalidOperationException("Calibrate Eye V2 before recentering.");

            presenterStarted = BeginPresenter("EYE V2-A RECENTER");
            lock (_sync) _samples.Remove(Recenter);
            await CaptureAsync(Recenter, "Look straight ahead and relax for a quick recenter.", 2,
                0, 5, progress, cancellationToken);
            var updated = EyeV2CalibrationFitter.Recenter(current, Snapshot(Recenter));
            await CommitCalibrationAsync(updated, cancellationToken);
            progress?.Report(new EyeV2CalibrationProgress("Eyes recentered.", 1));
            completed = true;
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
            if (presenterStarted)
                _presenter?.End(completed ? "Eye V2-A was recentered and saved." : null);
            _operation.Release();
        }
    }

    public async Task<EyeV2ValidityResult> CheckValidityAsync(
        IProgress<EyeV2CalibrationProgress>? progress = null,
        CancellationToken cancellationToken = default,
        bool useVrPresenter = true)
    {
        await _operation.WaitAsync(cancellationToken);
        var presenterStarted = false;
        try
        {
            var current = _manager.Calibration;
            if (current == null || !current.IsValid())
                throw new InvalidOperationException("No active Eye V2 calibration to check.");

            presenterStarted = useVrPresenter && BeginPresenter("EYE V2-A HEADSET CHECK");
            lock (_sync) _samples.Remove(Validity);
            await CaptureAsync(Validity, "Relax and look straight ahead while Eye V2 checks the headset position.",
                2, 0, 5, progress, cancellationToken);
            var result = EyeV2CalibrationFitter.AssessValidity(current, Snapshot(Validity));
            _manager.ReportStatus(result.Message);
            progress?.Report(new EyeV2CalibrationProgress(result.Message, 1));
            return result;
        }
        finally
        {
            lock (_sync) _activePhase = null;
            if (presenterStarted) _presenter?.End("Headset-position check complete.");
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
        while (true)
        {
            progress?.Report(new EyeV2CalibrationProgress(instruction,
                Math.Clamp(completedSeconds / totalSeconds, 0, 1), targetX, targetY));

            var retry = false;
            for (var elapsed = 0; elapsed < PrepMilliseconds; elapsed += UpdateMilliseconds)
            {
                var remaining = (PrepMilliseconds - elapsed) / 1000.0;
                PresentPhase(key, instruction, VrCalibrationPhase.Preparing,
                    completedSeconds / totalSeconds,
                    elapsed / (double)PrepMilliseconds, remaining, targetX, targetY);
                await _delay(UpdateMilliseconds, cancellationToken);
                retry = HandlePresenterAction();
                if (retry) break;
            }
            if (retry) continue;

            lock (_sync)
            {
                _samples[key] = [];
                _activePhase = null;
            }

            // Presenting the sampling state happens before the phase is published below, so a
            // frame can never receive a hold label while the headset still says "get ready".
            var samplingPhase = targetX.HasValue
                ? VrCalibrationPhase.Target
                : key is Squint or Wide ? VrCalibrationPhase.Hold : VrCalibrationPhase.Sampling;
            PresentPhase(key, instruction, samplingPhase,
                (completedSeconds + PrepMilliseconds / 1000.0) / totalSeconds,
                0, null, targetX, targetY);
            lock (_sync) _activePhase = key;

            var sampleMilliseconds = seconds * 1000;
            for (var elapsed = 0; elapsed < sampleMilliseconds; elapsed += UpdateMilliseconds)
            {
                await _delay(UpdateMilliseconds, cancellationToken);
                progress?.Report(new EyeV2CalibrationProgress(instruction,
                    Math.Clamp((completedSeconds + PrepMilliseconds / 1000.0 +
                                (elapsed + UpdateMilliseconds) / 1000.0) / totalSeconds, 0, 1),
                    targetX, targetY));
                PresentPhase(key, instruction, samplingPhase,
                    (completedSeconds + PrepMilliseconds / 1000.0 +
                     (elapsed + UpdateMilliseconds) / 1000.0) / totalSeconds,
                    (elapsed + UpdateMilliseconds) / (double)sampleMilliseconds,
                    null, targetX, targetY);

                retry = HandlePresenterAction();
                if (retry) break;
            }

            lock (_sync) _activePhase = null;
            if (retry) continue;
            return;
        }
    }

    private bool BeginPresenter(string title)
    {
        if (_presenter == null) return false; // Unit-test/headless compatibility seam.
        var result = _presenter.Begin(title);
        if (!result.Started)
            throw new InvalidOperationException(
                $"Eye V2 calibration needs the in-headset presenter. {result.Message} " +
                "Start SteamVR, confirm the headset is connected, and try again.");
        EnsurePresenterHealthy();
        return true;
    }

    private void PresentPhase(
        string key,
        string instruction,
        VrCalibrationPhase phase,
        double overallProgress,
        double phaseProgress,
        double? countdown,
        float? targetX,
        float? targetY)
    {
        if (_presenter == null) return;
        EnsurePresenterHealthy();

        var title = key switch
        {
            Relax => "RELAX YOUR EYES",
            Blinks => "BLINK SLOWLY THREE TIMES",
            Squint => "SQUINT AND HOLD",
            Wide => "OPEN YOUR EYES WIDE",
            Recenter => "LOOK STRAIGHT AHEAD",
            Validity => "HEADSET POSITION CHECK",
            _ when key.StartsWith("gaze:", StringComparison.Ordinal) =>
                $"GAZE — {key[5..].ToUpperInvariant()}",
            _ => "EYE V2-A CALIBRATION",
        };

        _presenter.Present(new VrCalibrationFrame(
            title, instruction, phase, overallProgress, phaseProgress, countdown,
            targetX, targetY, AllowRetry: true, AllowCancel: true));
        EnsurePresenterHealthy();
    }

    private bool HandlePresenterAction()
    {
        if (_presenter == null) return false;
        EnsurePresenterHealthy();
        var action = _presenter.ConsumeAction();
        EnsurePresenterHealthy();
        return action switch
        {
            VrCalibrationAction.Cancel => throw new OperationCanceledException(
                "Eye calibration cancelled from the headset."),
            VrCalibrationAction.Retry => true,
            _ => false,
        };
    }

    private void EnsurePresenterHealthy()
    {
        if (_presenter is { IsPresenting: true, IsHealthy: true }) return;
        throw new InvalidOperationException(
            $"The in-headset presenter stopped updating. {_presenter?.Status ?? "Unknown presenter error."}");
    }

    private async Task CommitCalibrationAsync(
        EyeV2Calibration calibration,
        CancellationToken cancellationToken)
    {
        // Disk replacement and mapper/settings activation are one logical commit. Save first so a
        // successfully activated manager can always load its calibration on restart; if activation
        // then fails, restore the exact prior bytes while the manager rolls its mapper back itself.
        var snapshot = await _store.CaptureSnapshotAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await _store.SaveAsync(calibration, cancellationToken).ConfigureAwait(false);
            _manager.Activate(calibration);
        }
        catch (Exception commitError)
        {
            try
            {
                await _store.RestoreSnapshotAsync(snapshot).ConfigureAwait(false);
            }
            catch (Exception rollbackError)
            {
                _logger.LogCritical(rollbackError,
                    "Eye V2 calibration file rollback failed after commit error");
                throw new AggregateException(
                    "Eye V2 calibration commit failed and the previous file could not be restored.",
                    commitError, rollbackError);
            }
            throw;
        }
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
