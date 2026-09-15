using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Baballonia.Services.Calibration;

public enum EyeCalibrationStep
{
    Relaxed,
    Closed,
    Wide,
}

/// <summary>
/// Captures the geometry-corrected, unfiltered eye stream. The inference callback only copies six
/// scalar values into the active session; percentile work and settings I/O happen off that thread.
/// </summary>
public sealed class EyeCalibrationRecorder : IDisposable
{
    private static readonly TimeSpan DiscardDuration = TimeSpan.FromSeconds(0.5);
    private const int MaximumSamplesPerStep = 300;

    private readonly ProcessingLoopService _processingLoop;
    private readonly ILocalSettingsService _settings;
    private readonly EyePipelineManager _pipelineManager;
    private readonly ILogger _logger;
    private readonly object _gate = new();

    private CaptureSession? _active;
    private IReadOnlyList<EyeCalibrationSample> _relaxed = [];
    private IReadOnlyList<EyeCalibrationSample> _closed = [];
    private IReadOnlyList<EyeCalibrationSample> _wide = [];

    public EyeCalibrationRecorder(
        ProcessingLoopService processingLoop,
        ILocalSettingsService settings,
        EyePipelineManager pipelineManager,
        ILogger<EyeCalibrationRecorder>? logger = null)
    {
        _processingLoop = processingLoop;
        _settings = settings;
        _pipelineManager = pipelineManager;
        _logger = logger ?? NullLogger<EyeCalibrationRecorder>.Instance;
        _processingLoop.ExpressionChangeEvent += OnExpressions;
    }

    public bool HasAllSteps
    {
        get
        {
            lock (_gate)
                return _relaxed.Count > 0 && _closed.Count > 0 && _wide.Count > 0;
        }
    }

    public async Task<int> CaptureStepAsync(
        EyeCalibrationStep step,
        CancellationToken cancellationToken = default)
    {
        var duration = step switch
        {
            EyeCalibrationStep.Relaxed => TimeSpan.FromSeconds(5),
            EyeCalibrationStep.Closed => TimeSpan.FromSeconds(3),
            EyeCalibrationStep.Wide => TimeSpan.FromSeconds(4),
            _ => throw new ArgumentOutOfRangeException(nameof(step)),
        };

        CaptureSession session;
        lock (_gate)
        {
            if (_active is not null)
                throw new InvalidOperationException("An eye-calibration step is already running.");
            session = new CaptureSession(DiscardDuration);
            _active = session;
        }

        try
        {
            await Task.Delay(duration, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            lock (_gate)
            {
                if (ReferenceEquals(_active, session))
                    _active = null;
            }
        }

        var captured = session.Snapshot();
        lock (_gate)
        {
            switch (step)
            {
                case EyeCalibrationStep.Relaxed: _relaxed = captured; break;
                case EyeCalibrationStep.Closed: _closed = captured; break;
                case EyeCalibrationStep.Wide: _wide = captured; break;
            }
        }
        return captured.Count;
    }

    public EyeCalibrationEstimate Estimate()
    {
        EyeCalibrationEstimate estimate;
        lock (_gate)
            estimate = EyeCalibrationEstimator.Estimate(_relaxed, _closed, _wide);

        // The six anchors are the whole result of a capture, and when one is wrong the user's only
        // other evidence is a sentence in the UI. Log them once per estimate so a failed setup can
        // be diagnosed after the fact.
        _logger.LogInformation(
            "Eye calibration anchors — left closed {LC:F3} relaxed {LN:F3} wide {LW:F3} " +
            "center ({LX:F3},{LY:F3}); right closed {RC:F3} relaxed {RN:F3} wide {RW:F3} " +
            "center ({RX:F3},{RY:F3}); valid={Valid}{Detail}",
            estimate.Left.OpennessClosed, estimate.Left.OpennessNeutral, estimate.Left.OpennessWide,
            estimate.Left.GazeCenterX, estimate.Left.GazeCenterY,
            estimate.Right.OpennessClosed, estimate.Right.OpennessNeutral, estimate.Right.OpennessWide,
            estimate.Right.GazeCenterX, estimate.Right.GazeCenterY,
            estimate.IsValid,
            estimate.Error is { } e ? $" — {e}" : estimate.Note is { } n ? $" — {n}" : "");

        return estimate;
    }

    /// <summary>
    /// Stores the capture in the library and makes it active. The library writes the two profile
    /// keys the post-processor reads, so activation and persistence cannot disagree.
    /// </summary>
    public EyeCalibrationEstimate Save(EyeCalibrationLibrary library)
    {
        var estimate = Estimate();
        if (!estimate.IsValid)
            return estimate;

        library.Add(estimate.Left, estimate.Right);
        _pipelineManager.LoadEyePostProcessor();
        return estimate;
    }

    private void OnExpressions(ProcessingLoopService.Expressions expressions)
    {
        lock (_gate)
        {
            if (_active is null)
                return;
        }

        var raw = expressions.EyeExpressionRaw;
        if (raw is null || !TryCopy(raw, out var sample))
            return;

        lock (_gate)
            _active?.Add(sample);
    }

    private static bool TryCopy(OrderedFloatMap raw, out EyeCalibrationSample sample)
    {
        sample = default;
        if (!raw.TryGetValue("/leftEyeLid", out var leftLid) ||
            !raw.TryGetValue("/leftEyeX", out var leftX) ||
            !raw.TryGetValue("/leftEyeY", out var leftY) ||
            !raw.TryGetValue("/rightEyeLid", out var rightLid) ||
            !raw.TryGetValue("/rightEyeX", out var rightX) ||
            !raw.TryGetValue("/rightEyeY", out var rightY))
            return false;

        sample = new EyeCalibrationSample(leftLid, leftX, leftY,
            rightLid, rightX, rightY);
        return float.IsFinite(leftLid) && float.IsFinite(leftX) && float.IsFinite(leftY) &&
               float.IsFinite(rightLid) && float.IsFinite(rightX) && float.IsFinite(rightY);
    }

    public void Dispose()
    {
        _processingLoop.ExpressionChangeEvent -= OnExpressions;
        lock (_gate)
            _active = null;
    }

    private sealed class CaptureSession(TimeSpan discardDuration)
    {
        private readonly long _startedAt = Stopwatch.GetTimestamp();
        private readonly List<EyeCalibrationSample> _samples = [];

        public void Add(EyeCalibrationSample sample)
        {
            if (Stopwatch.GetElapsedTime(_startedAt) < discardDuration)
                return;
            if (_samples.Count < MaximumSamplesPerStep)
                _samples.Add(sample);
        }

        public IReadOnlyList<EyeCalibrationSample> Snapshot() => _samples.ToArray();
    }
}
