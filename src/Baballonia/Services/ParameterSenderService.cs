using Baballonia.Contracts;
using Baballonia.Helpers;
using Baballonia.Services.Calibration;
using Baballonia.Services.Inference;
using Baballonia.Services.Personalization;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OscCore;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.Services;

public class ParameterSenderService : BackgroundService
{
    private readonly VrcftModuleSendService _vrcftModuleSendService;
    private readonly DfrSendService _dfrSendService;
    private readonly ILocalSettingsService _localSettingsService;
    private float _eyeSquintStrength = 1f;
    private readonly ICalibrationService _calibrationService;
    private readonly ILogger<ParameterSenderService> _logger;
    private readonly EyeStageTrace? _eyeTrace;

    // Written ~once/sec on the sender loop, read on the inference worker threads — volatile for a
    // well-defined cross-thread read.
    private volatile string _prefix = "";
    private volatile bool _sendNativeVrcEyeTracking;
    private volatile bool _useDfr;

    /// <summary>
    /// Supplies avatar-guided calibration targets. Null when personalization is not registered.
    /// </summary>
    private readonly IExpressionOverrideSource? _expressionOverrideSource;

    /// <summary>
    /// True while calibration targets are being sent instead of tracked face values. Written by the
    /// sender loop, read by the processing-tick handler.
    /// </summary>
    private volatile bool _faceOverrideActive;
    private readonly ConcurrentQueue<OscMessage> _vrcftQueue = new();
    private readonly ConcurrentQueue<OscMessage> _dfrQueue = new();
    // Reused drain buffer for the (single-threaded) sender loop.
    private readonly List<OscMessage> _sendBuffer = new();

    public ParameterSenderService(
        VrcftModuleSendService vrcftModuleSendService,
        DfrSendService dfrSendService,
        ILocalSettingsService localSettingsService,
        ICalibrationService calibrationService,
        ProcessingLoopService processingLoopService,
        ILogger<ParameterSenderService> logger,
        IExpressionOverrideSource? expressionOverrideSource = null,
        EyeStageTrace? eyeTrace = null)
    {
        this._vrcftModuleSendService = vrcftModuleSendService;
        this._dfrSendService = dfrSendService;
        this._localSettingsService = localSettingsService;
        this._calibrationService = calibrationService;
        this._logger = logger;
        this._expressionOverrideSource = expressionOverrideSource;
        _eyeTrace = eyeTrace;

        processingLoopService.ExpressionChangeEvent += ExpressionUpdateHandler;
    }

    protected override async Task ExecuteAsync(CancellationToken cancellationToken)
    {
        _logger.LogDebug("Starting Parameter Sender Service...");
        _logger.LogDebug("OSC parameter mapping initialized");

        long lastSettingsRead = 0;
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                // These settings change only on user edits, but ReadSetting deserializes JSON on
                // every call. Refreshing them on each 10 ms tick was ~300 deserializations/sec for
                // nothing; refresh roughly once per second instead.
                var now = Environment.TickCount64;
                if (now - lastSettingsRead >= 1000)
                {
                    lastSettingsRead = now;
                    _prefix = _localSettingsService.ReadSetting<string>("AppSettings_OSCPrefix");
                    _sendNativeVrcEyeTracking = _localSettingsService.ReadSetting<bool>("VRC_UseNativeTracking");
                    _useDfr = _localSettingsService.ReadSetting<bool>("AppSettings_UseDFR");
                    _eyeSquintStrength = Math.Clamp(
                        _localSettingsService.ReadSetting("AppSettings_EyeSquintStrength", 1f), 0f, 2f);
                }
                // Sampled every tick, not on the once-per-second settings cadence: the avatar
                // animates on the transmitting clock, independent of camera rate and UI jitter.
                var overrideTarget = _expressionOverrideSource?.SampleTarget();
                _faceOverrideActive = overrideTarget != null;
                if (overrideTarget != null)
                    EnqueueRawFaceVector(overrideTarget);

                await SendAndClearQueue(cancellationToken);
                await Task.Delay(10, cancellationToken);
            }
            catch (Exception)
            {
                // ignore
            }
        }
    }

    private void ExpressionUpdateHandler(ProcessingLoopService.Expressions expressions)
    {
        if (expressions.EyeExpression != null)
            ProcessEyeExpressionData(expressions.EyeExpression, expressions.EyeExpressionRaw);
        if (expressions.FaceExpression != null)
            ProcessFaceExpressionData(expressions.FaceExpression);
    }

    private void ProcessEyeExpressionData(OrderedFloatMap expressions, OrderedFloatMap? rawExpressions)
    {
        if (expressions is null) return;
        var traceOutput = _eyeTrace?.Enabled == true ? new OrderedFloatMap(expressions.Keys.ToArray()) : null;

        foreach (var expression in expressions)
        {
            float weight = expression.Value;
            var settings = _calibrationService.GetExpressionSettings(expression.Key);

            var value = CalibrateAndClampEye(expression.Key, weight, settings);
            value = ScaleSquint(expression.Key, value, _eyeSquintStrength);

            _vrcftQueue.Enqueue(new OscMessage(_prefix + expression.Key, value));
            if (traceOutput != null) traceOutput[expression.Key] = value;
        }
        if (traceOutput != null)
            _eyeTrace!.Add(0, "sender_queue_unpaired", traceOutput,
                state: "Frame=0: no transported sequence; enqueue observed, delivery/receiver not observed");

        // Native eye tracking (DFR / VRChat native) wants the raw, un-smoothed stream for the lowest
        // latency, so it bypasses the OneEuroFilter. Falls back to the filtered map if no raw is supplied.
        var nativeSource = rawExpressions ?? expressions;

        // Never let an OSC-fanout error escape into the inference worker: ProcessingLoopService's
        // EyeWorker catch tears down all cameras, so a single bad value here would kill eye tracking.
        try
        {
            if (_useDfr)
                ProcessNativeVrcEyeTracking(nativeSource, _dfrQueue);

            if (_sendNativeVrcEyeTracking)
                ProcessNativeVrcEyeTracking(nativeSource, _vrcftQueue);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error building native/DFR eye OSC parameters");
        }
    }

    private void ProcessNativeVrcEyeTracking(OrderedFloatMap expressions, ConcurrentQueue<OscMessage> queue)
    {
        // Keys produced by the eye runner are X/Y/Lid (see DefaultInferenceRunner); read them with the
        // non-throwing accessor so a future key-set change degrades to 0 instead of throwing.
        expressions.TryGetValue("/leftEyeX", out var leftEyeX);
        expressions.TryGetValue("/leftEyeY", out var leftEyeY);
        expressions.TryGetValue("/leftEyeLid", out var leftEyeLid);
        expressions.TryGetValue("/rightEyeX", out var rightEyeX);
        expressions.TryGetValue("/rightEyeY", out var rightEyeY);
        expressions.TryGetValue("/rightEyeLid", out var rightEyeLid);

        var leftEyeLidSettings = _calibrationService.GetExpressionSettings("/leftEyeLid");
        var rightEyeLidSettings = _calibrationService.GetExpressionSettings("/rightEyeLid");
        var weightedLeftEyeLid = CalibrateAndClampEye("/leftEyeLid", leftEyeLid, leftEyeLidSettings);
        var weightedRightEyeLid = CalibrateAndClampEye("/rightEyeLid", rightEyeLid, rightEyeLidSettings);
        var averageLid = (weightedLeftEyeLid + weightedRightEyeLid) / 2f;
        queue.Enqueue(new OscMessage("/tracking/eye/EyesClosedAmount", 1f - Math.Clamp(averageLid, 0f, 1f)));

        // Convert normalized eye positions to angles
        const float maxEyeAngle = 45f;
        leftEyeX *= maxEyeAngle;
        leftEyeY *= -maxEyeAngle; // Negative because Y is inverted (up is negative pitch)
        rightEyeX *= maxEyeAngle;
        rightEyeY *= -maxEyeAngle; // Negative because Y is inverted (up is negative pitch)
        queue.Enqueue(new OscMessage("/tracking/eye/LeftRightPitchYaw", leftEyeY, leftEyeX, rightEyeY, rightEyeX));
    }

    internal const string LeftSquintKey = "/leftEyeSquint";
    internal const string RightSquintKey = "/rightEyeSquint";

    /// <summary>
    /// Scales how strongly an already-detected squint is expressed.
    /// </summary>
    /// <remarks>
    /// <para>Applied at the last point the value is mutable: after
    /// <see cref="CalibrateAndClampEye"/> and immediately before the OSC enqueue. Everything
    /// downstream is a verbatim copy into <c>UnifiedTracking.Data</c>, so nothing can undo it.</para>
    ///
    /// <para>That placement is what makes this safe. Detection, the eyelid channel, blink shaping,
    /// gaze, gaze sync and the One Euro filter are all <em>upstream</em> and cannot see it. In
    /// particular the lid channel is an input to the gaze fusion, so scaling anything lid-derived
    /// earlier would leak into where the eyes appear to be looking; scaling the squint output here
    /// cannot. Scaling before the filter would be worse still, since the adaptive cutoff depends on
    /// the signal's own derivative and the smoothing would then change with the setting.</para>
    ///
    /// <para>The two eyes are scaled independently by the same factor, never averaged or coupled -
    /// that is what preserves the asymmetry the model detects. Exactly identity at 1, short-circuited
    /// rather than relying on a float multiply.</para>
    /// </remarks>
    internal static float ScaleSquint(string key, float value, float strength)
    {
        if (strength == 1f || key is not (LeftSquintKey or RightSquintKey))
            return value;

        if (!float.IsFinite(value) || !float.IsFinite(strength))
            return value;

        return Math.Clamp(value * strength, 0f, 1f);
    }

    /// <summary>The model's key for jaw opening, the one channel shaped before calibration.</summary>
    internal const string JawOpenKey = "/jawOpen";

    /// <summary>Opt-in output shaping, shared with the C2 recording and compatibility contract.</summary>
    internal static float ReadEffectiveJawOpenCurve(ILocalSettingsService? settings)
    {
        if (settings?.ReadSetting("AppSettings_JawOpenCurveEnabled", false) != true)
            return 1f;

        var curve = settings.ReadSetting("AppSettings_JawOpenCurve", 1f);
        return float.IsFinite(curve) ? Math.Clamp(curve, 0.5f, 3f) : 1f;
    }

    /// <summary>
    /// Bends how quickly an expression climbs, without changing where it ends up.
    /// </summary>
    /// <remarks>
    /// <para>Built for jaw open, where the complaint is "when I open my mouth a little bit, my whole
    /// jaw goes way down" - a small real movement arriving as most of the avatar's range, which
    /// reads as a caveman face rather than as speech.</para>
    ///
    /// <para>A gain cannot fix that: scaling the channel down to tame a small opening also caps a
    /// genuinely wide one, so the avatar can never open its mouth properly again. An exponent leaves
    /// both ends pinned - 0 stays 0 and 1 stays 1 - and only changes how fast the middle climbs.
    /// Above 1, small openings stay small while a wide one still reaches full; below 1, the reverse,
    /// for anyone with the opposite complaint.</para>
    ///
    /// <para>Applied before the user's Lower/Upper calibration, so that keeps working on top
    /// unchanged. Disabled by default, and exactly identity at 1 even when enabled. The calibration
    /// override path deliberately does not go through here: a commanded
    /// ground-truth target must reach the avatar verbatim.</para>
    /// </remarks>
    internal static float ShapeExpression(float value, float curve)
    {
        if (curve == 1f || !float.IsFinite(value) || !float.IsFinite(curve) || curve <= 0f)
            return value;

        var clamped = Math.Clamp(value, 0f, 1f);
        return MathF.Pow(clamped, curve);
    }

    internal static float CalibrateAndClampEye(
        string key,
        float value,
        CalibrationParameter settings)
    {
        var mapped = value.Remap(settings.Lower, settings.Upper, settings.Min, settings.Max);
        if (!float.IsFinite(mapped))
            mapped = EyeOutputPostProcessor.NeutralForKey(key);

        var minimum = Math.Min(settings.Min, settings.Max);
        var maximum = Math.Max(settings.Min, settings.Max);
        return Math.Clamp(mapped, minimum, maximum);
    }

    /// <summary>
    /// Sends an externally commanded calibration target verbatim: clamped to [0,1] but deliberately
    /// NOT passed through the user's calibration remap. Those ranges exist to correct the stock
    /// model's output; a commanded ground-truth target is already in canonical units, and remapping
    /// it would make the avatar show something other than the value we are about to record as the
    /// training label.
    /// </summary>
    /// <remarks>
    /// <paramref name="target"/> is positional, in <see cref="PersonalizationSchema"/> order. The
    /// OSC addresses are derived from that same schema, which is the one place the positional
    /// personalization world and the keyed pipeline world are allowed to meet.
    /// </remarks>
    private void EnqueueRawFaceVector(float[] target)
    {
        var count = Math.Min(target.Length, PersonalizationSchemaBinding.ExpectedKeys.Count);
        for (var i = 0; i < count; i++)
        {
            _vrcftQueue.Enqueue(new OscMessage(
                _prefix + PersonalizationSchemaBinding.ExpectedKeys[i],
                Math.Clamp(target[i], 0f, 1f)));
        }
    }

    private void ProcessFaceExpressionData(OrderedFloatMap expressions)
    {
        // Calibration owns the face channel while an override is active. Eye output is untouched:
        // ProcessEyeExpressionData still runs, so eye tracking keeps working during calibration.
        if (_faceOverrideActive) return;

        if (expressions == null) return;

        // Use current settings once per face batch, as C2 does for its live context guard.
        // The old one-second cache could lag behind a deliberate C2/curve setting change.
        var jawOpenCurve = ReadEffectiveJawOpenCurve(_localSettingsService);
        foreach (var expression in expressions)
        {
            float weight = expression.Value;
            if (expression.Key == JawOpenKey)
                weight = ShapeExpression(weight, jawOpenCurve);

            var settings = _calibrationService.GetExpressionSettings(expression.Key);

            var msg = new OscMessage(_prefix + expression.Key,
                Math.Clamp(
                    weight.Remap(settings.Lower, settings.Upper, settings.Min, settings.Max),
                    settings.Min,
                    settings.Max));
            _vrcftQueue.Enqueue(msg);
        }
    }

    private async Task SendAndClearQueue(CancellationToken cancellationToken)
    {
        await DrainAndSend(_vrcftQueue, _vrcftModuleSendService, cancellationToken);
        await DrainAndSend(_dfrQueue, _dfrSendService, cancellationToken);
    }

    /// <summary>
    /// Atomically drains the queue (via TryDequeue) and sends it. The previous ToArray()+Clear()
    /// was not atomic — anything enqueued between the snapshot and the Clear() was silently dropped.
    /// </summary>
    private async Task DrainAndSend(ConcurrentQueue<OscMessage> queue, OscSendService sender,
        CancellationToken cancellationToken)
    {
        if (queue.IsEmpty)
            return;

        _sendBuffer.Clear();
        while (queue.TryDequeue(out var message))
            _sendBuffer.Add(message);

        if (_sendBuffer.Count > 0)
            await sender.Send(_sendBuffer.ToArray(), cancellationToken);
    }
}
