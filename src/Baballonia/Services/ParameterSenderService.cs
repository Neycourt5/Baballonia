using Baballonia.Contracts;
using Baballonia.Helpers;
using Baballonia.Services.Personalization;
using Baballonia.Services.EyeV2;
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
    private readonly ICalibrationService _calibrationService;
    private readonly ILogger<ParameterSenderService> _logger;

    /// <summary>
    /// Supplies avatar-guided calibration targets. Null outside calibration builds/sessions.
    /// </summary>
    private readonly IExpressionOverrideSource? _expressionOverrideSource;

    /// <summary>
    /// True while calibration targets are being sent instead of tracked face values. Written by the
    /// sender loop, read by the processing-tick handler.
    /// </summary>
    private volatile bool _faceOverrideActive;

    private string _prefix = "";
    private bool _sendNativeVrcEyeTracking;
    private bool _useDfr;
    private readonly ConcurrentQueue<OscMessage> _vrcftQueue = new();
    private readonly ConcurrentQueue<OscMessage> _dfrQueue = new();
    private readonly object _vrcftQueueGate = new();
    private readonly object _dfrQueueGate = new();
    private EyeOscSendSnapshot? _lastEyeOscSendSnapshot;

    // Positional protocol ABI. Keep this explicit: Dictionary.ElementAt made correctness depend on
    // an implementation-specific enumeration order and obscured which V2 slot drove each address.
    public static IReadOnlyList<EyeOscParameter> EyeExpressionOrder { get; } = Array.AsReadOnly(new[]
    {
        new EyeOscParameter("LeftEyeX", "/LeftEyeX", true),
        new EyeOscParameter("LeftEyeY", "/LeftEyeY", true),
        new EyeOscParameter("LeftEyeLid", "/LeftEyeLid", false),
        new EyeOscParameter("RightEyeX", "/RightEyeX", true),
        new EyeOscParameter("RightEyeY", "/RightEyeY", true),
        new EyeOscParameter("RightEyeLid", "/RightEyeLid", false),
        // V2 appends these after the exact legacy six-value prefix.
        new EyeOscParameter("LeftEyeWiden", "/LeftEyeWiden", false),
        new EyeOscParameter("LeftEyeSquint", "/LeftEyeSquint", false),
        new EyeOscParameter("RightEyeWiden", "/RightEyeWiden", false),
        new EyeOscParameter("RightEyeSquint", "/RightEyeSquint", false),
    });

    /// <summary>The newest final eye batch and its local UDP transport state.</summary>
    public EyeOscSendSnapshot? LastEyeOscSendSnapshot =>
        Volatile.Read(ref _lastEyeOscSendSnapshot);

    /// <summary>Raised for queued and transport-completion snapshots.</summary>
    public event Action<EyeOscSendSnapshot>? EyeOscSendSnapshotChanged;

    public readonly Dictionary<string, string> FaceExpressionMap = new()
    {
        { "CheekPuffLeft", "/cheekPuffLeft" },
        { "CheekPuffRight", "/cheekPuffRight" },
        { "CheekSuckLeft", "/cheekSuckLeft" },
        { "CheekSuckRight", "/cheekSuckRight" },
        { "JawOpen", "/jawOpen" },
        { "JawForward", "/jawForward" },
        { "JawLeft", "/jawLeft" },
        { "JawRight", "/jawRight" },
        { "NoseSneerLeft", "/noseSneerLeft" },
        { "NoseSneerRight", "/noseSneerRight" },
        { "MouthFunnel", "/mouthFunnel" },
        { "MouthPucker", "/mouthPucker" },
        { "MouthLeft", "/mouthLeft" },
        { "MouthRight", "/mouthRight" },
        { "MouthRollUpper", "/mouthRollUpper" },
        { "MouthRollLower", "/mouthRollLower" },
        { "MouthShrugUpper", "/mouthShrugUpper" },
        { "MouthShrugLower", "/mouthShrugLower" },
        { "MouthClose", "/mouthClose" },
        { "MouthSmileLeft", "/mouthSmileLeft" },
        { "MouthSmileRight", "/mouthSmileRight" },
        { "MouthFrownLeft", "/mouthFrownLeft" },
        { "MouthFrownRight", "/mouthFrownRight" },
        { "MouthDimpleLeft", "/mouthDimpleLeft" },
        { "MouthDimpleRight", "/mouthDimpleRight" },
        { "MouthUpperUpLeft", "/mouthUpperUpLeft" },
        { "MouthUpperUpRight", "/mouthUpperUpRight" },
        { "MouthLowerDownLeft", "/mouthLowerDownLeft" },
        { "MouthLowerDownRight", "/mouthLowerDownRight" },
        { "MouthPressLeft", "/mouthPressLeft" },
        { "MouthPressRight", "/mouthPressRight" },
        { "MouthStretchLeft", "/mouthStretchLeft" },
        { "MouthStretchRight", "/mouthStretchRight" },
        { "TongueOut", "/tongueOut" },
        { "TongueUp", "/tongueUp" },
        { "TongueDown", "/tongueDown" },
        { "TongueLeft", "/tongueLeft" },
        { "TongueRight", "/tongueRight" },
        { "TongueRoll", "/tongueRoll" },
        { "TongueBendDown", "/tongueBendDown" },
        { "TongueCurlUp", "/tongueCurlUp" },
        { "TongueSquish", "/tongueSquish" },
        { "TongueFlat", "/tongueFlat" },
        { "TongueTwistLeft", "/tongueTwistLeft" },
        { "TongueTwistRight", "/tongueTwistRight" }
    };

    public ParameterSenderService(
        VrcftModuleSendService vrcftModuleSendService,
        DfrSendService dfrSendService,
        ILocalSettingsService localSettingsService,
        ICalibrationService calibrationService,
        ProcessingLoopService processingLoopService,
        ILogger<ParameterSenderService> logger,
        IExpressionOverrideSource? expressionOverrideSource = null)
    {
        this._vrcftModuleSendService = vrcftModuleSendService;
        this._dfrSendService = dfrSendService;
        this._localSettingsService = localSettingsService;
        this._calibrationService = calibrationService;
        this._logger = logger;
        this._expressionOverrideSource = expressionOverrideSource;

         processingLoopService.ExpressionChangeEvent += ExpressionUpdateHandler;
    }

    protected override async Task ExecuteAsync(CancellationToken cancellationToken)
    {
        _logger.LogDebug("Starting Parameter Sender Service...");
        _logger.LogDebug("OSC parameter mapping initialized with {EyeCount} eye expressions and {FaceCount} face expressions",
            EyeExpressionOrder.Count, FaceExpressionMap.Count);

        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                _prefix = _localSettingsService.ReadSetting<string>("AppSettings_OSCPrefix");
                _sendNativeVrcEyeTracking = _localSettingsService.ReadSetting<bool>("VRC_UseNativeTracking");
                _useDfr = _localSettingsService.ReadSetting<bool>("AppSettings_UseDFR");

                // Sampled here rather than pushed from the cue engine so the avatar animates on the
                // transmitting clock, independent of camera rate and UI tick jitter.
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
            ProcessEyeExpressionData(expressions.EyeExpression);
        if (expressions.FaceExpression != null)
            ProcessFaceExpressionData(expressions.FaceExpression);
    }

    private void ProcessEyeExpressionData(float[] expressions)
    {
        if (expressions is null) return;
        if (expressions.Length < EyeStateLayout.LegacyCount) return;

        var eyeV2 = expressions.Length >= EyeStateLayout.V2Count;
        var channelCount = eyeV2 ? EyeStateLayout.V2Count : EyeStateLayout.LegacyCount;
        var prefix = _prefix;
        var values = new EyeOscValue[channelCount];
        EyeOscSendSnapshot queuedSnapshot;

        lock (_vrcftQueueGate)
        {
            for (var i = 0; i < channelCount; i++)
            {
                var weight = expressions[i];
                var channel = EyeExpressionOrder[i];
                float sent;
                if (eyeV2)
                {
                    // V2 is already in canonical units and owns separate personal calibration.
                    // Never pass it through Default's CalibrationParams or the two modes cease to
                    // be independent. Gaze is signed; lid/wide/squint are unit weights.
                    sent = Math.Clamp(weight, channel.IsSigned ? -1f : 0f, 1f);
                }
                else
                {
                    var settings = _calibrationService.GetExpressionSettings(channel.ExpressionName);
                    sent = weight.Remap(settings.Lower, settings.Upper, settings.Min, settings.Max);
                }

                var address = prefix + channel.Address;
                _vrcftQueue.Enqueue(new OscMessage(address, sent));
                values[i] = new EyeOscValue(channel.ExpressionName, address, sent);
            }

            if (_sendNativeVrcEyeTracking)
                ProcessNativeVrcEyeTracking(expressions, _vrcftQueue);

            var now = DateTimeOffset.UtcNow;
            queuedSnapshot = new EyeOscSendSnapshot(
                Array.AsReadOnly(values),
                _vrcftModuleSendService?.Destination ?? "unconfigured",
                now,
                now,
                EyeOscTransportStatus.Queued,
                null);
            Volatile.Write(ref _lastEyeOscSendSnapshot, queuedSnapshot);
        }

        if (_useDfr)
        {
            lock (_dfrQueueGate)
                ProcessNativeVrcEyeTracking(expressions, _dfrQueue);
        }

        NotifyEyeOscSnapshotChanged(queuedSnapshot);
    }

    private void ProcessNativeVrcEyeTracking(float[] expressions, ConcurrentQueue<OscMessage> queue)
    {
        var leftEyeX = expressions[0];
        var leftEyeY = expressions[1];
        var leftEyeLid = expressions[2];
        var rightEyeX = expressions[3];
        var rightEyeY = expressions[4];
        var rightEyeLid = expressions[5];

        var eyeV2 = expressions.Length >= EyeStateLayout.V2Count;
        var weightedLeftEyeLid = leftEyeLid;
        var weightedRightEyeLid = rightEyeLid;
        if (!eyeV2)
        {
            var leftEyeLidSettings = _calibrationService.GetExpressionSettings("LeftEyeLid");
            var rightEyeLidSettings = _calibrationService.GetExpressionSettings("RightEyeLid");
            weightedLeftEyeLid = leftEyeLid.Remap(leftEyeLidSettings.Lower, leftEyeLidSettings.Upper, leftEyeLidSettings.Min, leftEyeLidSettings.Max);
            weightedRightEyeLid = rightEyeLid.Remap(rightEyeLidSettings.Lower, rightEyeLidSettings.Upper, rightEyeLidSettings.Min, rightEyeLidSettings.Max);
        }
        var averageLid = (weightedLeftEyeLid + weightedRightEyeLid) / 2f;
        queue.Enqueue(new OscMessage("/tracking/eye/EyesClosedAmount", 1f - Math.Clamp(averageLid, 0f, 1f)));

        // Convert normalized eye positions to angles
        const float maxEyeAngle = 45f;
        leftEyeX *= maxEyeAngle;
        leftEyeY *= -maxEyeAngle; // Negative because Y is inverted (up is negative pitch)
        rightEyeX *= maxEyeAngle;
        rightEyeY *= -maxEyeAngle; // Negative because Y is inverted (up is negative pitch)
        queue.Enqueue(new OscMessage("/tracking/eye/LeftRightPitchYaw", leftEyeY, rightEyeX, rightEyeY, leftEyeX));
    }

    /// <summary>
    /// Sends an externally commanded calibration target verbatim: clamped to [0,1] but deliberately
    /// NOT passed through the user's calibration remap. Those ranges exist to correct the stock
    /// model's output; a commanded ground-truth target is already in canonical units, and remapping
    /// it would make the avatar show something other than the value we are about to record as the
    /// training label.
    /// </summary>
    private void EnqueueRawFaceVector(float[] target)
    {
        lock (_vrcftQueueGate)
        {
            for (var i = 0; i < Math.Min(target.Length, FaceExpressionMap.Count); i++)
            {
                var faceElement = FaceExpressionMap.ElementAt(i);
                _vrcftQueue.Enqueue(new OscMessage(
                    _prefix + faceElement.Value,
                    Math.Clamp(target[i], 0f, 1f)));
            }
        }
    }

    private void ProcessFaceExpressionData(float[] expressions)
    {
        // Calibration owns the face channel while an override is active. Eye output is untouched:
        // ProcessEyeExpressionData still runs, so eye tracking keeps working during calibration.
        if (_faceOverrideActive) return;

        if (expressions == null) return;
        if (expressions.Length == 0) return;

        lock (_vrcftQueueGate)
        {
            for (var i = 0; i < Math.Min(expressions.Length, FaceExpressionMap.Count); i++)
            {
                var weight = expressions[i];
                var faceElement = FaceExpressionMap.ElementAt(i);
                var settings = _calibrationService.GetExpressionSettings(faceElement.Key);

                var msg = new OscMessage(_prefix + faceElement.Value,
                    Math.Clamp(
                        weight.Remap(settings.Lower, settings.Upper, settings.Min, settings.Max),
                        settings.Min,
                        settings.Max));
                _vrcftQueue.Enqueue(msg);
            }
        }
    }

    private async Task SendAndClearQueue(CancellationToken cancellationToken)
    {
        OscMessage[] vrcftMessages;
        EyeOscSendSnapshot? eyeSnapshotAtDispatch;
        lock (_vrcftQueueGate)
        {
            vrcftMessages = _vrcftQueue.ToArray();
            _vrcftQueue.Clear();
            eyeSnapshotAtDispatch = Volatile.Read(ref _lastEyeOscSendSnapshot);
        }

        if (vrcftMessages.Length > 0)
        {
            OscDispatchResult result;
            try
            {
                result = await _vrcftModuleSendService.Send(vrcftMessages, cancellationToken);
            }
            catch (Exception ex)
            {
                result = new OscDispatchResult(
                    false,
                    _vrcftModuleSendService?.Destination ?? "unconfigured",
                    DateTimeOffset.UtcNow,
                    vrcftMessages.Length,
                    ex.Message);
            }

            CompleteEyeOscSnapshot(eyeSnapshotAtDispatch, result);
        }

        OscMessage[] dfrMessages;
        lock (_dfrQueueGate)
        {
            dfrMessages = _dfrQueue.ToArray();
            _dfrQueue.Clear();
        }

        if (dfrMessages.Length > 0)
            await _dfrSendService.Send(dfrMessages, cancellationToken);
    }

    private void CompleteEyeOscSnapshot(
        EyeOscSendSnapshot? queuedSnapshot,
        OscDispatchResult result)
    {
        if (queuedSnapshot is not { TransportStatus: EyeOscTransportStatus.Queued })
            return;

        var updated = queuedSnapshot with
        {
            Destination = result.Destination,
            UpdatedAtUtc = result.CompletedAtUtc,
            TransportStatus = result.SentToUdpSocket
                ? EyeOscTransportStatus.SentToUdpSocket
                : EyeOscTransportStatus.TransportError,
            Error = result.Error,
        };

        var previous = Interlocked.CompareExchange(
            ref _lastEyeOscSendSnapshot,
            updated,
            queuedSnapshot);
        if (ReferenceEquals(previous, queuedSnapshot))
            NotifyEyeOscSnapshotChanged(updated);
    }

    private void NotifyEyeOscSnapshotChanged(EyeOscSendSnapshot snapshot)
    {
        var handlers = EyeOscSendSnapshotChanged;
        if (handlers == null)
            return;

        foreach (Action<EyeOscSendSnapshot> handler in handlers.GetInvocationList())
        {
            try
            {
                handler(snapshot);
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "Eye OSC diagnostic subscriber failed");
            }
        }
    }
}
