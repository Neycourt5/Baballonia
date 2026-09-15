using System;
using Baballonia.Services.Inference.BlinkGuard;
using System.Collections.Generic;
using System.Linq;
using Baballonia.Services.Calibration;

namespace Baballonia.Services.Inference;

/// <summary>
/// Final safety, calibration and shaping boundary for the filtered eye-expression stream.
/// </summary>
/// <remarks>
/// Runs after the OneEuro filter and the geometry pass, and before the event that reaches OSC. The
/// order inside is deliberate and each step depends on the one before it:
///
/// <list type="number">
/// <item><description>finite guard and range clamp - nothing downstream should ever have to
/// wonder;</description></item>
/// <item><description>per-eye openness calibration, so a fixed threshold means the same thing for
/// every face;</description></item>
/// <item><description>per-eye gaze recentre and gain;</description></item>
/// <item><description>widen/squint - the model's own if it has them, derived from calibrated
/// openness only if it does not;</description></item>
/// <item><description>blink shaping on the final lid value, last, so that everything which
/// reasons about openness saw the unshaped value.</description></item>
/// </list>
///
/// The raw/DFR path never comes through here: native tracking wants the model's own output at the
/// lowest possible latency, and the user's manual Lower/Upper calibration remains the final trim
/// further downstream.
/// </remarks>
public sealed class EyeOutputPostProcessor
{
    public EyeStageTrace? Trace { get; set; }
    public long TraceFrame { get; set; }
    private const string LeftLid = "/leftEyeLid";
    private const string RightLid = "/rightEyeLid";
    private const string LeftWiden = "/leftEyeWiden";
    private const string RightWiden = "/rightEyeWiden";
    private const string LeftSquint = "/leftEyeSquint";
    private const string RightSquint = "/rightEyeSquint";

    private static readonly string[] DerivedKeys = [LeftWiden, LeftSquint, RightWiden, RightSquint];

    private readonly Dictionary<string, float> _lastGood = new(StringComparer.Ordinal);
    private readonly EyeCalibrationProfile? _leftCalibration;
    private readonly EyeCalibrationProfile? _rightCalibration;
    private readonly bool _deriveWidenSquint;

    private readonly DerivedEyeShapeEstimator _leftShapes = new();
    private readonly DerivedEyeShapeEstimator _rightShapes = new();
    private readonly BlinkReopenLimiter _leftBlink = new();
    private readonly BlinkReopenLimiter _rightBlink = new();
    private readonly bool _shapeBlinks;
    private readonly EyeLidSynchronizer _lidSync;
    private readonly EyeSyncSource _syncSource;
    private readonly float _squintSyncAmount;
    private readonly float _widenSyncAmount;
    private readonly float _gazeSyncAmount;

    /// <summary>
    /// Post-blink gaze stabilization. Null when disabled, which is the default.
    /// </summary>
    /// <remarks>
    /// Lives here rather than in the sender because the eye pipeline snapshots
    /// <c>RawEyeResult</c> for the native/DFR path <em>before</em> calling this post-processor. That
    /// makes "social gaze only, never foveated rendering" a property of the pipeline's shape rather
    /// than a rule someone has to remember.
    /// </remarks>
    public BlinkGuardFilter? BlinkGuard { get; set; }

    // Built once per source key-set and reused, so the per-frame path allocates nothing.
    private OrderedFloatMap? _augmentedSource;
    private OrderedFloatMap? _augmented;

    public EyeOutputPostProcessor(
        EyeCalibrationProfile? leftCalibration = null,
        EyeCalibrationProfile? rightCalibration = null,
        bool deriveWidenSquint = true,
        bool shapeBlinks = true,
        float lidSyncAmount = 0f,
        float squintSyncAmount = 0f,
        float widenSyncAmount = 0f,
        float gazeSyncAmount = 0f,
        EyeSyncSource syncSource = EyeSyncSource.Average,
        bool detectWinks = true)
    {
        _leftCalibration = leftCalibration;
        _rightCalibration = rightCalibration;
        _deriveWidenSquint = deriveWidenSquint;
        _shapeBlinks = shapeBlinks;
        _syncSource = syncSource;
        _squintSyncAmount = Math.Clamp(squintSyncAmount, 0f, 1f);
        _gazeSyncAmount = Math.Clamp(gazeSyncAmount, 0f, 1f);
        _widenSyncAmount = Math.Clamp(widenSyncAmount, 0f, 1f);
        _lidSync = new EyeLidSynchronizer(lidSyncAmount, syncSource, detectWinks);
    }

    /// <summary>
    /// True once <see cref="Apply"/> has seen a model that does not emit widen/squint and has begun
    /// supplying them. False while the model's own channels are being passed through. Diagnostics
    /// only.
    /// </summary>
    public bool IsDerivingWidenSquint { get; private set; }

    public DerivedEyeShapeEstimator.Shape LeftDerivedShape => _leftShapes.State;

    public DerivedEyeShapeEstimator.Shape RightDerivedShape => _rightShapes.State;

    /// <summary>True while a wink has released lid coupling. Diagnostics only.</summary>
    public bool IsWinking => _lidSync.IsReleased;

    /// <summary>
    /// Openness as the model reported it for the most recent frame, before calibration. Diagnostics
    /// only - shown next to the calibrated value so a bad calibration is visible as a divergence
    /// between the two rather than as vaguely wrong tracking.
    /// </summary>
    public float LeftRawOpenness { get; private set; }

    public float RightRawOpenness { get; private set; }

    public void Reset()
    {
        _lastGood.Clear();
        _leftShapes.Reset();
        _rightShapes.Reset();
        _leftBlink.Reset();
        _rightBlink.Reset();
        _lidSync.Reset();
    }

    /// <summary>
    /// Sanitizes and calibrates <paramref name="map"/> in place, and returns the map that should be
    /// published.
    /// </summary>
    /// <returns>
    /// <paramref name="map"/> itself in the ordinary case. When the loaded model emits no
    /// widen/squint channels and derivation is enabled, a wider map owned by this post-processor,
    /// carrying every source value plus the four derived ones. It is reused between frames, exactly
    /// like the runner's own buffer, so subscribers must copy rather than retain it.
    /// </returns>
    /// <param name="dt">Seconds since the previous call; drives dwell and slew.</param>
    public OrderedFloatMap Apply(OrderedFloatMap map, float dt)
    {
        foreach (var key in map.Keys)
        {
            var value = map[key];
            if (!float.IsFinite(value))
            {
                if (!_lastGood.TryGetValue(key, out value))
                {
                    // This is already output-space neutral. Do not feed it through a gaze-center
                    // offset: first-sample corruption must still mean centered gaze and open lids.
                    map[key] = NeutralForKey(key);
                    continue;
                }
            }

            value = ClampForKey(key, value);
            _lastGood[key] = value;

            if (key == LeftLid)
                LeftRawOpenness = value;
            else if (key == RightLid)
                RightRawOpenness = value;

            map[key] = ClampForKey(key, ApplyCalibration(key, value));
        }

        // Yoke the lids before anything reads openness, so derived shapes and blink shaping both see
        // the same synchronized value rather than each eye's independent guess.
        Trace?.Add(TraceFrame, "calibrated_before_sync", map);
        SynchronizeLids(map, dt);
        Trace?.Add(TraceFrame, "lids_synchronized", map);
        // Before gaze sync, so the two eyes are coupled using gaze that has already been vetted:
        // coupling a held eye to a snapping one would spread the snap to both.
        StabilizeBlinkGaze(map, dt);
        if (Trace?.Enabled == true)
            Trace.Add(TraceFrame, "blink_guard", map, state: BlinkGuard == null ? "off" :
                $"left={BlinkGuard.LeftState}; right={BlinkGuard.RightState}; left_reacquiring_s={BlinkGuard.LeftReacquiringSeconds}; right_reacquiring_s={BlinkGuard.RightReacquiringSeconds}");
        SynchronizeGaze(map);
        Trace?.Add(TraceFrame, "gaze_synchronized", map);

        // The model's own widen/squint always win. A value the network predicted knows things
        // openness cannot: brow position, lid shape, the difference between a squint and a blink.
        var modelSuppliesShapes = map.ContainsKey(LeftWiden) || map.ContainsKey(RightWiden) ||
                                  map.ContainsKey(LeftSquint) || map.ContainsKey(RightSquint);

        OrderedFloatMap result;
        if (!_deriveWidenSquint || modelSuppliesShapes)
        {
            IsDerivingWidenSquint = false;
            result = map;
        }
        else
        {
            IsDerivingWidenSquint = true;
            // Derivation reads the calibrated-but-unshaped lid deliberately. Shaping exists to make
            // the value pleasant to watch; classifying a squint should use the value the model
            // actually produced, not a smoothed version of it.
            result = Derive(map, dt);
        }

        // After derivation, so it covers both the model's own channels and the derived fallback.
        SynchronizeExpressions(result);
        Trace?.Add(TraceFrame, "expressions_synchronized", result);

        ShapeBlinks(result, dt);

        // Blink shaping keeps one rate limiter per eye, each with its own history, so two lids that
        // arrived here identical can leave it apart - most visibly while a squint holds them in the
        // middle of their range, where the limiters have the most room to disagree. Pull them back
        // together afterwards so the coupling describes what is actually sent.
        ResynchronizeLids(result);
        return result;
    }

    /// <summary>
    /// Pulls the two lids together, yielding to a deliberate wink. No-op when both lid keys are not
    /// present, or when the amount is zero.
    /// </summary>
    private void SynchronizeLids(OrderedFloatMap map, float dt)
    {
        if (!map.TryGetValue(LeftLid, out var left) || !map.TryGetValue(RightLid, out var right))
            return;

        var (syncedLeft, syncedRight) = _lidSync.Apply(left, right, dt);
        map[LeftLid] = ClampForKey(LeftLid, syncedLeft);
        map[RightLid] = ClampForKey(RightLid, syncedRight);
    }

    /// <summary>
    /// Runs BlinkGuard over the two eyes' gaze, holding it through a blink and vetting what comes
    /// back afterwards.
    /// </summary>
    /// <remarks>
    /// Reads the calibrated lid, which at this point is openness (1 = open), and converts to the
    /// closedness BlinkGuard's thresholds are stated in. Validity is the finite check: this pipeline
    /// has no confidence channel, so temporal consistency does the work instead.
    /// </remarks>
    private void StabilizeBlinkGaze(OrderedFloatMap map, float dt)
    {
        if (BlinkGuard is not { } guard)
            return;

        if (!map.TryGetValue(LeftLid, out var leftOpenness) ||
            !map.TryGetValue(RightLid, out var rightOpenness) ||
            !map.TryGetValue("/leftEyeX", out var leftX) ||
            !map.TryGetValue("/leftEyeY", out var leftY) ||
            !map.TryGetValue("/rightEyeX", out var rightX) ||
            !map.TryGetValue("/rightEyeY", out var rightY))
        {
            return;
        }

        var left = new GazeSample(leftX, leftY, 1f - leftOpenness, float.IsFinite(leftX) && float.IsFinite(leftY), 1f);
        var right = new GazeSample(rightX, rightY, 1f - rightOpenness, float.IsFinite(rightX) && float.IsFinite(rightY), 1f);

        var (outLeft, outRight) = guard.Process(left, right, dt);

        map["/leftEyeX"] = ClampForKey("/leftEyeX", outLeft.X);
        map["/leftEyeY"] = ClampForKey("/leftEyeY", outLeft.Y);
        map["/rightEyeX"] = ClampForKey("/rightEyeX", outRight.X);
        map["/rightEyeY"] = ClampForKey("/rightEyeY", outRight.Y);
    }

    /// <summary>
    /// Pulls the two eyes' gaze together, after per-eye centring has made them comparable.
    /// </summary>
    /// <remarks>
    /// <para>Here rather than in <c>ProcessExpressions</c>, which is where the older
    /// <c>GazeConjugateAmount</c> dial lives, because geometry runs <em>before</em>
    /// <see cref="EyeCalibrationProfile.MapGazeX"/>. At that point the two eyes still carry their own
    /// resting offsets - for the reporting user, 0.53 and 0.33 raw, a systematic 0.20 apart.
    /// Averaging two differently-biased signals and then subtracting different biases from the
    /// result leaves the eyes further apart than it found them. Once each eye is centred they are in
    /// one coordinate frame and averaging means something.</para>
    ///
    /// <para>Measured on a recorded session from the reporting user's own hardware: with a 9-frame
    /// smooth, the two eyes' horizontal motion correlates at only <b>0.29</b> (vertical, which
    /// geometry already forces to a single shared value, correlates at 0.59). The best time
    /// alignment is <b>0 ms</b> - the eyes are not lagging each other, they are moving
    /// semi-independently, which is what reads as one eye arriving before the other.</para>
    ///
    /// <para><see cref="EyeSyncSource.Stronger"/> falls back to Average: "stronger" on a gaze axis
    /// would mean "further from centre", which would drive both eyes to whichever extreme either
    /// camera guessed.</para>
    /// </remarks>
    private void SynchronizeGaze(OrderedFloatMap map)
    {
        // A wink makes one eye's gaze unreliable, and coupling would spread that onto the good eye.
        if (_gazeSyncAmount <= 0f || _lidSync.IsReleased)
            return;

        var source = _syncSource == EyeSyncSource.Stronger ? EyeSyncSource.Average : _syncSource;
        Pull("/leftEyeX", "/rightEyeX");
        Pull("/leftEyeY", "/rightEyeY");

        void Pull(string leftKey, string rightKey)
        {
            if (!map.TryGetValue(leftKey, out var left) || !map.TryGetValue(rightKey, out var right))
                return;

            var (syncedLeft, syncedRight) =
                EyeLidSynchronizer.Pull(left, right, _gazeSyncAmount, source);

            map[leftKey] = ClampForKey(leftKey, syncedLeft);
            map[rightKey] = ClampForKey(rightKey, syncedRight);
        }
    }

    /// <summary>
    /// Re-applies the lid pull after blink shaping, without re-running the wink state machine.
    /// </summary>
    /// <remarks>
    /// Deliberately pull-only. The state machine has already advanced once for this frame, and
    /// running it again would double-count <c>dt</c> against the re-couple dwell, making the
    /// coupling re-engage in half the time it claims to.
    /// </remarks>
    private void ResynchronizeLids(OrderedFloatMap map)
    {
        if (_lidSync.Amount <= 0f || _lidSync.IsReleased)
            return;

        if (!map.TryGetValue(LeftLid, out var left) || !map.TryGetValue(RightLid, out var right))
            return;

        var source = _lidSync.Source == EyeSyncSource.Stronger
            ? EyeSyncSource.Average
            : _lidSync.Source;
        var (syncedLeft, syncedRight) =
            EyeLidSynchronizer.Pull(left, right, _lidSync.Amount, source);

        map[LeftLid] = ClampForKey(LeftLid, syncedLeft);
        map[RightLid] = ClampForKey(RightLid, syncedRight);
    }

    /// <summary>
    /// Pulls squint and widen toward the chosen eye, for the same reason the lids are pulled: two
    /// cameras disagreeing about one face is not information.
    /// </summary>
    /// <remarks>
    /// <para>Off by default, unlike lid coupling. Which eye reads an expression better is personal —
    /// it depends on how the cameras happen to sit — so there is no default that is right for
    /// everyone, and averaging two channels when one is simply better drags the good one down.</para>
    ///
    /// <para>Released during a wink along with the lids. A wink genuinely does make the two eyes
    /// differ, including in squint, and copying the open eye's squint onto the winking one would
    /// undo the wink.</para>
    /// </remarks>
    private void SynchronizeExpressions(OrderedFloatMap map)
    {
        if (_lidSync.IsReleased)
            return;

        Pull(map, LeftSquint, RightSquint, _squintSyncAmount);
        Pull(map, LeftWiden, RightWiden, _widenSyncAmount);

        void Pull(OrderedFloatMap m, string leftKey, string rightKey, float amount)
        {
            if (amount <= 0f ||
                !m.TryGetValue(leftKey, out var left) || !m.TryGetValue(rightKey, out var right))
                return;

            var (syncedLeft, syncedRight) =
                EyeLidSynchronizer.Pull(left, right, amount, _syncSource);

            m[leftKey] = ClampForKey(leftKey, syncedLeft);
            m[rightKey] = ClampForKey(rightKey, syncedRight);
        }
    }

    /// <summary>
    /// Last stage: crisp blink onset, rate-limited reopen. Applied to whichever map is published,
    /// after everything that reasons about openness has already read it.
    /// </summary>
    private void ShapeBlinks(OrderedFloatMap map, float dt)
    {
        if (!_shapeBlinks)
            return;

        if (map.TryGetValue(LeftLid, out var left))
            map[LeftLid] = ClampForKey(LeftLid, _leftBlink.Apply(left, dt));

        if (map.TryGetValue(RightLid, out var right))
            map[RightLid] = ClampForKey(RightLid, _rightBlink.Apply(right, dt));
    }

    private OrderedFloatMap Derive(OrderedFloatMap map, float dt)
    {
        var augmented = GetAugmented(map);

        // Copy positionally: the augmented map's first Count entries are the source keys in order.
        map.ValuesSpan.CopyTo(augmented.ValuesSpan);

        _leftShapes.Update(map.TryGetValue(LeftLid, out var leftLid) ? leftLid : 1f, dt);
        _rightShapes.Update(map.TryGetValue(RightLid, out var rightLid) ? rightLid : 1f, dt);

        // Deriving "wider than open" from a channel that saturates at open is reading noise. Squint
        // still works there, because it lives below relaxed where the channel does have range.
        augmented[LeftWiden] = CanDeriveWiden(_leftCalibration) ? _leftShapes.Widen : 0f;
        augmented[LeftSquint] = _leftShapes.Squint;
        augmented[RightWiden] = CanDeriveWiden(_rightCalibration) ? _rightShapes.Widen : 0f;
        augmented[RightSquint] = _rightShapes.Squint;

        return augmented;
    }

    /// <summary>Uncalibrated eyes keep the old behaviour; nothing here knows better than they did.</summary>
    private static bool CanDeriveWiden(EyeCalibrationProfile? profile) =>
        profile is null || !profile.HasUsableOpennessRange || profile.HasWideHeadroom;

    private OrderedFloatMap GetAugmented(OrderedFloatMap source)
    {
        if (ReferenceEquals(_augmentedSource, source) && _augmented != null)
            return _augmented;

        // Source keys keep their positions, so the positional copy above stays valid.
        var keys = source.Keys.Concat(DerivedKeys.Where(k => !source.ContainsKey(k))).ToArray();
        _augmented = new OrderedFloatMap(keys);
        _augmentedSource = source;
        return _augmented;
    }

    private float ApplyCalibration(string key, float value)
    {
        var profile = key.StartsWith("/leftEye", StringComparison.Ordinal)
            ? _leftCalibration
            : key.StartsWith("/rightEye", StringComparison.Ordinal)
                ? _rightCalibration
                : null;
        if (profile is null)
            return value;

        return key switch
        {
            LeftLid or RightLid => profile.MapOpenness(value),
            "/leftEyeX" or "/rightEyeX" => profile.MapGazeX(value),
            "/leftEyeY" or "/rightEyeY" => profile.MapGazeY(value),
            _ => value,
        };
    }

    /// <summary>Output-space neutral: gaze centered, lids open, other eye shapes inactive.</summary>
    public static float NeutralForKey(string key) =>
        key is LeftLid or RightLid ? 1f : 0f;

    public static float ClampForKey(string key, float value) =>
        key is "/leftEyeX" or "/leftEyeY" or "/rightEyeX" or "/rightEyeY"
            ? Math.Clamp(value, -1f, 1f)
            : Math.Clamp(value, 0f, 1f);
}
