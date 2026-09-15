using System;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Personalization.Eye;

/// <summary>How one channel of one pose should be supervised.</summary>
/// <param name="Dim">Index into <see cref="EyePersonalizationSchema"/>.</param>
/// <param name="Target">Target in raw model space, or null to anchor at whatever the model said.</param>
/// <param name="Weight">Relative loss weight.</param>
public readonly record struct EyeChannelSupervision(int Dim, float? Target, float Weight)
{
    /// <summary>Pulls the residual to zero without asserting a value — "the model is right here".</summary>
    public bool IsAnchor => Target is null;
}

/// <summary>
/// One guided pose: what to do, optionally where to look, and what that licenses us to teach.
/// </summary>
/// <remarks>
/// Every channel not listed in <see cref="Supervision"/> is masked — the pose says nothing about it
/// and training must stay silent. That is the whole reason supervision is per-channel rather than a
/// whole vector: "look left" is ground truth about gaze and says nothing whatsoever about your
/// eyelids, and a vector-shaped label would quietly assert both.
/// </remarks>
public sealed record EyePose(
    string Id,
    string DisplayName,
    string Instruction,
    float? DotXDegrees,
    float? DotYDegrees,
    double HoldSeconds,
    int Repetitions,
    IReadOnlyList<EyeChannelSupervision> Supervision);

/// <summary>
/// The guided routine: which poses are recorded, and what each one is allowed to teach.
/// </summary>
/// <remarks>
/// <para>Everything is in <em>raw model space</em>, because that is where the corrector runs. Gaze
/// is a sigmoid where 0.5 is centre and full scale is
/// <see cref="EyePersonalizationSchema.GazeRangeDegrees"/>; lid is <em>closedness</em>, so a shut
/// eye is 1; widen, squint and brow pass through the geometry stage untouched, so raw and output
/// space agree for them.</para>
///
/// <para>Dot positions stay inside roughly ±18° horizontally and +6°/−13° vertically, which is what
/// the headset overlay panel can actually reach. A target the user cannot look at directly is worse
/// than no target: they would look at the edge of the panel and we would record that as ground
/// truth for a position they never fixated.</para>
///
/// <para>The masks are the interesting part of the table, and most of them exist because of a
/// specific way a naive label would lie:</para>
/// <list type="bullet">
/// <item><description>Looking down genuinely lowers the eyelids, so lid stays masked on every gaze
/// pose — otherwise "look down" would teach the model that looking down means eyes open.</description></item>
/// <item><description>A hard squint genuinely narrows the eye, so lid is masked there too.</description></item>
/// <item><description>Closing hard squeezes the eye, which looks like squint; squint is masked
/// while closed.</description></item>
/// <item><description>Blinking is not held, so it teaches no lid value at all. That pose exists
/// only to say "squint should not fire while you blink", at low weight.</description></item>
/// </list>
/// </remarks>
public static class EyeGuidedCues
{
    // Schema indices, named so the table below reads as intent rather than arithmetic.
    private const int RightY = 0, RightX = 1, RightLid = 2, RightWiden = 3, RightSquint = 4, RightBrow = 5;
    private const int LeftY = 6, LeftX = 7, LeftLid = 8, LeftWiden = 9, LeftSquint = 10, LeftBrow = 11;

    private static readonly int[] GazeDims = [RightY, RightX, LeftY, LeftX];
    private static readonly int[] WidenDims = [RightWiden, LeftWiden];
    private static readonly int[] SquintDims = [RightSquint, LeftSquint];
    private static readonly int[] LidDims = [RightLid, LeftLid];
    private static readonly int[] BrowDims = [RightBrow, LeftBrow];

    /// <summary>Raw value for a gaze angle: 0.5 is centre, full scale is the model's gaze range.</summary>
    public static float RawFromDegrees(float degrees) =>
        Math.Clamp(degrees / EyePersonalizationSchema.GazeRangeDegrees, -1f, 1f) * 0.5f + 0.5f;

    /// <summary>Inverse of <see cref="RawFromDegrees"/>, for reading recorded targets back.</summary>
    public static float DegreesFromRaw(float raw) =>
        (Math.Clamp(raw, 0f, 1f) - 0.5f) * 2f * EyePersonalizationSchema.GazeRangeDegrees;

    /// <summary>Panel coordinate for the headset dot, −1..1 across the overlay's reach.</summary>
    public const float PanelHalfWidthDegrees = 20f;

    public static float PanelFromDegrees(float degrees) =>
        Math.Clamp(degrees / PanelHalfWidthDegrees, -1f, 1f);

    /// <summary>
    /// Vertical panel coordinate, which is <em>screen</em>-space and therefore inverted.
    /// </summary>
    /// <remarks>
    /// Gaze degrees are positive upward; the overlay draws at <c>y = centre + TargetY * scale</c> in
    /// canvas coordinates, where positive is downward. Without the negation a "look up" cue would
    /// put the dot below centre and the recorded fixation would be labelled with the opposite of
    /// where the user actually looked - a sign error that produces a confidently inverted vertical
    /// correction rather than an obvious failure.
    /// </remarks>
    public static float PanelYFromDegrees(float degrees) => PanelFromDegrees(-degrees);

    private static EyeChannelSupervision At(int dim, float target, float weight = 1f) =>
        new(dim, target, weight);

    private static IEnumerable<EyeChannelSupervision> All(IEnumerable<int> dims, float target, float weight) =>
        dims.Select(dim => new EyeChannelSupervision(dim, target, weight));

    private static IEnumerable<EyeChannelSupervision> Anchor(IEnumerable<int> dims, float weight) =>
        dims.Select(dim => new EyeChannelSupervision(dim, null, weight));

    private static EyePose Gaze(string id, string name, float xDegrees, float yDegrees) => new(
        Id: id,
        DisplayName: name,
        Instruction: "Look at the dot and hold still.",
        DotXDegrees: xDegrees,
        DotYDegrees: yDegrees,
        HoldSeconds: 2.5,
        Repetitions: 1,
        Supervision:
        [
            At(RightX, RawFromDegrees(xDegrees)), At(LeftX, RawFromDegrees(xDegrees)),
            At(RightY, RawFromDegrees(yDegrees)), At(LeftY, RawFromDegrees(yDegrees)),
            // Lid, widen, squint and brow are all masked: where you look says nothing about them,
            // and looking down genuinely lowers the lids.
        ]);

    /// <summary>The routine, in the order it is performed.</summary>
    public static IReadOnlyList<EyePose> Poses { get; } =
    [
        new("relaxed_center", "Relax, look ahead",
            "Relax your face and look straight at the dot.",
            0f, 0f, 3.0, 1,
            [
                .. All(GazeDims, RawFromDegrees(0f), 1f),
                // A resting face is the one moment we can assert the absence of an expression.
                .. All(WidenDims, 0f, 0.5f),
                .. All(SquintDims, 0f, 0.5f),
            ]),

        Gaze("gaze_left", "Look left", -18f, 0f),
        Gaze("gaze_right", "Look right", 18f, 0f),
        Gaze("gaze_up", "Look up", 0f, 6f),
        Gaze("gaze_down", "Look down", 0f, -13f),
        Gaze("gaze_up_left", "Look up-left", -13f, 5f),
        Gaze("gaze_up_right", "Look up-right", 13f, 5f),
        Gaze("gaze_down_left", "Look down-left", -13f, -10f),
        Gaze("gaze_down_right", "Look down-right", 13f, -10f),

        new("eyes_normal", "Eyes normally open",
            "Open your eyes normally, without staring or squinting.",
            null, null, 3.0, 1,
            [
                .. All(WidenDims, 0f, 0.75f),
                .. All(SquintDims, 0f, 0.75f),
                .. Anchor(LidDims, 0.5f),
                .. Anchor(BrowDims, 0.3f),
            ]),

        new("wide_moderate", "Open a little wide",
            "Open your eyes moderately wide, as if mildly surprised.",
            0f, 0f, 2.5, 1,
            [
                .. All(WidenDims, 0.5f, 1f),
                .. All(SquintDims, 0f, 0.75f),
                .. All(GazeDims, RawFromDegrees(0f), 0.5f),
            ]),

        new("wide_strong", "Open very wide",
            "Open your eyes as wide as you comfortably can.",
            0f, 0f, 2.5, 1,
            [
                .. All(WidenDims, 1f, 1f),
                .. All(SquintDims, 0f, 0.75f),
                .. All(GazeDims, RawFromDegrees(0f), 0.5f),
            ]),

        new("squint_light", "Squint slightly",
            "Narrow your eyes slightly, as if in soft light.",
            0f, 0f, 2.5, 1,
            [
                .. All(SquintDims, 0.4f, 1f),
                .. All(WidenDims, 0f, 0.75f),
                .. All(GazeDims, RawFromDegrees(0f), 0.5f),
            ]),

        new("squint_strong", "Squint hard",
            "Squint hard, as if in bright sun - but keep your eyes open.",
            null, null, 2.5, 1,
            [
                .. All(SquintDims, 0.9f, 1f),
                .. All(WidenDims, 0f, 0.75f),
                // Lid masked: a hard squint really does narrow the eye, and gaze masked because
                // it is hard to aim accurately while squinting.
            ]),

        new("eyes_closed", "Close your eyes",
            "Close both eyes gently and keep them closed.",
            null, null, 3.0, 1,
            [
                .. All(LidDims, 1f, 1f),
                .. All(WidenDims, 0f, 0.5f),
                // Squint masked: squeezing shut looks exactly like squinting.
            ]),

        new("wink_left", "Wink your left eye",
            "Close your LEFT eye only, keeping the right one open and relaxed.",
            null, null, 2.0, 2,
            [
                At(LeftLid, 1f),
                new(RightLid, null, 0.75f),
                At(RightY, RawFromDegrees(0f), 0.5f), At(RightX, RawFromDegrees(0f), 0.5f),
                At(RightWiden, 0f, 0.5f),
            ]),

        new("wink_right", "Wink your right eye",
            "Close your RIGHT eye only, keeping the left one open and relaxed.",
            null, null, 2.0, 2,
            [
                At(RightLid, 1f),
                new(LeftLid, null, 0.75f),
                At(LeftY, RawFromDegrees(0f), 0.5f), At(LeftX, RawFromDegrees(0f), 0.5f),
                At(LeftWiden, 0f, 0.5f),
            ]),

        new("blinks", "Blink normally",
            "Blink normally, about five times, at your own pace.",
            null, null, 8.0, 1,
            [
                // No lid supervision at all: a commanded value describes a moving lid badly. This
                // pose exists to say squint must not fire during a blink, and nothing else.
                .. All(SquintDims, 0f, 0.25f),
            ]),
    ];

    /// <summary>Phase lengths. Short compared with the face routine: a dot needs no preparation.</summary>
    public sealed record Timing(
        double PrepSeconds = 1.5,
        double TransitionSeconds = 0.7,
        double RestSeconds = 0.6)
    {
        public static readonly Timing Default = new();
    }

    /// <summary>
    /// The neutral raw vector a pose departs from and returns to: gaze centred, everything else at
    /// rest. Lid rest is 0 because raw lid is closedness.
    /// </summary>
    public static float[] NeutralVector()
    {
        var vector = new float[EyePersonalizationSchema.ExpressionCount];
        foreach (var dim in GazeDims)
            vector[dim] = RawFromDegrees(0f);
        return vector;
    }

    /// <summary>The commanded vector for a pose: its explicit targets over the neutral vector.</summary>
    public static float[] TargetVector(EyePose pose)
    {
        var vector = NeutralVector();
        foreach (var channel in pose.Supervision)
        {
            if (channel.Target is { } target)
                vector[channel.Dim] = target;
        }
        return vector;
    }

    /// <summary>
    /// Expands the pose list into the step sequence <see cref="GuidedCaptureRoutine"/> executes.
    /// </summary>
    /// <remarks>
    /// Each repetition is prep, transition in, hold, transition out, rest. Only the hold is
    /// trustworthy — the transitions are the face still arriving — and the trainer additionally
    /// discards <see cref="GuidedCaptureRoutine.HoldSettleTrimSeconds"/> from the front of it.
    /// </remarks>
    public static IReadOnlyList<CueStep> BuildSteps(
        IReadOnlyList<EyePose>? poses = null, Timing? timing = null)
    {
        poses ??= Poses;
        timing ??= Timing.Default;

        var steps = new List<CueStep>();
        var neutral = NeutralVector();

        foreach (var pose in poses)
        {
            var target = TargetVector(pose);
            var dims = pose.Supervision.Select(channel => channel.Dim).ToArray();

            for (var repetition = 0; repetition < Math.Max(1, pose.Repetitions); repetition++)
            {
                steps.Add(new CueStep(pose.Id, "prep", dims, neutral, neutral,
                    timing.PrepSeconds, 1f, repetition,
                    $"Next: {pose.DisplayName}", pose.DisplayName, pose.Repetitions));

                steps.Add(new CueStep(pose.Id, "transition", dims, neutral, target,
                    timing.TransitionSeconds, 1f, repetition,
                    pose.Instruction, pose.DisplayName, pose.Repetitions));

                steps.Add(new CueStep(pose.Id, "hold", dims, target, target,
                    pose.HoldSeconds, 1f, repetition,
                    pose.Instruction, pose.DisplayName, pose.Repetitions));

                steps.Add(new CueStep(pose.Id, "transition", dims, target, neutral,
                    timing.TransitionSeconds, 1f, repetition,
                    "Relax.", pose.DisplayName, pose.Repetitions));

                steps.Add(new CueStep(pose.Id, "rest", dims, neutral, neutral,
                    timing.RestSeconds, 1f, repetition,
                    "Relax.", pose.DisplayName, pose.Repetitions));
            }
        }

        return steps;
    }

    /// <summary>Total wall time of the routine, for telling the user what they are signing up for.</summary>
    public static double TotalSeconds(
        IReadOnlyList<EyePose>? poses = null, Timing? timing = null) =>
        BuildSteps(poses, timing).Sum(step => step.DurationSeconds);
}
