using System;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Personalization;

/// <summary>
/// One expression the avatar can demonstrate: which dimensions it drives, at what intensities, and
/// what to tell the user while it happens.
/// </summary>
/// <remarks>
/// Data, not code. Adding an expression to guided capture is a row in <see cref="GuidedCues"/>, and
/// everything downstream - the cue engine, the recorder's stamps, the labeller's weighting, the
/// coverage report - already handles arbitrary cued dimensions.
///
/// <para><b>On intensity.</b> 1.0 means "the user's own maximum", not the avatar's rendered
/// extremity. The avatar teaches the *shape*; the commanded fraction defines the label. That is what
/// keeps the resulting model avatar-agnostic, and it is why an expression nobody can modulate
/// (a tongue is either out or it is not) declares a single level instead of pretending otherwise.</para>
/// </remarks>
/// <param name="Id">Stable prefix for the stamped cue id, e.g. "Smile" becomes "Smile50"/"Smile100".</param>
/// <param name="DisplayName">What the routine picker shows.</param>
/// <param name="Dims">Expression indices this cue deliberately drives.</param>
/// <param name="Action">Imperative used to build instructions, e.g. "Smile" or "Open your jaw".</param>
/// <param name="Levels">Commanded intensities. Defaults to 50 % and 100 %.</param>
/// <param name="DimScale">
/// Per-dimension multiplier applied to the level. Combination cues use it so a secondary expression
/// can be commanded gently while the primary goes to full - "smile with your mouth a bit open" is a
/// pose a person can actually hold, whereas both at maximum usually is not.
/// </param>
public sealed record GuidedCue(
    string Id,
    string DisplayName,
    IReadOnlyList<int> Dims,
    string Action,
    IReadOnlyList<float>? Levels = null,
    IReadOnlyDictionary<int, float>? DimScale = null)
{
    /// <summary>Intensities actually commanded, falling back to the shared default ladder.</summary>
    public IReadOnlyList<float> EffectiveLevels =>
        Levels is { Count: > 0 } ? Levels : GuidedCaptureRoutine.DefaultLevels;

    /// <summary>Whether this expression is commanded at a single intensity (on/off).</summary>
    public bool IsBinary => EffectiveLevels.Count == 1;

    /// <summary>The full 45-vector commanded at <paramref name="level"/>.</summary>
    public float[] TargetAt(float level)
    {
        var target = new float[PersonalizationSchema.ExpressionCount];

        foreach (var dim in Dims)
        {
            if (dim < 0 || dim >= target.Length)
                continue;

            var scale = DimScale is not null && DimScale.TryGetValue(dim, out var s) ? s : 1f;
            target[dim] = Math.Clamp(level * scale, 0f, 1f);
        }

        return target;
    }

    public string LeadInInstruction => $"Next: {DisplayName}. Relax your face and watch your avatar.";

    /// <summary>
    /// Shown during the countdown before an attempt. Names the expression in words as well as
    /// showing it on the avatar: a subtle shape on someone's avatar - a small smile especially - is
    /// easy to miss entirely, and a cue the user never noticed still gets recorded as though they
    /// performed it.
    /// </summary>
    public string PrepInstruction => $"Get ready to {Action.ToLowerInvariant()}. Watch your avatar.";

    public string RestInstruction => "Rest. Let your face relax completely.";

    public string ApproachInstruction(float level) =>
        IsBinary ? $"{Action} to match the avatar..." : $"{Action} to match the avatar ({Percent(level)}%)...";

    public string HoldInstruction(float level) =>
        IsBinary ? "Hold it - match the avatar." : $"Hold it - match the avatar ({Percent(level)}%).";

    private static int Percent(float level) => (int)Math.Round(level * 100);

    /// <summary>Length of a routine built from this cue alone, for showing the user up front.</summary>
    public double EstimatedSeconds(int repetitions, GuidedCaptureRoutine.CueTiming? timing = null)
    {
        timing ??= GuidedCaptureRoutine.CueTiming.Default;
        var perBlock = timing.PrepSeconds + timing.TransitionSeconds * 2 +
                       timing.HoldSeconds + timing.RestSeconds;
        return timing.LeadInSeconds + perBlock * EffectiveLevels.Count * repetitions;
    }
}

/// <summary>
/// The expressions guided capture can teach, and the passes that group them.
/// </summary>
/// <remarks>
/// Chosen for what a person can actually reproduce on demand while watching an avatar, which is a
/// stricter filter than "what the schema has a slot for". A cue nobody can imitate reliably produces
/// confident wrong labels at full weight - worse than no cue at all - so the marginal ones
/// (asymmetric smiles, tongue) are offered separately rather than folded into the standard pass.
/// </remarks>
public static class GuidedCues
{
    private static int Ix(string name) => PersonalizationSchema.IndexOf(name);

    /// <summary>Single commanded level: an expression that is either happening or not.</summary>
    private static readonly float[] Binary = [1.0f];

    // ---- single expressions --------------------------------------------------------------------

    /// <summary>The complaint this whole phase exists to fix.</summary>
    public static GuidedCue JawOpen { get; } = new(
        "JawOpen", "Jaw open", [Ix("JawOpen")], "Open your jaw");

    /// <summary>
    /// Both corners together. Symmetric because a deliberate one-sided smile is a skill, and an
    /// unreproducible cue is worse than a missing one; the asymmetric variants below are opt-in.
    /// </summary>
    public static GuidedCue Smile { get; } = new(
        "Smile", "Smile", [Ix("MouthSmileLeft"), Ix("MouthSmileRight")], "Smile");

    public static GuidedCue Frown { get; } = new(
        "Frown", "Frown", [Ix("MouthFrownLeft"), Ix("MouthFrownRight")], "Frown");

    public static GuidedCue Pucker { get; } = new(
        "Pucker", "Pucker", [Ix("MouthPucker")], "Pucker your lips");

    public static GuidedCue Funnel { get; } = new(
        "Funnel", "Funnel", [Ix("MouthFunnel")], "Make an 'oh' shape");

    public static GuidedCue MouthLeft { get; } = new(
        "MouthLeft", "Mouth left", [Ix("MouthLeft")], "Slide your mouth left");

    public static GuidedCue MouthRight { get; } = new(
        "MouthRight", "Mouth right", [Ix("MouthRight")], "Slide your mouth right");

    /// <summary>
    /// Binary, and deliberately so. Tongue extension has no reproducible middle setting, and the
    /// labeller already discounts tongue holds because the camera sees them poorly.
    /// </summary>
    public static GuidedCue TongueOut { get; } = new(
        "TongueOut", "Tongue out", [Ix("TongueOut")], "Stick your tongue out", Binary);

    // ---- asymmetric, opt-in --------------------------------------------------------------------

    /// <summary>
    /// Left corner only. Without a cue that separates the corners, nothing in the corpus ever
    /// distinguishes them, so a personal model can only ever mirror them. Offered separately because
    /// not everyone can hold a one-sided smile, and a cue the user cannot follow is poison.
    /// </summary>
    public static GuidedCue SmileLeft { get; } = new(
        "SmileLeft", "Smile (left only)", [Ix("MouthSmileLeft")], "Smile with the left side only");

    public static GuidedCue SmileRight { get; } = new(
        "SmileRight", "Smile (right only)", [Ix("MouthSmileRight")], "Smile with the right side only");

    // ---- combinations --------------------------------------------------------------------------
    //
    // Combinations exist because cross-talk is a property of expressions happening *together*, and a
    // corpus of one-at-a-time cues cannot teach it. The jaw is commanded gently (0.6) so the pose
    // stays one a person can actually hold and imitate.

    private static IReadOnlyDictionary<int, float> GentleJaw() =>
        new Dictionary<int, float> { [Ix("JawOpen")] = 0.6f };

    public static GuidedCue SmileWithJaw { get; } = new(
        "SmileJaw", "Smile + open jaw",
        [Ix("MouthSmileLeft"), Ix("MouthSmileRight"), Ix("JawOpen")],
        "Smile with your mouth open", DimScale: GentleJaw());

    public static GuidedCue FrownWithJaw { get; } = new(
        "FrownJaw", "Frown + open jaw",
        [Ix("MouthFrownLeft"), Ix("MouthFrownRight"), Ix("JawOpen")],
        "Frown with your mouth open", DimScale: GentleJaw());

    public static GuidedCue PuckerWithJaw { get; } = new(
        "PuckerJaw", "Pucker + open jaw",
        [Ix("MouthPucker"), Ix("JawOpen")],
        "Pucker with your jaw a little open", DimScale: GentleJaw());

    // ---- passes --------------------------------------------------------------------------------

    /// <summary>Everything, for the picker.</summary>
    public static IReadOnlyList<GuidedCue> All { get; } =
    [
        JawOpen, Smile, Frown, Pucker, Funnel, MouthLeft, MouthRight, TongueOut,
        SmileLeft, SmileRight, SmileWithJaw, FrownWithJaw, PuckerWithJaw
    ];

    /// <summary>
    /// The standard pass: the expressions most worth having, in an order that alternates mouth
    /// shapes so the face is not held in one configuration for minutes at a time.
    /// </summary>
    public static IReadOnlyList<GuidedCue> CorePass { get; } =
    [
        JawOpen, Smile, Frown, Pucker, Funnel, MouthLeft, MouthRight, TongueOut
    ];

    /// <summary>Combination cues, which teach cross-talk that one-at-a-time cues cannot.</summary>
    public static IReadOnlyList<GuidedCue> CombinationPass { get; } =
    [
        SmileWithJaw, FrownWithJaw, PuckerWithJaw
    ];

    /// <summary>Lookup by <see cref="GuidedCue.Id"/>, for restoring a saved picker selection.</summary>
    public static GuidedCue? ById(string id) =>
        All.FirstOrDefault(c => string.Equals(c.Id, id, StringComparison.OrdinalIgnoreCase));
}
