using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Personalization;

/// <summary>
/// A pickable guided session: what it covers, how it reads in the dropdown, and how long it takes.
/// </summary>
/// <remarks>
/// Separate from <see cref="GuidedCue"/> so the UI can offer sensible *groupings* without the cue
/// definitions knowing anything about menus. The user picks an outcome ("everything", "just the
/// jaw"), not a list of expression indices.
/// </remarks>
public sealed record GuidedRoutineChoice(
    string Id,
    string DisplayName,
    string Description,
    IReadOnlyList<GuidedCue> Cues,
    int Repetitions = 3)
{
    /// <summary>Total length, so the picker can say what the user is committing to.</summary>
    public double EstimatedSeconds =>
        Cues.Sum(c => c.EstimatedSeconds(Repetitions));

    public IReadOnlyList<CueStep> Build() =>
        GuidedCaptureRoutine.BuildRoutine(Cues, Repetitions);

    /// <summary>
    /// The offered sessions, most useful first.
    /// </summary>
    /// <remarks>
    /// The full pass uses two repetitions rather than three: at eight expressions the third
    /// repetition adds several minutes for frames that are highly correlated with the first two,
    /// and a pass the user abandons halfway teaches nothing. Single-expression sessions keep three,
    /// where the extra repetition is cheap and the added independence is worth having.
    /// </remarks>
    public static IReadOnlyList<GuidedRoutineChoice> All { get; } = BuildAll();

    private static IReadOnlyList<GuidedRoutineChoice> BuildAll()
    {
        var choices = new List<GuidedRoutineChoice>
        {
            new("core", "Full pass (recommended)",
                "Covers the eight expressions worth teaching, one after another.",
                GuidedCues.CorePass, Repetitions: 2),

            new("jaw", "Jaw only",
                "Just the jaw - the expression most likely to misbehave.",
                [GuidedCues.JawOpen]),

            new("combos", "Combinations",
                "Two expressions at once, which teaches how they interfere with each other.",
                GuidedCues.CombinationPass, Repetitions: 2),

            new("asymmetric", "One-sided smiles",
                "Left and right smiles separately, so they stop mirroring each other. " +
                "Skip this if you cannot hold a one-sided smile.",
                [GuidedCues.SmileLeft, GuidedCues.SmileRight], Repetitions: 2),
        };

        // Every single expression, for topping up one weak spot without redoing the whole pass.
        choices.AddRange(GuidedCues.CorePass
            .Where(cue => cue.Id != GuidedCues.JawOpen.Id)
            .Select(cue => new GuidedRoutineChoice(
                cue.Id, $"{cue.DisplayName} only",
                $"Just {cue.DisplayName.ToLowerInvariant()}.",
                [cue])));

        return choices;
    }

    public override string ToString() => DisplayName;
}
