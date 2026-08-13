using System;
using System.Collections.Generic;

namespace Baballonia.Services.Personalization;

/// <summary>
/// An immutable snapshot of what the guided-calibration cue engine wants the avatar to be doing
/// right now. The engine swaps whole snapshots rather than mutating shared state, so the sender
/// loop can sample it from another thread without locking.
///
/// The target is interpolated from <see cref="FromVector"/> to <see cref="ToVector"/> across
/// <see cref="DurationSeconds"/>. A hold is simply a phase where the two vectors are equal.
/// </summary>
/// <param name="CueId">Stable identifier of the cue, stamped into the dataset (e.g. "SmileHold50").</param>
/// <param name="PhaseName">Phase within the cue: prep, transition, hold, rest. Drives label weighting.</param>
/// <param name="Dims">
/// Indices this cue deliberately drives. Needed alongside the vector because a commanded 0 on a
/// cued dimension ("relax your jaw") means something different from an uncommanded dimension.
/// </param>
/// <param name="FromVector">Full 45-length start vector, canonical schema order.</param>
/// <param name="ToVector">Full 45-length end vector, canonical schema order.</param>
/// <param name="DurationSeconds">Phase length; 0 or less pins the target at <see cref="ToVector"/>.</param>
/// <param name="StartTimestamp">Stopwatch timestamp when the phase began.</param>
/// <param name="Level">Nominal intensity of the cue (0..1), recorded as a label convenience.</param>
/// <param name="RepetitionIndex">Which repetition of the cue this is.</param>
public sealed record CuePhase(
    string CueId,
    string PhaseName,
    IReadOnlyList<int> Dims,
    float[] FromVector,
    float[] ToVector,
    double DurationSeconds,
    long StartTimestamp,
    float Level = 0f,
    int RepetitionIndex = 0)
{
    /// <summary>Validates lengths early; a malformed cue must not reach the avatar.</summary>
    public void Validate()
    {
        if (FromVector.Length != PersonalizationSchema.ExpressionCount ||
            ToVector.Length != PersonalizationSchema.ExpressionCount)
        {
            throw new ArgumentException(
                $"Cue '{CueId}' vectors must have length {PersonalizationSchema.ExpressionCount} " +
                $"(got {FromVector.Length}/{ToVector.Length}).");
        }
    }
}
