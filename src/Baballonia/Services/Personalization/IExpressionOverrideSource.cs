namespace Baballonia.Services.Personalization;

/// <summary>
/// Supplies an independently generated face expression target that temporarily replaces live
/// tracked output on the way to VRChat.
///
/// This is what makes avatar-guided calibration possible: during a guided session the cue system
/// commands the avatar directly (through Baballonia's existing OSC -> VRCFT path), the user imitates
/// what they see, and the commanded vector - never the model's own prediction - becomes the
/// training label. Keeping generation here, entirely separate from inference, is what keeps the
/// training loop non-circular.
///
/// The camera, stock inference and dataset recording all keep running normally while an override is
/// active; only the values sent downstream are replaced.
/// </summary>
public interface IExpressionOverrideSource
{
    /// <summary>
    /// The face expression vector to send right now, or null to send live tracked values.
    ///
    /// Called from the sender loop (roughly every 10-16 ms) rather than pushed from the cue engine,
    /// so the animation is sampled on the same clock that transmits it. Implementations must be
    /// non-blocking and safe to call from a background thread.
    /// </summary>
    /// <returns>
    /// A vector of length <see cref="PersonalizationSchema.ExpressionCount"/> in canonical schema
    /// order, with values in [0,1], or null when no override is in effect.
    /// </returns>
    float[]? SampleTarget();
}
