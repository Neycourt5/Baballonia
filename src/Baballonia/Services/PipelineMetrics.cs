using Baballonia.Services.Inference.VideoSources;

namespace Baballonia.Services;

/// <summary>
/// Monotonic throughput counters for the pipeline, surfaced on the Debug page. Producers just increment
/// on their own thread; the Debug view-model samples deltas against a monotonic clock to derive rates.
/// </summary>
public sealed class PipelineMetrics
{
    public long UiTicks;
    public long EyeInferences;
    public long FaceInferences;

    /// <summary>
    /// Frames the corruption detector rejected before inference. Previously dropped silently, which
    /// made "tracking went quiet in dim light" indistinguishable from "the camera stalled".
    /// </summary>
    public long EyeCorruptFrames;

    // Active camera sources, published by the workers; the sampler reads frame stats off them live.
    // A single split eye feed sets only EyeLeftSource (EyeDual false); two independent feeds set both.
    public volatile SingleCameraSource? EyeLeftSource;
    public volatile SingleCameraSource? EyeRightSource;
    public volatile bool EyeDual;
    public volatile SingleCameraSource? FaceSource;

    // Per-stage processing time (EWMA, milliseconds). Written by the respective inference worker while
    // it holds the pipeline's SyncRoot (so writes are serialised); read on the UI thread for display.
    // Together with per-thread CPU%, these answer "which stage of a hot thread is the hotspot".
    public double EyeCaptureMs;
    public double EyeTransformMs;
    public double EyeInferenceMs;
    public double EyePostMs;

    public double FaceCaptureMs;
    public double FaceTransformMs;
    public double FaceInferenceMs;
    public double FacePostMs;

    // Live eye output, sampled once per frame for the Debug page. Plain floats: a 32-bit aligned
    // write is atomic, a torn *set* across fields is harmless on a display refreshed twice a second,
    // and this keeps the per-frame path allocation-free.
    public float EyeLeftRawOpenness;
    public float EyeLeftOpenness;
    public float EyeLeftWiden;
    public float EyeLeftSquint;
    public float EyeLeftGazeX;
    public float EyeLeftGazeY;

    public float EyeRightRawOpenness;
    public float EyeRightOpenness;
    public float EyeRightWiden;
    public float EyeRightSquint;
    public float EyeRightGazeX;
    public float EyeRightGazeY;

    /// <summary>True while widen/squint are being derived because the model does not supply them.</summary>
    public volatile bool EyeWidenSquintDerived;

    /// <summary>
    /// True while a wink has released eye coupling. Worth showing: a coupling that has latched
    /// itself off looks exactly like a coupling that was never enabled.
    /// </summary>
    public volatile bool EyeSyncReleased;

    /// <summary>BlinkGuard's live state, for the Debug page. Diagnostics only.</summary>
    public volatile bool EyeBlinkGuardEnabled;
    public volatile bool EyeBlinkGuardIntervening;
    public long EyeBlinkGuardGlitches;

    /// <summary>Time spent in the personalized eye corrector (EWMA, milliseconds).</summary>
    public double EyeCorrectMs;

    /// <summary>
    /// Mean absolute change the eye corrector made this frame. Zero when none is installed, which
    /// is what tells "the corrector is off" apart from "the corrector is doing nothing".
    /// </summary>
    public float EyeCorrectorDelta;

    /// <summary>Exponential moving average that seeds from the first sample (so it converges quickly).</summary>
    public static double Ewma(double previous, double sample) =>
        previous <= 0 ? sample : previous * 0.9 + sample * 0.1;
}
