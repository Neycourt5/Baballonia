namespace Baballonia.Services.EyeV2;

/// <summary>
/// Optional final eye-state stage. <paramref name="stockState"/> is the existing postprocessed
/// six-vector; <paramref name="filteredRawState"/> preserves the per-eye model values from the same
/// tick. Keeping both lets V2 improve per-eye gaze without moving the stage ahead of stock
/// postprocessing. A null mapper is the exact legacy path.
/// </summary>
public interface IEyeStateMapper
{
    float[] Map(float[] stockState, float[] filteredRawState, long timestampTicks);
    void Reset();
}
