namespace Baballonia.Services.Personalization;

/// <summary>
/// The single mapping between the Train page's model dropdown, the trainer's <c>--model</c> flag,
/// and the <c>adapter_type</c> the exported ONNX carries.
///
/// It lives in one place because the three names for the same thing are easy to drift apart, and a
/// drift is invisible: picking "B" and silently training A produces a perfectly plausible-looking
/// result. The round trip (dropdown index -> flag -> adapter type -> label) is covered by tests.
/// </summary>
public static class TrainingModelChoice
{
    /// <summary>Dropdown index of the output-only adapter. This is the default.</summary>
    public const int OutputOnlyIndex = 0;

    /// <summary>Dropdown index of the image-conditioned adapter.</summary>
    public const int ImageConditionedIndex = 1;

    public const string OutputOnlyAdapterType = "output_mlp_v1";
    public const string ImageConditionedAdapterType = "image_residual_v1";

    /// <summary>The trainer's <c>--model</c> value for a dropdown index.</summary>
    public static string KindForIndex(int index) =>
        index == ImageConditionedIndex ? "b" : "a";

    /// <summary>The <c>adapter_type</c> a given dropdown index is expected to produce.</summary>
    public static string AdapterTypeForIndex(int index) =>
        index == ImageConditionedIndex ? ImageConditionedAdapterType : OutputOnlyAdapterType;

    /// <summary>Human-readable name for an <c>adapter_type</c> read back from a trained model.</summary>
    public static string DisplayName(string? adapterType) => adapterType switch
    {
        OutputOnlyAdapterType => "Model A (expressions only)",
        ImageConditionedAdapterType => "Model B (expressions + camera image)",
        null or "" or "unknown" => "unknown model",
        _ => adapterType
    };
}
