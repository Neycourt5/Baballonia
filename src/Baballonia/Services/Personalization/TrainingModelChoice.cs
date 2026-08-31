using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Personalization;

/// <summary>
/// The single mapping between the Train page, the trainer's <c>--model</c> flag, and the adapter
/// metadata written into the exported ONNX. Keeping the three names together prevents a dropdown
/// choice from silently training a different architecture.
/// </summary>
public static class TrainingModelChoice
{
    public const int OutputOnlyIndex = 0;
    public const int ImageConditionedIndex = 1;
    public const int SharedFeaturesIndex = 2;

    public const string OutputOnlyAdapterType = "output_mlp_v1";
    public const string ImageConditionedAdapterType = "image_residual_v1";
    public const string SharedFeaturesAdapterType = "embedding_head_v1";

    public sealed record Option(
        int Index,
        string Kind,
        string AdapterType,
        string Label,
        string DisplayName,
        string Description,
        bool RequiresEmbedding);

    public static IReadOnlyList<Option> Options { get; } =
    [
        new(
            OutputOnlyIndex,
            "a",
            OutputOnlyAdapterType,
            "A — Expressions only (simple)",
            "Model A (expressions only)",
            "Learns from the 45 expression values only. It is fast and small, but never sees " +
            "your face. This is the simple reference model, not the recommended winner.",
            false),
        new(
            ImageConditionedIndex,
            "b",
            ImageConditionedAdapterType,
            "B — Expressions + camera image (current best)",
            "Model B (camera image)",
            "Learns its own small camera-image network alongside the 45 expression values. It " +
            "takes longer to train, but it is the proven real-world baseline for the B-vs-C test.",
            false),
        new(
            SharedFeaturesIndex,
            "c",
            SharedFeaturesAdapterType,
            "C — Shared visual features (experimental)",
            "Model C (shared visual features)",
            "Reuses visual features already computed inside the stock face model instead of " +
            "running a second camera CNN. Experimental until it beats Model B in the home test.",
            true)
    ];

    public static Option ForIndex(int index) =>
        index >= 0 && index < Options.Count ? Options[index] : Options[OutputOnlyIndex];

    public static string KindForIndex(int index) => ForIndex(index).Kind;

    public static string AdapterTypeForIndex(int index) => ForIndex(index).AdapterType;

    public static Option? ForAdapterType(string? adapterType) =>
        Options.FirstOrDefault(option => option.AdapterType == adapterType);

    public static bool RequiresEmbedding(string? kind) =>
        string.Equals(kind, Options[SharedFeaturesIndex].Kind, System.StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Guard shared by the GUI state and the train command. A/B keep their existing requirements;
    /// C cannot start until both its runtime and recorded feature sidecars are ready.
    /// </summary>
    public static string? TrainingUnavailableReason(
        string? kind,
        bool ordinaryPrerequisitesReady,
        bool modelCReady)
    {
        if (kind is not ("a" or "A" or "b" or "B" or "c" or "C"))
            return $"Unknown training model '{kind}'. Choose A, B, or C.";

        if (!ordinaryPrerequisitesReady)
            return "The normal recording and training-tool requirements are not ready.";

        if (RequiresEmbedding(kind) && !modelCReady)
        {
            return "Model C needs the stock face model's visual features. Choose Prepare Model C " +
                   "to enable its embedding runner and generate features for every recording.";
        }

        return null;
    }

    public static string DisplayName(string? adapterType) => adapterType switch
    {
        OutputOnlyAdapterType => Options[OutputOnlyIndex].DisplayName,
        ImageConditionedAdapterType => Options[ImageConditionedIndex].DisplayName,
        SharedFeaturesAdapterType => Options[SharedFeaturesIndex].DisplayName,
        null or "" or "unknown" => "unknown model",
        _ => adapterType
    };
}
