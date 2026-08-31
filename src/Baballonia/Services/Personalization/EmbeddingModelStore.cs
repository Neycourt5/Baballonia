using System;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Locates and validates the derived face model that exposes the stock visual embedding.
///
/// The derived model is an exact copy of the stock graph with one extra output. That makes it
/// trivially correct while it matches the stock file, and quietly catastrophic once it does not: an
/// embedding produced by a *different* network describes a different feature space, so a personal
/// head trained against the old one would keep running and keep emitting confident nonsense. No
/// exception, no obviously broken output - just a face that is subtly wrong.
///
/// So the derived model is only ever used when a sidecar records the MD5 of the stock file it was
/// built from and that hash still matches. Checking costs one hash of a 24 MB file at load time,
/// which is nothing next to the failure it prevents.
/// </summary>
public static class EmbeddingModelStore
{
    /// <summary>Lives in the writable models directory, never beside the shipped stock model.</summary>
    public const string FileName = "faceModelWithEmbedding.onnx";

    /// <summary>Graph output the derived model adds, matching the Python side.</summary>
    public const string EmbeddingOutputName = "embedding";

    public static string ModelPath => Path.Combine(PersonalizationPaths.ModelsRoot, FileName);

    public static string SidecarPath =>
        Path.Combine(PersonalizationPaths.ModelsRoot,
            Path.ChangeExtension(FileName, ".json"));

    /// <summary>Why a derived model was refused, phrased for a log line or the UI.</summary>
    public sealed record ValidationResult(bool Valid, string Message, string? Path = null);

    /// <summary>
    /// Returns the derived model's path when it exists and was built from the current stock model.
    /// </summary>
    /// <param name="stockModelPath">The stock face model actually in use.</param>
    public static ValidationResult TryGetValid(string stockModelPath)
    {
        var model = ModelPath;
        var sidecar = SidecarPath;

        if (!File.Exists(model))
            return new ValidationResult(false, "No embedding model has been built yet.");

        if (!File.Exists(sidecar))
        {
            return new ValidationResult(false,
                "The embedding model has no provenance record, so it cannot be matched against the " +
                "stock model. Rebuild it.");
        }

        if (!File.Exists(stockModelPath))
            return new ValidationResult(false, "The stock face model is missing.");

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(sidecar));

            if (!document.RootElement.TryGetProperty("base_model_md5", out var recorded) ||
                recorded.ValueKind != JsonValueKind.String)
            {
                return new ValidationResult(false, "The embedding model's record is unreadable.");
            }

            var actual = ComputeMd5(stockModelPath);
            if (!string.Equals(recorded.GetString(), actual, StringComparison.OrdinalIgnoreCase))
            {
                return new ValidationResult(false,
                    "The embedding model was built from a different version of the face model. " +
                    "Rebuild it before using a model C adapter.");
            }

            return new ValidationResult(true, "", model);
        }
        catch (Exception ex)
        {
            return new ValidationResult(false, $"Could not validate the embedding model: {ex.Message}");
        }
    }

    public static string ComputeMd5(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(MD5.HashData(stream)).ToLowerInvariant();
    }
}
