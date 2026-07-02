using BabblePersonalizer.Core.Inventory;
using BabblePersonalizer.Core.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace BabblePersonalizer.Core.Onnx;

public sealed class StockModelInspector
{
    public Task<ModelValidationResult> InspectAsync(string path, CancellationToken cancellationToken = default) =>
        Task.Run(() => Inspect(path, cancellationToken), cancellationToken);

    private static ModelValidationResult Inspect(string path, CancellationToken cancellationToken)
    {
        var errors = new List<string>(); var warnings = new List<string>();
        if (!File.Exists(path)) return new(null, new[] { "The selected model does not exist." });
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            var file = new FileInfo(path);
            using var hashStream = File.OpenRead(path);
            var hash = Convert.ToHexString(SHA256.HashData(hashStream));
            using var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                InterOpNumThreads = 1,
                IntraOpNumThreads = 1
            };
            options.AppendExecutionProvider_CPU();
            using var session = new InferenceSession(path, options);
            if (session.InputMetadata.Count != 1) errors.Add("Exactly one image input is required.");
            if (session.OutputMetadata.Count != 1) errors.Add("Exactly one direct-expression output is required.");
            if (errors.Count > 0) return new(null, errors);

            var inputEntry = session.InputMetadata.Single();
            var outputEntry = session.OutputMetadata.Single();
            var input = ToContract(inputEntry);
            var output = ToContract(outputEntry);
            if (input.ElementType != typeof(float).FullName) errors.Add("Only float32 image input is currently supported.");
            if (output.ElementType != typeof(float).FullName) errors.Add("Only float32 expression output is currently supported.");
            if (input.Dimensions.Count != 4) errors.Add("The image input must be rank 4 (NCHW).");
            else if (input.Dimensions[1] != 1) errors.Add("Baballonia compatibility requires one grayscale input channel.");
            var outputCount = output.Dimensions.LastOrDefault();
            if (outputCount <= 0) errors.Add("The direct-expression output count must be fixed and positive.");

            var metadata = session.ModelMetadata.CustomMetadataMap.ToDictionary(x => x.Key, x => x.Value);
            var expressionNames = ResolveNames(metadata, checked((int)Math.Max(0, outputCount)), warnings, errors);
            var parameters = expressionNames.Count == 0
                ? Array.Empty<FaceParameterDefinition>()
                : LegacyBaballoniaFaceCatalog.CreateDefinitions(expressionNames).ToArray();
            if (expressionNames.Count != LegacyBaballoniaFaceCatalog.OrderedNames.Count && expressionNames.Count > 0)
                warnings.Add("The metadata-defined output count differs from this repository's legacy 45-output sender. Preserve it exactly, but verify compatibility with the target Baballonia build.");
            var expressionHash = HashNames(expressionNames);

            if (errors.Count == 0)
            {
                var dimensions = inputEntry.Value.Dimensions.Select(x => x > 0 ? x : 1).ToArray();
                var tensor = new DenseTensor<float>(dimensions);
                using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputEntry.Key, tensor) });
                var values = results.Single().AsEnumerable<float>().ToArray();
                if (values.Length != outputCount) errors.Add($"Test inference returned {values.Length} values; expected {outputCount}.");
                if (values.Any(x => !float.IsFinite(x))) errors.Add("Test inference produced NaN or infinity.");
            }

            var contract = new ModelContract
            {
                Path = System.IO.Path.GetFullPath(path), FileSize = file.Length, Sha256 = hash,
                OpsetVersion = OnnxProtobufHeaderReader.ReadDefaultOpset(path), Input = input, Output = output,
                Metadata = metadata, Parameters = parameters, ExpressionListHash = expressionHash,
                IsCompatible = errors.Count == 0, Warnings = warnings
            };
            return new(contract, errors);
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex) { return new(null, new[] { $"ONNX validation failed: {ex.Message}" }); }
    }

    private static TensorContract ToContract(KeyValuePair<string, NodeMetadata> value) =>
        new(value.Key, value.Value.ElementType.FullName ?? value.Value.ElementType.Name,
            value.Value.Dimensions.Select(x => (long)x).ToArray());

    private static IReadOnlyList<string> ResolveNames(
        IReadOnlyDictionary<string, string> metadata, int count, List<string> warnings, List<string> errors)
    {
        if (metadata.TryGetValue("blendshape_names", out var json))
        {
            try
            {
                var names = JsonSerializer.Deserialize<string[]>(json) ?? Array.Empty<string>();
                if (names.Length != count) errors.Add($"blendshape_names has {names.Length} names but output has {count} values.");
                if (names.Distinct(StringComparer.Ordinal).Count() != names.Length) errors.Add("blendshape_names contains duplicates.");
                return errors.Count == 0 ? names : Array.Empty<string>();
            }
            catch (JsonException ex) { errors.Add($"blendshape_names is invalid JSON: {ex.Message}"); return Array.Empty<string>(); }
        }
        if (count == LegacyBaballoniaFaceCatalog.OrderedNames.Count)
        {
            warnings.Add("The model has no blendshape_names metadata. Using the validated legacy Baballonia 45-output ordering.");
            return LegacyBaballoniaFaceCatalog.OrderedNames;
        }
        errors.Add($"The model has {count} outputs and no blendshape_names metadata; expression identity cannot be established safely.");
        return Array.Empty<string>();
    }

    private static string HashNames(IReadOnlyList<string> names) => Convert.ToHexString(
        SHA256.HashData(Encoding.UTF8.GetBytes(string.Join("\n", names))));
}
