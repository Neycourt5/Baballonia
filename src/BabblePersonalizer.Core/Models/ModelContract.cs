namespace BabblePersonalizer.Core.Models;

public sealed record TensorContract(string Name, string ElementType, IReadOnlyList<long> Dimensions);

public sealed class ModelContract
{
    public required string Path { get; init; }
    public required long FileSize { get; init; }
    public required string Sha256 { get; init; }
    public long? OpsetVersion { get; init; }
    public required TensorContract Input { get; init; }
    public required TensorContract Output { get; init; }
    public required IReadOnlyDictionary<string, string> Metadata { get; init; }
    public required IReadOnlyList<FaceParameterDefinition> Parameters { get; init; }
    public required string ExpressionListHash { get; init; }
    public required bool IsCompatible { get; init; }
    public required IReadOnlyList<string> Warnings { get; init; }
    public string Provider { get; init; } = "CPUExecutionProvider";
}

public sealed record ModelValidationResult(ModelContract? Contract, IReadOnlyList<string> Errors)
{
    public bool IsValid => Contract?.IsCompatible == true && Errors.Count == 0;
}
