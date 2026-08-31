using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;

namespace Baballonia.Contracts;

public interface IInferenceRunner
{
    public float[]? Run();
    public DenseTensor<float> GetInputTensor();
}

/// <summary>
/// Optional metadata for inference runners whose output has named elements.
/// </summary>
public interface INamedInferenceOutput
{
    public IReadOnlyList<string>? OutputNames { get; }
}
