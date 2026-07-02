using BabblePersonalizer.Core.Camera;
using BabblePersonalizer.Core.Models;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace BabblePersonalizer.Core.Onnx;

public sealed class FaceInferenceSession : IDisposable
{
    private readonly InferenceSession _session;
    private readonly ModelContract _contract;
    private readonly BaballoniaCompatiblePreprocessor _preprocessor;

    public FaceInferenceSession(ModelContract contract, BaballoniaCompatiblePreprocessor preprocessor)
    {
        if (!contract.IsCompatible) throw new ArgumentException("The model contract is not compatible.", nameof(contract));
        _contract = contract; _preprocessor = preprocessor;
        _session = new InferenceSession(contract.Path);
    }

    public InferenceFrame Run(Mat cameraFrame, CameraConfiguration configuration)
    {
        var height = checked((int)_contract.Input.Dimensions[2]);
        var width = checked((int)_contract.Input.Dimensions[3]);
        var transformed = _preprocessor.Transform(cameraFrame, configuration, width, height);
        var tensor = _preprocessor.ToTensor(transformed);
        using var output = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_contract.Input.Name, tensor) });
        return new InferenceFrame(transformed, output.Single().AsEnumerable<float>().ToArray());
    }

    public void Dispose() => _session.Dispose();
}

public sealed record InferenceFrame(Mat TransformedImage, float[] RawOutput) : IDisposable
{
    public void Dispose() => TransformedImage.Dispose();
}
