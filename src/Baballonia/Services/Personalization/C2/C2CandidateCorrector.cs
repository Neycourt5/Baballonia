using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Baballonia.Services.Personalization.C2;

/// <summary>Separate experimental stage. Its fallback is the already computed working C output.</summary>
public sealed class C2CandidateCorrector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly DenseTensor<float> _reference = new([1,45]);
    private readonly List<NamedOnnxValue> _inputs = new(2);
    private readonly int _embeddingDim;
    private readonly HashSet<int>? _supportedChannels;
    private bool _disposed;
    private volatile float _contribution = 1f;
    public string? Failure { get; private set; }
    public float Contribution { get => _contribution; set => _contribution = float.IsFinite(value) ? Math.Clamp(value, 0, 1) : 0; }

    public bool CorrectsChannel(int channel) => _supportedChannels?.Contains(channel) ?? true;

    public C2CandidateCorrector(string path, string contextSha256, int embeddingDim = 1280,
        IReadOnlySet<int>? supportedChannels = null)
    {
        _supportedChannels = supportedChannels?.ToHashSet();
        if (_supportedChannels != null && (_supportedChannels.Count == 0 || _supportedChannels.Any(i => i < 0 || i >= 45)))
            throw new InvalidDataException("C2 supported expressions are invalid.");
        using var options = new SessionOptions { IntraOpNumThreads = 1, InterOpNumThreads = 1 };
        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        _session = new InferenceSession(path, options);
        _embeddingDim = embeddingDim;
        try
        {
            var metadata = _session.ModelMetadata.CustomMetadataMap;
            if (metadata.GetValueOrDefault("c2_contract") != C2Contract.Version ||
                metadata.GetValueOrDefault("c2_label_recipe") != C2Contract.LabelRecipe ||
                metadata.GetValueOrDefault("c2_context_sha256") != contextSha256 ||
                metadata.GetValueOrDefault("expression_schema_sha256") != PersonalizationSchema.Sha256)
                throw new InvalidDataException("C2 model contract, reference, labels or schema differ. Revalidate the candidate.");
            ValidateTensor(_session.InputMetadata, "reference", 45);
            ValidateTensor(_session.InputMetadata, "embedding", embeddingDim);
            ValidateTensor(_session.OutputMetadata, "personal", 45);
            if (_session.InputMetadata.Count != 2) throw new InvalidDataException("Unexpected C2 inputs.");
            // Load-time finite probe before publication. The actual data export parity is a separate gate.
            Correct(new float[45], new DenseTensor<float>([1, embeddingDim]));
            if (Failure != null) throw new InvalidDataException(Failure);
        }
        catch { _session.Dispose(); throw; }
    }

    private static void ValidateTensor(IReadOnlyDictionary<string, NodeMetadata> tensors, string name, int width)
    {
        if (!tensors.TryGetValue(name, out var t) || t.ElementType != typeof(float) ||
            t.Dimensions.Length != 2 || t.Dimensions[1] != width || t.Dimensions[0] > 1)
            throw new InvalidDataException($"C2 tensor '{name}' must be float [1,{width}].");
    }

    public float[] Correct(float[] reference, DenseTensor<float>? embedding)
    {
        var amount = _contribution;
        if (_disposed || Failure != null || amount == 0) return reference;
        try
        {
            if (reference.Length != 45 || embedding?.Length != _embeddingDim ||
                reference.Any(v => !float.IsFinite(v)) || embedding.Any(v => !float.IsFinite(v)))
                throw new InvalidDataException("C2 reference/features are unavailable or non-finite.");
            reference.AsSpan().CopyTo(_reference.Buffer.Span);
            _inputs.Clear();
            _inputs.Add(NamedOnnxValue.CreateFromTensor("reference", _reference));
            _inputs.Add(NamedOnnxValue.CreateFromTensor("embedding", embedding));
            using var results = _session.Run(_inputs, ["personal"]);
            var candidate = results.Single(r => r.Name == "personal").AsTensor<float>().ToArray();
            if (candidate.Length != 45 || candidate.Any(v => !float.IsFinite(v)))
                throw new InvalidDataException("C2 returned invalid output; using your working C.");
            // Enforce the report's preservation boundary in the runtime too. Calibration of
            // these untouched channels can then change during everyday use independently of C2.
            for (var i = 0; i < 45; i++)
                if (!CorrectsChannel(i)) candidate[i] = reference[i];
            if (amount == 1) return candidate;
            for (var i = 0; i < 45; i++) candidate[i] = reference[i] + (candidate[i] - reference[i]) * amount;
            return candidate;
        }
        catch (Exception ex) { Failure = ex.Message; return reference; }
    }

    public void Invalidate(string reason) => Failure ??= reason;
    public void Dispose() { if (_disposed) return; _disposed = true; _session.Dispose(); }
}
