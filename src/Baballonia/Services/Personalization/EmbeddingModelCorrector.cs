using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Model C at runtime: corrects using the stock network's own visual features rather than the frame.
///
/// A near-mirror of <see cref="PersonalModelCorrector"/> - same blend arithmetic, same permanent
/// self-disable on first failure - differing only in what it feeds the session. It is a separate
/// class rather than a branch inside the existing one so that models A and B keep running through
/// code that has not changed at all.
///
/// The embedding is borrowed, not copied: it belongs to the inference runner and is overwritten on
/// the next tick. Everything here happens within the same tick, so that is safe, and it keeps the
/// per-frame cost to the head's own arithmetic - about 0.4 MFLOP - with no allocation.
/// </summary>
public sealed class EmbeddingModelCorrector : IEmbeddingAwareCorrector, IPersonalCorrector
{
    public const string StockInputName = "stock";
    public const string EmbeddingInputName = "embedding";
    public const string OutputName = "personal";

    private readonly InferenceSession _session;
    private readonly ILogger _logger;
    private readonly DenseTensor<float> _stockTensor = new([1, PersonalizationSchema.ExpressionCount]);
    private readonly List<NamedOnnxValue> _inputs = new(2);

    private volatile float _blend = 1f;
    private volatile bool _failed;
    private bool _disposed;
    private bool _warnedAboutMissingEmbedding;

    public PersonalModelMetadata Metadata { get; }

    public bool HasFailed => _failed;

    public EmbeddingModelCorrector(InferenceSession session, PersonalModelMetadata metadata,
                                   ILogger logger)
    {
        _session = session;
        Metadata = metadata;
        _logger = logger;
    }

    /// <inheritdoc />
    public float Blend
    {
        get => _blend;
        set => _blend = Math.Clamp(value, 0f, 1f);
    }

    /// <summary>
    /// Model C cannot work without features, so this overload is always a passthrough.
    /// </summary>
    /// <remarks>
    /// Reached only if something installed this corrector into a pipeline that does not know to
    /// supply an embedding. Returning stock is the honest answer - the alternative would be feeding
    /// the model zeros and shipping whatever it made of them.
    /// </remarks>
    public float[] Correct(DenseTensor<float> image, float[] stock) => Correct(image, stock, null);

    /// <inheritdoc />
    public float[] Correct(DenseTensor<float> image, float[] stock, DenseTensor<float> embedding)
    {
        if (_failed || _disposed)
            return (float[])stock.Clone();

        var blend = _blend;
        if (blend <= 0f)
            return (float[])stock.Clone();

        if (embedding == null)
        {
            if (!_warnedAboutMissingEmbedding)
            {
                _warnedAboutMissingEmbedding = true;
                _logger.LogWarning(
                    "Personal model needs the stock visual embedding but none is available; " +
                    "running stock. Enable the embedding runner under Advanced.");
            }

            return (float[])stock.Clone();
        }

        try
        {
            for (var i = 0; i < stock.Length && i < _stockTensor.Length; i++)
                _stockTensor.SetValue(i, stock[i]);

            _inputs.Clear();
            _inputs.Add(NamedOnnxValue.CreateFromTensor(StockInputName, _stockTensor));
            _inputs.Add(NamedOnnxValue.CreateFromTensor(EmbeddingInputName, embedding));

            using var results = _session.Run(_inputs);
            var personal = results[0].AsEnumerable<float>().ToArray();

            if (personal.Length != stock.Length)
            {
                _failed = true;
                _logger.LogError(
                    "Personal model returned {Actual} values, expected {Expected}; disabling it",
                    personal.Length, stock.Length);
                return (float[])stock.Clone();
            }

            var output = new float[stock.Length];
            for (var i = 0; i < output.Length; i++)
                output[i] = stock[i] + (personal[i] - stock[i]) * blend;

            return output;
        }
        catch (Exception ex)
        {
            // One failure is enough: a model that throws once will throw every tick, and logging at
            // 100 Hz would be worse than the original problem.
            _failed = true;
            _logger.LogError(ex, "Personal model inference failed; falling back to stock tracking");
            return (float[])stock.Clone();
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        _session.Dispose();
    }
}
