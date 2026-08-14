using System;
using System.Collections.Generic;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Runs the trained personal adapter and blends its output with the stock prediction.
///
/// Pinned to the CPU execution provider regardless of the app's GPU setting. The model is tiny
/// (tens of thousands of parameters), so there is nothing to gain from an accelerator, and it
/// avoids a second DirectML session competing with the stock model for the same device.
///
/// A broken personal model must never break tracking, so any inference failure permanently disables
/// this corrector and the pipeline continues on pure stock output.
/// </summary>
public sealed class PersonalModelCorrector : IExpressionCorrector, IPersonalCorrector
{
    public const string ImageInputName = "image";
    public const string StockInputName = "stock";
    public const string OutputName = "personal";

    private readonly InferenceSession _session;
    private readonly ILogger _logger;
    private readonly DenseTensor<float> _stockTensor = new([1, PersonalizationSchema.ExpressionCount]);
    private readonly List<NamedOnnxValue> _inputs = new(2);

    private volatile float _blend = 1f;
    private volatile bool _failed;
    private bool _disposed;

    /// <summary>Metadata read from the model file, surfaced for logging and the debug view.</summary>
    public PersonalModelMetadata Metadata { get; }

    /// <summary>True once inference has thrown; the corrector is inert from that point on.</summary>
    public bool HasFailed => _failed;

    public PersonalModelCorrector(InferenceSession session, PersonalModelMetadata metadata, ILogger logger)
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

    /// <inheritdoc />
    public float[] Correct(DenseTensor<float> image, float[] stock)
    {
        // Fast path once broken, and when the user has dialled personalization out entirely.
        if (_failed || _disposed)
            return (float[])stock.Clone();

        var blend = _blend;
        if (blend <= 0f)
            return (float[])stock.Clone();

        try
        {
            for (var i = 0; i < stock.Length && i < _stockTensor.Length; i++)
                _stockTensor.SetValue(i, stock[i]);

            _inputs.Clear();
            // The image tensor is the one the stock model just consumed: inference has completed,
            // and the whole pipeline runs on a single tick, so borrowing it is safe and copy-free.
            _inputs.Add(NamedOnnxValue.CreateFromTensor(ImageInputName, image));
            _inputs.Add(NamedOnnxValue.CreateFromTensor(StockInputName, _stockTensor));

            using var results = _session.Run(_inputs);
            var personal = results[0].AsEnumerable<float>();

            var output = new float[stock.Length];
            var index = 0;
            foreach (var value in personal)
            {
                if (index >= output.Length)
                    break;

                // lerp(stock, personal, blend): 0 is exactly stock, 1 is fully personalized.
                output[index] = stock[index] + (value - stock[index]) * blend;
                index++;
            }

            if (index != output.Length)
            {
                _failed = true;
                _logger.LogError(
                    "Personal model returned {Actual} values, expected {Expected}. Reverting to stock output.",
                    index, output.Length);
                return (float[])stock.Clone();
            }

            return output;
        }
        catch (Exception ex)
        {
            // Disable rather than log every frame at ~100 Hz.
            _failed = true;
            _logger.LogError(ex, "Personal model inference failed. Falling back to stock output for " +
                                 "the rest of this session.");
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
