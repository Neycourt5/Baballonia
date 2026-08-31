using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Exercises the personal-model runtime against real ONNX files produced by the Python trainer
/// (Assets/PersonalModels, generated with random weights plus two deliberate biases).
///
/// Using genuine model files rather than mocks is the point: this is where a schema mismatch, a
/// shape mismatch or a blend bug would actually surface, and all three would otherwise show up as
/// "tracking feels wrong" rather than as an error.
/// </summary>
[TestClass]
[TestSubject(typeof(PersonalModelCorrector))]
public class PersonalModelCorrectorTest
{
    private const int N = PersonalizationSchema.ExpressionCount;
    private static readonly int JawOpen = PersonalizationSchema.IndexOf("JawOpen");
    private static readonly int SmileLeft = PersonalizationSchema.IndexOf("MouthSmileLeft");

    private static string AssetPath(string name) =>
        Path.Combine(AppContext.BaseDirectory, "Assets", "PersonalModels", name);

    private static PersonalModelCorrector Load(string fileName)
    {
        var options = new SessionOptions();
        options.AppendExecutionProvider_CPU();
        var session = new InferenceSession(AssetPath(fileName), options);
        var metadata = PersonalModelMetadata.FromSession(session);
        return new PersonalModelCorrector(session, metadata, NullLogger.Instance);
    }

    private static DenseTensor<float> Image() => new([1, 1, 224, 224]);

    private static float[] Stock(float value = 0.5f) => Enumerable.Repeat(value, N).ToArray();

    [TestMethod]
    public void Fixtures_ArePresent()
    {
        Assert.IsTrue(File.Exists(AssetPath("validAdapter.onnx")),
            "Test fixture missing. Regenerate with training/ (see WORK_PROGRESS.md).");
    }

    [TestMethod]
    public void ValidModel_LoadsWithExpectedMetadata()
    {
        using var corrector = Load("validAdapter.onnx");

        Assert.AreEqual(PersonalizationSchema.Sha256, corrector.Metadata.SchemaSha256);
        Assert.AreEqual("output_mlp_v1", corrector.Metadata.AdapterType);
        Assert.AreEqual(224, corrector.Metadata.InputSize);
        Assert.AreEqual(1, corrector.Metadata.AdapterVersion);
        Assert.AreEqual("gray_div255", corrector.Metadata.InputNormalization);
    }

    [TestMethod]
    public void Correct_AppliesTheModelsLearnedChanges()
    {
        using var corrector = Load("validAdapter.onnx");
        var stock = Stock();

        var personal = corrector.Correct(Image(), stock);

        Assert.AreEqual(N, personal.Length);
        // The fixture was built with a strong negative bias on JawOpen and positive on SmileLeft.
        Assert.IsTrue(personal[JawOpen] < stock[JawOpen] - 0.1f,
            $"expected JawOpen to drop, got {stock[JawOpen]} -> {personal[JawOpen]}");
        Assert.IsTrue(personal[SmileLeft] > stock[SmileLeft] + 0.1f,
            $"expected MouthSmileLeft to rise, got {stock[SmileLeft]} -> {personal[SmileLeft]}");
    }

    [TestMethod]
    public void Correct_OutputStaysInUnitRange()
    {
        using var corrector = Load("validAdapter.onnx");

        foreach (var level in new[] { 0f, 0.5f, 1f })
        {
            var personal = corrector.Correct(Image(), Stock(level));
            Assert.IsTrue(personal.All(v => v is >= 0f and <= 1f),
                $"values outside [0,1] at stock={level}: {string.Join(",", personal.Take(6))}");
        }
    }

    [TestMethod]
    public void Blend_Zero_IsExactlyStock()
    {
        using var corrector = Load("validAdapter.onnx");
        corrector.Blend = 0f;
        var stock = Stock();

        var result = corrector.Correct(Image(), stock);

        CollectionAssert.AreEqual(stock, result,
            "Blend 0 must be byte-identical to stock; it is the A side of every A/B comparison.");
    }

    [TestMethod]
    public void Blend_InterpolatesBetweenStockAndPersonal()
    {
        using var corrector = Load("validAdapter.onnx");
        var stock = Stock();

        corrector.Blend = 1f;
        var full = corrector.Correct(Image(), stock);

        corrector.Blend = 0.5f;
        var half = corrector.Correct(Image(), stock);

        var expected = stock[JawOpen] + (full[JawOpen] - stock[JawOpen]) * 0.5f;
        Assert.AreEqual(expected, half[JawOpen], 1e-5,
            "Half blend should sit exactly halfway between stock and personal.");
    }

    [TestMethod]
    public void Blend_IsClampedToUnitRange()
    {
        using var corrector = Load("validAdapter.onnx");

        corrector.Blend = 5f;
        Assert.AreEqual(1f, corrector.Blend);

        corrector.Blend = -2f;
        Assert.AreEqual(0f, corrector.Blend);
    }

    [TestMethod]
    public void Correct_DoesNotMutateTheStockArray()
    {
        using var corrector = Load("validAdapter.onnx");
        var stock = Stock();
        var original = (float[])stock.Clone();

        var result = corrector.Correct(Image(), stock);

        CollectionAssert.AreEqual(original, stock,
            "The stock array is aliased by the corrected event payload and must not be modified.");
        Assert.AreNotSame(stock, result, "Returning the input array would corrupt the One Euro filter state.");
    }

    [TestMethod]
    public void Correct_IsStableAcrossManyCalls()
    {
        using var corrector = Load("validAdapter.onnx");
        var stock = Stock();

        var first = corrector.Correct(Image(), stock);
        for (var i = 0; i < 200; i++)
            corrector.Correct(Image(), stock);
        var last = corrector.Correct(Image(), stock);

        CollectionAssert.AreEqual(first, last, "Repeated inference must be deterministic.");
        Assert.IsFalse(corrector.HasFailed);
    }

    /// <summary>
    /// Measures the real added cost per frame. The plan budgets well under the 10 ms processing
    /// tick; this records what it actually is rather than trusting the estimate.
    /// </summary>
    [TestMethod]
    public void Correct_LatencyIsSmallEnoughForTheProcessingTick()
    {
        using var corrector = Load("validAdapter.onnx");
        var stock = Stock();
        var image = Image();

        for (var i = 0; i < 50; i++)
            corrector.Correct(image, stock);

        const int iterations = 500;
        var samples = new double[iterations];
        for (var i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            corrector.Correct(image, stock);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(samples);
        var p50 = samples[iterations / 2];
        var p95 = samples[(int)(iterations * 0.95)];
        Console.WriteLine($"PersonalModelCorrector latency: p50 {p50:F3} ms, p95 {p95:F3} ms");

        // Generous bound: this is a regression guard, not a benchmark assertion. The processing
        // tick is 10 ms and already runs the stock face and eye models.
        Assert.IsTrue(p95 < 3.0,
            $"Personal inference p95 {p95:F3} ms is too slow for the 10 ms tick.");
    }

    /// <summary>
    /// Same measurement for the image-conditioned adapter, which does real convolution work on a
    /// 224x224 frame. This is the number that decides whether model B is affordable on the
    /// processing tick alongside the stock face and eye models.
    /// </summary>
    [TestMethod]
    public void ImageConditionedModel_LatencyIsAffordable()
    {
        using var corrector = Load("imageAdapter.onnx");
        Assert.AreEqual("image_residual_v1", corrector.Metadata.AdapterType);

        var stock = Stock();
        var image = Image();

        for (var i = 0; i < 20; i++)
            corrector.Correct(image, stock);

        const int iterations = 200;
        var samples = new double[iterations];
        for (var i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            corrector.Correct(image, stock);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(samples);
        var p50 = samples[iterations / 2];
        var p95 = samples[(int)(iterations * 0.95)];
        Console.WriteLine($"ImageResidualAdapter latency: p50 {p50:F3} ms, p95 {p95:F3} ms");

        Assert.IsFalse(corrector.HasFailed, "Image-conditioned inference failed.");
        Assert.IsTrue(p95 < 5.0,
            $"Image adapter p95 {p95:F3} ms leaves too little room in the 10 ms tick.");
    }
}
