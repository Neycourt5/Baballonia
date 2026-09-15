using System;
using System.Collections.Generic;
using System.Linq;
using Baballonia.Services.Personalization.Eye;
using JetBrains.Annotations;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The no-machine-learning half of eye personalization: six coefficients per eye, least squares.
/// </summary>
/// <remarks>
/// Every refusal path matters as much as the fit itself. A gaze correction that is merely wrong
/// does not look broken — it looks like the tracking drifted — so the fitter returns identity
/// rather than a bad answer whenever it cannot be confident.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeAffineFitter))]
public class EyeAffineFitterTest
{
    /// <summary>The nine dot positions the guided routine records, in raw units.</summary>
    private static IReadOnlyList<(float X, float Y)> DotGrid() =>
        EyeGuidedCues.Poses
            .Where(pose => pose.DotXDegrees is not null && pose.DotYDegrees is not null)
            .Select(pose => (EyeGuidedCues.RawFromDegrees(pose.DotXDegrees!.Value),
                             EyeGuidedCues.RawFromDegrees(pose.DotYDegrees!.Value)))
            .Distinct()
            .ToList();

    /// <summary>Observations from an eye whose true behaviour is a known affine distortion.</summary>
    private static List<EyeGazeObservation> Distorted(EyeAffineMap truth, float noise = 0f, int seed = 7)
    {
        var random = new Random(seed);
        return DotGrid().Select(dot =>
        {
            // `truth` maps measured -> target, so the measurement is the inverse. Build it forwards
            // instead: pick a measured value that `truth` maps onto the dot.
            var measured = Invert(truth, dot.X, dot.Y);
            var jitterX = noise * (float)(random.NextDouble() - 0.5);
            var jitterY = noise * (float)(random.NextDouble() - 0.5);
            return new EyeGazeObservation(measured.X + jitterX, measured.Y + jitterY, dot.X, dot.Y);
        }).ToList();
    }

    /// <summary>Closed-form inverse of a 2x2-plus-offset affine.</summary>
    private static (float X, float Y) Invert(EyeAffineMap map, float x, float y)
    {
        var determinant = map.Ax * map.By - map.Bx * map.Ay;
        var dx = x - map.Cx;
        var dy = y - map.Cy;
        return ((map.By * dx - map.Bx * dy) / determinant,
                (map.Ax * dy - map.Ay * dx) / determinant);
    }

    [TestMethod]
    public void AKnownDistortionIsRecoveredExactly()
    {
        var truth = new EyeAffineMap(1.10f, 0.04f, -0.03f, -0.02f, 1.07f, 0.01f);

        var result = EyeAffineFitter.Fit(Distorted(truth));

        Assert.IsTrue(result.Fitted, result.Reason);
        Assert.AreEqual(truth.Ax, result.Map.Ax, 1e-4);
        Assert.AreEqual(truth.Bx, result.Map.Bx, 1e-4);
        Assert.AreEqual(truth.Cx, result.Map.Cx, 1e-4);
        Assert.AreEqual(truth.Ay, result.Map.Ay, 1e-4);
        Assert.AreEqual(truth.By, result.Map.By, 1e-4);
        Assert.AreEqual(truth.Cy, result.Map.Cy, 1e-4);
    }

    [TestMethod]
    public void APerfectEyeFitsToNearIdentity()
    {
        var observations = DotGrid()
            .Select(dot => new EyeGazeObservation(dot.X, dot.Y, dot.X, dot.Y))
            .ToList();

        var result = EyeAffineFitter.Fit(observations);

        Assert.IsTrue(result.Fitted, result.Reason);
        Assert.AreEqual(1f, result.Map.Ax, 1e-4);
        Assert.AreEqual(0f, result.Map.Bx, 1e-4);
        Assert.AreEqual(0f, result.Map.Cx, 1e-4);
    }

    [TestMethod]
    public void ARotatedCameraIsCorrectedByTheCrossTerms()
    {
        // The reason for a full 2x3 rather than gains and offsets: a reseated headset rotates the
        // camera, which is exactly a cross term.
        const float angle = 0.08f;
        var truth = new EyeAffineMap(
            MathF.Cos(angle), -MathF.Sin(angle), 0.5f * (1 - MathF.Cos(angle) + MathF.Sin(angle)),
            MathF.Sin(angle), MathF.Cos(angle), 0.5f * (1 - MathF.Cos(angle) - MathF.Sin(angle)));

        var result = EyeAffineFitter.Fit(Distorted(truth));

        Assert.IsTrue(result.Fitted, result.Reason);
        Assert.IsTrue(Math.Abs(result.Map.Bx) > 0.05f, "the cross term should have been recovered");
        Assert.AreEqual(truth.Bx, result.Map.Bx, 1e-3);
    }

    [TestMethod]
    public void NoiseIsAveragedRatherThanFollowed()
    {
        var truth = new EyeAffineMap(1.08f, 0f, -0.02f, 0f, 1.05f, 0.01f);

        var result = EyeAffineFitter.Fit(Distorted(truth, noise: 0.01f));

        Assert.IsTrue(result.Fitted, result.Reason);
        Assert.AreEqual(truth.Ax, result.Map.Ax, 0.05);
        Assert.AreEqual(truth.By, result.Map.By, 0.05);
    }

    [TestMethod]
    public void TooFewFixationsIsRefused()
    {
        var observations = Distorted(EyeAffineMap.Identity).Take(3).ToList();

        var result = EyeAffineFitter.Fit(observations);

        Assert.IsFalse(result.Fitted);
        Assert.IsTrue(result.Map.IsIdentity, "a refusal must leave the base model alone");
        StringAssert.Contains(result.Reason, "fixations");
    }

    [TestMethod]
    public void FixationsAlongOneLineAreRefused()
    {
        // Points on a line leave the cross terms unconstrained; the solver would happily return
        // enormous ones that only look right on the line they were fitted to.
        var observations = Enumerable.Range(0, 9)
            .Select(i => 0.3f + i * 0.05f)
            .Select(x => new EyeGazeObservation(x, 0.5f, x, 0.5f))
            .ToList();

        var result = EyeAffineFitter.Fit(observations);

        Assert.IsFalse(result.Fitted);
        Assert.IsTrue(result.Map.IsIdentity);
        StringAssert.Contains(result.Reason, "span");
    }

    [TestMethod]
    public void AllFixationsAtOnePointAreRefused()
    {
        var observations = Enumerable.Repeat(
            new EyeGazeObservation(0.5f, 0.5f, 0.5f, 0.5f), 9).ToList();

        var result = EyeAffineFitter.Fit(observations);

        Assert.IsFalse(result.Fitted);
        Assert.IsTrue(result.Map.IsIdentity);
    }

    [TestMethod]
    public void AnImplausiblyLargeCorrectionIsRefused()
    {
        // Someone who looked away from the dots produces a fit that "works" on the data and is
        // nonsense on a face.
        var observations = DotGrid()
            .Select(dot => new EyeGazeObservation(1f - dot.X, 1f - dot.Y, dot.X, dot.Y))
            .ToList();

        var result = EyeAffineFitter.Fit(observations);

        Assert.IsFalse(result.Fitted);
        Assert.IsTrue(result.Map.IsIdentity);
        StringAssert.Contains(result.Reason, "implausibly large");
    }

    [TestMethod]
    public void NonFiniteObservationsAreRefused()
    {
        var observations = Distorted(EyeAffineMap.Identity);
        observations[0] = observations[0] with { MeasuredX = float.NaN };

        Assert.IsFalse(EyeAffineFitter.Fit(observations).Fitted);
    }

    [TestMethod]
    public void EvaluateReportsTheImprovementInDegrees()
    {
        var truth = new EyeAffineMap(1.25f, 0.03f, -0.13f, -0.02f, 1.20f, -0.09f);
        var observations = Distorted(truth);

        var (before, after) = EyeAffineFitter.Evaluate(observations, EyeAffineFitter.Fit(observations).Map);

        Assert.IsTrue(before > 1f, $"the distorted eye should have visible error, got {before:F2}°");
        Assert.IsTrue(after < before / 10f,
            $"an exact affine distortion should be almost entirely removed: {before:F2}° -> {after:F2}°");
    }

    [TestMethod]
    public void EvaluateOnIdentityReportsNoChange()
    {
        var observations = Distorted(new EyeAffineMap(1.1f, 0f, -0.05f, 0f, 1.1f, -0.05f));

        var (before, after) = EyeAffineFitter.Evaluate(observations, EyeAffineMap.Identity);

        Assert.AreEqual(before, after, 1e-4);
    }
}

/// <summary>Applying a fit: gaze only, clamped, and instantly switchable.</summary>
[TestClass]
[TestSubject(typeof(EyeAffineCorrector))]
public class EyeAffineCorrectorTest
{
    private const int RightY = 0, RightX = 1, RightLid = 2, RightWiden = 3;
    private const int LeftY = 6, LeftX = 7, LeftLid = 8;

    private static readonly DenseTensor<float> AnyImage = new([1, 8, 8, 8]);

    private static float[] Stock()
    {
        var stock = new float[EyePersonalizationSchema.ExpressionCount];
        for (var i = 0; i < stock.Length; i++)
            stock[i] = 0.5f;
        stock[RightLid] = 0.2f;
        stock[LeftLid] = 0.25f;
        stock[RightWiden] = 0.3f;
        return stock;
    }

    private static EyeAffineProfile ProfileWith(EyeAffineMap left, EyeAffineMap right) =>
        EyeAffineProfile.Identity with { Left = left, Right = right };

    [TestMethod]
    public void OnlyGazeChannelsAreTouched()
    {
        // The fit comes from fixations, which say nothing about lids or expression.
        var corrector = new EyeAffineCorrector(ProfileWith(
            new EyeAffineMap(1f, 0f, 0.05f, 0f, 1f, 0.05f),
            new EyeAffineMap(1f, 0f, 0.05f, 0f, 1f, 0.05f)));

        var stock = Stock();
        var corrected = corrector.Correct(AnyImage, stock);

        Assert.AreNotEqual(stock[LeftX], corrected[LeftX]);
        Assert.AreEqual(stock[LeftLid], corrected[LeftLid], 1e-6);
        Assert.AreEqual(stock[RightLid], corrected[RightLid], 1e-6);
        Assert.AreEqual(stock[RightWiden], corrected[RightWiden], 1e-6);
    }

    [TestMethod]
    public void EachEyeUsesItsOwnFit()
    {
        var corrector = new EyeAffineCorrector(ProfileWith(
            left: new EyeAffineMap(1f, 0f, 0.10f, 0f, 1f, 0f),
            right: new EyeAffineMap(1f, 0f, -0.10f, 0f, 1f, 0f)));

        var corrected = corrector.Correct(AnyImage, Stock());

        Assert.AreEqual(0.60f, corrected[LeftX], 1e-5);
        Assert.AreEqual(0.40f, corrected[RightX], 1e-5);
    }

    [TestMethod]
    public void BlendZeroIsExactlyTheBaseModel()
    {
        var corrector = new EyeAffineCorrector(ProfileWith(
            new EyeAffineMap(1.2f, 0f, 0.1f, 0f, 1.2f, 0.1f),
            new EyeAffineMap(1.2f, 0f, 0.1f, 0f, 1.2f, 0.1f)))
        { Blend = 0f };

        var stock = Stock();
        CollectionAssert.AreEqual(stock, corrector.Correct(AnyImage, stock));
    }

    [TestMethod]
    public void BlendScalesTheCorrectionLinearly()
    {
        var map = new EyeAffineMap(1f, 0f, 0.10f, 0f, 1f, 0f);
        var full = new EyeAffineCorrector(ProfileWith(map, map)) { Blend = 1f }
            .Correct(AnyImage, Stock())[LeftX];
        var half = new EyeAffineCorrector(ProfileWith(map, map)) { Blend = 0.5f }
            .Correct(AnyImage, Stock())[LeftX];

        Assert.AreEqual(0.60f, full, 1e-5);
        Assert.AreEqual(0.55f, half, 1e-5);
    }

    [TestMethod]
    public void ACorrectionIsClampedNoMatterWhatTheFitSays()
    {
        // Belt and braces: the fitter already refuses large corrections, but a hand-edited settings
        // file must not be able to send gaze anywhere it likes.
        var wild = new EyeAffineMap(5f, 0f, -1.5f, 0f, 5f, -1.5f);
        var corrected = new EyeAffineCorrector(ProfileWith(wild, wild)).Correct(AnyImage, Stock());

        Assert.AreEqual(0.5f + EyeAffineFitter.MaximumCorrection, corrected[LeftX], 1e-5);
        Assert.IsTrue(corrected[LeftY] is >= 0f and <= 1f);
    }

    [TestMethod]
    public void OutputStaysInTheRawRange()
    {
        var map = new EyeAffineMap(1f, 0f, 0.2f, 0f, 1f, 0.2f);
        var stock = Stock();
        stock[LeftX] = 0.95f;
        stock[LeftY] = 0.95f;

        var corrected = new EyeAffineCorrector(ProfileWith(map, map)).Correct(AnyImage, stock);

        Assert.IsTrue(corrected[LeftX] is >= 0f and <= 1f);
        Assert.IsTrue(corrected[LeftY] is >= 0f and <= 1f);
    }

    [TestMethod]
    public void TheInputArrayIsNeverMutated()
    {
        // The caller publishes `stock` as the base half of the A/B comparison, and the One Euro
        // filter keys its buffers to array identity.
        var corrector = new EyeAffineCorrector(ProfileWith(
            new EyeAffineMap(1f, 0f, 0.1f, 0f, 1f, 0.1f), EyeAffineMap.Identity));

        var stock = Stock();
        var before = (float[])stock.Clone();
        var corrected = corrector.Correct(AnyImage, stock);

        CollectionAssert.AreEqual(before, stock);
        Assert.AreNotSame(stock, corrected);
    }

    [TestMethod]
    public void NonFiniteGazeIsLeftAlone()
    {
        var map = new EyeAffineMap(1f, 0f, 0.1f, 0f, 1f, 0.1f);
        var stock = Stock();
        stock[LeftX] = float.NaN;

        var corrected = new EyeAffineCorrector(ProfileWith(map, map)).Correct(AnyImage, stock);

        Assert.IsTrue(float.IsNaN(corrected[LeftX]), "the finite guard downstream owns this");
        Assert.AreEqual(0.6f, corrected[RightX], 1e-5, "the other eye is unaffected");
    }
}

/// <summary>The gates that decide whether a saved fit may be applied at all.</summary>
[TestClass]
[TestSubject(typeof(EyeAffineProfile))]
public class EyeAffineProfileTest
{
    private static EyeAffineProfile Saved(string md5 = "abc123", int version = 1, string? schema = null) =>
        new(version, schema ?? EyePersonalizationSchema.Sha256, md5,
            DateTime.UtcNow.ToString("o"), "session",
            new EyeAffineMap(1.05f, 0f, 0f, 0f, 1.05f, 0f), EyeAffineMap.Identity);

    [TestMethod]
    public void AMatchingProfileApplies()
    {
        Assert.IsTrue(Saved().AppliesTo("abc123", out var reason), reason);
        Assert.IsNull(reason);
    }

    [TestMethod]
    public void ADifferentBaseModelIsRefused()
    {
        // Retraining the eye model in VR replaces it underneath the fit. The coefficients would
        // become arbitrary numbers added to real tracking.
        Assert.IsFalse(Saved().AppliesTo("a-different-model", out var reason));
        StringAssert.Contains(reason, "different eye model");
    }

    [TestMethod]
    public void ADifferentSchemaIsRefused()
    {
        Assert.IsFalse(Saved(schema: "0".PadLeft(64, '0')).AppliesTo("abc123", out var reason));
        StringAssert.Contains(reason, "layout");
    }

    [TestMethod]
    public void ANewerFormatIsRefusedRatherThanGuessedAt()
    {
        Assert.IsFalse(Saved(version: EyeAffineProfile.CurrentVersion + 1)
            .AppliesTo("abc123", out var reason));
        StringAssert.Contains(reason, "version");
    }

    [TestMethod]
    public void ACorruptProfileIsRefused()
    {
        var corrupt = Saved() with { Left = new EyeAffineMap(float.NaN, 0f, 0f, 0f, 1f, 0f) };

        Assert.IsFalse(corrupt.AppliesTo("abc123", out var reason));
        StringAssert.Contains(reason, "invalid");
    }
}
