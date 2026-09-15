using System;
using System.Collections.Generic;
using System.Linq;
using Baballonia.Contracts;
using Baballonia.Services.Calibration;
using Baballonia.Services.Personalization.Eye;
using JetBrains.Annotations;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// Response curves: making a squint that the model barely registers actually read as a squint.
/// </summary>
/// <remarks>
/// The complaint this exists for is "my hard squint does nothing". The usual cause is that the model
/// emits about 0.35 for the strongest squint a person makes, which downstream is a faint squint, and
/// no threshold tuning turns that into a full one without also turning noise into one. A curve
/// anchored on measured poses raises the strong pose without raising the resting value — which a
/// plain gain cannot do, because it scales rest too.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeResponseCurve))]
public class EyeResponseCurveTest
{
    [TestMethod]
    public void IdentityLeavesEverythingAlone()
    {
        foreach (var value in new[] { 0f, 0.25f, 0.5f, 0.75f, 1f })
            Assert.AreEqual(value, EyeResponseCurve.Identity.Apply(value), 1e-6);
    }

    [TestMethod]
    public void AWeakSquintIsRaisedWithoutRaisingRest()
    {
        // The whole point, stated directly: rest stays at zero while the strong pose reaches full.
        var curve = new EyeResponseCurve([0f, 0.04f, 0.18f, 0.35f, 1f], [0f, 0f, 0.4f, 0.9f, 1f]);

        Assert.AreEqual(0f, curve.Apply(0.04f), 1e-5, "a resting face must stay at rest");
        Assert.AreEqual(0.4f, curve.Apply(0.18f), 1e-5);
        Assert.AreEqual(0.9f, curve.Apply(0.35f), 1e-5, "the strongest pose should nearly max out");
    }

    [TestMethod]
    public void AGainWouldHaveRaisedRestToo()
    {
        // Contrast with the naive fix. Tripling 0.35 to reach ~1.0 also triples the resting 0.04
        // into a permanent faint squint; the curve does not.
        var curve = new EyeResponseCurve([0f, 0.04f, 0.35f, 1f], [0f, 0f, 0.9f, 1f]);

        const float naiveGain = 0.9f / 0.35f;
        Assert.IsTrue(0.04f * naiveGain > 0.10f, "the naive gain really would lift rest");
        Assert.AreEqual(0f, curve.Apply(0.04f), 1e-6);
    }

    [TestMethod]
    public void ValuesBetweenAnchorsAreInterpolated()
    {
        var curve = new EyeResponseCurve([0f, 0.2f, 0.6f], [0f, 0.5f, 1f]);

        Assert.AreEqual(0.25f, curve.Apply(0.1f), 1e-5);
        Assert.AreEqual(0.75f, curve.Apply(0.4f), 1e-5);
    }

    [TestMethod]
    public void OutsideTheMeasuredRangeTheCurveHoldsFlat()
    {
        // Extrapolating past the strongest pose someone performed is invention, and invention in
        // the direction of overshoot.
        var curve = new EyeResponseCurve([0.1f, 0.5f], [0f, 1f]);

        Assert.AreEqual(0f, curve.Apply(0f), 1e-6);
        Assert.AreEqual(0f, curve.Apply(-5f), 1e-6);
        Assert.AreEqual(1f, curve.Apply(0.9f), 1e-6);
        Assert.AreEqual(1f, curve.Apply(50f), 1e-6);
    }

    [TestMethod]
    public void ACurveIsMonotone()
    {
        var curve = new EyeResponseCurve([0f, 0.05f, 0.2f, 0.4f, 1f], [0f, 0f, 0.4f, 0.9f, 1f]);

        var previous = -1f;
        for (var value = 0f; value <= 1f; value += 0.01f)
        {
            var mapped = curve.Apply(value);
            Assert.IsTrue(mapped >= previous - 1e-5,
                $"the curve went backwards at {value}: {mapped} after {previous}");
            previous = mapped;
        }
    }

    [TestMethod]
    public void AnUnusableCurveIsARefusalNotAMisbehaviour()
    {
        // Anchors that are not separated, out of order, or non-finite all pass values through.
        var tooClose = new EyeResponseCurve([0.30f, 0.301f], [0f, 1f]);
        var backwards = new EyeResponseCurve([0.5f, 0.2f], [0f, 1f]);
        var broken = new EyeResponseCurve([0f, float.NaN], [0f, 1f]);

        Assert.IsFalse(tooClose.IsUsable);
        Assert.IsFalse(backwards.IsUsable);
        Assert.IsFalse(broken.IsUsable);
        Assert.AreEqual(0.42f, tooClose.Apply(0.42f), 1e-6);
        Assert.AreEqual(0.42f, broken.Apply(0.42f), 1e-6);
    }

    [TestMethod]
    public void NonFiniteInputIsPassedThrough()
    {
        var curve = new EyeResponseCurve([0f, 0.5f], [0f, 1f]);
        Assert.IsTrue(float.IsNaN(curve.Apply(float.NaN)));
    }
}

/// <summary>Building a curve from measured poses, and refusing when they do not support one.</summary>
[TestClass]
[TestSubject(typeof(EyeResponseCurveFitter))]
public class EyeResponseCurveFitterTest
{
    /// <summary>A model that under-reports squint: rest 0.04, light 0.18, hard 0.35.</summary>
    private static List<EyeResponseAnchor> UnderReportingSquint() =>
    [
        new(0.04f, 0f),
        new(0.18f, 0.4f),
        new(0.35f, 0.9f),
    ];

    [TestMethod]
    public void AnUnderReportingChannelIsFittedAndAmplified()
    {
        var result = EyeResponseCurveFitter.Fit(UnderReportingSquint());

        Assert.IsTrue(result.Fitted, result.Reason);
        Assert.AreEqual(0f, result.Curve.Apply(0.04f), 1e-5);
        Assert.AreEqual(0.9f, result.Curve.Apply(0.35f), 1e-5);

        var gain = EyeResponseCurveFitter.StrongestGain(UnderReportingSquint(), result.Curve);
        Assert.IsTrue(gain > 2f, $"the strong pose should read much stronger, got {gain:F2}x");
    }

    [TestMethod]
    public void AnchorsAreSortedByIntendedStrengthNotRecordingOrder()
    {
        // Poses can be retried, so recording order is not strength order.
        var shuffled = new List<EyeResponseAnchor>
        {
            new(0.35f, 0.9f),
            new(0.04f, 0f),
            new(0.18f, 0.4f),
        };

        var result = EyeResponseCurveFitter.Fit(shuffled);

        Assert.IsTrue(result.Fitted, result.Reason);
        Assert.AreEqual(0f, result.Curve.Apply(0.04f), 1e-5);
    }

    [TestMethod]
    public void IndistinguishablePosesAreRefused()
    {
        // If the model reports nearly the same value for a light and a hard squint, a curve through
        // them would amplify noise into expression.
        var flat = new List<EyeResponseAnchor>
        {
            new(0.20f, 0f),
            new(0.21f, 0.4f),
            new(0.22f, 0.9f),
        };

        var result = EyeResponseCurveFitter.Fit(flat);

        Assert.IsFalse(result.Fitted);
        Assert.IsTrue(result.Curve.IsIdentity, "a refusal must leave the channel alone");
        StringAssert.Contains(result.Reason, "too similar");
    }

    [TestMethod]
    public void AChannelWithNoPosesIsRefused()
    {
        Assert.IsFalse(EyeResponseCurveFitter.Fit([]).Fitted);
        Assert.IsFalse(EyeResponseCurveFitter.Fit([new EyeResponseAnchor(0.1f, 0f)]).Fitted);
    }

    [TestMethod]
    public void NonFiniteAnchorsAreRefused()
    {
        var anchors = UnderReportingSquint();
        anchors[1] = new EyeResponseAnchor(float.NaN, 0.4f);

        Assert.IsFalse(EyeResponseCurveFitter.Fit(anchors).Fitted);
    }

    [TestMethod]
    public void AnAlreadyWellScaledChannelIsBarelyChanged()
    {
        // Someone whose model already reports honestly should not have their expression distorted.
        var honest = new List<EyeResponseAnchor>
        {
            new(0.02f, 0f),
            new(0.40f, 0.4f),
            new(0.88f, 0.9f),
        };

        var result = EyeResponseCurveFitter.Fit(honest);

        Assert.IsTrue(result.Fitted, result.Reason);
        Assert.IsTrue(EyeResponseCurveFitter.StrongestGain(honest, result.Curve) < 1.1f);
        foreach (var value in new[] { 0.1f, 0.3f, 0.6f })
            Assert.AreEqual(value, result.Curve.Apply(value), 0.06,
                $"an honest model should be left roughly alone at {value}");
    }
}

/// <summary>The curves as the corrector applies them.</summary>
[TestClass]
[TestSubject(typeof(EyeAffineCorrector))]
public class EyeExpressionCorrectionTest
{
    private const int RightLid = 2, RightWiden = 3, RightSquint = 4, RightBrow = 5;
    private const int LeftWiden = 9, LeftSquint = 10;

    private static readonly DenseTensor<float> AnyImage = new(new[] { 1, 8, 8, 8 });

    private static float[] Stock(float widen = 0.35f, float squint = 0.30f)
    {
        var stock = new float[EyePersonalizationSchema.ExpressionCount];
        for (var i = 0; i < stock.Length; i++)
            stock[i] = 0.5f;
        stock[RightLid] = 0.2f;
        stock[RightWiden] = widen;
        stock[RightSquint] = squint;
        stock[LeftWiden] = widen;
        stock[LeftSquint] = squint;
        stock[RightBrow] = 0.4f;
        return stock;
    }

    private static EyeAffineProfile WithCurves(EyeExpressionCurves curves) =>
        EyeAffineProfile.Identity with { LeftCurves = curves, RightCurves = curves };

    private static EyeExpressionCurves Amplifying() => new(
        Widen: new EyeResponseCurve([0f, 0.05f, 0.35f, 1f], [0f, 0f, 1f, 1f]),
        Squint: new EyeResponseCurve([0f, 0.04f, 0.30f, 1f], [0f, 0f, 0.9f, 1f]));

    [TestMethod]
    public void AWeakSquintReachesFullStrength()
    {
        var corrected = new EyeAffineCorrector(WithCurves(Amplifying()))
            .Correct(AnyImage, Stock());

        Assert.AreEqual(0.9f, corrected[RightSquint], 1e-4);
        Assert.AreEqual(0.9f, corrected[LeftSquint], 1e-4);
        Assert.AreEqual(1f, corrected[RightWiden], 1e-4);
    }

    [TestMethod]
    public void LidIsLeftToTheQuickEyeSetupProfile()
    {
        // Two systems correcting one channel is how a correction ends up fighting a calibration.
        var corrected = new EyeAffineCorrector(WithCurves(Amplifying())).Correct(AnyImage, Stock());

        Assert.AreEqual(0.2f, corrected[RightLid], 1e-6);
    }

    [TestMethod]
    public void BrowIsNeverTouched()
    {
        // No pose in the routine demonstrates a brow position, so nothing licenses correcting it.
        var corrected = new EyeAffineCorrector(WithCurves(Amplifying())).Correct(AnyImage, Stock());

        Assert.AreEqual(0.4f, corrected[RightBrow], 1e-6);
    }

    [TestMethod]
    public void BlendScalesTheExpressionCorrectionToo()
    {
        var half = new EyeAffineCorrector(WithCurves(Amplifying())) { Blend = 0.5f }
            .Correct(AnyImage, Stock());

        // Halfway between the model's 0.30 and the curve's 0.9.
        Assert.AreEqual(0.6f, half[RightSquint], 1e-4);
    }

    [TestMethod]
    public void BlendZeroIsExactlyTheBaseModel()
    {
        var stock = Stock();
        var corrected = new EyeAffineCorrector(WithCurves(Amplifying())) { Blend = 0f }
            .Correct(AnyImage, stock);

        CollectionAssert.AreEqual(stock, corrected);
    }

    [TestMethod]
    public void AGazeOnlyProfileLeavesExpressionsAlone()
    {
        // Version 1 profiles deserialize with no curves; they must keep working as gaze-only.
        var stock = Stock();
        var corrected = new EyeAffineCorrector(EyeAffineProfile.Identity).Correct(AnyImage, stock);

        Assert.AreEqual(stock[RightSquint], corrected[RightSquint], 1e-6);
        Assert.AreEqual(stock[RightWiden], corrected[RightWiden], 1e-6);
    }

    [TestMethod]
    public void AnUnusableCurveLeavesItsChannelAlone()
    {
        var broken = new EyeExpressionCurves(
            Widen: new EyeResponseCurve([0.3f, 0.301f], [0f, 1f]),
            Squint: EyeResponseCurve.Identity);

        var stock = Stock();
        var corrected = new EyeAffineCorrector(WithCurves(broken)).Correct(AnyImage, stock);

        Assert.AreEqual(stock[RightWiden], corrected[RightWiden], 1e-6);
    }

    [TestMethod]
    public void CorrectedExpressionsStayInRange()
    {
        var wild = new EyeExpressionCurves(
            Widen: new EyeResponseCurve([0f, 0.1f], [0f, 1f]),
            Squint: new EyeResponseCurve([0f, 0.1f], [0f, 1f]));

        var corrected = new EyeAffineCorrector(WithCurves(wild))
            .Correct(AnyImage, Stock(widen: 0.9f, squint: 0.9f));

        foreach (var index in new[] { RightWiden, RightSquint, LeftWiden, LeftSquint })
            Assert.IsTrue(corrected[index] is >= 0f and <= 1f, $"channel {index} left [0,1]");
    }
}

/// <summary>
/// The prompt that stops a good gaze fit from looking like a bad one.
/// </summary>
/// <remarks>
/// Two systems correct gaze centre at different points: the affine corrector in raw model space
/// before geometry, and <c>EyeCalibrationProfile.MapGazeX</c> afterwards. That composes only when
/// the centre was measured through whatever correction is installed. A centre captured before a fit
/// is applied on top of a signal that has since been re-centred, so the two stack — about 10 degrees
/// of gaze error for the reporting user, which reads as the fit having made tracking worse.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyePersonalizationManager))]
public class EyeCalibrationStalenessTest
{
    private sealed class Settings : ILocalSettingsService
    {
        private readonly Dictionary<string, object?> _values = new(StringComparer.Ordinal);

        public T ReadSetting<T>(string key, T? defaultValue = default, bool forceLocal = false) =>
            _values.TryGetValue(key, out var v) && v is T t ? t : defaultValue!;

        public void SaveSetting<T>(string key, T value, bool forceLocal = false) =>
            _values[key] = value;

        public void Save(object target) { }
        public void Load(object target) { }
        public void ForceSave() { }
    }

    [TestMethod]
    public void AFreshInstallIsNotStale()
    {
        Assert.IsFalse(new Settings().ReadSetting(
            EyePersonalizationManager.CalibrationStaleSetting, false));
    }

    [TestMethod]
    public void CapturingACalibrationClearsTheFlag()
    {
        var settings = new Settings();
        settings.SaveSetting(EyePersonalizationManager.CalibrationStaleSetting, true);

        var library = new EyeCalibrationLibrary(settings);
        library.Add(EyeCalibrationProfile.Default, EyeCalibrationProfile.Default, "test");

        Assert.IsFalse(settings.ReadSetting(
            EyePersonalizationManager.CalibrationStaleSetting, false),
            "capturing a calibration answers the prompt");
    }

    [TestMethod]
    public void SwitchingSavedCalibrationsAlsoClearsIt()
    {
        var settings = new Settings();
        var library = new EyeCalibrationLibrary(settings);
        var entry = library.Add(EyeCalibrationProfile.Default, EyeCalibrationProfile.Default, "a");

        settings.SaveSetting(EyePersonalizationManager.CalibrationStaleSetting, true);
        library.Activate(entry.Id);

        Assert.IsFalse(settings.ReadSetting(
            EyePersonalizationManager.CalibrationStaleSetting, false));
    }

    [TestMethod]
    public void GoingBackToUncalibratedClearsItToo()
    {
        // Choosing "no calibration" is also a deliberate answer: there is no stale centre left.
        var settings = new Settings();
        settings.SaveSetting(EyePersonalizationManager.CalibrationStaleSetting, true);

        new EyeCalibrationLibrary(settings).Activate(EyeCalibrationLibrary.DefaultId);

        Assert.IsFalse(settings.ReadSetting(
            EyePersonalizationManager.CalibrationStaleSetting, false));
    }
}
