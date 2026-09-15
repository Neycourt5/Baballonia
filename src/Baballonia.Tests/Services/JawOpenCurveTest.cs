using System.Reflection;
using Baballonia.Services;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services;

/// <summary>
/// Bending how fast jaw open climbs, without moving where it ends.
/// </summary>
/// <remarks>
/// Reported as "when I open my mouth a little bit, my whole mouth and jaw will go way down… it feels
/// like I'm making a caveman face for a split second". A small real movement arriving as most of the
/// avatar's range. The user asked for it as an option specifically so nothing else would change,
/// which is why the default is exactly identity.
/// </remarks>
[TestClass]
[TestSubject(typeof(ParameterSenderService))]
public class JawOpenCurveTest
{
    // Reached by reflection, matching ParameterSenderEyeSafetyTest — these stay internal.
    private static float Shape(float v, float curve) =>
        (float)typeof(ParameterSenderService)
            .GetMethod("ShapeExpression", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, [v, curve])!;

    private static string JawOpenKey =>
        (string)typeof(ParameterSenderService)
            .GetField("JawOpenKey", BindingFlags.NonPublic | BindingFlags.Static)!
            .GetRawConstantValue()!;

    [TestMethod]
    public void TheDefaultIsExactlyIdentity()
    {
        // The user's condition for accepting this at all: untouched means unchanged.
        foreach (var v in new[] { 0f, 0.13f, 0.5f, 0.77f, 1f })
            Assert.AreEqual(v, Shape(v, 1f), 0f, "the default must not alter a single value");
    }

    [TestMethod]
    public void ASmallOpeningStopsDroppingTheWholeJaw()
    {
        // A quarter-open mouth reading as a quarter of the avatar's range is the complaint;
        // at 2.0 it becomes a sixteenth.
        Assert.AreEqual(0.0625f, Shape(0.25f, 2f), 1e-5);
        Assert.IsTrue(Shape(0.25f, 1.5f) < 0.25f);
    }

    [TestMethod]
    public void OpeningWideStillReachesFull()
    {
        // The whole reason this is an exponent and not a gain or a ceiling.
        foreach (var curve in new[] { 0.5f, 1f, 1.5f, 2f, 3f })
            Assert.AreEqual(1f, Shape(1f, curve), 1e-6, $"curve {curve} capped a wide opening");
    }

    [TestMethod]
    public void AClosedMouthStaysClosed()
    {
        foreach (var curve in new[] { 0.5f, 1.5f, 3f })
            Assert.AreEqual(0f, Shape(0f, curve), 1e-6);
    }

    [TestMethod]
    public void AGainWouldHaveCappedTheWideOpening()
    {
        // Contrast with the obvious alternative. Scaling by 0.25 tames the small opening the same
        // amount, but the avatar can then never open its mouth past a quarter.
        // Pick the gain that tames a half-open mouth by the same amount the curve does.
        const float gain = 0.5f;
        Assert.AreEqual(Shape(0.5f, 2f), 0.5f * gain, 1e-6, "matched at the small end by construction");

        // At the wide end they diverge completely, and that is the whole argument.
        Assert.AreEqual(0.5f, 1f * gain, 1e-6, "the gain halves a fully open mouth");
        Assert.AreEqual(1f, Shape(1f, 2f), 1e-6, "the curve leaves it fully open");
    }

    [TestMethod]
    public void TheCurveIsMonotone()
    {
        foreach (var curve in new[] { 0.5f, 1.5f, 2f, 3f })
        {
            var previous = -1f;
            for (var v = 0f; v <= 1f; v += 0.01f)
            {
                var shaped = Shape(v, curve);
                Assert.IsTrue(shaped >= previous - 1e-6, $"curve {curve} went backwards at {v}");
                Assert.IsTrue(shaped is >= 0f and <= 1f, $"curve {curve} left [0,1] at {v}");
                previous = shaped;
            }
        }
    }

    [TestMethod]
    public void BelowOneMakesItMoreEager()
    {
        // The opposite complaint, for someone whose jaw barely registers.
        Assert.IsTrue(Shape(0.25f, 0.5f) > 0.25f);
        Assert.AreEqual(1f, Shape(1f, 0.5f), 1e-6);
    }

    [TestMethod]
    public void GarbageIsPassedThroughUntouched()
    {
        Assert.IsTrue(float.IsNaN(Shape(float.NaN, 2f)));
        Assert.AreEqual(0.5f, Shape(0.5f, float.NaN), 1e-6);
        Assert.AreEqual(0.5f, Shape(0.5f, 0f), 1e-6, "a zero exponent must not become a constant 1");
        Assert.AreEqual(0.5f, Shape(0.5f, -2f), 1e-6);
    }

    [TestMethod]
    public void OnlyJawOpenIsShaped()
    {
        // Named explicitly so a rename upstream fails here rather than silently disabling the fix.
        Assert.AreEqual("JawOpen", JawOpenKey);
    }
}
