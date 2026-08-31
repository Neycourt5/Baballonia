using Baballonia.Services.EyeV2;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Baballonia.Tests.Services.EyeV2;

[TestClass]
public class EyeV2CalibrationAndMapperTest
{
    [TestMethod]
    public void FivePointFit_RecoversPerEyeAffineGazeMapping()
    {
        var calibration = EyeV2CalibrationFitter.Fit(SyntheticCapture());

        AssertTarget(calibration.Left.Gaze, RawForTarget(EyeV2GazeTarget.Left, EyeSide.Left), -0.7f, 0f);
        AssertTarget(calibration.Left.Gaze, RawForTarget(EyeV2GazeTarget.Up, EyeSide.Left), 0f, -0.7f);
        AssertTarget(calibration.Right.Gaze, RawForTarget(EyeV2GazeTarget.Right, EyeSide.Right), 0.7f, 0f);
        AssertTarget(calibration.Right.Gaze, RawForTarget(EyeV2GazeTarget.Down, EyeSide.Right), 0f, 0.7f);
        Assert.AreNotEqual(calibration.Left.Gaze.OffsetX, calibration.Right.Gaze.OffsetX,
            "Each eye must be fitted independently.");
    }

    [TestMethod]
    public void Fit_CreatesPersonalNeutralClosedSquintAndWideAnchors()
    {
        var calibration = EyeV2CalibrationFitter.Fit(SyntheticCapture());

        Assert.AreEqual(0.75f, calibration.Left.Lid.Neutral, 0.02f);
        Assert.AreEqual(0.08f, calibration.Left.Lid.Closed, 0.03f);
        Assert.AreEqual(0.45f, calibration.Left.Lid.Squint, 0.02f);
        Assert.AreEqual(0.95f, calibration.Left.Lid.Wide, 0.02f);
        Assert.AreEqual(0.70f, calibration.Right.Lid.Neutral, 0.02f);
        Assert.IsTrue(calibration.Left.Lid.NeutralHigh >= 0.78f,
            "ordinary openness changes at gaze targets belong in the neutral envelope");
    }

    [TestMethod]
    public void Fit_RejectsMissingOrDegenerateCalibration()
    {
        Assert.ThrowsExactly<InvalidOperationException>(() =>
            EyeV2CalibrationFitter.Fit(new EyeV2CalibrationCapture()));

        var capture = SyntheticCapture();
        var same = Samples(30, 0, 0, 0.75f, 0, 0, 0.70f);
        capture = capture with
        {
            Gaze = EyeV2GazeTarget.FivePoint.ToDictionary(x => x.Name,
                _ => (IReadOnlyList<EyeV2Sample>)same),
        };
        Assert.ThrowsExactly<InvalidOperationException>(() => EyeV2CalibrationFitter.Fit(capture));
    }

    [TestMethod]
    public void Recenter_UpdatesOffsetsOnly()
    {
        var calibration = EyeV2CalibrationFitter.Fit(SyntheticCapture());
        var shifted = Samples(30, 0.20f, -0.15f, 0.75f, -0.12f, 0.10f, 0.70f);

        var recentered = EyeV2CalibrationFitter.Recenter(calibration, shifted);

        Assert.AreEqual(calibration.Left.Gaze.XX, recentered.Left.Gaze.XX);
        Assert.AreEqual(calibration.Left.Gaze.XY, recentered.Left.Gaze.XY);
        Assert.AreEqual(calibration.Left.Lid, recentered.Left.Lid);
        AssertTarget(recentered.Left.Gaze, new EyeRawState(0.20f, -0.15f, 0.75f), 0f, 0f);
        AssertTarget(recentered.Right.Gaze, new EyeRawState(-0.12f, 0.10f, 0.70f), 0f, 0f);
    }

    [TestMethod]
    public void Validity_DistinguishesRecenterFromFullCalibration()
    {
        var calibration = IdentityCalibration();
        var gazeShift = Samples(30, 0.30f, 0f, 0.75f, 0.25f, 0f, 0.70f);
        var lidShift = Samples(30, 0f, 0f, 0.40f, 0f, 0f, 0.38f);

        Assert.AreEqual(EyeV2ValidityKind.RecenterRecommended,
            EyeV2CalibrationFitter.AssessValidity(calibration, gazeShift).Kind);
        Assert.AreEqual(EyeV2ValidityKind.FullCalibrationRecommended,
            EyeV2CalibrationFitter.AssessValidity(calibration, lidShift).Kind);
    }

    [TestMethod]
    public void Mapper_NormalOpenStateHasNoSquintWideOrBlink()
    {
        EyeV2Diagnostics? diagnostics = null;
        var mapper = new EyeV2Mapper(IdentityCalibration(), d => diagnostics = d);

        var output = Map(mapper, 0, 0, 0.75f, 0, 0, 0.70f, 0);

        Assert.AreEqual(1f, output[EyeStateLayout.LeftLid], 0.001f);
        Assert.AreEqual(0f, output[EyeStateLayout.LeftWide], 0.001f);
        Assert.AreEqual(0f, output[EyeStateLayout.LeftSquint], 0.001f);
        Assert.IsFalse(diagnostics!.Left.Blink);
    }

    [TestMethod]
    public void Mapper_WideIsRelativeToPersonalNeutralEnvelope()
    {
        var mapper = new EyeV2Mapper(IdentityCalibration());

        var ordinaryGazeOpen = Map(mapper, 0.7f, 0, 0.80f, -0.7f, 0, 0.75f, 0);
        var wide = Map(mapper, 0, 0, 0.95f, 0, 0, 0.91f, 100);

        Assert.AreEqual(0f, ordinaryGazeOpen[EyeStateLayout.LeftWide], 0.001f);
        Assert.IsTrue(wide[EyeStateLayout.LeftWide] > 0.95f);
        Assert.IsTrue(wide[EyeStateLayout.RightWide] > 0.95f);
    }

    [TestMethod]
    public void Mapper_SustainedPartialClosureBecomesSquint()
    {
        var mapper = new EyeV2Mapper(IdentityCalibration());
        float[] output = [];
        for (var ms = 0; ms <= 700; ms += 100)
            output = Map(mapper, 0, 0, 0.45f, 0, 0, 0.42f, ms);

        Assert.IsTrue(output[EyeStateLayout.LeftSquint] > 0.9f);
        Assert.IsTrue(output[EyeStateLayout.RightSquint] > 0.9f);
    }

    [TestMethod]
    public void Mapper_ShortBlinkNeverBecomesSquint()
    {
        EyeV2Diagnostics? diagnostics = null;
        var mapper = new EyeV2Mapper(IdentityCalibration(), d => diagnostics = d);

        Assert.AreEqual(0f, Map(mapper, 0, 0, 0.45f, 0, 0, 0.42f, 0)[EyeStateLayout.LeftSquint]);
        Assert.AreEqual(0f, Map(mapper, 0, 0, 0.08f, 0, 0, 0.12f, 100)[EyeStateLayout.LeftSquint]);
        Assert.IsTrue(diagnostics!.Left.Blink);
        Assert.AreEqual(0f, Map(mapper, 0, 0, 0.75f, 0, 0, 0.70f, 200)[EyeStateLayout.LeftSquint]);
    }

    [TestMethod]
    public void Mapper_SlowBlinkThatReachesClosedIsSuppressed()
    {
        var mapper = new EyeV2Mapper(IdentityCalibration());
        var sequence = new[] { 0.60f, 0.50f, 0.42f, 0.08f, 0.20f, 0.55f, 0.75f };
        for (var i = 0; i < sequence.Length; i++)
        {
            var output = Map(mapper, 0, 0, sequence[i], 0, 0, sequence[i], i * 100);
            Assert.AreEqual(0f, output[EyeStateLayout.LeftSquint], 0.01f,
                $"slow blink emitted Squint at step {i}");
        }
    }

    [TestMethod]
    public void Mapper_SquintRecoversSmoothlyAfterOpening()
    {
        var mapper = new EyeV2Mapper(IdentityCalibration());
        float[] held = [];
        for (var ms = 0; ms <= 700; ms += 100)
            held = Map(mapper, 0, 0, 0.45f, 0, 0, 0.42f, ms);
        var firstRecovery = Map(mapper, 0, 0, 0.75f, 0, 0, 0.70f, 800);
        var recovered = Map(mapper, 0, 0, 0.75f, 0, 0, 0.70f, 1400);

        Assert.IsTrue(firstRecovery[EyeStateLayout.LeftSquint] < held[EyeStateLayout.LeftSquint]);
        Assert.AreEqual(0f, recovered[EyeStateLayout.LeftSquint], 0.01f);
    }

    [TestMethod]
    public void Mapper_PreservesAsymmetricEyeBehavior()
    {
        var mapper = new EyeV2Mapper(IdentityCalibration());
        float[] output = [];
        for (var ms = 0; ms <= 700; ms += 100)
            output = Map(mapper, 0.2f, -0.1f, 0.45f, -0.3f, 0.4f, 0.70f, ms);

        Assert.IsTrue(output[EyeStateLayout.LeftSquint] > 0.9f);
        Assert.AreEqual(0f, output[EyeStateLayout.RightSquint], 0.001f);
        Assert.AreEqual(0.2f, output[EyeStateLayout.LeftX], 0.001f);
        Assert.AreEqual(-0.3f, output[EyeStateLayout.RightX], 0.001f);
        Assert.AreEqual(-0.1f, output[EyeStateLayout.LeftY], 0.001f);
        Assert.AreEqual(0.4f, output[EyeStateLayout.RightY], 0.001f);
    }

    [TestMethod]
    public void Mapper_StaysWellInsideTheProcessingTickBudget()
    {
        var mapper = new EyeV2Mapper(IdentityCalibration());
        var raw = Raw(0.2f, -0.1f, 0.74f, -0.3f, 0.4f, 0.69f);
        var stock = new float[EyeStateLayout.LegacyCount];

        // Warm the JIT before measuring the steady-state mapping cost.
        for (var i = 0; i < 1_000; i++)
            mapper.Map(stock, raw, DateTime.UnixEpoch.Ticks + i * TimeSpan.TicksPerMillisecond);

        const int iterations = 20_000;
        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < iterations; i++)
            mapper.Map(stock, raw, DateTime.UnixEpoch.Ticks + i * TimeSpan.TicksPerMillisecond);
        stopwatch.Stop();

        var averageMilliseconds = stopwatch.Elapsed.TotalMilliseconds / iterations;
        Assert.IsTrue(averageMilliseconds < 0.20,
            $"Eye V2 averaged {averageMilliseconds:F4} ms per tick; expected under 0.20 ms.");
    }

    private static void AssertTarget(EyeGazeMap map, EyeRawState raw, float x, float y)
    {
        var mapped = map.Map(raw.X, raw.Y);
        Assert.AreEqual(x, mapped.X, 0.015f);
        Assert.AreEqual(y, mapped.Y, 0.015f);
    }

    private static float[] Map(EyeV2Mapper mapper,
        float leftX, float leftY, float leftOpen,
        float rightX, float rightY, float rightOpen,
        int milliseconds)
    {
        var raw = Raw(leftX, leftY, leftOpen, rightX, rightY, rightOpen);
        return mapper.Map(new float[EyeStateLayout.LegacyCount], raw,
            DateTime.UnixEpoch.Ticks + milliseconds * TimeSpan.TicksPerMillisecond);
    }

    private static EyeV2CalibrationCapture SyntheticCapture()
    {
        var gaze = new Dictionary<string, IReadOnlyList<EyeV2Sample>>();
        foreach (var target in EyeV2GazeTarget.FivePoint)
        {
            var left = RawForTarget(target, EyeSide.Left);
            var right = RawForTarget(target, EyeSide.Right);
            var opennessBump = target.Name == "Up" ? 0.03f : 0f;
            gaze[target.Name] = Samples(30,
                left.X, left.Y, 0.75f + opennessBump,
                right.X, right.Y, 0.70f + opennessBump);
        }

        var blink = new List<EyeV2Sample>();
        for (var cycle = 0; cycle < 3; cycle++)
        {
            blink.AddRange(Samples(6, 0, 0, 0.75f, 0, 0, 0.70f, blink.Count * 33));
            blink.AddRange(Samples(8, 0, 0, 0.08f, 0, 0, 0.12f, blink.Count * 33));
            blink.AddRange(Samples(6, 0, 0, 0.75f, 0, 0, 0.70f, blink.Count * 33));
        }

        return new EyeV2CalibrationCapture
        {
            Relax = Samples(60, 0, 0, 0.75f, 0, 0, 0.70f),
            Blinks = blink,
            Squint = Samples(60, 0, 0, 0.45f, 0, 0, 0.42f),
            Wide = Samples(60, 0, 0, 0.95f, 0, 0, 0.91f),
            Gaze = gaze,
        };
    }

    private static EyeRawState RawForTarget(EyeV2GazeTarget target, EyeSide side)
    {
        // Different personal camera bias/gain/cross-coupling for each eye.
        return side == EyeSide.Left
            ? new EyeRawState(0.15f + target.X / 1.4f + 0.10f * target.Y,
                -0.10f + target.Y / 1.2f - 0.05f * target.X, 0.75f)
            : new EyeRawState(-0.08f + target.X / 1.1f - 0.07f * target.Y,
                0.12f + target.Y / 1.35f + 0.04f * target.X, 0.70f);
    }

    internal static EyeV2Calibration IdentityCalibration() => new()
    {
        Left = new EyeV2PerEyeCalibration
        {
            Gaze = new EyeGazeMap(),
            Lid = new EyeLidAnchors
            {
                Closed = 0.08f, Neutral = 0.75f, NeutralLow = 0.68f, NeutralHigh = 0.82f,
                Squint = 0.45f, Wide = 0.95f, TypicalBlinkMilliseconds = 400f,
            },
        },
        Right = new EyeV2PerEyeCalibration
        {
            Gaze = new EyeGazeMap(),
            Lid = new EyeLidAnchors
            {
                Closed = 0.12f, Neutral = 0.70f, NeutralLow = 0.63f, NeutralHigh = 0.78f,
                Squint = 0.42f, Wide = 0.91f, TypicalBlinkMilliseconds = 400f,
            },
        },
    };

    internal static List<EyeV2Sample> Samples(int count,
        float leftX, float leftY, float leftOpen,
        float rightX, float rightY, float rightOpen,
        int startMilliseconds = 0)
    {
        return Enumerable.Range(0, count)
            .Select(i => new EyeV2Sample(
                DateTime.UnixEpoch.Ticks + (startMilliseconds + i * 33) * TimeSpan.TicksPerMillisecond,
                Raw(leftX, leftY, leftOpen, rightX, rightY, rightOpen)))
            .ToList();
    }

    internal static float[] Raw(float leftX, float leftY, float leftOpen,
        float rightX, float rightY, float rightOpen) =>
    [
        (rightY + 1f) / 2f,
        (rightX + 1f) / 2f,
        1f - rightOpen,
        (leftY + 1f) / 2f,
        (leftX + 1f) / 2f,
        1f - leftOpen,
    ];
}
