using Baballonia.Services.Personalization;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The C#/Python contract for summary.json.
///
/// This is a cross-language seam, and the failure mode it guards against is silent: rename a key on
/// the Python side and nothing throws, a number simply stops appearing on the results screen. The
/// literal JSON in these tests mirrors what <c>train.py</c> actually writes.
///
/// Backward compatibility is tested as deliberately as the new fields. A user who trained before
/// this milestone still has v1 runs on disk, and opening one must show the numbers it does have
/// rather than failing to read it at all.
/// </summary>
[TestClass]
public class TrainingSummaryReaderTest
{
    /// <summary>A v1 summary: no jaw_open, no cross_talk, no hard_examples, no summary_version.</summary>
    private const string V1Json = """
    {
      "summary_version": 1,
      "adapter_type": "image_residual_v1",
      "parameters": 44389,
      "validated_on_held_out_sessions": true,
      "scored_expressions": 45,
      "expressions_improved": 45,
      "expressions_regressed": 0,
      "mean_stock_mae": 0.0962,
      "mean_personal_mae": 0.0068,
      "worst_regressions": [],
      "neutral": {
        "frames": 1655,
        "stock_false_activation_rate": 0.179,
        "personal_false_activation_rate": 0.0031,
        "stock_jitter": 0.0130,
        "personal_jitter": 0.0141
      },
      "verdict": "better"
    }
    """;

    private const string V2Json = """
    {
      "summary_version": 2,
      "adapter_type": "image_residual_v1",
      "parameters": 44389,
      "validated_on_held_out_sessions": true,
      "expressions_improved": 40,
      "expressions_regressed": 2,
      "mean_stock_mae": 0.0962,
      "mean_personal_mae": 0.0068,
      "worst_regressions": [{"name": "MouthSmileLeft", "stock_mae": 0.01, "personal_mae": 0.02}],
      "neutral": {
        "frames": 1655,
        "stock_false_activation_rate": 0.179,
        "personal_false_activation_rate": 0.0031,
        "stock_jitter": 0.0130,
        "personal_jitter": 0.0141
      },
      "jaw_open": {
        "name": "JawOpen",
        "dim": 4,
        "closed_frames": 1555,
        "stock_fp_rate": 0.231,
        "personal_fp_rate": 0.0019,
        "stock_runs": {"count": 12, "runs_per_minute": 14.0, "mean_seconds": 0.8,
                       "p95_seconds": 2.1, "max_seconds": 2.4},
        "personal_runs": {"count": 1, "runs_per_minute": 1.2, "mean_seconds": 0.1,
                          "p95_seconds": 0.1, "max_seconds": 0.13},
        "range_retention": 0.96,
        "suppression_warning": false
      },
      "tongue_out": {
        "name": "TongueOut",
        "dim": 33,
        "closed_frames": 1555,
        "stock_fp_rate": 0.02,
        "personal_fp_rate": 0.0,
        "stock_runs": {"max_seconds": 0.3},
        "personal_runs": {"max_seconds": 0.0},
        "range_retention": 0.4,
        "suppression_warning": true
      },
      "cross_talk": {"stock": 0.21, "personal": 0.08},
      "hard_examples": {"sessions": ["20260814_a_correction"], "frames": 150},
      "verdict": "better"
    }
    """;

    [TestMethod]
    public void ReadsTheV1FieldsEveryRunHas()
    {
        var summary = TrainingSummaryReader.Parse(V1Json);

        Assert.AreEqual("image_residual_v1", summary.AdapterType);
        Assert.AreEqual(44389, summary.Parameters);
        Assert.IsTrue(summary.ValidatedOnHeldOutSessions);
        Assert.AreEqual(45, summary.ExpressionsImproved);
        Assert.AreEqual("better", summary.Verdict);
        Assert.AreEqual(0.179, summary.NeutralStockFalseActivation!.Value, 1e-9);
        Assert.AreEqual(0.0141, summary.NeutralPersonalJitter!.Value, 1e-9);
    }

    [TestMethod]
    public void AV1SummaryStillLoadsAndSimplyHasNoWatchedExpressions()
    {
        var summary = TrainingSummaryReader.Parse(V1Json);

        Assert.AreEqual(1, summary.SummaryVersion);
        Assert.IsNull(summary.JawOpen, "v1 has no jaw_open section");
        Assert.IsNull(summary.TongueOut);
        Assert.IsNull(summary.CrossTalkStock);
        Assert.AreEqual(0, summary.HardExampleFrames);
    }

    [TestMethod]
    public void ReadsTheWatchedExpressionBlock()
    {
        var summary = TrainingSummaryReader.Parse(V2Json);
        var jaw = summary.JawOpen;

        Assert.IsNotNull(jaw);
        Assert.AreEqual("JawOpen", jaw.Name);
        Assert.AreEqual(1555, jaw.ClosedFrames);
        Assert.AreEqual(0.231, jaw.StockFalseActivation!.Value, 1e-9);
        Assert.AreEqual(0.0019, jaw.PersonalFalseActivation!.Value, 1e-9);
        Assert.IsFalse(jaw.SuppressionWarning);
    }

    [TestMethod]
    public void ReadsFalseActivationDurationsWhichRateAloneCannotExpress()
    {
        var jaw = TrainingSummaryReader.Parse(V2Json).JawOpen!;

        Assert.AreEqual(2.4, jaw.StockLongestFalseRunSeconds!.Value, 1e-9);
        Assert.AreEqual(0.13, jaw.PersonalLongestFalseRunSeconds!.Value, 1e-9);
        Assert.IsTrue(jaw.PersonalLongestFalseRunSeconds < jaw.StockLongestFalseRunSeconds,
            "fixture describes a model that shortened its false activations");
    }

    [TestMethod]
    public void CarriesTheSuppressionWarningThrough()
    {
        // The whole point of the guardrail: TongueOut fired less, but only because it stopped
        // moving. That must survive the trip from Python to the results screen.
        var tongue = TrainingSummaryReader.Parse(V2Json).TongueOut!;

        Assert.AreEqual(0.0, tongue.PersonalFalseActivation!.Value, 1e-9);
        Assert.AreEqual(0.4, tongue.RangeRetention!.Value, 1e-9);
        Assert.IsTrue(tongue.SuppressionWarning,
            "a model that stopped moving the expression must not read as a clean win");
    }

    [TestMethod]
    public void ReadsCrossTalkAndHardExampleCounts()
    {
        var summary = TrainingSummaryReader.Parse(V2Json);

        Assert.AreEqual(0.21, summary.CrossTalkStock!.Value, 1e-9);
        Assert.AreEqual(0.08, summary.CrossTalkPersonal!.Value, 1e-9);
        Assert.AreEqual(150, summary.HardExampleFrames);
    }

    [TestMethod]
    public void MissingRunSectionYieldsNullRatherThanZero()
    {
        // Zero would read as "no false activations", which is a claim. Null means "not measured".
        const string json = """
        {"jaw_open": {"name": "JawOpen", "closed_frames": 10, "stock_fp_rate": 0.5}}
        """;

        var jaw = TrainingSummaryReader.Parse(json).JawOpen!;

        Assert.IsNull(jaw.StockLongestFalseRunSeconds);
        Assert.IsNull(jaw.RangeRetention);
        Assert.AreEqual(0.5, jaw.StockFalseActivation!.Value, 1e-9);
    }

    [TestMethod]
    public void NullJsonValuesAreTreatedAsAbsent()
    {
        // The trainer writes an explicit null for cross_talk when there were no guided sessions.
        const string json = """
        {"summary_version": 2, "cross_talk": null, "hard_examples": null, "verdict": "unclear"}
        """;

        var summary = TrainingSummaryReader.Parse(json);

        Assert.IsNull(summary.CrossTalkStock);
        Assert.AreEqual(0, summary.HardExampleFrames);
        Assert.AreEqual("unclear", summary.Verdict);
    }

    [TestMethod]
    public void AnEmptyObjectParsesToConservativeDefaults()
    {
        var summary = TrainingSummaryReader.Parse("{}");

        Assert.AreEqual("", summary.AdapterType);
        Assert.AreEqual("unclear", summary.Verdict, "no verdict means no claim");
        Assert.IsFalse(summary.ValidatedOnHeldOutSessions);
        Assert.AreEqual(1, summary.SummaryVersion);
        Assert.AreEqual(0, summary.WorstRegressions.Count);
    }
}
