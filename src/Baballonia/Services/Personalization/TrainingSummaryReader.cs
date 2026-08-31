using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Reads the trainer's <c>summary.json</c> into <see cref="TrainingSummary"/>.
/// </summary>
/// <remarks>
/// Separate from <see cref="PersonalTrainingService"/> so the format contract can be tested against
/// literal JSON without spawning Python or touching the filesystem. That matters more than usual
/// here: this is the seam between two languages, and the failure it guards against is a silent one -
/// a renamed key does not crash, it just makes a number quietly disappear from the results screen.
///
/// Every field is read defensively. Summaries written by an older trainer are missing whole sections
/// (v1 has no <c>jaw_open</c>, no <c>cross_talk</c>, no <c>hard_examples</c>), and the right
/// behaviour is to report what is there rather than to reject the file.
/// </remarks>
public static class TrainingSummaryReader
{
    /// <summary>Parses summary JSON. Throws only if the text is not JSON at all.</summary>
    public static TrainingSummary Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;

        double? Number(string name) =>
            root.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.Number
                ? element.GetDouble()
                : null;

        double? Nested(string parent, string name) =>
            root.TryGetProperty(parent, out var section) &&
            section.ValueKind == JsonValueKind.Object &&
            section.TryGetProperty(name, out var element) &&
            element.ValueKind == JsonValueKind.Number
                ? element.GetDouble()
                : null;

        var regressions = root.TryGetProperty("worst_regressions", out var list) &&
                          list.ValueKind == JsonValueKind.Array
            ? list.EnumerateArray()
                .Select(e => e.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "")
                .Where(s => s.Length > 0)
                .ToList()
            : [];

        return new TrainingSummary(
            AdapterType: root.TryGetProperty("adapter_type", out var t) ? t.GetString() ?? "" : "",
            Parameters: (int)(Number("parameters") ?? 0),
            ValidatedOnHeldOutSessions: root.TryGetProperty("validated_on_held_out_sessions", out var v)
                                        && v.ValueKind == JsonValueKind.True,
            ExpressionsImproved: (int)(Number("expressions_improved") ?? 0),
            ExpressionsRegressed: (int)(Number("expressions_regressed") ?? 0),
            MeanStockMae: Number("mean_stock_mae"),
            MeanPersonalMae: Number("mean_personal_mae"),
            NeutralStockFalseActivation: Nested("neutral", "stock_false_activation_rate"),
            NeutralPersonalFalseActivation: Nested("neutral", "personal_false_activation_rate"),
            NeutralStockJitter: Nested("neutral", "stock_jitter"),
            NeutralPersonalJitter: Nested("neutral", "personal_jitter"),
            Verdict: root.TryGetProperty("verdict", out var verdict)
                ? verdict.GetString() ?? "unclear"
                : "unclear",
            WorstRegressions: regressions,
            SummaryVersion: (int)(Number("summary_version") ?? 1),
            JawOpen: ReadWatched(root, "jaw_open"),
            TongueOut: ReadWatched(root, "tongue_out"),
            CrossTalkStock: Nested("cross_talk", "stock"),
            CrossTalkPersonal: Nested("cross_talk", "personal"),
            HardExampleFrames: (int)(Nested("hard_examples", "frames") ?? 0));
    }

    private static WatchedExpressionSummary? ReadWatched(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var section) || section.ValueKind != JsonValueKind.Object)
            return null;

        double? Value(string name) =>
            section.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.Number
                ? element.GetDouble()
                : null;

        double? RunValue(string group, string name) =>
            section.TryGetProperty(group, out var runs) &&
            runs.ValueKind == JsonValueKind.Object &&
            runs.TryGetProperty(name, out var element) &&
            element.ValueKind == JsonValueKind.Number
                ? element.GetDouble()
                : null;

        return new WatchedExpressionSummary(
            Name: section.TryGetProperty("name", out var n) ? n.GetString() ?? key : key,
            ClosedFrames: (int)(Value("closed_frames") ?? 0),
            StockFalseActivation: Value("stock_fp_rate"),
            PersonalFalseActivation: Value("personal_fp_rate"),
            StockLongestFalseRunSeconds: RunValue("stock_runs", "max_seconds"),
            PersonalLongestFalseRunSeconds: RunValue("personal_runs", "max_seconds"),
            RangeRetention: Value("range_retention"),
            SuppressionWarning: section.TryGetProperty("suppression_warning", out var warn)
                                && warn.ValueKind == JsonValueKind.True);
    }
}
