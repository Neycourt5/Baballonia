using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Proves that the running face model's keyed output lines up, position for position, with
/// <see cref="PersonalizationSchema"/>.
/// </summary>
/// <remarks>
/// Personalization is positional and the pipeline is keyed. Every recorded dataset frame, every
/// trained adapter and every <c>expression_schema_sha256</c> stamped into an exported ONNX file
/// assumes index <c>i</c> means <see cref="PersonalizationSchema.ExpressionNames"/><c>[i]</c>. The
/// pipeline, meanwhile, hands out an <see cref="OrderedFloatMap"/> keyed by OSC address, whose order
/// comes from the model's metadata or from <c>DefaultInferenceRunner</c>'s known-layout table.
///
/// If those two orders ever drift apart, nothing crashes: expressions simply get fed to the wrong
/// slots, the personal model produces confident nonsense, and 20,000+ recorded training frames
/// silently stop describing reality. So this fails loudly instead, at the first frame after a model
/// is installed, naming the exact mismatch.
///
/// The mapping between the two spellings is mechanical: the map's keys are the schema names with a
/// <c>/</c> prefix and a lowercased first letter (<c>CheekPuffLeft</c> ↔ <c>/cheekPuffLeft</c>).
/// </remarks>
public static class PersonalizationSchemaBinding
{
    /// <summary>
    /// The map keys the personalization schema expects, in schema order.
    /// </summary>
    public static IReadOnlyList<string> ExpectedKeys { get; } =
        PersonalizationSchema.ExpressionNames.Select(ToKey).ToArray();

    /// <summary>OSC-style key for a schema expression name, e.g. <c>CheekPuffLeft</c> → <c>/cheekPuffLeft</c>.</summary>
    public static string ToKey(string expressionName) =>
        "/" + char.ToLowerInvariant(expressionName[0]) + expressionName[1..];

    /// <summary>
    /// Throws <see cref="InvalidOperationException"/> unless <paramref name="map"/>'s first
    /// <see cref="PersonalizationSchema.ExpressionCount"/> keys are exactly
    /// <see cref="ExpectedKeys"/>, in order.
    /// </summary>
    public static void Bind(OrderedFloatMap map)
    {
        ArgumentNullException.ThrowIfNull(map);

        var actual = map.Keys.ToArray();
        if (actual.Length < PersonalizationSchema.ExpressionCount)
        {
            throw new InvalidOperationException(
                $"Face model output has {actual.Length} values but the personalization schema " +
                $"(version {PersonalizationSchema.Version}, sha256 {PersonalizationSchema.Sha256}) " +
                $"requires {PersonalizationSchema.ExpressionCount}. Personalization cannot run " +
                "against this model - every trained adapter and recorded dataset assumes the " +
                "45-expression layout.");
        }

        var mismatches = new List<string>();
        for (var i = 0; i < PersonalizationSchema.ExpressionCount; i++)
        {
            if (!string.Equals(actual[i], ExpectedKeys[i], StringComparison.Ordinal))
                mismatches.Add($"  [{i}] expected '{ExpectedKeys[i]}', model has '{actual[i]}'");
        }

        if (mismatches.Count == 0)
            return;

        var message = new StringBuilder()
            .AppendLine("Face expression ordering does not match the personalization schema.")
            .AppendLine($"Schema version {PersonalizationSchema.Version}, sha256 {PersonalizationSchema.Sha256}.")
            .AppendLine($"{mismatches.Count} of {PersonalizationSchema.ExpressionCount} positions differ:")
            .AppendLine(string.Join(Environment.NewLine, mismatches.Take(10)));

        if (mismatches.Count > 10)
            message.AppendLine($"  ... and {mismatches.Count - 10} more");

        message.Append(
            "Refusing to run personalization against a mismatched layout: it would feed every " +
            "expression to the wrong slot and invalidate every trained personal model and every " +
            "recorded training frame.");

        throw new InvalidOperationException(message.ToString());
    }

    /// <summary>
    /// Non-throwing form for startup diagnostics and tests.
    /// </summary>
    public static bool TryBind(OrderedFloatMap map, out string? error)
    {
        try
        {
            Bind(map);
            error = null;
            return true;
        }
        catch (InvalidOperationException ex)
        {
            error = ex.Message;
            return false;
        }
    }
}
