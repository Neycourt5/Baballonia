using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace Baballonia.Services.Personalization.Eye;

/// <summary>
/// Canonical, order-sensitive schema for the tuned eye model's twelve-value output.
/// </summary>
/// <remarks>
/// <para>The eye correction layer is positional — a trained adapter maps twelve numbers to twelve
/// numbers and has no idea what any of them mean. The pipeline is keyed by OSC address. This class
/// is the contract between the two, and it is exactly as load-bearing as
/// <see cref="PersonalizationSchema"/> is for the face: if the order ever drifts, a trained adapter
/// applies every learned correction to the wrong channel, which reads as erratic tracking rather
/// than as an error.</para>
///
/// <para>The order is the tuned model's own <c>blendshape_names</c> metadata, right eye first. The
/// map keys are these names with a <c>/</c> prefix, which is added by
/// <c>DefaultInferenceRunner</c>.</para>
///
/// <para>The stock six-output eye model has no widen/squint/brow at all and therefore cannot be
/// personalized by this layer — <see cref="EyePersonalizationSchemaBinding"/> refuses it rather
/// than silently correcting the wrong six values.</para>
/// </remarks>
public static class EyePersonalizationSchema
{
    /// <summary>Bumped whenever <see cref="ExpressionNames"/> changes in any way.</summary>
    public const int Version = 1;

    public const int ExpressionCount = 12;

    /// <summary>Full-scale gaze angle in degrees, from the tuned model's <c>gaze_range_deg</c>.</summary>
    public const float GazeRangeDegrees = 45f;

    private static readonly string[] Names =
    [
        "rightEyeY",
        "rightEyeX",
        "rightEyeLid",
        "rightEyeWiden",
        "rightEyeSquint",
        "rightEyeBrow",
        "leftEyeY",
        "leftEyeX",
        "leftEyeLid",
        "leftEyeWiden",
        "leftEyeSquint",
        "leftEyeBrow",
    ];

    /// <summary>The twelve names in model-output order. Index i corresponds to output[i].</summary>
    public static IReadOnlyList<string> ExpressionNames => Names;

    /// <summary>The map keys those names produce, in the same order.</summary>
    public static IReadOnlyList<string> ExpressionKeys { get; } =
        Names.Select(name => "/" + name).ToArray();

    private static readonly Lazy<string> LazyHash = new(ComputeSha256);

    /// <summary>
    /// Lowercase hex SHA-256 over the names joined by a single LF, UTF-8, no trailing newline and
    /// no BOM — the same recipe <see cref="PersonalizationSchema"/> uses, so the Python side can
    /// share one implementation. Exported adapters embed this as <c>eye_schema_sha256</c> and are
    /// refused at load if it disagrees.
    /// </summary>
    public static string Sha256 => LazyHash.Value;

    private static string ComputeSha256() =>
        Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join("\n", Names))));

    /// <summary>Index of a name in the output vector, or -1. Setup paths only, not per-frame.</summary>
    public static int IndexOf(string expressionName)
    {
        for (var i = 0; i < Names.Length; i++)
        {
            if (string.Equals(Names[i], expressionName, StringComparison.Ordinal))
                return i;
        }

        return -1;
    }
}

/// <summary>
/// Proves a running eye model's keyed output lines up, position for position, with
/// <see cref="EyePersonalizationSchema"/>.
/// </summary>
/// <remarks>
/// Unlike the face equivalent this is expected to fail in normal use: the stock six-output eye
/// model is a perfectly valid model that simply cannot be personalized by an adapter trained on
/// twelve channels. So the caller is given a reason rather than an exception on the hot path, and
/// personalization stays switched off until a twelve-output model is loaded.
/// </remarks>
public static class EyePersonalizationSchemaBinding
{
    /// <summary>
    /// True when <paramref name="map"/> is exactly the twelve schema keys in schema order.
    /// </summary>
    public static bool TryBind(OrderedFloatMap? map, out string? error)
    {
        if (map is null)
        {
            error = "No eye model output to bind to.";
            return false;
        }

        var actual = map.Keys.ToArray();
        var expected = EyePersonalizationSchema.ExpressionKeys;

        if (actual.Length != expected.Count)
        {
            error =
                $"The loaded eye model produces {actual.Length} values; personalized eye correction " +
                $"needs the {expected.Count}-output model (per-eye gaze, lid, widen, squint and brow). " +
                "The stock eye model cannot be personalized.";
            return false;
        }

        for (var i = 0; i < expected.Count; i++)
        {
            if (string.Equals(actual[i], expected[i], StringComparison.Ordinal))
                continue;

            error =
                $"The loaded eye model's output order does not match the personalization schema at " +
                $"position {i}: expected '{expected[i]}', model has '{actual[i]}'. Correcting it " +
                "would apply every learned change to the wrong channel.";
            return false;
        }

        error = null;
        return true;
    }
}
