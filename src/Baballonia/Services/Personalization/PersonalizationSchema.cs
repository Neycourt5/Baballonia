using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Canonical, order-sensitive schema for the stock face model's 45-value output vector.
///
/// The stock model (faceModel.onnx) emits a positional float[45]; nothing in the ONNX file
/// names those outputs. The de-facto source of truth is the insertion order of
/// <see cref="ParameterSenderService.FaceExpressionMap"/>. This class mirrors that order so
/// personalization code (dataset recording, training, personal-model validation) can depend on a
/// stable contract without editing upstream code. PersonalizationSchemaTests fails loudly if the
/// two ever diverge, which would otherwise silently corrupt every personal model trained before
/// the change.
/// </summary>
public static class PersonalizationSchema
{
    /// <summary>Bumped whenever <see cref="ExpressionNames"/> changes in any way.</summary>
    public const int Version = 1;

    /// <summary>Number of face expression values produced by the stock model. Matches Utils.FaceRawExpressions.</summary>
    public const int ExpressionCount = 45;

    private static readonly string[] Names =
    [
        "CheekPuffLeft",
        "CheekPuffRight",
        "CheekSuckLeft",
        "CheekSuckRight",
        "JawOpen",
        "JawForward",
        "JawLeft",
        "JawRight",
        "NoseSneerLeft",
        "NoseSneerRight",
        "MouthFunnel",
        "MouthPucker",
        "MouthLeft",
        "MouthRight",
        "MouthRollUpper",
        "MouthRollLower",
        "MouthShrugUpper",
        "MouthShrugLower",
        "MouthClose",
        "MouthSmileLeft",
        "MouthSmileRight",
        "MouthFrownLeft",
        "MouthFrownRight",
        "MouthDimpleLeft",
        "MouthDimpleRight",
        "MouthUpperUpLeft",
        "MouthUpperUpRight",
        "MouthLowerDownLeft",
        "MouthLowerDownRight",
        "MouthPressLeft",
        "MouthPressRight",
        "MouthStretchLeft",
        "MouthStretchRight",
        "TongueOut",
        "TongueUp",
        "TongueDown",
        "TongueLeft",
        "TongueRight",
        "TongueRoll",
        "TongueBendDown",
        "TongueCurlUp",
        "TongueSquish",
        "TongueFlat",
        "TongueTwistLeft",
        "TongueTwistRight"
    ];

    /// <summary>
    /// The 45 expression names in model-output order. Index i corresponds to output[i].
    /// </summary>
    public static IReadOnlyList<string> ExpressionNames => Names;

    private static readonly Lazy<string> LazyHash = new(ComputeSha256);

    /// <summary>
    /// Lowercase hex SHA-256 over the canonical serialization of <see cref="ExpressionNames"/>:
    /// the names joined by a single LF ('\n'), UTF-8 encoded, with no trailing newline and no BOM.
    ///
    /// This exact recipe is reproduced by the Python trainer (training/babble_personal/schema.py)
    /// and embedded in exported personal models as "expression_schema_sha256". A personal model
    /// whose stored hash does not match this value is rejected at load time.
    /// </summary>
    public static string Sha256 => LazyHash.Value;

    private static string ComputeSha256()
    {
        var canonical = string.Join("\n", Names);
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(canonical));
        return Convert.ToHexStringLower(hash);
    }

    /// <summary>
    /// Index of <paramref name="expressionName"/> in the output vector, or -1 if unknown.
    /// Linear scan; intended for setup/validation paths, not per-frame use.
    /// </summary>
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
