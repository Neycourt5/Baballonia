using BabblePersonalizer.Core.Models;
using System.Security.Cryptography;
using System.Text;

namespace BabblePersonalizer.Core.Inventory;

public static class LegacyBaballoniaFaceCatalog
{
    public static readonly IReadOnlyList<string> OrderedNames = new[]
    {
        "CheekPuffLeft", "CheekPuffRight", "CheekSuckLeft", "CheekSuckRight",
        "JawOpen", "JawForward", "JawLeft", "JawRight", "NoseSneerLeft", "NoseSneerRight",
        "MouthFunnel", "MouthPucker", "MouthLeft", "MouthRight", "MouthRollUpper", "MouthRollLower",
        "MouthShrugUpper", "MouthShrugLower", "MouthClose", "MouthSmileLeft", "MouthSmileRight",
        "MouthFrownLeft", "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight",
        "MouthUpperUpLeft", "MouthUpperUpRight", "MouthLowerDownLeft", "MouthLowerDownRight",
        "MouthPressLeft", "MouthPressRight", "MouthStretchLeft", "MouthStretchRight",
        "TongueOut", "TongueUp", "TongueDown", "TongueLeft", "TongueRight", "TongueRoll",
        "TongueBendDown", "TongueCurlUp", "TongueSquish", "TongueFlat", "TongueTwistLeft", "TongueTwistRight"
    };

    public static string ExpressionListHash => Convert.ToHexString(SHA256.HashData(
        Encoding.UTF8.GetBytes(string.Join("\n", OrderedNames))));

    public static IReadOnlyList<FaceParameterDefinition> CreateDefinitions(IReadOnlyList<string>? modelNames = null)
    {
        var names = modelNames ?? OrderedNames;
        return names.Select((name, index) =>
        {
            var known = OrderedNames.Contains(name, StringComparer.Ordinal);
            return new FaceParameterDefinition(
                name, Humanize(name), name, index, known ? Category(name) : "Unresolved",
                known ? FaceParameterType.UnsignedContinuous : FaceParameterType.Passthrough,
                0, 1, known ? Counterpart(name) : null,
                known ? Strategy(name) : CalibrationStrategy.UnsupportedPassthrough, true,
                known ? "/" + char.ToLowerInvariant(name[0]) + name[1..] : null,
                known ? Aliases(name) : Array.Empty<string>(),
                !known ? "Not present in the validated Baballonia catalog; developer mapping required." :
                Strategy(name) == CalibrationStrategy.ManualReviewOnly
                    ? "Physical isolation is ambiguous; preserve unless manually reviewed." : null);
        }).ToArray();
    }

    public static IReadOnlyList<(int Left, int Right)> LeftRightPairs(IReadOnlyList<FaceParameterDefinition> parameters)
    {
        var byName = parameters.ToDictionary(x => x.CanonicalName, x => x.OutputIndex);
        return parameters.Where(x => x.CanonicalName.EndsWith("Left", StringComparison.Ordinal))
            .Select(x => (x.OutputIndex, Right: byName.GetValueOrDefault(x.CanonicalName[..^4] + "Right", -1)))
            .Where(x => x.Right >= 0).ToArray();
    }

    private static CalibrationStrategy Strategy(string name) => name switch
    {
        "MouthClose" => CalibrationStrategy.MaximumOnly,
        "TongueSquish" or "TongueFlat" or "TongueTwistLeft" or "TongueTwistRight" => CalibrationStrategy.ManualReviewOnly,
        _ when name.EndsWith("Left", StringComparison.Ordinal) => CalibrationStrategy.LeftGuidedRamp,
        _ when name.EndsWith("Right", StringComparison.Ordinal) => CalibrationStrategy.RightGuidedRamp,
        _ => CalibrationStrategy.SymmetricGuidedRamp
    };

    private static string Category(string name) => name.StartsWith("Cheek") ? "Cheek" :
        name.StartsWith("Jaw") ? "Jaw" : name.StartsWith("Nose") ? "Nose" :
        name.StartsWith("Tongue") ? "Tongue" : "Mouth/Lip";

    private static string? Counterpart(string name) => name.EndsWith("Left", StringComparison.Ordinal) ? name[..^4] + "Right" :
        name.EndsWith("Right", StringComparison.Ordinal) ? name[..^5] + "Left" : null;

    private static string[] Aliases(string name) => name switch
    {
        "MouthFunnel" => new[] { "LipFunnelLowerLeft", "LipFunnelLowerRight", "LipFunnelUpperLeft", "LipFunnelUpperRight" },
        "MouthPucker" => new[] { "LipPuckerLowerLeft", "LipPuckerLowerRight", "LipPuckerUpperLeft", "LipPuckerUpperRight" },
        "MouthRollUpper" => new[] { "LipSuckUpperLeft", "LipSuckUpperRight" },
        "MouthRollLower" => new[] { "LipSuckLowerLeft", "LipSuckLowerRight" },
        "MouthSmileLeft" => new[] { "MouthCornerPullLeft" },
        "MouthSmileRight" => new[] { "MouthCornerPullRight" },
        "MouthClose" => new[] { "MouthClosed" },
        _ => Array.Empty<string>()
    };

    private static string Humanize(string name) => string.Concat(name.Select((c, i) => i > 0 && char.IsUpper(c) ? " " + c : c.ToString()));
}
