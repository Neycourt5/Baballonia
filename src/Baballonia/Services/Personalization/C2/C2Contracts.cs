using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using Baballonia.Contracts;

namespace Baballonia.Services.Personalization.C2;

public static class C2Contract
{
    public const string Version = "c2-raw-reference-v1";
    public const string LabelRecipe = "c2-reviewed-holds-v1";
    public static string Root => Path.Combine(Utils.PersistentDataDirectory, "C2");
    public static string Recordings => Path.Combine(Root, "Recordings");
    public static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    public static void WriteAtomic<T>(string path, T value)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(value, PersonalizationPaths.IndentedJson),
            PersonalizationPaths.Utf8NoBom);
        File.Move(temporary, path, true);
    }
}

/// <summary>Frozen meaning of the output. Values enter the existing filter BEFORE this remap.</summary>
public sealed record C2OutputContext(float[][] Ranges, float EffectiveJawExponent,
    bool FilterEnabled, float MinCutoff, float Beta)
{
    public static C2OutputContext Capture(ICalibrationService calibration, ILocalSettingsService settings)
    {
        var ranges = PersonalizationSchemaBinding.ExpectedKeys.Select(key =>
        {
            var r = calibration.GetExpressionSettings(key);
            return new[] { r.Lower, r.Upper, r.Min, r.Max };
        }).ToArray();
        // The current sender compares "JawOpen", whereas active schema keys are "/jawOpen".
        // Preserve the real legacy path (effective exponent 1), not the UI's inert value.
        return new C2OutputContext(ranges, 1f,
            settings.ReadSetting<bool>("AppSettings_OneEuroEnabled"),
            settings.ReadSetting<float>("AppSettings_OneEuroMinFreqCutoff"),
            settings.ReadSetting<float>("AppSettings_OneEuroSpeedCutoff"));
    }

    public void Validate()
    {
        if (Ranges.Length != 45 || Ranges.Any(r => r.Length != 4 || r.Any(v => !float.IsFinite(v)) ||
            r[1] <= r[0] || r[3] <= r[2]) || !float.IsFinite(EffectiveJawExponent) || EffectiveJawExponent <= 0 ||
            !float.IsFinite(MinCutoff) || !float.IsFinite(Beta) || MinCutoff <= 0 || Beta < 0)
            throw new InvalidDataException("Face ranges cannot be inverted safely. Review the face calibration ranges first.");
    }

    public float ToRawTarget(int channel, float canonical)
    {
        Validate();
        var r = Ranges[channel];
        if (!float.IsFinite(canonical) || canonical < r[2] || canonical > r[3])
            throw new InvalidDataException("This cue is outside the saved output range.");
        var raw = r[0] + (canonical - r[2]) / (r[3] - r[2]) * (r[1] - r[0]);
        if (channel == 4) raw = MathF.Pow(Math.Clamp(raw, 0, 1), 1 / EffectiveJawExponent);
        if (raw < 0 || raw > 1) throw new InvalidDataException("This cue is unreachable under the saved face ranges.");
        return raw;
    }

    public float Emit(int channel, float raw)
    {
        var r = Ranges[channel];
        if (channel == 4) raw = MathF.Pow(Math.Clamp(raw, 0, 1), EffectiveJawExponent);
        return Math.Clamp(r[2] + (raw - r[0]) / (r[1] - r[0]) * (r[3] - r[2]), r[2], r[3]);
    }
}

public sealed record C2CaptureContext(
    string OriginId, string Role, string TaskId, string ContractPath,
    string ContractSha256, int EmbeddingDim, long StopwatchFrequency,
    string Clock = "Stopwatch inference completion; NOT exposure time",
    string LabelRecipe = C2Contract.LabelRecipe);

public sealed record C2ReferenceContract(
    string StockPath, string StockSha256, string FeaturePath, string FeatureSha256,
    string ReferencePath, string ReferenceSha256, float ReferenceBlend,
    C2OutputContext Output, string CameraSettings,
    int EmbeddingDim = 1280, string Version = C2Contract.Version,
    string SchemaSha256 = "", string InputNormalization = "gray_div255");

public sealed record C2Task(string Id, string Name, string Purpose, string Instruction,
    int[] Dims, float[] Values, bool Optional = false, double Seconds = 4)
{
    public override string ToString() => Name;
    public float[] Target()
    {
        var result = new float[45];
        for (var i = 0; i < Dims.Length; i++) result[Dims[i]] = Values[i];
        return result;
    }

    public static IReadOnlyList<C2Task> All { get; } =
    [
        new("rest", "Relaxed jaw and smile", "Checks unwanted jaw and smile movement at rest; other channels remain unknown.",
            "Let your jaw rest without deliberately opening it. Relax your smile. Do not press your lips together.", [4,19,20], [0,0,0]),
        new("jaw-small", "Small jaw opening", "Teaches gentle onset without assuming that parted lips mean an open jaw.",
            "Lower your jaw a small comfortable amount. Keep it steady; do not stretch into a smile.", [4], [.25f]),
        new("jaw-comfortable", "Comfortable jaw opening", "Checks that reducing rest movement does not flatten intentional opening.",
            "Lower your jaw comfortably farther than the small opening. Hold without strain.", [4], [.65f]),
        new("smile-gentle", "Gentle smile", "Teaches subtle smile onset. Jaw and lip lift are not inferred from the smile.",
            "Make a gentle natural smile. Let your jaw and lips do what feels natural.", [19,20], [.25f,.25f]),
        new("smile-natural", "Comfortable natural smile", "Checks a stronger natural smile, including natural coupled movement.",
            "Make a comfortable fuller smile. Do not force your mouth shut or open.", [19,20], [.65f,.65f]),
        new("speech", "Talking and return to rest", "Preservation/replay only: speech is not measured ground truth.",
            "Say a sentence naturally, then relax. No microphone recording is needed.", [], [], Seconds: 8),
        new("toothy", "Toothy smile", "Replay of tooth visibility; this does not assert JawOpen or lip-lift values.",
            "Smile naturally with teeth visible, then relax. Do not open your jaw merely to show teeth.", [], [], true, 6),
        new("smile-speech", "Smile while talking", "Checks mixed expressions without inventing exact speech targets.",
            "Say a sentence while smiling naturally, then relax.", [], [], true, 8),
        new("asymmetry", "One-sided expressions", "Checks that unsupported asymmetric behavior stays with your working model.",
            "Move one corner of your mouth, then the other. Use your normal range.", [], [], true, 8)
    ];
}

public sealed record C2Review(string SessionDirectory, string OriginId, string Role,
    string TaskId, bool Accepted, string Reason, string ReviewedUtc,
    string? CueId = null, int? Repetition = null, int? Attempt = null,
    long? StartTicks = null, long? EndTicks = null,
    string LabelRecipe = C2Contract.LabelRecipe,
    string? SessionSha256 = null, string? LabelsSha256 = null, string? EmbeddingsSha256 = null);

public sealed record C2Recording(string Directory, string Id, string TaskId, string Role,
    string OriginId, string State, string Reason, int Frames, bool Legacy)
{
    public override string ToString() => $"{Id} — {State}";
}
