using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Baballonia.Services.Personalization;

/// <summary>What the user was doing during a recording, which decides how labels are built.</summary>
public enum SessionType
{
    /// <summary>Relaxed face. Supplies the high-confidence all-zero anchors.</summary>
    Neutral,

    /// <summary>Cue-driven expressions (avatar-guided or on-screen bar). Commanded target is the label.</summary>
    Guided,

    /// <summary>Natural speech. No per-frame priors; used for realistic expression combinations.</summary>
    Speech,

    /// <summary>
    /// A stretch the user flagged as wrong while using the tracker, saved from the rolling buffer.
    /// Supervises only the dimensions the correction names; every other expression is left unlabelled
    /// because the user asserted one specific thing and may have been emoting freely otherwise.
    /// </summary>
    Correction
}

/// <summary>
/// What the user asserted when they flagged a moment as wrong. Written as correction.json beside the
/// frames, and the reason a correction session can be trusted more than any automatic label: it is
/// the one source where a human looked at the result and said "that was not my face".
/// </summary>
public sealed class CorrectionMetadata
{
    public int Version { get; init; } = 1;

    /// <summary>Machine-readable action, e.g. <c>mouth_closed</c>.</summary>
    public string Kind { get; init; } = "";

    /// <summary>Expression indices this correction speaks about. Everything else stays unlabelled.</summary>
    public IReadOnlyList<int> CorrectedDims { get; init; } = [];

    /// <summary>Value those dimensions should have had.</summary>
    public float Target { get; init; }

    /// <summary>How far back the user said the mistake extended.</summary>
    public double WindowSeconds { get; init; }

    public string FlaggedUtc { get; init; } = "";

    /// <summary>Always <c>hard_example</c> today; distinguishes provenance if other sources appear.</summary>
    public string Source { get; init; } = "hard_example";

    /// <summary>
    /// Which model produced the mistake. Without it a correction is unattributable: a flag captured
    /// against an old model may already be fixed, and retraining on it would chase a stale failure.
    /// </summary>
    public ModelProvenance? Model { get; init; }

    public sealed class ModelProvenance
    {
        public string? AdapterType { get; init; }
        public string? TrainedUtc { get; init; }
        public float Blend { get; init; }
    }
}

/// <summary>
/// Written once per session as session.json. Everything the trainer needs to interpret the frames
/// without guessing, and enough provenance to tell whether two sessions are comparable.
/// </summary>
public sealed class SessionMetadata
{
    /// <summary>Bumped when the on-disk layout changes in a way readers must notice.</summary>
    public int SchemaVersion { get; init; } = 1;

    public string SessionId { get; init; } = "";
    public string SessionType { get; init; } = "";
    public string StartedUtc { get; init; } = "";
    public string? EndedUtc { get; set; }

    public string AppVersion { get; init; } = "";

    /// <summary>Canonical expression order plus its hash, so a trainer can refuse mismatched data.</summary>
    public IReadOnlyList<string> ExpressionNames { get; init; } = PersonalizationSchema.ExpressionNames;
    public string ExpressionSchemaSha256 { get; init; } = PersonalizationSchema.Sha256;
    public int ExpressionSchemaVersion { get; init; } = PersonalizationSchema.Version;

    /// <summary>Frame geometry as recorded (post-transform, exactly what the model consumed).</summary>
    public int ImageWidth { get; init; }
    public int ImageHeight { get; init; }
    public string ImageFormat { get; init; } = "jpeg";
    public int JpegQuality { get; init; }
    public string ImageColor { get; init; } = "gray8";
    public string InputNormalization { get; init; } = "gray_div255";

    /// <summary>
    /// Camera and crop settings in force. A personal model is only valid for roughly the geometry it
    /// was trained on, so this is what lets the runtime warn about a changed ROI.
    /// </summary>
    public CameraGeometry? Camera { get; init; }

    /// <summary>Free-form note from the user (e.g. "headset slightly higher").</summary>
    public string? Notes { get; set; }

    /// <summary>Populated on clean stop; lets a reader spot truncated sessions.</summary>
    public int? FrameCount { get; set; }

    /// <summary>Measured unique frames per second, for diagnosing camera rate vs tick rate.</summary>
    public double? EffectiveFps { get; set; }

    public sealed class CameraGeometry
    {
        public string? Address { get; init; }
        public string? Backend { get; init; }
        public int NativeWidth { get; init; }
        public int NativeHeight { get; init; }
        public double RoiX { get; init; }
        public double RoiY { get; init; }
        public double RoiWidth { get; init; }
        public double RoiHeight { get; init; }
        public double RotationRadians { get; init; }
        public double Gamma { get; init; }
        public bool HorizontalFlip { get; init; }
        public bool VerticalFlip { get; init; }
    }
}

/// <summary>
/// One line of labels.jsonl. Kept compact because there is one per frame, and deliberately flat so
/// the file stays greppable and diffable while debugging a session.
/// </summary>
public sealed class FrameLabel
{
    /// <summary>Frame index; matches the zero-padded image filename.</summary>
    [JsonPropertyName("i")]
    public int Index { get; init; }

    /// <summary>UTC ticks captured at inference time, for cue/response alignment.</summary>
    [JsonPropertyName("t")]
    public long TimestampTicks { get; init; }

    /// <summary>Raw pre-filter stock prediction: the adapter's runtime input, never a label by itself.</summary>
    [JsonPropertyName("stock")]
    public float[] Stock { get; init; } = [];

    /// <summary>
    /// What the personal model actually emitted for this frame, when one was active. Recorded only
    /// for corrections, where the whole point is that this value was wrong; normal sessions omit it
    /// so labels.jsonl does not double in size for data no trainer reads.
    /// </summary>
    [JsonPropertyName("personal")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public float[]? Personal { get; init; }

    /// <summary>Cue state when guided, otherwise null.</summary>
    [JsonPropertyName("cue")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public CueLabel? Cue { get; init; }

    public sealed class CueLabel
    {
        [JsonPropertyName("id")] public string Id { get; init; } = "";
        [JsonPropertyName("phase")] public string Phase { get; init; } = "";

        /// <summary>Indices this cue drives; distinguishes a commanded 0 from an uncommanded value.</summary>
        [JsonPropertyName("dims")] public IReadOnlyList<int> Dims { get; init; } = [];

        /// <summary>Full commanded 45-vector at this instant - the supervision target.</summary>
        [JsonPropertyName("target")] public float[] Target { get; init; } = [];

        [JsonPropertyName("level")] public float Level { get; init; }
        [JsonPropertyName("rep")] public int Repetition { get; init; }

        /// <summary>"avatar" when the VRChat avatar was the visual teacher, otherwise "bar".</summary>
        [JsonPropertyName("source")] public string Source { get; init; } = "bar";
    }
}

/// <summary>Filesystem layout for personalization data. Everything lives outside the repo.</summary>
public static class PersonalizationPaths
{
    /// <summary>
    /// Datasets live under APPDATA, next to the existing ModelData directory - deliberately not the
    /// Documents folder, which is commonly OneDrive-synced. Thousands of small JPEGs in a synced
    /// folder cause sync churn, file locks and cloud-quota surprises.
    /// </summary>
    public static string DatasetRoot => Path.Combine(Utils.PersistentDataDirectory, "PersonalDataset");

    /// <summary>Trained personal models sit beside the tuned eye models the app already manages.</summary>
    public static string ModelsRoot => Utils.ModelsDirectory;

    public static string DefaultPersonalModelPath => Path.Combine(ModelsRoot, "personalFaceModel.onnx");

    /// <summary>Dedicated slots keep training one architecture from destroying another.</summary>
    public static string PersonalModelPath(string modelKind) => Path.Combine(
        ModelsRoot,
        modelKind.ToLowerInvariant() switch
        {
            "a" => "personalFaceModel-a.onnx",
            "b" => "personalFaceModel-b.onnx",
            "c" => "personalFaceModel-c.onnx",
            _ => throw new ArgumentOutOfRangeException(nameof(modelKind), "Expected model A, B, or C.")
        });

    public static string SessionDirectory(string sessionId) => Path.Combine(DatasetRoot, sessionId);
    public static string FramesDirectory(string sessionId) => Path.Combine(SessionDirectory(sessionId), "frames");
    public static string SessionMetadataPath(string sessionId) => Path.Combine(SessionDirectory(sessionId), "session.json");
    public static string LabelsPath(string sessionId) => Path.Combine(SessionDirectory(sessionId), "labels.jsonl");
    public static string CorrectionPath(string sessionId) => Path.Combine(SessionDirectory(sessionId), "correction.json");

    /// <summary>e.g. 20260813_193000_guided - sorts chronologically and says what it is at a glance.</summary>
    public static string NewSessionId(SessionType type, DateTime utcNow) =>
        $"{utcNow:yyyyMMdd_HHmmss}_{type.ToString().ToLowerInvariant()}";

    /// <summary>
    /// UTF-8 with no byte-order mark, for every personalization JSON file we write.
    /// <para>
    /// <c>Encoding.UTF8</c> is <i>not</i> this: its preamble is the BOM, so a <see cref="StreamWriter"/>
    /// built with it emits <c>EF BB BF</c> at the head of a new file. JSON and JSONL readers are not
    /// required to tolerate that, and Python's <c>json.loads</c> rejects it outright - which is
    /// exactly how it broke the trainer. Writing no BOM keeps the files valid for every reader.
    /// </para>
    /// <para>
    /// Also passed on append. <see cref="StreamWriter"/> only skips the preamble when it can seek and
    /// finds a non-empty file, so a BOM-emitting encoding on a fresh-but-reopened file could put one
    /// mid-stream. With an empty preamble that hazard does not exist at all, by construction rather
    /// than by relying on framework behavior.
    /// </para>
    /// </summary>
    public static readonly UTF8Encoding Utf8NoBom = new(encoderShouldEmitUTF8Identifier: false);

    public static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    public static readonly JsonSerializerOptions IndentedJson = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };
}
