using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace Baballonia.Services.Personalization.Eye;

/// <summary>Where recorded eye-personalization data lives.</summary>
public static class EyeDatasetPaths
{
    /// <summary>
    /// Beside the face dataset, under APPDATA rather than Documents — thousands of small files in a
    /// cloud-synced folder cause sync churn and file locks.
    /// </summary>
    public static string DatasetRoot =>
        Path.Combine(Utils.PersistentDataDirectory, "EyePersonalDataset");

    public static string SessionDirectory(string sessionId) =>
        Path.Combine(DatasetRoot, sessionId);

    public static string LabelsPath(string sessionId) =>
        Path.Combine(SessionDirectory(sessionId), "labels.jsonl");

    public static string MetadataPath(string sessionId) =>
        Path.Combine(SessionDirectory(sessionId), "session.json");

    public static string QualityPath(string sessionId) =>
        Path.Combine(SessionDirectory(sessionId), "quality.json");

    /// <summary>Sorts chronologically, stays unique if two sessions start in the same tick.</summary>
    public static string NewSessionId(DateTime utcNow) =>
        $"{utcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture)}_" +
        Guid.NewGuid().ToString("N")[..8];
}

/// <summary>What the cue engine was commanding when a frame was recorded.</summary>
public sealed record EyeCueLabel(string id, string phase, int rep, int attempt);

/// <summary>One recorded frame: the model's own twelve values, and what was being asked for.</summary>
/// <remarks>
/// Short property names because this is one line per frame at ~90 Hz. No image: the correction layer
/// reads the model's output, not pixels, so a session is kilobytes rather than megabytes — the one
/// genuine simplification the eye path has over the face one.
/// </remarks>
public sealed record EyeFrameLabel(int i, long t, float[] stock, EyeCueLabel? cue);

/// <summary>Supervision for one channel, written into the session so the trainer cannot drift.</summary>
public sealed record EyeChannelSupervisionRecord(int dim, float? target, float weight);

/// <summary>A pose as recorded, including exactly what it licenses training to assert.</summary>
public sealed record EyePoseRecord(
    string id,
    string displayName,
    float? dotXDegrees,
    float? dotYDegrees,
    double holdSeconds,
    int repetitions,
    IReadOnlyList<EyeChannelSupervisionRecord> supervision);

/// <summary>
/// The session header, written when recording starts and finalised when it ends.
/// </summary>
/// <remarks>
/// <para><c>baseEyeModelMd5</c> is the load-bearing field. These recordings are the *output of a
/// specific model*; against a different base model the numbers describe a different function and
/// training on them would teach corrections for a model that is no longer running. The trainer
/// refuses to mix sessions whose base model differs.</para>
///
/// <para><c>poses</c> is written here rather than hard-coded in the trainer so the supervision table
/// has exactly one source. A C# table and a Python table would be free to drift, and the drift
/// would be silent — labels quietly asserting the wrong thing.</para>
/// </remarks>
public sealed record EyeSessionMetadata(
    int schemaVersion,
    string sessionId,
    string startedUtc,
    string? endedUtc,
    string appVersion,
    IReadOnlyList<string> eyeSchemaNames,
    string eyeSchemaSha256,
    float gazeRangeDegrees,
    string baseEyeModelPath,
    string baseEyeModelMd5,
    string presentation,
    IReadOnlyList<EyePoseRecord> poses,
    int frameCount,
    double effectiveFps);

/// <summary>Per-pose outcome, so the trainer can distrust what the user could not perform.</summary>
public sealed record EyePoseQuality(
    string id,
    int repetition,
    int attempt,
    int heldFrames,
    bool valid,
    string? reason);

public sealed record EyeSessionQuality(
    IReadOnlyList<EyePoseQuality> attempts,
    bool cameraDisturbed,
    bool completed);
