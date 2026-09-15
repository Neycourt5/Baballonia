using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Baballonia.Services.Personalization.C2;

/// <summary>Source data is read-only. Every decision and derived artifact lives under C2.</summary>
public sealed class C2Workspace(string root, string legacyRoot)
{
    public string Root { get; } = root;
    public string RecordingsRoot => Path.Combine(Root, "Recordings");
    public string OriginPath => Path.Combine(Root, "wearing-session.json");
    public string ReviewPath(string directory) => Path.Combine(Root, "Reviews",
        Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(Path.GetFullPath(directory).ToUpperInvariant()))) + ".json");

    public string GetOrCreateOrigin(bool newWearing = false)
    {
        if (!newWearing && File.Exists(OriginPath))
            return JsonSerializer.Deserialize<string>(File.ReadAllText(OriginPath))!;
        var id = Guid.NewGuid().ToString("N");
        C2Contract.WriteAtomic(OriginPath, id);
        return id;
    }

    public IReadOnlyList<C2Recording> Inventory()
    {
        var result = new List<C2Recording>();
        foreach (var (source, legacy) in new[] { (legacyRoot, true), (RecordingsRoot, false) })
        {
            if (!Directory.Exists(source)) continue;
            foreach (var directory in Directory.EnumerateDirectories(source).Order())
            {
                try
                {
                    var path = Path.Combine(directory, "session.json");
                    if (!File.Exists(path))
                    {
                        result.Add(new(directory, Path.GetFileName(directory), "unknown", "none", "unknown",
                            "Incomplete", "No session metadata; original files retained. No labels guessed.", 0, legacy));
                        continue;
                    }
                    var m = JsonSerializer.Deserialize<SessionMetadata>(File.ReadAllText(path))!;
                    var review = ReadReview(directory);
                    var valid = m.ExpressionSchemaSha256 == PersonalizationSchema.Sha256 &&
                        m.InputNormalization == "gray_div255" && m.ImageWidth == 224 && m.ImageHeight == 224;
                    var complete = m.EndedUtc != null && m.FrameCount > 0 && File.Exists(Path.Combine(directory, "labels.jsonl"));
                    var unchanged = complete && review?.SessionSha256 == C2Contract.Hash(path) &&
                        review.LabelsSha256 == C2Contract.Hash(Path.Combine(directory, "labels.jsonl")) &&
                        (!File.Exists(Path.Combine(directory, "embeddings.f32")) ||
                            review.EmbeddingsSha256 == C2Contract.Hash(Path.Combine(directory, "embeddings.f32")));
                    var state = !complete ? "Incomplete" : !valid ? "Incompatible" : review?.Accepted == true ?
                        (!unchanged ? "Needs review" : legacy ? "Replay / preparation" : "Accepted") : review != null ? "Excluded" : "Needs review";
                    var reason = !valid ? "Image/schema contract differs; keep the source and record this task again." :
                        !complete ? "Capture did not finish cleanly. Review intact files; repeat this attempt only." :
                        review?.Accepted == true && !unchanged ? "Source changed since review, or review provenance is missing. Review this attempt again before reuse." :
                        legacy ? LegacyReason(m.SessionType) : "Review the attempt before it can teach C2. Unknown channels stay unknown.";
                    result.Add(new(directory, m.SessionId, m.C2?.TaskId ?? m.SessionType,
                        m.C2?.Role ?? "replay", m.C2?.OriginId ?? "legacy-unknown",
                        state, reason, m.FrameCount ?? 0, legacy));
                }
                catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
                {
                    result.Add(new(directory, Path.GetFileName(directory), "unknown", "none", "unknown",
                        "Unreadable", "Cannot read this recording: " + ex.Message, 0, legacy));
                }
            }
        }
        return result;
    }

    private static string LegacyReason(string type) => type.ToLowerInvariant() switch
    {
        "guided" => "Reusable images and cues after local preparation/review. Missing wearing-session and feature provenance; never independent check evidence.",
        "speech" => "Useful for C preservation and replay; speech predictions are a teacher prior, not ground truth.",
        "neutral" => "Useful for replay. The session name does not verify 45 zero targets; new explicit jaw/smile-rest evidence is still needed.",
        "correction" => "Replay only until the marked interval and assertion are reviewed. Closed lips do not prove JawOpen = 0.",
        _ => "Unknown supervision; retained for review, not automatically labelled."
    };

    public C2Review? ReadReview(string directory) => File.Exists(ReviewPath(directory))
        ? JsonSerializer.Deserialize<C2Review>(File.ReadAllText(ReviewPath(directory))) : null;

    public void Review(C2Recording recording, bool accept, string reason)
    {
        if (accept && recording.State is "Incompatible" or "Incomplete" or "Unreadable")
            throw new InvalidOperationException(recording.Reason);
        if (accept && !recording.Legacy)
        {
            var rows = File.ReadLines(Path.Combine(recording.Directory, "labels.jsonl"))
                .Where(s => !string.IsNullOrWhiteSpace(s)).Select(s => JsonSerializer.Deserialize<FrameLabel>(s)!).ToArray();
            var holds = rows.Where(r => r.Cue?.Phase == "hold").ToArray();
            var metadata = JsonSerializer.Deserialize<SessionMetadata>(File.ReadAllText(Path.Combine(recording.Directory,"session.json")))!;
            var frequency = metadata.C2?.StopwatchFrequency ?? 0;
            if (holds.Length < 2 || frequency <= 0 ||
                (holds[^1].InferenceTimestamp - holds[0].InferenceTimestamp) / (double)frequency < 1)
                throw new InvalidOperationException("Not enough fresh steady-hold time was saved. Repeat this attempt; the original remains available.");
            if (holds.Zip(holds.Skip(1)).Any(pair => pair.Second.InferenceTimestamp <= pair.First.InferenceTimestamp ||
                (pair.Second.InferenceTimestamp - pair.First.InferenceTimestamp) / (double)frequency > .5))
                throw new InvalidOperationException("A capture gap interrupted this hold. Repeat it; this is a capture issue, not a judgment of your expression.");
        }
        C2Contract.WriteAtomic(ReviewPath(recording.Directory), new C2Review(recording.Directory,
            recording.OriginId, recording.Role, recording.TaskId, accept, reason, DateTime.UtcNow.ToString("o"),
            SessionSha256: accept ? C2Contract.Hash(Path.Combine(recording.Directory,"session.json")) : null,
            LabelsSha256: accept ? C2Contract.Hash(Path.Combine(recording.Directory,"labels.jsonl")) : null,
            EmbeddingsSha256: accept && File.Exists(Path.Combine(recording.Directory,"embeddings.f32"))
                ? C2Contract.Hash(Path.Combine(recording.Directory,"embeddings.f32")) : null));
    }

    public string NextAction(IReadOnlyList<C2Recording> inventory, string role)
    {
        var accepted = inventory.Where(r => !r.Legacy && r.Role == role && r.State == "Accepted").ToArray();
        var next = C2Task.All.FirstOrDefault(t => !t.Optional && accepted.All(r => r.TaskId != t.Id));
        return next == null ? role == "practice"
            ? "Practice core saved. Train an exploratory candidate; record a separate wearing session before judging it."
            : "Check core saved. Compare on these untouched examples; hardware/avatar validation is still required."
            : $"Record {next.Name}: {next.Purpose}";
    }

    public string WriteManifest(string contractPath)
    {
        var rows = Inventory();
        var accepted = rows.Where(r => ReadReview(r.Directory)?.Accepted == true &&
            r.State is "Accepted" or "Replay / preparation").ToArray();
        var manifest = Path.Combine(Root, "Manifests", Guid.NewGuid().ToString("N") + ".json");
        C2Contract.WriteAtomic(manifest, new
        {
            Version = C2Contract.Version, LabelRecipe = C2Contract.LabelRecipe,
            ContractPath = contractPath, ContractSha256 = C2Contract.Hash(contractPath),
            Sessions = accepted.Select(r => new
            {
                r.Directory, r.OriginId, r.Role, r.TaskId, r.Legacy,
                SessionSha256 = C2Contract.Hash(Path.Combine(r.Directory, "session.json")),
                LabelsSha256 = C2Contract.Hash(Path.Combine(r.Directory, "labels.jsonl")),
                EmbeddingsSha256 = File.Exists(Path.Combine(r.Directory,"embeddings.f32"))
                    ? C2Contract.Hash(Path.Combine(r.Directory,"embeddings.f32")) : null,
                Review = ReadReview(r.Directory)
            }).ToArray()
        });
        return manifest;
    }
}
