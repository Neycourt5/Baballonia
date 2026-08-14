using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using OpenCvSharp;

namespace Baballonia.Services.Personalization;

/// <summary>
/// What the user is asserting when they flag a moment. Adding a case here is the whole cost of
/// supporting a new correction, because nothing downstream is JawOpen-specific: the dimensions and
/// target travel with the data.
/// </summary>
public sealed record CorrectionKind(string Id, string Label, IReadOnlyList<int> Dims, float Target)
{
    /// <summary>The one that matters today: the avatar's jaw hung open while the mouth was closed.</summary>
    public static readonly CorrectionKind MouthClosed = new(
        "mouth_closed",
        "My mouth was closed",
        [PersonalizationSchema.IndexOf("JawOpen")],
        0f);

    /// <summary>Defined but not offered in the UI yet - the rarer of the two reported problems.</summary>
    public static readonly CorrectionKind TongueNotOut = new(
        "tongue_not_out",
        "My tongue was not out",
        [PersonalizationSchema.IndexOf("TongueOut")],
        0f);

    public static IReadOnlyList<CorrectionKind> All => [MouthClosed, TongueNotOut];
}

/// <summary>
/// Saves a flagged stretch of the rolling buffer as a training session.
///
/// The output is an ordinary session directory, so the trainer discovers it with no special casing,
/// plus a correction.json recording exactly what the user claimed and which model got it wrong. That
/// provenance is what keeps the label honest later: a correction captured against a model that has
/// since been retrained may describe a failure that no longer exists.
///
/// Only the named dimensions are supervised. The user said "my mouth was closed" - they said nothing
/// about whether they were smiling, and in VR they very well might have been. Labelling the rest of
/// the face as neutral would invent supervision the user never gave, which is exactly the kind of
/// confident wrong label that is worse than no label at all.
/// </summary>
public sealed class HardExampleService
{
    /// <summary>How far back to save when the user does not choose. Long enough to cover noticing.</summary>
    public static readonly TimeSpan DefaultWindow = TimeSpan.FromSeconds(5);

    /// <summary>Selectable windows, shortest first.</summary>
    public static readonly IReadOnlyList<TimeSpan> WindowChoices =
    [
        TimeSpan.FromSeconds(2),
        TimeSpan.FromSeconds(5),
        TimeSpan.FromSeconds(10)
    ];

    /// <summary>Two flags in quick succession are one mistake noticed twice, not two examples.</summary>
    private static readonly TimeSpan Debounce = TimeSpan.FromSeconds(3);

    private const int JpegQuality = 95;

    private readonly HardExampleBuffer _buffer;
    private readonly PersonalModelManager _modelManager;
    private readonly ILogger<HardExampleService> _logger;

    private DateTime _lastFlagUtc = DateTime.MinValue;

    public HardExampleService(HardExampleBuffer buffer, PersonalModelManager modelManager,
                              ILogger<HardExampleService> logger)
    {
        _buffer = buffer;
        _modelManager = modelManager;
        _logger = logger;
    }

    /// <summary>Outcome of a flag, phrased so the caller can show it directly.</summary>
    public sealed record FlagResult(bool Success, string Message, string? SessionId = null, int Frames = 0);

    /// <summary>
    /// Saves the last <paramref name="window"/> of tracking as a correction session.
    /// </summary>
    public async Task<FlagResult> FlagAsync(CorrectionKind kind, TimeSpan? window = null)
    {
        var effectiveWindow = window ?? DefaultWindow;
        var now = DateTime.UtcNow;

        if (now - _lastFlagUtc < Debounce)
            return new FlagResult(false, "Just saved one - give it a few seconds.");

        if (!_buffer.Enabled)
            return new FlagResult(false, "Quick correction is turned off in Advanced.");

        var snapshot = _buffer.Take(effectiveWindow);
        if (snapshot.Frames.Count == 0)
        {
            return new FlagResult(false,
                "Nothing captured yet - the face camera needs to be running for a few seconds first.");
        }

        _lastFlagUtc = now;

        try
        {
            var result = await Task.Run(() => Write(kind, effectiveWindow, snapshot, now));
            _logger.LogInformation("Hard example saved: {Session} ({Frames} frames, {Kind})",
                result.SessionId, result.Frames, kind.Id);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Could not save hard example");
            return new FlagResult(false, "Could not save that correction. See the log for details.");
        }
    }

    private FlagResult Write(CorrectionKind kind, TimeSpan window,
                             HardExampleBuffer.Snapshot snapshot, DateTime flaggedUtc)
    {
        var sessionId = PersonalizationPaths.NewSessionId(SessionType.Correction, flaggedUtc);
        var directory = PersonalizationPaths.SessionDirectory(sessionId);
        var framesDirectory = PersonalizationPaths.FramesDirectory(sessionId);

        Directory.CreateDirectory(framesDirectory);

        var encodeParams = new[] { new ImageEncodingParam(ImwriteFlags.JpegQuality, JpegQuality) };
        var written = 0;

        using (var labels = new StreamWriter(PersonalizationPaths.LabelsPath(sessionId),
                                             append: false, PersonalizationPaths.Utf8NoBom))
        {
            for (var i = 0; i < snapshot.Frames.Count; i++)
            {
                var frame = snapshot.Frames[i];

                // Rebuild the Mat from raw bytes only now, on a background thread, so the capture
                // path never paid for an encode it might not need.
                using var mat = Mat.FromPixelData(snapshot.Height, snapshot.Width, MatType.CV_8UC1, frame.Pixels);
                mat.SaveImage(Path.Combine(framesDirectory, $"{i:D6}.jpg"), encodeParams);

                var label = new FrameLabel
                {
                    Index = i,
                    TimestampTicks = frame.TimestampTicks,
                    Stock = frame.Stock,
                    Personal = frame.Personal
                };

                labels.WriteLine(JsonSerializer.Serialize(label, PersonalizationPaths.Json));
                written++;
            }
        }

        var metadata = new SessionMetadata
        {
            SessionId = sessionId,
            SessionType = SessionType.Correction.ToString(),
            StartedUtc = new DateTime(snapshot.Frames[0].TimestampTicks, DateTimeKind.Utc)
                .ToString("O"),
            EndedUtc = new DateTime(snapshot.Frames[^1].TimestampTicks, DateTimeKind.Utc)
                .ToString("O"),
            AppVersion = typeof(HardExampleService).Assembly.GetName().Version?.ToString() ?? "",
            ImageWidth = snapshot.Width,
            ImageHeight = snapshot.Height,
            JpegQuality = JpegQuality,
            FrameCount = written,
            Notes = $"User correction: {kind.Label}"
        };

        File.WriteAllText(PersonalizationPaths.SessionMetadataPath(sessionId),
            JsonSerializer.Serialize(metadata, PersonalizationPaths.IndentedJson),
            PersonalizationPaths.Utf8NoBom);

        var loaded = _modelManager.LoadedMetadata;
        var correction = new CorrectionMetadata
        {
            Kind = kind.Id,
            CorrectedDims = kind.Dims,
            Target = kind.Target,
            WindowSeconds = window.TotalSeconds,
            FlaggedUtc = flaggedUtc.ToString("O"),
            Model = new CorrectionMetadata.ModelProvenance
            {
                AdapterType = loaded?.AdapterType,
                TrainedUtc = loaded?.TrainedUtc,
                Blend = _modelManager.Blend
            }
        };

        File.WriteAllText(PersonalizationPaths.CorrectionPath(sessionId),
            JsonSerializer.Serialize(correction, PersonalizationPaths.IndentedJson),
            PersonalizationPaths.Utf8NoBom);

        var seconds = written > 1
            ? (snapshot.Frames[^1].TimestampTicks - snapshot.Frames[0].TimestampTicks) / (double)TimeSpan.TicksPerSecond
            : 0;

        return new FlagResult(
            true,
            $"Saved {written} frames ({seconds:F1}s). Press Train My Face Model when you have a few.",
            sessionId,
            written);
    }

    /// <summary>How many corrections are already on disk, for the UI's "you have N saved" line.</summary>
    public static int CountSaved()
    {
        var root = PersonalizationPaths.DatasetRoot;
        if (!Directory.Exists(root))
            return 0;

        var count = 0;
        foreach (var directory in Directory.EnumerateDirectories(root))
        {
            if (File.Exists(Path.Combine(directory, "correction.json")))
                count++;
        }

        return count;
    }
}
