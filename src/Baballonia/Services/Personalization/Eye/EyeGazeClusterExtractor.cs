using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Baballonia.Services.Personalization.Eye;


/// <summary>What one eye's expression poses measured, ready to fit a response curve from.</summary>
public sealed record EyeExpressionAnchors(
    IReadOnlyList<EyeResponseAnchor> Widen,
    IReadOnlyList<EyeResponseAnchor> Squint)
{
    public static EyeExpressionAnchors Empty { get; } = new([], []);
}

public sealed record EyeGazeClusters(
    IReadOnlyList<EyeGazeObservation> Left,
    IReadOnlyList<EyeGazeObservation> Right,
    string BaseEyeModelMd5,
    string SessionId,
    string? Error)
{
    /// <summary>Expression poses, which say nothing about gaze but everything about widen/squint.</summary>
    public EyeExpressionAnchors LeftExpressions { get; init; } = EyeExpressionAnchors.Empty;
    public EyeExpressionAnchors RightExpressions { get; init; } = EyeExpressionAnchors.Empty;

    public bool IsUsable => Error is null;

    public static EyeGazeClusters Failed(string error) =>
        new([], [], string.Empty, string.Empty, error);
}

/// <summary>
/// Turns recorded frames into one fixation per gaze pose per eye.
/// </summary>
/// <remarks>
/// <para>The reduction from ~220 frames to one point is where most of the honesty lives. A hold
/// contains the saccade that arrived, the overshoot that followed, at least one blink, and only
/// then the fixation actually being asked for. Three filters in order:</para>
///
/// <list type="number">
/// <item><description><em>Settle trim</em> — the first
/// <see cref="GuidedCaptureRoutine.HoldSettleTrimSeconds"/> of each hold is discarded outright, the
/// same window the trainer discards.</description></item>
/// <item><description><em>Blink and saccade rejection</em> — frames where an eye is substantially
/// closed cannot be looking anywhere, and frames where gaze is moving fast are mid-saccade rather
/// than fixating.</description></item>
/// <item><description><em>Median, not mean</em> — one stray frame cannot move a median, and after
/// the first two filters what remains is a tight cluster whose middle is the answer.</description></item>
/// </list>
/// </remarks>
public static class EyeGazeClusterExtractor
{
    private const int RightY = 0, RightX = 1, RightLid = 2, RightWiden = 3, RightSquint = 4;
    private const int LeftY = 6, LeftX = 7, LeftLid = 8, LeftWiden = 9, LeftSquint = 10;

    /// <summary>Above this raw closedness the eye is blinking, not fixating.</summary>
    public const float MaximumLidClosedness = 0.5f;

    /// <summary>Raw gaze movement per frame above which the eye is mid-saccade.</summary>
    public const float MaximumFrameMovement = 0.02f;

    /// <summary>Fewer usable frames than this and the pose was not really held.</summary>
    public const int MinimumFramesPerPose = 30;

    /// <summary>Reads a recorded session and reduces its gaze poses to fixations.</summary>
    public static EyeGazeClusters FromSession(string sessionId)
    {
        var metadataPath = EyeDatasetPaths.MetadataPath(sessionId);
        var labelsPath = EyeDatasetPaths.LabelsPath(sessionId);

        if (!File.Exists(metadataPath) || !File.Exists(labelsPath))
            return EyeGazeClusters.Failed($"Session {sessionId} is missing its files.");

        EyeSessionMetadata? metadata;
        try
        {
            metadata = JsonSerializer.Deserialize<EyeSessionMetadata>(File.ReadAllText(metadataPath));
        }
        catch (JsonException ex)
        {
            return EyeGazeClusters.Failed($"Session {sessionId} has an unreadable header: {ex.Message}");
        }

        if (metadata is null)
            return EyeGazeClusters.Failed($"Session {sessionId} has an empty header.");

        if (!string.Equals(metadata.eyeSchemaSha256, EyePersonalizationSchema.Sha256,
                StringComparison.OrdinalIgnoreCase))
        {
            return EyeGazeClusters.Failed(
                $"Session {sessionId} was recorded against a different eye output layout.");
        }

        var frames = ReadFrames(labelsPath).ToList();
        var clusters = Extract(frames, metadata);
        return clusters with { BaseEyeModelMd5 = metadata.baseEyeModelMd5, SessionId = sessionId };
    }

    private static IEnumerable<EyeFrameLabel> ReadFrames(string path)
    {
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            EyeFrameLabel? frame = null;
            try
            {
                frame = JsonSerializer.Deserialize<EyeFrameLabel>(line);
            }
            catch (JsonException)
            {
                // A torn last line is expected if a session was interrupted; skip it rather than
                // discarding everything recorded before it.
                continue;
            }

            if (frame is { stock.Length: EyePersonalizationSchema.ExpressionCount })
                yield return frame;
        }
    }

    /// <summary>The pure half, so the filtering can be tested without touching a disk.</summary>
    public static EyeGazeClusters Extract(
        IReadOnlyList<EyeFrameLabel> frames, EyeSessionMetadata metadata)
    {
        var left = new List<EyeGazeObservation>();
        var right = new List<EyeGazeObservation>();

        var gazePoses = metadata.poses
            .Where(pose => pose.dotXDegrees is not null && pose.dotYDegrees is not null)
            .ToDictionary(pose => pose.id, StringComparer.Ordinal);

        var byAttempt = frames
            .Where(frame => frame.cue is { phase: "hold" } && gazePoses.ContainsKey(frame.cue.id))
            .GroupBy(frame => (frame.cue!.id, frame.cue.rep, frame.cue.attempt));

        foreach (var group in byAttempt)
        {
            var pose = gazePoses[group.Key.id];
            var ordered = group.OrderBy(frame => frame.t).ToList();
            var usable = Trim(ordered);
            if (usable.Count < MinimumFramesPerPose)
                continue;

            var targetX = EyeGuidedCues.RawFromDegrees(pose.dotXDegrees!.Value);
            var targetY = EyeGuidedCues.RawFromDegrees(pose.dotYDegrees!.Value);

            left.Add(new EyeGazeObservation(
                Median(usable, LeftX), Median(usable, LeftY), targetX, targetY));
            right.Add(new EyeGazeObservation(
                Median(usable, RightX), Median(usable, RightY), targetX, targetY));
        }

        var (leftExpressions, rightExpressions) = ExtractExpressions(frames, metadata);

        return new EyeGazeClusters(left, right, metadata.baseEyeModelMd5, metadata.sessionId, null)
        {
            LeftExpressions = leftExpressions,
            RightExpressions = rightExpressions,
        };
    }

    /// <summary>
    /// Reduces the expression poses to one measurement each, per eye.
    /// </summary>
    /// <remarks>
    /// The intended value comes from the pose's own supervision, read out of the session file rather
    /// than a table in here — the recording already says what each pose was asking for, and a second
    /// copy of that knowledge is a second thing to drift.
    ///
    /// Blinks and winks are excluded: they are not held, and a curve anchor needs a value the user
    /// was actually holding still at.
    /// </remarks>
    private static (EyeExpressionAnchors Left, EyeExpressionAnchors Right) ExtractExpressions(
        IReadOnlyList<EyeFrameLabel> frames, EyeSessionMetadata metadata)
    {
        var leftWiden = new List<EyeResponseAnchor>();
        var leftSquint = new List<EyeResponseAnchor>();
        var rightWiden = new List<EyeResponseAnchor>();
        var rightSquint = new List<EyeResponseAnchor>();

        var poses = metadata.poses.ToDictionary(pose => pose.id, StringComparer.Ordinal);

        var byAttempt = frames
            .Where(frame => frame.cue is { phase: "hold" } &&
                            poses.ContainsKey(frame.cue.id) &&
                            !frame.cue.id.StartsWith("wink", StringComparison.Ordinal) &&
                            frame.cue.id != "blinks")
            .GroupBy(frame => (frame.cue!.id, frame.cue.rep, frame.cue.attempt));

        foreach (var group in byAttempt)
        {
            var pose = poses[group.Key.id];
            var usable = TrimForExpressions(group.OrderBy(frame => frame.t).ToList());
            if (usable.Count < MinimumFramesPerPose)
                continue;

            Add(pose, LeftWiden, leftWiden, usable);
            Add(pose, LeftSquint, leftSquint, usable);
            Add(pose, RightWiden, rightWiden, usable);
            Add(pose, RightSquint, rightSquint, usable);
        }

        return (new EyeExpressionAnchors(leftWiden, leftSquint),
                new EyeExpressionAnchors(rightWiden, rightSquint));

        static void Add(EyePoseRecord pose, int dim,
            List<EyeResponseAnchor> into, IReadOnlyList<EyeFrameLabel> frames)
        {
            // Only where the pose actually commanded a value for this channel. An anchored or
            // masked channel is one the pose says nothing about.
            var supervision = pose.supervision.FirstOrDefault(c => c.dim == dim);
            if (supervision?.target is not { } intended)
                return;

            into.Add(new EyeResponseAnchor(Median(frames, dim), intended));
        }
    }

    /// <summary>
    /// Same settle trim as the gaze path, without the saccade gate — an expression is not a
    /// fixation, and gaze wandering during a squint says nothing about the squint.
    /// </summary>
    private static List<EyeFrameLabel> TrimForExpressions(List<EyeFrameLabel> hold)
    {
        if (hold.Count == 0)
            return hold;

        var settleTicks = (long)(GuidedCaptureRoutine.HoldSettleTrimSeconds * TimeSpan.TicksPerSecond);
        var start = hold[0].t + settleTicks;

        return hold
            .Where(frame => frame.t >= start && frame.stock.All(float.IsFinite))
            .ToList();
    }

    private static List<EyeFrameLabel> Trim(List<EyeFrameLabel> hold)
    {
        if (hold.Count == 0)
            return hold;

        var settleTicks = (long)(GuidedCaptureRoutine.HoldSettleTrimSeconds * TimeSpan.TicksPerSecond);
        var start = hold[0].t + settleTicks;

        var usable = new List<EyeFrameLabel>(hold.Count);
        for (var i = 0; i < hold.Count; i++)
        {
            var frame = hold[i];
            if (frame.t < start)
                continue;

            // A closed eye is not looking anywhere.
            if (frame.stock[LeftLid] > MaximumLidClosedness || frame.stock[RightLid] > MaximumLidClosedness)
                continue;

            if (!frame.stock.All(float.IsFinite))
                continue;

            // Mid-saccade frames are in transit, not fixating. Compared against the previous frame
            // of the same hold, so a gap left by the filters above cannot look like movement.
            if (usable.Count > 0)
            {
                var previous = usable[^1];
                if (Moved(previous, frame, LeftX, LeftY) || Moved(previous, frame, RightX, RightY))
                    continue;
            }

            usable.Add(frame);
        }

        return usable;
    }

    private static bool Moved(EyeFrameLabel a, EyeFrameLabel b, int xIndex, int yIndex) =>
        Math.Abs(b.stock[xIndex] - a.stock[xIndex]) > MaximumFrameMovement ||
        Math.Abs(b.stock[yIndex] - a.stock[yIndex]) > MaximumFrameMovement;

    private static float Median(IReadOnlyList<EyeFrameLabel> frames, int index)
    {
        var values = frames.Select(frame => frame.stock[index]).Order().ToArray();
        var middle = values.Length / 2;
        return values.Length % 2 == 1
            ? values[middle]
            : (values[middle - 1] + values[middle]) / 2f;
    }
}
