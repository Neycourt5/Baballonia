using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Baballonia.Services;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.Eye;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The guided pose table: what each pose teaches, and — more importantly — what it stays silent about.
/// </summary>
/// <remarks>
/// Most of these test masks rather than targets. A label that asserts something the pose does not
/// actually demonstrate is not a smaller amount of training signal, it is a confidently wrong one:
/// "look down" would teach that looking down means eyes wide open, because the lids really do
/// lower and nobody told the trainer to ignore that.
/// </remarks>
[TestClass]
[TestSubject(typeof(EyeGuidedCues))]
public class EyeGuidedCuesTest
{
    private const int RightY = 0, RightX = 1, RightLid = 2, RightWiden = 3, RightSquint = 4, RightBrow = 5;
    private const int LeftY = 6, LeftX = 7, LeftLid = 8, LeftWiden = 9, LeftSquint = 10, LeftBrow = 11;

    private static EyePose Pose(string id) =>
        EyeGuidedCues.Poses.Single(pose => pose.Id == id);

    private static int[] SupervisedDims(string id) =>
        Pose(id).Supervision.Select(channel => channel.Dim).ToArray();

    [TestMethod]
    public void EveryPoseTheUserWasPromisedIsPresent()
    {
        var ids = EyeGuidedCues.Poses.Select(pose => pose.Id).ToArray();

        CollectionAssert.AreEquivalent(
            new[]
            {
                "relaxed_center",
                "gaze_left", "gaze_right", "gaze_up", "gaze_down",
                "gaze_up_left", "gaze_up_right", "gaze_down_left", "gaze_down_right",
                "eyes_normal", "wide_moderate", "wide_strong",
                "squint_light", "squint_strong", "eyes_closed",
                "wink_left", "wink_right", "blinks",
            },
            ids);
    }

    [TestMethod]
    public void GazeDegreesConvertToRawAndBack()
    {
        Assert.AreEqual(0.5f, EyeGuidedCues.RawFromDegrees(0f), 1e-6);
        Assert.AreEqual(1f, EyeGuidedCues.RawFromDegrees(45f), 1e-6);
        Assert.AreEqual(0f, EyeGuidedCues.RawFromDegrees(-45f), 1e-6);

        foreach (var degrees in new[] { -18f, -13f, 0f, 6f, 13f })
            Assert.AreEqual(degrees, EyeGuidedCues.DegreesFromRaw(EyeGuidedCues.RawFromDegrees(degrees)), 1e-3);
    }

    [TestMethod]
    public void GazeBeyondTheModelsRangeIsClampedRatherThanWrapped()
    {
        Assert.AreEqual(1f, EyeGuidedCues.RawFromDegrees(90f), 1e-6);
        Assert.AreEqual(0f, EyeGuidedCues.RawFromDegrees(-90f), 1e-6);
    }

    [TestMethod]
    public void EveryDotStaysInsideThePanelsReach()
    {
        // A target the user cannot look at directly is worse than none: they would fixate the edge
        // of the panel and we would record that as ground truth for a position never looked at.
        foreach (var pose in EyeGuidedCues.Poses)
        {
            if (pose.DotXDegrees is not { } x || pose.DotYDegrees is not { } y)
                continue;

            Assert.IsTrue(Math.Abs(x) <= 18f, $"{pose.Id} dot X {x} is beyond the panel");
            Assert.IsTrue(y is <= 6f and >= -13f, $"{pose.Id} dot Y {y} is beyond the panel");
            Assert.IsTrue(Math.Abs(EyeGuidedCues.PanelFromDegrees(x)) <= 1f);
            Assert.IsTrue(Math.Abs(EyeGuidedCues.PanelFromDegrees(y)) <= 1f);
        }
    }

    [TestMethod]
    public void GazePosesTeachBothEyesTheSameDirection()
    {
        var left = Pose("gaze_left");
        var targets = left.Supervision.ToDictionary(c => c.Dim, c => c.Target);

        Assert.AreEqual(targets[RightX], targets[LeftX]);
        Assert.AreEqual(targets[RightY], targets[LeftY]);
        Assert.AreEqual(EyeGuidedCues.RawFromDegrees(-18f), targets[LeftX]!.Value, 1e-6);
        Assert.AreEqual(0.5f, targets[LeftY]!.Value, 1e-6, "looking left is not looking up or down");
    }

    [TestMethod]
    public void NoGazePoseEverTeachesAnEyelidValue()
    {
        // Looking down genuinely lowers the lids. Supervising lid here would teach the model that
        // looking down means eyes open.
        foreach (var pose in EyeGuidedCues.Poses.Where(p => p.Id.StartsWith("gaze_")))
        {
            var dims = pose.Supervision.Select(c => c.Dim).ToArray();
            CollectionAssert.DoesNotContain(dims, RightLid, $"{pose.Id} supervises the right lid");
            CollectionAssert.DoesNotContain(dims, LeftLid, $"{pose.Id} supervises the left lid");
        }
    }

    [TestMethod]
    public void AHardSquintDoesNotTeachAnEyelidValue()
    {
        // Squinting really does narrow the eye; the lid reading is legitimately low.
        var dims = SupervisedDims("squint_strong");
        CollectionAssert.DoesNotContain(dims, RightLid);
        CollectionAssert.DoesNotContain(dims, LeftLid);
    }

    [TestMethod]
    public void ClosingDoesNotTeachASquintValue()
    {
        // Squeezing the eyes shut looks exactly like squinting.
        var dims = SupervisedDims("eyes_closed");
        CollectionAssert.DoesNotContain(dims, RightSquint);
        CollectionAssert.DoesNotContain(dims, LeftSquint);
        CollectionAssert.Contains(dims, RightLid);
        CollectionAssert.Contains(dims, LeftLid);
    }

    [TestMethod]
    public void ClosedTeachesFullClosedness()
    {
        // Raw lid is closedness, not openness: a shut eye is 1.
        var targets = Pose("eyes_closed").Supervision.ToDictionary(c => c.Dim, c => c.Target);
        Assert.AreEqual(1f, targets[LeftLid]!.Value, 1e-6);
        Assert.AreEqual(1f, targets[RightLid]!.Value, 1e-6);
    }

    [TestMethod]
    public void BlinkingTeachesNoLidValueAtAll()
    {
        // A commanded value describes a moving lid badly. This pose exists only to say that squint
        // must not fire during a blink.
        var supervision = Pose("blinks").Supervision;
        var dims = supervision.Select(c => c.Dim).ToArray();

        CollectionAssert.DoesNotContain(dims, LeftLid);
        CollectionAssert.DoesNotContain(dims, RightLid);
        CollectionAssert.AreEquivalent(new[] { RightSquint, LeftSquint }, dims);
        Assert.IsTrue(supervision.All(c => c.Weight <= 0.25f),
            "the blink pose is a nudge, not a lesson");
    }

    [TestMethod]
    public void WinksTeachOneEyeShutAndAnchorTheOther()
    {
        var left = Pose("wink_left").Supervision.ToDictionary(c => c.Dim, c => c);

        Assert.AreEqual(1f, left[LeftLid].Target!.Value, 1e-6, "the winking eye closes");
        Assert.IsTrue(left[RightLid].IsAnchor,
            "the open eye is anchored, not commanded: we know it should not change, not what it is");
        CollectionAssert.DoesNotContain(left.Keys.ToArray(), LeftX,
            "aiming the winking eye is not something we can ask for");
        CollectionAssert.Contains(left.Keys.ToArray(), RightX);
    }

    [TestMethod]
    public void TheTwoWinksAreMirrorImages()
    {
        var left = Pose("wink_left").Supervision.ToDictionary(c => c.Dim, c => c);
        var right = Pose("wink_right").Supervision.ToDictionary(c => c.Dim, c => c);

        Assert.AreEqual(left[LeftLid].Target, right[RightLid].Target);
        Assert.AreEqual(left[RightLid].IsAnchor, right[LeftLid].IsAnchor);
        Assert.AreEqual(left.Count, right.Count);
    }

    [TestMethod]
    public void WinksAreRepeatedBecauseTheyAreHardToDo()
    {
        Assert.AreEqual(2, Pose("wink_left").Repetitions);
        Assert.AreEqual(2, Pose("wink_right").Repetitions);
    }

    [TestMethod]
    public void WideAndSquintTeachOppositeChannelsAndSilenceTheOther()
    {
        var wide = Pose("wide_strong").Supervision.ToDictionary(c => c.Dim, c => c.Target);
        Assert.AreEqual(1f, wide[LeftWiden]!.Value, 1e-6);
        Assert.AreEqual(0f, wide[LeftSquint]!.Value, 1e-6, "you are not squinting while wide-eyed");

        var squint = Pose("squint_strong").Supervision.ToDictionary(c => c.Dim, c => c.Target);
        Assert.AreEqual(0.9f, squint[LeftSquint]!.Value, 1e-6);
        Assert.AreEqual(0f, squint[LeftWiden]!.Value, 1e-6);
    }

    [TestMethod]
    public void AnchorsCarryNoTargetButStillCarryWeight()
    {
        var anchor = Pose("eyes_normal").Supervision.First(c => c.Dim == LeftLid);

        Assert.IsTrue(anchor.IsAnchor);
        Assert.IsNull(anchor.Target);
        Assert.IsTrue(anchor.Weight > 0);
    }

    [TestMethod]
    public void EverySupervisedDimIsARealChannel()
    {
        foreach (var pose in EyeGuidedCues.Poses)
        {
            foreach (var channel in pose.Supervision)
            {
                Assert.IsTrue(channel.Dim is >= 0 and < EyePersonalizationSchema.ExpressionCount,
                    $"{pose.Id} supervises dim {channel.Dim}");
                Assert.IsTrue(channel.Weight > 0, $"{pose.Id} dim {channel.Dim} has no weight");
                if (channel.Target is { } target)
                    Assert.IsTrue(target is >= 0f and <= 1f, $"{pose.Id} target {target} is not raw");
            }
        }
    }

    [TestMethod]
    public void NoPoseSupervisesTheSameChannelTwice()
    {
        foreach (var pose in EyeGuidedCues.Poses)
        {
            var dims = pose.Supervision.Select(c => c.Dim).ToArray();
            Assert.AreEqual(dims.Length, dims.Distinct().Count(),
                $"{pose.Id} supervises a channel more than once, so which target wins is undefined");
        }
    }

    [TestMethod]
    public void TheTargetVectorCarriesTheDotPositionForThePresenter()
    {
        var vector = EyeGuidedCues.TargetVector(Pose("gaze_right"));

        Assert.AreEqual(EyeGuidedCues.RawFromDegrees(18f), vector[RightX], 1e-6);
        Assert.AreEqual(EyeGuidedCues.RawFromDegrees(18f), vector[LeftX], 1e-6);
        Assert.AreEqual(EyePersonalizationSchema.ExpressionCount, vector.Length);
    }

    [TestMethod]
    public void AnchoredChannelsAreLeftAtNeutralInTheCommandedVector()
    {
        // An anchor has no value to command, so the vector must not invent one.
        var neutral = EyeGuidedCues.NeutralVector();
        var vector = EyeGuidedCues.TargetVector(Pose("eyes_normal"));

        Assert.AreEqual(neutral[LeftLid], vector[LeftLid], 1e-6);
        Assert.AreEqual(neutral[LeftBrow], vector[LeftBrow], 1e-6);
    }

    [TestMethod]
    public void TheNeutralVectorIsCentredGazeAndNoExpression()
    {
        var neutral = EyeGuidedCues.NeutralVector();

        Assert.AreEqual(0.5f, neutral[LeftX], 1e-6);
        Assert.AreEqual(0.5f, neutral[RightY], 1e-6);
        Assert.AreEqual(0f, neutral[LeftLid], 1e-6, "raw lid is closedness, so open is 0");
        Assert.AreEqual(0f, neutral[LeftWiden], 1e-6);
    }

    [TestMethod]
    public void EachRepetitionIsPrepTransitionHoldTransitionRest()
    {
        var steps = EyeGuidedCues.BuildSteps([Pose("gaze_left")]);

        CollectionAssert.AreEqual(
            new[] { "prep", "transition", "hold", "transition", "rest" },
            steps.Select(step => step.Phase).ToArray());
        Assert.AreEqual(2.5, steps[2].DurationSeconds, 1e-6, "the hold is the pose's own length");
    }

    [TestMethod]
    public void ARepeatedPoseProducesOneBlockPerRepetition()
    {
        var steps = EyeGuidedCues.BuildSteps([Pose("wink_left")]);

        Assert.AreEqual(10, steps.Count);
        CollectionAssert.AreEqual(new[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 },
            steps.Select(step => step.Repetition).ToArray());
    }

    [TestMethod]
    public void OnlyTheHoldCommandsThePose()
    {
        var steps = EyeGuidedCues.BuildSteps([Pose("eyes_closed")]);
        var neutral = EyeGuidedCues.NeutralVector();

        CollectionAssert.AreEqual(neutral, steps[0].From, "prep starts and ends neutral");
        CollectionAssert.AreEqual(neutral, steps[0].To);
        CollectionAssert.AreEqual(steps[2].From, steps[2].To, "a hold does not move");
        CollectionAssert.AreEqual(neutral, steps[4].To, "rest returns to neutral");
    }

    [TestMethod]
    public void TheWholeRoutineIsAboutTwoMinutes()
    {
        var seconds = EyeGuidedCues.TotalSeconds();

        // Long enough to cover every pose, short enough that people will actually finish it.
        Assert.IsTrue(seconds is > 60 and < 150,
            $"the routine takes {seconds:F0}s, which is outside the range anyone will sit through");
    }

    [TestMethod]
    public void TheRoutineRunsToCompletionOnAnInjectedClock()
    {
        var now = 0L;
        var routine = new GuidedCaptureRoutine(
            EyeGuidedCues.BuildSteps(), () => now, ticksPerSecond: 1000);

        var phases = new List<string>();
        for (var elapsed = 0; elapsed < 200_000 && !routine.IsFinished; elapsed += 50)
        {
            now = elapsed;
            if (routine.Tick() is { } phase)
                phases.Add(phase.PhaseName);
        }

        Assert.IsTrue(routine.IsFinished, "the routine never completed");
        Assert.AreEqual(EyeGuidedCues.BuildSteps().Count, phases.Count,
            "every step should have been commanded exactly once");
        Assert.AreEqual(1.0, routine.Progress, 1e-6);
    }
}

/// <summary>The on-disk dataset: what gets written, and what the trainer needs from it.</summary>
[TestClass]
[TestSubject(typeof(EyeDatasetRecorder))]
public class EyeDatasetRecorderTest
{
    private EyePipelineEventBus _bus = null!;
    private EyeDatasetRecorder _recorder = null!;
    private string _sessionId = null!;

    [TestInitialize]
    public void Initialize()
    {
        _bus = new EyePipelineEventBus();
        _recorder = new EyeDatasetRecorder(_bus);
    }

    [TestCleanup]
    public void Cleanup()
    {
        _recorder.Dispose();
        if (_sessionId is not null)
        {
            var directory = EyeDatasetPaths.SessionDirectory(_sessionId);
            if (Directory.Exists(directory))
                Directory.Delete(directory, recursive: true);
        }
    }

    private EyeSessionMetadata NewMetadata()
    {
        _sessionId = EyeDatasetPaths.NewSessionId(DateTime.UtcNow);
        return new EyeSessionMetadata(
            schemaVersion: EyePersonalizationSchema.Version,
            sessionId: _sessionId,
            startedUtc: DateTime.UtcNow.ToString("o"),
            endedUtc: null,
            appVersion: "test",
            eyeSchemaNames: EyePersonalizationSchema.ExpressionNames.ToArray(),
            eyeSchemaSha256: EyePersonalizationSchema.Sha256,
            gazeRangeDegrees: EyePersonalizationSchema.GazeRangeDegrees,
            baseEyeModelPath: @"C:\models\tuned.onnx",
            baseEyeModelMd5: "0123456789abcdef0123456789abcdef",
            presentation: "headset",
            poses: EyeGuidedCues.Poses.Select(pose => new EyePoseRecord(
                pose.Id, pose.DisplayName, pose.DotXDegrees, pose.DotYDegrees,
                pose.HoldSeconds, pose.Repetitions,
                pose.Supervision.Select(c => new EyeChannelSupervisionRecord(c.Dim, c.Target, c.Weight))
                    .ToArray())).ToArray(),
            frameCount: 0,
            effectiveFps: 0);
    }

    private void PublishFrame(float seed) =>
        _bus.Publish(new Baballonia.Services.events.EyePipelineEvents.NewRawEyeExpressionsEvent(
            Enumerable.Range(0, EyePersonalizationSchema.ExpressionCount)
                .Select(i => seed + i * 0.01f).ToArray(),
            DateTime.UtcNow.Ticks));

    private static EyeSessionQuality NoQuality() => new([], false, true);

    [TestMethod]
    public async Task AFrameBecomesAJsonlLine()
    {
        _recorder.Start(NewMetadata());
        _recorder.CurrentCue = new EyeCueLabel("gaze_left", "hold", 0, 0);
        PublishFrame(0.5f);
        var finalized = await _recorder.StopAsync(NoQuality());

        Assert.IsNotNull(finalized);
        var lines = File.ReadAllLines(EyeDatasetPaths.LabelsPath(_sessionId));
        Assert.AreEqual(1, lines.Length);

        var frame = JsonSerializer.Deserialize<EyeFrameLabel>(lines[0])!;
        Assert.AreEqual(EyePersonalizationSchema.ExpressionCount, frame.stock.Length);
        Assert.AreEqual(0.5f, frame.stock[0], 1e-6);
        Assert.AreEqual("gaze_left", frame.cue!.id);
        Assert.AreEqual("hold", frame.cue.phase);
    }

    [TestMethod]
    public async Task NothingIsRecordedOutsideASession()
    {
        PublishFrame(0.5f);
        _recorder.Start(NewMetadata());
        PublishFrame(0.6f);
        await _recorder.StopAsync(NoQuality());
        PublishFrame(0.7f);

        var lines = File.ReadAllLines(EyeDatasetPaths.LabelsPath(_sessionId));
        Assert.AreEqual(1, lines.Length, "only frames during the session belong to it");
    }

    [TestMethod]
    public async Task FramesAreCopiedNotAliased()
    {
        // The event payload is the pipeline's reusable scratch buffer; keeping a reference would
        // record the same twelve numbers over and over.
        _recorder.Start(NewMetadata());
        var reused = new float[EyePersonalizationSchema.ExpressionCount];

        for (var i = 0; i < 5; i++)
        {
            Array.Fill(reused, i * 0.1f);
            _bus.Publish(new Baballonia.Services.events.EyePipelineEvents.NewRawEyeExpressionsEvent(
                reused, DateTime.UtcNow.Ticks));
        }

        await _recorder.StopAsync(NoQuality());

        var values = File.ReadAllLines(EyeDatasetPaths.LabelsPath(_sessionId))
            .Select(line => JsonSerializer.Deserialize<EyeFrameLabel>(line)!.stock[0])
            .ToArray();

        CollectionAssert.AreEqual(new[] { 0f, 0.1f, 0.2f, 0.3f, 0.4f }, values, "frames were aliased");
    }

    [TestMethod]
    public async Task TheHeaderExistsBeforeTheSessionEnds()
    {
        // A session interrupted by a crash must still leave an interpretable directory.
        _recorder.Start(NewMetadata());

        Assert.IsTrue(File.Exists(EyeDatasetPaths.MetadataPath(_sessionId)));

        await _recorder.StopAsync(NoQuality());
    }

    [TestMethod]
    public async Task TheHeaderCarriesTheBaseModelIdentityAndSchema()
    {
        _recorder.Start(NewMetadata());
        PublishFrame(0.5f);
        await _recorder.StopAsync(NoQuality());

        var metadata = JsonSerializer.Deserialize<EyeSessionMetadata>(
            File.ReadAllText(EyeDatasetPaths.MetadataPath(_sessionId)))!;

        // These recordings are the output of one specific model; against another they describe a
        // different function entirely.
        Assert.AreEqual("0123456789abcdef0123456789abcdef", metadata.baseEyeModelMd5);
        Assert.AreEqual(EyePersonalizationSchema.Sha256, metadata.eyeSchemaSha256);
        Assert.AreEqual(1, metadata.frameCount);
        Assert.IsNotNull(metadata.endedUtc);
    }

    [TestMethod]
    public async Task TheHeaderCarriesTheSupervisionTableSoTheTrainerCannotDrift()
    {
        _recorder.Start(NewMetadata());
        await _recorder.StopAsync(NoQuality());

        var metadata = JsonSerializer.Deserialize<EyeSessionMetadata>(
            File.ReadAllText(EyeDatasetPaths.MetadataPath(_sessionId)))!;

        Assert.AreEqual(EyeGuidedCues.Poses.Count, metadata.poses.Count);
        var blinks = metadata.poses.Single(pose => pose.id == "blinks");
        Assert.IsTrue(blinks.supervision.All(c => c.weight <= 0.25f));
        var normal = metadata.poses.Single(pose => pose.id == "eyes_normal");
        Assert.IsTrue(normal.supervision.Any(c => c.target is null), "anchors must survive the round trip");
    }

    [TestMethod]
    public async Task TheQualitySidecarIsWritten()
    {
        _recorder.Start(NewMetadata());
        await _recorder.StopAsync(new EyeSessionQuality(
            [new EyePoseQuality("gaze_left", 0, 1, 12, false, "retried")], true, false));

        var quality = JsonSerializer.Deserialize<EyeSessionQuality>(
            File.ReadAllText(EyeDatasetPaths.QualityPath(_sessionId)))!;

        Assert.AreEqual(1, quality.attempts.Count);
        Assert.IsFalse(quality.attempts[0].valid);
        Assert.AreEqual("retried", quality.attempts[0].reason);
        Assert.IsTrue(quality.cameraDisturbed);
        Assert.IsFalse(quality.completed);
    }

    [TestMethod]
    public async Task TheJsonlHasNoByteOrderMark()
    {
        // Encoding.UTF8's preamble breaks Python's json.loads on the first line.
        _recorder.Start(NewMetadata());
        PublishFrame(0.5f);
        await _recorder.StopAsync(NoQuality());

        var bytes = File.ReadAllBytes(EyeDatasetPaths.LabelsPath(_sessionId));
        Assert.IsFalse(bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF);
    }

    [TestMethod]
    public void StartingTwiceIsRefused()
    {
        _recorder.Start(NewMetadata());
        Assert.ThrowsExactly<InvalidOperationException>(() => _recorder.Start(NewMetadata()));
    }

    [TestMethod]
    public async Task StoppingWithoutStartingIsHarmless()
    {
        Assert.IsNull(await _recorder.StopAsync(NoQuality()));
        _sessionId = null!;
    }

    [TestMethod]
    public async Task ABacklogDropsOldFramesRatherThanStallingTracking()
    {
        // Losing a frame is survivable; blocking the inference thread is not.
        _recorder.Start(NewMetadata());
        for (var i = 0; i < EyeDatasetRecorder.QueueCapacity * 4; i++)
            PublishFrame(i * 0.001f);

        await _recorder.StopAsync(NoQuality());

        var written = File.ReadAllLines(EyeDatasetPaths.LabelsPath(_sessionId)).Length;
        Assert.IsTrue(written > 0, "the writer should have kept up with most of it");
    }
}
