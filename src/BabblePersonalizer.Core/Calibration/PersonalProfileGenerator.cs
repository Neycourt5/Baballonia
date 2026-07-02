using BabblePersonalizer.Core.Camera;
using BabblePersonalizer.Core.Inventory;
using BabblePersonalizer.Core.Models;
using BabblePersonalizer.Core.Storage;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace BabblePersonalizer.Core.Calibration;

public static class PersonalProfileGenerator
{
    private const int MinimumNeutralSamples = 30;
    private const int MinimumActiveSamples = 45;

    public static PersonalCalibrationProfile Generate(
        IReadOnlyList<CalibrationSample> samples, SessionMetadata metadata)
    {
        var neutral = samples.Where(x => !x.IsValidation && x.StepType == "NeutralHold").ToArray();
        var profiles = new List<PersonalParameterProfile>(metadata.Model.Parameters.Count);
        foreach (var parameter in metadata.Model.Parameters)
            profiles.Add(BuildParameter(parameter, samples, neutral));
        ApplyLeftRightScale(profiles);
        var profile = new PersonalCalibrationProfile
        {
            ProfileId = $"Profile-{DateTime.UtcNow:yyyyMMdd-HHmmss}", SessionId = metadata.SessionId,
            CreatedUtc = DateTimeOffset.UtcNow, ApplicationVersion = metadata.ApplicationVersion,
            StockModelHash = metadata.Model.Sha256, StockInput = metadata.Model.Input,
            StockOutput = metadata.Model.Output, ExpressionListHash = metadata.Model.ExpressionListHash,
            CameraConfigurationHash = HashCamera(metadata.Camera), Camera = metadata.Camera,
            Parameters = profiles
        };
        profile.CrossActivations.AddRange(FindCrossActivations(samples, profiles));
        profile.Validation = HeldOutValidator.Validate(profile, samples);
        return profile;
    }

    public static string HashCamera(CameraConfiguration configuration) => Convert.ToHexString(SHA256.HashData(
        Encoding.UTF8.GetBytes(JsonSerializer.Serialize(configuration))));

    private static PersonalParameterProfile BuildParameter(
        FaceParameterDefinition parameter, IReadOnlyList<CalibrationSample> all,
        IReadOnlyList<CalibrationSample> neutralSamples)
    {
        var index = parameter.OutputIndex;
        var neutral = neutralSamples.Where(x => x.RawOutput.Length > index).Select(x => x.RawOutput[index]).ToArray();
        var baseline = RobustStatistics.Median(neutral);
        var noiseMad = RobustStatistics.MedianAbsoluteDeviation(neutral) * 1.4826f;
        var automatic = parameter.CalibrationStrategy is not
            (CalibrationStrategy.ManualReviewOnly or CalibrationStrategy.UnsupportedPassthrough);
        var candidates = all.Where(x => !x.IsValidation && x.CanonicalTarget == parameter.CanonicalName &&
                                        x.RawOutput.Length > index && x.StepType.EndsWith("Ramp", StringComparison.Ordinal) &&
                                        x.RequestedIntensity > 0).ToArray();
        var latestAttempt = candidates.Length == 0 ? 0 : candidates.Max(x => x.Attempt);
        var targeted = candidates.Where(x => x.Attempt == latestAttempt).ToArray();
        var raw = targeted.Select(x => x.RawOutput[index]).ToArray();
        var rejection = RobustStatistics.RejectOutliersAndSpikes(raw);
        var accepted = rejection.Accepted;
        var reliableMin = accepted.Count == 0 ? baseline : RobustStatistics.Percentile(accepted, .05f);
        var reliableMax = accepted.Count == 0 ? baseline : RobustStatistics.Percentile(accepted, .95f);
        var activeRange = parameter.ParameterType == FaceParameterType.SignedContinuous
            ? Math.Max(Math.Max(Math.Abs(reliableMin - baseline), Math.Abs(reliableMax - baseline)), .001f)
            : Math.Max(reliableMax - baseline, .001f);
        var repetitionPeaks = targeted.GroupBy(x => x.Repetition).Where(x => x.Key > 0)
            .Select(x => RobustStatistics.Percentile(x.Select(y => y.RawOutput[index]), .9f)).ToArray();
        var repetitionConsistency = repetitionPeaks.Length < 3 ? 0 :
            Math.Clamp(1 - RobustStatistics.MedianAbsoluteDeviation(repetitionPeaks) / activeRange, 0, 1);
        var trajectory = targeted.Select(x => (x.RequestedIntensity, x.RawOutput[index])).ToArray();
        var monotonicity = RobustStatistics.SpearmanLikeMonotonicity(trajectory);
        var hysteresis = CalculateHysteresis(targeted, index, activeRange);
        var stabilitySamples = targeted.Where(x => x.RampDirection == "Hold" && x.RequestedIntensity >= .99f)
            .Select(x => x.Stability).Where(x => x > 0).ToArray();
        var stability = stabilitySamples.Length == 0 ? RobustStatistics.Stability(accepted) : RobustStatistics.Median(stabilitySamples);
        var signalToNoise = Math.Clamp(activeRange / Math.Max(noiseMad * 8, .04f), 0, 1);
        var sampleSupport = Math.Min(1, accepted.Count / (float)MinimumActiveSamples);
        var neutralSupport = Math.Min(1, neutral.Length / (float)MinimumNeutralSamples);
        var repetitionSupport = Math.Min(1, repetitionPeaks.Length / 3f);
        var confidence = automatic
            ? neutralSupport * sampleSupport * repetitionSupport *
              (.2f + .8f * signalToNoise) * (.4f + .6f * repetitionConsistency) * (.5f + .5f * monotonicity)
            : 0;
        confidence = Math.Clamp(confidence, 0, 1);
        var passthrough = parameter.PassthroughReason;
        if (automatic && neutral.Length < MinimumNeutralSamples) passthrough = "Insufficient stable neutral samples.";
        else if (automatic && accepted.Count < MinimumActiveSamples) passthrough = "Insufficient accepted active samples.";
        else if (automatic && signalToNoise < .2f) passthrough = "Expression signal was not reliably above neutral noise.";
        else if (automatic && confidence < .35f) passthrough = "Calibration confidence is below the safe correction threshold.";

        var rejectionReasons = rejection.Reasons.ToDictionary(x => x.Key, x => x.Value);
        if (candidates.Length > targeted.Length) rejectionReasons["SupersededRetryAttempt"] = candidates.Length - targeted.Length;
        return new PersonalParameterProfile
        {
            CanonicalName = parameter.CanonicalName, OriginalOutputName = parameter.ModelOutputName,
            OriginalOutputIndex = index, Category = parameter.Category, ParameterType = parameter.ParameterType,
            ValidMinimum = parameter.Minimum, ValidMaximum = parameter.Maximum,
            CalibrationStrategy = parameter.CalibrationStrategy, LeftRightCounterpart = parameter.Counterpart,
            NeutralMedian = baseline, NeutralP05 = RobustStatistics.Percentile(neutral, .05f),
            NeutralP95 = RobustStatistics.Percentile(neutral, .95f), NeutralNoiseMad = noiseMad,
            DeadZone = Math.Max(.005f, noiseMad * 3), ReliableMinimum = reliableMin,
            ReliableMaximum = reliableMax, ActiveRange = activeRange, Stability = stability,
            RepetitionConsistency = repetitionConsistency, RampMonotonicity = monotonicity,
            Hysteresis = hysteresis, Confidence = confidence, AcceptedSamples = accepted.Count,
            RejectedSamples = raw.Length - accepted.Count + candidates.Length - targeted.Length,
            RejectionReasons = rejectionReasons,
            ResponseCurve = BuildCurve(targeted, index, baseline, activeRange,
                parameter.ParameterType == FaceParameterType.SignedContinuous),
            Enabled = passthrough == null && confidence >= .35f, PassthroughReason = passthrough
        };
    }

    private static List<ResponseCurvePoint> BuildCurve(
        IReadOnlyList<CalibrationSample> samples, int index, float baseline, float range, bool signed)
    {
        if (samples.Count < MinimumActiveSamples) return new() { new(0, 0), new(1, 1) };
        var points = samples.GroupBy(x => x.RequestedIntensity).Select(group =>
        {
            var median = RobustStatistics.Median(group.Select(x => x.RawOutput[index]));
            var observed = Math.Clamp((signed ? Math.Abs(median - baseline) : median - baseline) / range, 0, 1);
            var softTarget = Math.Clamp(.65f * group.Key + .35f * observed, 0, 1);
            return new ResponseCurvePoint(observed, softTarget);
        }).ToList();
        points.Add(new(0, 0)); points.Add(new(1, 1));
        return RobustStatistics.IsotonicCurve(points);
    }

    private static float CalculateHysteresis(
        IReadOnlyList<CalibrationSample> samples, int index, float range)
    {
        var differences = new List<float>();
        foreach (var level in new[] { .25f, .5f, .75f })
        {
            var up = samples.Where(x => x.RampDirection == "Increasing" && Math.Abs(x.RequestedIntensity - level) < .01f);
            var down = samples.Where(x => x.RampDirection == "Decreasing" && Math.Abs(x.RequestedIntensity - level) < .01f);
            if (!up.Any() || !down.Any()) continue;
            differences.Add(Math.Abs(RobustStatistics.Median(up.Select(x => x.RawOutput[index])) -
                                     RobustStatistics.Median(down.Select(x => x.RawOutput[index]))) / range);
        }
        return differences.Count == 0 ? 0 : Math.Clamp(RobustStatistics.Median(differences), 0, 1);
    }

    private static void ApplyLeftRightScale(IReadOnlyList<PersonalParameterProfile> profiles)
    {
        var byName = profiles.ToDictionary(x => x.CanonicalName);
        foreach (var left in profiles.Where(x => x.CanonicalName.EndsWith("Left", StringComparison.Ordinal)))
        {
            if (left.LeftRightCounterpart == null || !byName.TryGetValue(left.LeftRightCounterpart, out var right)) continue;
            if (left.Confidence < .5f || right.Confidence < .5f) continue;
            var target = (left.ActiveRange + right.ActiveRange) / 2;
            left.LeftRightScale = Math.Clamp(target / left.ActiveRange, .67f, 1.5f);
            right.LeftRightScale = Math.Clamp(target / right.ActiveRange, .67f, 1.5f);
        }
    }

    private static IEnumerable<CrossActivationDiagnostic> FindCrossActivations(
        IReadOnlyList<CalibrationSample> samples, IReadOnlyList<PersonalParameterProfile> profiles)
    {
        foreach (var instructed in profiles.Where(x => x.Confidence >= .5f))
        {
            var high = samples.Where(x => !x.IsValidation && x.CanonicalTarget == instructed.CanonicalName &&
                                          x.RequestedIntensity >= .75f).ToArray();
            if (high.Length < 30) continue;
            foreach (var other in profiles.Where(x => x.CanonicalName != instructed.CanonicalName &&
                                                       x.Category != instructed.Category && x.ActiveRange > .01f))
            {
                var repetitionValues = high.GroupBy(x => x.Repetition).Where(x => x.Key > 0)
                    .Select(rep => RobustStatistics.Median(rep.Select(x =>
                        Math.Max(0, x.RawOutput[other.OriginalOutputIndex] - other.NeutralMedian) / other.ActiveRange))).ToArray();
                if (repetitionValues.Length < 3) continue;
                var activation = RobustStatistics.Median(repetitionValues);
                var consistency = 1 - Math.Clamp(RobustStatistics.MedianAbsoluteDeviation(repetitionValues) /
                                                 Math.Max(activation, .01f), 0, 1);
                if (activation < .15f) continue;
                var candidate = activation >= .25f && consistency >= .75f;
                yield return new CrossActivationDiagnostic(instructed.CanonicalName, other.CanonicalName,
                    activation, consistency, repetitionValues.Length, candidate,
                    candidate ? "Strong and repeated; retain for Phase 4/manual review before correction."
                              : "Recorded diagnostically; too weak or inconsistent for correction.");
            }
        }
    }
}
