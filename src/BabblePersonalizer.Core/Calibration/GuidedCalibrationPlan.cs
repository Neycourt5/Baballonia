using BabblePersonalizer.Core.Models;

namespace BabblePersonalizer.Core.Calibration;

public sealed record CalibrationSegment(
    string PoseName, string CanonicalTarget, string Instruction, int Repetition,
    float RequestedIntensity, string RampDirection, string StepType,
    TimeSpan Duration, bool IsValidation = false);

public sealed class GuidedCalibrationPlan
{
    public required IReadOnlyList<CalibrationSegment> Segments { get; init; }
    public TimeSpan TotalDuration => TimeSpan.FromTicks(Segments.Sum(x => x.Duration.Ticks));

    public static GuidedCalibrationPlan Create(IReadOnlyList<FaceParameterDefinition> parameters)
    {
        var segments = new List<CalibrationSegment>();
        segments.Add(new("Neutral", "", "Relax your face completely. Breathe normally and avoid speaking.",
            0, 0, "None", "NeutralHold", TimeSpan.FromSeconds(15)));
        foreach (var parameter in parameters.Where(IsGuided)) AddParameter(segments, parameter, false, 1, 3);
        segments.Add(new("NormalSpeech", "", "Speak naturally and vary your words and mouth shapes.",
            0, .6f, "Natural", "Speech", TimeSpan.FromSeconds(30)));
        segments.Add(new("ExaggeratedSpeech", "", "Speak with deliberately exaggerated articulation.",
            0, 1, "Natural", "Speech", TimeSpan.FromSeconds(20)));
        segments.Add(new("ValidationNeutral", "", "Hold a relaxed neutral face for held-out validation.",
            0, 0, "None", "ValidationNeutral", TimeSpan.FromSeconds(8), true));
        foreach (var parameter in parameters.Where(IsGuided)) AddParameter(segments, parameter, true, 4, 4);
        return new GuidedCalibrationPlan { Segments = segments };
    }

    private static bool IsGuided(FaceParameterDefinition parameter) => parameter.CalibrationStrategy is not
        (CalibrationStrategy.ManualReviewOnly or CalibrationStrategy.UnsupportedPassthrough or
         CalibrationStrategy.SpeechDerived or CalibrationStrategy.NeutralBaseline);

    private static void AddParameter(List<CalibrationSegment> segments, FaceParameterDefinition parameter,
        bool validation, int firstRepetition, int lastRepetition)
    {
        var prefix = validation ? "Validation" : "Training";
        segments.Add(new(parameter.CanonicalName, parameter.CanonicalName,
            $"Prepare for {parameter.DisplayName}. {InstructionFor(parameter.CanonicalName)}",
            firstRepetition, 0, "None", prefix + "Transition", TimeSpan.FromSeconds(2), validation));
        var levels = parameter.CalibrationStrategy == CalibrationStrategy.MaximumOnly
            ? new[] { 0f, 1f, 0f }
            : new[] { 0f, .25f, .5f, .75f, 1f, .75f, .5f, .25f, 0f };
        for (var repetition = firstRepetition; repetition <= lastRepetition; repetition++)
        {
            for (var i = 0; i < levels.Length; i++)
            {
                var peak = Array.IndexOf(levels, 1f);
                var direction = i == 0 || i == levels.Length - 1 || i == peak ? "Hold" : i < peak ? "Increasing" : "Decreasing";
                var duration = i == peak ? 1.1 : i == 0 || i == levels.Length - 1 ? .8 : .9;
                segments.Add(new(parameter.CanonicalName, parameter.CanonicalName,
                    $"{InstructionFor(parameter.CanonicalName)} Target {levels[i]:P0}; move slowly and comfortably.",
                    repetition, levels[i], direction, prefix + "Ramp", TimeSpan.FromSeconds(duration), validation));
            }
            segments.Add(new("Rest", parameter.CanonicalName, "Relax to neutral before the next repetition.",
                repetition, 0, "None", prefix + "Rest", TimeSpan.FromSeconds(1), validation));
        }
    }

    private static string InstructionFor(string name) => name switch
    {
        "CheekPuffLeft" => "Puff only your left cheek.", "CheekPuffRight" => "Puff only your right cheek.",
        "CheekSuckLeft" => "Gently suck in only your left cheek.", "CheekSuckRight" => "Gently suck in only your right cheek.",
        "JawOpen" => "Open your jaw without stretching your lips.", "JawForward" => "Move your lower jaw forward.",
        "JawLeft" => "Move your lower jaw left.", "JawRight" => "Move your lower jaw right.",
        "NoseSneerLeft" => "Raise and wrinkle the left side of your nose.", "NoseSneerRight" => "Raise and wrinkle the right side of your nose.",
        "MouthFunnel" => "Funnel your lips forward with a relaxed opening.", "MouthPucker" => "Pucker your lips tightly forward.",
        "MouthLeft" => "Shift both lips left.", "MouthRight" => "Shift both lips right.",
        "MouthRollUpper" => "Roll your upper lip inward.", "MouthRollLower" => "Roll your lower lip inward.",
        "MouthShrugUpper" => "Raise the center of your upper lip.", "MouthShrugLower" => "Raise your lower lip toward the upper lip.",
        "MouthClose" => "Close and press your lips while keeping the jaw relaxed.",
        "MouthSmileLeft" => "Smile using only the left corner.", "MouthSmileRight" => "Smile using only the right corner.",
        "MouthFrownLeft" => "Pull only the left mouth corner downward.", "MouthFrownRight" => "Pull only the right mouth corner downward.",
        "MouthDimpleLeft" => "Pull the left mouth corner sideways into a dimple.", "MouthDimpleRight" => "Pull the right mouth corner sideways into a dimple.",
        "MouthUpperUpLeft" => "Raise the left side of your upper lip.", "MouthUpperUpRight" => "Raise the right side of your upper lip.",
        "MouthLowerDownLeft" => "Lower the left side of your lower lip.", "MouthLowerDownRight" => "Lower the right side of your lower lip.",
        "MouthPressLeft" => "Press the left side of your lips together.", "MouthPressRight" => "Press the right side of your lips together.",
        "MouthStretchLeft" => "Stretch the left mouth corner sideways.", "MouthStretchRight" => "Stretch the right mouth corner sideways.",
        "TongueOut" => "Extend your tongue forward.", "TongueUp" => "Point your visible tongue upward.",
        "TongueDown" => "Point your visible tongue downward.", "TongueLeft" => "Move your visible tongue left.",
        "TongueRight" => "Move your visible tongue right.", "TongueRoll" => "Roll the sides of your tongue upward.",
        "TongueBendDown" => "Extend your tongue and bend the tip downward.", "TongueCurlUp" => "Curl the tip of your tongue upward.",
        _ => "This expression has no safe automatic instruction and should be reviewed manually."
    };
}

public sealed class GuidedCalibrationEngine
{
    private readonly GuidedCalibrationPlan _plan;
    private readonly long[] _elapsedTicksBefore;
    private int _index;
    private TimeSpan _segmentElapsed;
    private long _lastTimestamp;
    public bool IsRunning { get; private set; }
    public bool IsComplete => _index >= _plan.Segments.Count;
    public CalibrationSegment? Current => IsComplete ? null : _plan.Segments[_index];
    public double OverallProgress => IsComplete ? 1 :
        Math.Clamp((_elapsedTicksBefore[_index] + _segmentElapsed.Ticks) / (double)_plan.TotalDuration.Ticks, 0, 1);
    public double SegmentProgress => Current == null ? 1 :
        Math.Clamp(_segmentElapsed.TotalSeconds / Current.Duration.TotalSeconds, 0, 1);

    public GuidedCalibrationEngine(GuidedCalibrationPlan plan)
    {
        _plan = plan; _elapsedTicksBefore = new long[plan.Segments.Count];
        for (var i = 1; i < _elapsedTicksBefore.Length; i++)
            _elapsedTicksBefore[i] = _elapsedTicksBefore[i - 1] + plan.Segments[i - 1].Duration.Ticks;
    }

    public void Start(long timestamp)
    {
        _index = 0; _segmentElapsed = TimeSpan.Zero; _lastTimestamp = timestamp; IsRunning = true;
    }

    public void Update(long timestamp)
    {
        if (!IsRunning || IsComplete) return;
        // A camera stall pauses the routine instead of silently skipping unrecorded steps.
        var delta = Math.Clamp((timestamp - _lastTimestamp) / (double)System.Diagnostics.Stopwatch.Frequency, 0, 1);
        _lastTimestamp = timestamp; _segmentElapsed += TimeSpan.FromSeconds(delta);
        while (!IsComplete && _segmentElapsed >= _plan.Segments[_index].Duration)
        {
            _segmentElapsed -= _plan.Segments[_index].Duration; _index++;
        }
        if (IsComplete) IsRunning = false;
    }

    public void Cancel() => IsRunning = false;

    public void SkipCurrentParameter()
    {
        var target = Current?.CanonicalTarget; if (target == null) return;
        while (_index < _plan.Segments.Count && _plan.Segments[_index].CanonicalTarget == target) _index++;
        _segmentElapsed = TimeSpan.Zero;
    }

    public void RetryCurrentParameter()
    {
        var target = Current?.CanonicalTarget; if (string.IsNullOrEmpty(target)) return;
        while (_index > 0 && _plan.Segments[_index - 1].CanonicalTarget == target &&
               _plan.Segments[_index - 1].IsValidation == Current!.IsValidation) _index--;
        _segmentElapsed = TimeSpan.Zero;
    }

}
