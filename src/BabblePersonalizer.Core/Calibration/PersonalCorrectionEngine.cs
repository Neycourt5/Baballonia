namespace BabblePersonalizer.Core.Calibration;

public sealed class PersonalCorrectionEngine
{
    private readonly PersonalCalibrationProfile _profile;
    public PersonalCorrectionEngine(PersonalCalibrationProfile profile) => _profile = profile;

    public float[] Apply(IReadOnlyList<float> raw)
    {
        var output = raw.ToArray(); Apply(raw, output); return output;
    }

    public void Apply(IReadOnlyList<float> raw, Span<float> output)
    {
        if (output.Length < raw.Count) throw new ArgumentException("Output buffer is too small.", nameof(output));
        for (var i = 0; i < raw.Count; i++) output[i] = raw[i];
        foreach (var parameter in _profile.Parameters)
        {
            var index = parameter.OriginalOutputIndex;
            if (index < 0 || index >= raw.Count || !parameter.Enabled || parameter.Confidence < .35f ||
                parameter.ParameterType is Models.FaceParameterType.Binary or Models.FaceParameterType.Passthrough)
                continue;
            var signal = raw[index] - parameter.NeutralMedian;
            if (parameter.ParameterType == Models.FaceParameterType.SignedContinuous)
            {
                var sign = Math.Sign(signal); var magnitude = Math.Abs(signal);
                magnitude = magnitude <= parameter.DeadZone ? 0 : magnitude - parameter.DeadZone;
                var usableRange = Math.Max(.001f, parameter.ActiveRange - parameter.DeadZone);
                var normalized = Math.Clamp(magnitude / usableRange * parameter.LeftRightScale, 0, 1);
                var mapped = RobustStatistics.Interpolate(parameter.ResponseCurve, normalized) * sign;
                output[index] = Math.Clamp(mapped, parameter.ValidMinimum, parameter.ValidMaximum);
            }
            else
            {
                signal = signal <= parameter.DeadZone ? 0 : signal - parameter.DeadZone;
                var usableRange = Math.Max(.001f, parameter.ReliableMaximum - parameter.NeutralMedian - parameter.DeadZone);
                var normalized = Math.Clamp(signal / usableRange * parameter.LeftRightScale, 0, 1);
                var mapped = RobustStatistics.Interpolate(parameter.ResponseCurve, normalized);
                output[index] = Math.Clamp(mapped, parameter.ValidMinimum, parameter.ValidMaximum);
            }
        }
    }
}
