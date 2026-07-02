using BabblePersonalizer.Core.Camera;
using BabblePersonalizer.Core.Models;

namespace BabblePersonalizer.Core.Calibration;

public sealed class PersonalCalibrationProfile
{
    public const int CurrentSchemaVersion = 2;
    public int SchemaVersion { get; init; } = CurrentSchemaVersion;
    public string CorrectionMethodVersion { get; init; } = "piecewise-linear-v1";
    public required string ProfileId { get; init; }
    public required string SessionId { get; init; }
    public required DateTimeOffset CreatedUtc { get; init; }
    public required string ApplicationVersion { get; init; }
    public required string StockModelHash { get; init; }
    public required TensorContract StockInput { get; init; }
    public required TensorContract StockOutput { get; init; }
    public required string ExpressionListHash { get; init; }
    public required string CameraConfigurationHash { get; init; }
    public required CameraConfiguration Camera { get; init; }
    public required List<PersonalParameterProfile> Parameters { get; init; }
    public List<CrossActivationDiagnostic> CrossActivations { get; init; } = new();
    public HeldOutValidationReport? Validation { get; set; }
}

public sealed class PersonalParameterProfile
{
    public required string CanonicalName { get; init; }
    public required string OriginalOutputName { get; init; }
    public required int OriginalOutputIndex { get; init; }
    public required string Category { get; init; }
    public required FaceParameterType ParameterType { get; init; }
    public required float ValidMinimum { get; init; }
    public required float ValidMaximum { get; init; }
    public required CalibrationStrategy CalibrationStrategy { get; init; }
    public string? LeftRightCounterpart { get; init; }
    public float NeutralMedian { get; init; }
    public float NeutralP05 { get; init; }
    public float NeutralP95 { get; init; }
    public float NeutralNoiseMad { get; init; }
    public float DeadZone { get; init; }
    public float ReliableMinimum { get; init; }
    public float ReliableMaximum { get; init; }
    public float ActiveRange { get; init; }
    public float LeftRightScale { get; set; } = 1;
    public float Stability { get; init; }
    public float RepetitionConsistency { get; init; }
    public float RampMonotonicity { get; init; }
    public float Hysteresis { get; init; }
    public float Confidence { get; init; }
    public int AcceptedSamples { get; init; }
    public int RejectedSamples { get; init; }
    public Dictionary<string, int> RejectionReasons { get; init; } = new();
    public List<ResponseCurvePoint> ResponseCurve { get; init; } = new();
    public bool Enabled { get; set; } = true;
    public string? PassthroughReason { get; init; }
    public string? Notes { get; init; }
}

public sealed record CrossActivationDiagnostic(
    string InstructedExpression, string ActivatedExpression, float MedianActivation,
    float RepetitionConsistency, int SupportingRepetitions, bool CorrectionCandidate,
    string DecisionReason);

public sealed class HeldOutValidationReport
{
    public required DateTimeOffset CreatedUtc { get; init; }
    public required List<ParameterValidationMetric> Parameters { get; init; }
    public float StockScore { get; init; }
    public float PersonalizedScore { get; init; }
    public bool ImprovementSupported { get; init; }
    public string Summary { get; init; } = "";
}

public sealed record ParameterValidationMetric(
    string Name, int SampleCount, float StockTargetError, float PersonalizedTargetError,
    float StockMonotonicity, float PersonalizedMonotonicity, float StockSpikeRate,
    float PersonalizedSpikeRate, bool Improved);
