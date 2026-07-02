namespace BabblePersonalizer.Core.Models;

public enum FaceParameterType { UnsignedContinuous, SignedContinuous, Binary, Passthrough }
public enum CalibrationStrategy
{
    NeutralBaseline, SymmetricGuidedRamp, LeftGuidedRamp, RightGuidedRamp,
    BidirectionalSigned, MaximumOnly, SpeechDerived, ManualReviewOnly, UnsupportedPassthrough
}

public sealed record FaceParameterDefinition(
    string CanonicalName,
    string DisplayName,
    string ModelOutputName,
    int OutputIndex,
    string Category,
    FaceParameterType ParameterType,
    float Minimum,
    float Maximum,
    string? Counterpart,
    CalibrationStrategy CalibrationStrategy,
    bool DirectModelOutput = true,
    string? SenderDestination = null,
    string[]? Aliases = null,
    string? PassthroughReason = null);
