using System;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.EyeV2;

public enum EyeTrackingMode
{
    DefaultBaballonia = 0,
    ExperimentalV2 = 1,
    GeometryHybridV2B = 2,
}

public enum EyeSide
{
    Left,
    Right,
}

/// <summary>
/// Eye output keeps the stock six values as an exact prefix. V2 appends four values, which lets
/// every legacy consumer continue to read [LX, LY, LLid, RX, RY, RLid] unchanged.
/// </summary>
public static class EyeStateLayout
{
    public const int LegacyCount = 6;
    public const int V2Count = 10;

    public const int LeftX = 0;
    public const int LeftY = 1;
    public const int LeftLid = 2;
    public const int RightX = 3;
    public const int RightY = 4;
    public const int RightLid = 5;
    public const int LeftWide = 6;
    public const int LeftSquint = 7;
    public const int RightWide = 8;
    public const int RightSquint = 9;
}

public readonly record struct EyeRawState(float X, float Y, float Openness)
{
    /// <summary>Converts the shipped model's documented right-eye-first sigmoid layout.</summary>
    public static EyeRawState FromModel(float[] raw, EyeSide side)
    {
        if (raw.Length < EyeStateLayout.LegacyCount)
            throw new ArgumentException("The eye model must emit six values.", nameof(raw));

        return side == EyeSide.Left
            ? new EyeRawState(raw[4] * 2f - 1f, raw[3] * 2f - 1f, 1f - raw[5])
            : new EyeRawState(raw[1] * 2f - 1f, raw[0] * 2f - 1f, 1f - raw[2]);
    }
}

public sealed record EyeGazeMap
{
    public float XX { get; init; } = 1f;
    public float XY { get; init; }
    public float YX { get; init; }
    public float YY { get; init; } = 1f;
    public float OffsetX { get; init; }
    public float OffsetY { get; init; }

    public (float X, float Y) Map(float x, float y) =>
        (XX * x + XY * y + OffsetX, YX * x + YY * y + OffsetY);

    public EyeGazeMap Recenter(float rawCenterX, float rawCenterY) => this with
    {
        OffsetX = -(XX * rawCenterX + XY * rawCenterY),
        OffsetY = -(YX * rawCenterX + YY * rawCenterY),
    };

    public bool IsValid() =>
        AllFinite(XX, XY, YX, YY, OffsetX, OffsetY) &&
        Math.Abs(XX * YY - XY * YX) > 0.02f &&
        Math.Abs(XX) < 10f && Math.Abs(XY) < 10f &&
        Math.Abs(YX) < 10f && Math.Abs(YY) < 10f;

    private static bool AllFinite(params float[] values) => values.All(float.IsFinite);
}

public sealed record EyeLidAnchors
{
    public float Closed { get; init; }
    public float Neutral { get; init; }
    public float NeutralLow { get; init; }
    public float NeutralHigh { get; init; }
    public float Squint { get; init; }
    public float Wide { get; init; }
    public float TypicalBlinkMilliseconds { get; init; } = 450f;

    public bool IsValid() =>
        AllFinite(Closed, Neutral, NeutralLow, NeutralHigh, Squint, Wide, TypicalBlinkMilliseconds) &&
        Closed >= -0.1f && Wide <= 1.1f &&
        Neutral - Closed >= 0.08f &&
        NeutralLow > Closed && NeutralHigh >= NeutralLow &&
        Squint < Neutral - 0.02f &&
        Wide > NeutralHigh + 0.015f &&
        TypicalBlinkMilliseconds is >= 100f and <= 2500f;

    private static bool AllFinite(params float[] values) => values.All(float.IsFinite);
}

public sealed record EyeV2PerEyeCalibration
{
    public EyeGazeMap Gaze { get; init; } = new();
    public EyeLidAnchors Lid { get; init; } = new();

    public bool IsValid() => Gaze.IsValid() && Lid.IsValid();
}

public sealed record EyeV2CaptureMetadata
{
    public string Protocol { get; init; } = "relax5-blinks3-squint5-wide5-gaze5x2";
    public string AppVersion { get; init; } = "unknown";
    public DateTime CreatedUtc { get; init; } = DateTime.UtcNow;
    public int RelaxSamples { get; init; }
    public int BlinkSamples { get; init; }
    public int SquintSamples { get; init; }
    public int WideSamples { get; init; }
    public int GazeSamples { get; init; }
}

public sealed record EyeV2Calibration
{
    public const int CurrentSchemaVersion = 1;

    public int SchemaVersion { get; init; } = CurrentSchemaVersion;
    public EyeV2PerEyeCalibration Left { get; init; } = new();
    public EyeV2PerEyeCalibration Right { get; init; } = new();
    public EyeV2CaptureMetadata Capture { get; init; } = new();

    public bool IsValid() =>
        SchemaVersion == CurrentSchemaVersion && Left.IsValid() && Right.IsValid();
}

public readonly record struct EyeV2Sample(long TimestampTicks, float[] Raw)
{
    public EyeRawState Eye(EyeSide side) => EyeRawState.FromModel(Raw, side);
}

public readonly record struct EyeV2GazeTarget(string Name, float X, float Y)
{
    public static readonly EyeV2GazeTarget Center = new("Center", 0f, 0f);
    public static readonly EyeV2GazeTarget Left = new("Left", -0.7f, 0f);
    public static readonly EyeV2GazeTarget Right = new("Right", 0.7f, 0f);
    public static readonly EyeV2GazeTarget Up = new("Up", 0f, -0.7f);
    public static readonly EyeV2GazeTarget Down = new("Down", 0f, 0.7f);

    public static IReadOnlyList<EyeV2GazeTarget> FivePoint { get; } =
        [Center, Left, Right, Up, Down];
}

public sealed record EyeV2CalibrationCapture
{
    public IReadOnlyList<EyeV2Sample> Relax { get; init; } = [];
    public IReadOnlyList<EyeV2Sample> Blinks { get; init; } = [];
    public IReadOnlyList<EyeV2Sample> Squint { get; init; } = [];
    public IReadOnlyList<EyeV2Sample> Wide { get; init; } = [];
    public IReadOnlyDictionary<string, IReadOnlyList<EyeV2Sample>> Gaze { get; init; } =
        new Dictionary<string, IReadOnlyList<EyeV2Sample>>();
}

public enum EyeV2ValidityKind
{
    Valid,
    RecenterRecommended,
    FullCalibrationRecommended,
}

public sealed record EyeV2ValidityResult(
    EyeV2ValidityKind Kind,
    string Message,
    float LeftGazeShift,
    float RightGazeShift,
    float LeftLidShift,
    float RightLidShift);

public sealed record EyeV2PerEyeDiagnostics(
    float RawX,
    float RawY,
    float MappedX,
    float MappedY,
    float RawOpenness,
    float NormalizedOpenness,
    float Squint,
    float Wide,
    bool Blink,
    float FixationJitter);

public sealed record EyeV2Diagnostics(
    long TimestampTicks,
    EyeV2PerEyeDiagnostics Left,
    EyeV2PerEyeDiagnostics Right,
    EyeV2Calibration Calibration);

public sealed record EyeV2CalibrationProgress(
    string Instruction,
    double Fraction,
    float? TargetX = null,
    float? TargetY = null);
