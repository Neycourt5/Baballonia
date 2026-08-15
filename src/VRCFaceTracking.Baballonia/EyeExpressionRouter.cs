using System;
using System.Collections.Generic;

namespace VRCFaceTracking.Baballonia;

/// <summary>An OSC address and its exact slot in <see cref="BabbleOsc.EyeExpressions"/>.</summary>
public readonly record struct EyeExpressionRoute(string Address, ExpressionMapping Destination);

/// <summary>
/// Dependency-free eye address router shared by the live receiver and focused contract tests.
/// </summary>
public static class EyeExpressionRouter
{
    public static IReadOnlyList<EyeExpressionRoute> Routes { get; } = Array.AsReadOnly(new[]
    {
        new EyeExpressionRoute("/LeftEyeX", ExpressionMapping.EyeLeftX),
        new EyeExpressionRoute("/LeftEyeY", ExpressionMapping.EyeLeftY),
        new EyeExpressionRoute("/LeftEyeLid", ExpressionMapping.EyeLeftLid),
        new EyeExpressionRoute("/RightEyeX", ExpressionMapping.EyeRightX),
        new EyeExpressionRoute("/RightEyeY", ExpressionMapping.EyeRightY),
        new EyeExpressionRoute("/RightEyeLid", ExpressionMapping.EyeRightLid),
        new EyeExpressionRoute("/LeftEyeWiden", ExpressionMapping.EyeLeftWiden),
        new EyeExpressionRoute("/LeftEyeSquint", ExpressionMapping.EyeLeftSquint),
        new EyeExpressionRoute("/RightEyeWiden", ExpressionMapping.EyeRightWiden),
        new EyeExpressionRoute("/RightEyeSquint", ExpressionMapping.EyeRightSquint),
    });

    /// <summary>
    /// Applies one supported eye value. Address matching remains case-insensitive to preserve the
    /// installed 3.2.0 module's compatibility with older lowercase senders.
    /// </summary>
    public static bool TryApply(string? address, float value, float[] destination)
    {
        ArgumentNullException.ThrowIfNull(destination);

        foreach (var route in Routes)
        {
            if (!string.Equals(address, route.Address, StringComparison.OrdinalIgnoreCase))
                continue;

            var index = (int)route.Destination;
            if (index >= destination.Length)
                throw new ArgumentException("Eye expression destination is too small.", nameof(destination));

            destination[index] = value;
            return true;
        }

        return false;
    }
}
