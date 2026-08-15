using System;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Personalization.Audio;

/// <summary>
/// A microphone exposed by the platform capture backend.
/// </summary>
/// <param name="Id">
/// Opaque, persistable identifier. Consumers must not interpret it; the platform backend owns its
/// format and may recover a renamed/re-enumerated device by <paramref name="DisplayName"/>.
/// </param>
/// <param name="DisplayName">Human-readable name suitable for a device picker.</param>
/// <param name="IsDefault">
/// True only when the platform backend can authoritatively identify the current default capture
/// endpoint. Backends such as WinMM that expose only an ordered device list must leave this false.
/// </param>
public sealed record AudioInputDevice(string Id, string DisplayName, bool IsDefault = false)
{
    public override string ToString() => DisplayName;
}

/// <summary>How a persisted microphone preference resolved against the devices available now.</summary>
public enum AudioInputResolutionKind
{
    Exact,
    RecoveredByName,
    Automatic,
    Unavailable
}

/// <summary>Result of resolving a persisted microphone preference.</summary>
public readonly record struct AudioInputDeviceResolution(
    AudioInputDevice? Device,
    AudioInputResolutionKind Kind,
    string Message)
{
    public bool IsAvailable => Device != null;

    /// <summary>
    /// True only when a specific saved preference could not be opened exactly. An automatic choice
    /// with no saved preference is normal, not a fallback warning.
    /// </summary>
    public bool UsedFallback { get; init; }
}

/// <summary>Enumerates usable input devices without exposing a platform audio API to the UI.</summary>
public interface IAudioInputDeviceCatalog
{
    IReadOnlyList<AudioInputDevice> GetDevices();
}

/// <summary>Creates a capture source for one device returned by <see cref="IAudioInputDeviceCatalog"/>.</summary>
public interface IAudioFeatureSourceFactory
{
    IAudioFeatureSource Create(AudioInputDevice device);
}

/// <summary>
/// Shared deterministic selection policy. Kept out of the Windows backend so missing-device and
/// driver-ID-change behaviour can be verified without microphones or an operating-system prompt.
/// </summary>
public static class AudioInputDeviceResolver
{
    public static AudioInputDeviceResolution Resolve(
        IEnumerable<AudioInputDevice>? available,
        string? preferredId,
        string? preferredName)
    {
        var devices = available?.Where(device =>
                !string.IsNullOrWhiteSpace(device.Id) &&
                !string.IsNullOrWhiteSpace(device.DisplayName))
            .ToArray() ?? [];

        if (devices.Length == 0)
        {
            return new AudioInputDeviceResolution(
                null,
                AudioInputResolutionKind.Unavailable,
                "No microphone was found.")
            {
                UsedFallback = !string.IsNullOrWhiteSpace(preferredId)
            };
        }

        if (!string.IsNullOrWhiteSpace(preferredId))
        {
            var exact = devices.FirstOrDefault(device =>
                string.Equals(device.Id, preferredId, StringComparison.Ordinal));
            if (exact != null)
            {
                return new AudioInputDeviceResolution(
                    exact,
                    AudioInputResolutionKind.Exact,
                    $"Using {exact.DisplayName}.");
            }

            // WinMM identifiers are as stable as the driver allows, but a driver reinstall can
            // legitimately change them. An exact name match is safer than silently abandoning the
            // user's headset microphone for the first enumerated laptop array.
            if (!string.IsNullOrWhiteSpace(preferredName))
            {
                var byName = devices.FirstOrDefault(device =>
                    string.Equals(device.DisplayName, preferredName, StringComparison.OrdinalIgnoreCase));
                if (byName != null)
                {
                    return new AudioInputDeviceResolution(
                        byName,
                        AudioInputResolutionKind.RecoveredByName,
                        $"The saved microphone ID changed; matched {byName.DisplayName} by name.")
                    {
                        UsedFallback = true
                    };
                }
            }
        }

        var knownDefault = devices.FirstOrDefault(device => device.IsDefault);
        var fallback = knownDefault ?? devices[0];
        var choiceDescription = knownDefault != null
            ? $"the platform default, {fallback.DisplayName}"
            : $"the first available input, {fallback.DisplayName}";
        var hadPreference = !string.IsNullOrWhiteSpace(preferredId);
        return new AudioInputDeviceResolution(
            fallback,
            AudioInputResolutionKind.Automatic,
            hadPreference
                ? $"The selected microphone is unavailable; using {choiceDescription}."
                : $"Automatically using {choiceDescription}.")
        {
            UsedFallback = hadPreference
        };
    }
}

/// <summary>Empty catalog used on platforms without a microphone capture backend.</summary>
public sealed class NullAudioInputDeviceCatalog : IAudioInputDeviceCatalog
{
    public IReadOnlyList<AudioInputDevice> GetDevices() => Array.Empty<AudioInputDevice>();
}

/// <summary>Safe factory counterpart to <see cref="NullAudioInputDeviceCatalog"/>.</summary>
public sealed class NullAudioFeatureSourceFactory : IAudioFeatureSourceFactory
{
    public IAudioFeatureSource Create(AudioInputDevice device) => new NullAudioFeatureSource();
}
