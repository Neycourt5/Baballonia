#if WINDOWS
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Baballonia.Services.Personalization.Audio;
using Microsoft.Extensions.Logging;
using NAudio.Wave;

namespace Baballonia.Desktop.Audio;

/// <summary>
/// Windows microphone catalog and source factory backed by NAudio/WinMM.
/// </summary>
/// <remarks>
/// WinMM does not expose the modern Windows endpoint ID. Its WAVEINCAPS2 name/product GUIDs are the
/// most stable identity available through this capture API, so the persisted ID uses those plus a
/// normalized product name. <see cref="AudioInputDeviceResolver"/> retains a name-based recovery
/// path for driver reinstalls that replace the GUID.
/// </remarks>
public sealed class NAudioInputBackend : IAudioInputDeviceCatalog, IAudioFeatureSourceFactory
{
    private readonly ILoggerFactory _loggerFactory;

    public NAudioInputBackend(ILoggerFactory loggerFactory) => _loggerFactory = loggerFactory;

    public IReadOnlyList<AudioInputDevice> GetDevices()
    {
        var candidates = new List<DeviceCandidate>();

        for (var index = 0; index < WaveInEvent.DeviceCount; index++)
        {
            try
            {
                var capabilities = WaveInEvent.GetCapabilities(index);
                if (capabilities.Channels <= 0 || string.IsNullOrWhiteSpace(capabilities.ProductName))
                    continue;

                candidates.Add(new DeviceCandidate(
                    capabilities.ProductName.Trim(),
                    BuildBaseId(capabilities)));
            }
            catch (Exception ex)
            {
                // One broken driver entry must not hide the other microphones from the selector.
                _loggerFactory.CreateLogger<NAudioInputBackend>().LogDebug(
                    ex,
                    "Skipping unusable microphone entry {DeviceNumber}",
                    index);
            }
        }

        var duplicateIds = candidates
            .GroupBy(candidate => candidate.BaseId, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal);
        var duplicateNames = candidates
            .GroupBy(candidate => candidate.Name, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.OrdinalIgnoreCase);

        var seenIds = new Dictionary<string, int>(StringComparer.Ordinal);
        var seenNames = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var result = new List<AudioInputDevice>(candidates.Count);
        foreach (var candidate in candidates)
        {
            seenIds.TryGetValue(candidate.BaseId, out var idOrdinal);
            idOrdinal++;
            seenIds[candidate.BaseId] = idOrdinal;

            seenNames.TryGetValue(candidate.Name, out var nameOrdinal);
            nameOrdinal++;
            seenNames[candidate.Name] = nameOrdinal;

            var id = duplicateIds[candidate.BaseId] > 1
                ? $"{candidate.BaseId}:{idOrdinal}"
                : candidate.BaseId;
            var displayName = duplicateNames[candidate.Name] > 1
                ? $"{candidate.Name} ({nameOrdinal})"
                : candidate.Name;

            // WinMM exposes an ordered list, not an authoritative Windows default-endpoint marker.
            // Device zero is merely the first WaveIn entry, so never label it as the Windows
            // default. The shared resolver may still choose the first entry automatically.
            result.Add(new AudioInputDevice(id, displayName));
        }

        return result;
    }

    public IAudioFeatureSource Create(AudioInputDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);

        var devices = EnumerateWithIndexes();
        var match = devices.FirstOrDefault(entry =>
            string.Equals(entry.Device.Id, device.Id, StringComparison.Ordinal));

        // Enumeration can change between painting the selector and pressing Apply. A unique name
        // is a useful last recovery attempt; ambiguity deliberately fails instead of opening an
        // arbitrary microphone.
        if (match == null)
        {
            var sameName = devices.Where(entry =>
                    string.Equals(entry.Device.DisplayName, device.DisplayName,
                        StringComparison.OrdinalIgnoreCase))
                .ToArray();
            if (sameName.Length == 1)
                match = sameName[0];
        }

        if (match == null)
            throw new InvalidOperationException($"Microphone '{device.DisplayName}' is no longer available.");

        return new MicrophoneFeatureSource(
            match.Index,
            match.Device.DisplayName,
            _loggerFactory.CreateLogger<MicrophoneFeatureSource>());
    }

    private IReadOnlyList<IndexedDevice> EnumerateWithIndexes()
    {
        // GetDevices deliberately exposes no native index. Recreate the same ordered mapping here;
        // WinMM enumeration is stable for the duration of this call and Create handles a race by
        // throwing back to AudioAssistService's visual-safe failure path.
        var publicDevices = GetDevices();
        var result = new List<IndexedDevice>(publicDevices.Count);
        var publicIndex = 0;

        for (var nativeIndex = 0;
             nativeIndex < WaveInEvent.DeviceCount && publicIndex < publicDevices.Count;
             nativeIndex++)
        {
            try
            {
                var capabilities = WaveInEvent.GetCapabilities(nativeIndex);
                if (capabilities.Channels <= 0 || string.IsNullOrWhiteSpace(capabilities.ProductName))
                    continue;

                result.Add(new IndexedDevice(nativeIndex, publicDevices[publicIndex++]));
            }
            catch
            {
                // Mirrored from GetDevices: unusable entries do not consume a public-device slot.
            }
        }

        return result;
    }

    private static string BuildBaseId(WaveInCapabilities capabilities)
    {
        var guid = capabilities.NameGuid != Guid.Empty
            ? capabilities.NameGuid
            : capabilities.ProductGuid;
        var normalizedName = Normalize(capabilities.ProductName);

        return guid != Guid.Empty
            ? $"winmm:{guid:N}:{normalizedName}"
            : $"winmm:name:{normalizedName}";
    }

    private static string Normalize(string value)
    {
        var result = new StringBuilder(value.Length);
        var separatorPending = false;

        foreach (var character in value.Trim().ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(character))
            {
                if (separatorPending && result.Length > 0)
                    result.Append('-');
                result.Append(character);
                separatorPending = false;
            }
            else
            {
                separatorPending = true;
            }
        }

        return result.Length == 0 ? "microphone" : result.ToString();
    }

    private sealed record DeviceCandidate(string Name, string BaseId);
    private sealed record IndexedDevice(int Index, AudioInputDevice Device);
}
#endif
