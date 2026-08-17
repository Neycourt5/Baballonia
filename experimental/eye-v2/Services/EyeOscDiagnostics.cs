using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;

namespace Baballonia.Services;

/// <summary>
/// The most recent state of an eye OSC batch at Baballonia's UDP transport boundary.
/// </summary>
public enum EyeOscTransportStatus
{
    Queued,
    SentToUdpSocket,
    TransportError,
}

/// <summary>An exact expression/address/value tuple prepared for the VRCFT Baballonia module.</summary>
public sealed record EyeOscValue(string ExpressionName, string Address, float Value);

/// <summary>A channel in the canonical final eye-vector/OSC order.</summary>
public sealed record EyeOscParameter(string ExpressionName, string Address, bool IsSigned);

/// <summary>
/// A detached diagnostic snapshot of one final six- or ten-channel eye batch.
/// </summary>
/// <remarks>
/// <see cref="EyeOscTransportStatus.SentToUdpSocket"/> means the local UDP send completed. UDP has
/// no receiver acknowledgement, so this status must never be presented as proof that VRCFT or
/// VRChat received or applied the values.
/// </remarks>
public sealed record EyeOscSendSnapshot(
    IReadOnlyList<EyeOscValue> Values,
    string Destination,
    DateTimeOffset QueuedAtUtc,
    DateTimeOffset UpdatedAtUtc,
    EyeOscTransportStatus TransportStatus,
    string? Error);

/// <summary>The result of handing an OSC batch to the local UDP transport.</summary>
public sealed record OscDispatchResult(
    bool SentToUdpSocket,
    string Destination,
    DateTimeOffset CompletedAtUtc,
    int MessageCount,
    string? Error);

/// <summary>Best-effort identity of the Baballonia module installed in VRCFaceTracking.</summary>
public sealed record VrcftBaballoniaModuleIdentity(
    bool Found,
    string Directory,
    string? Version,
    bool? IsLocal,
    string? ModuleName,
    string? ProductVersion,
    string? Sha256,
    string Status);

/// <summary>
/// Reads the on-disk VRCFT module identity. This is deliberately not called a connection check:
/// a running VRCFT process may still have an older DLL loaded until it is restarted.
/// </summary>
public static class VrcftBaballoniaModuleInspector
{
    public const string ModuleId = "360b014b-b57b-450f-8f12-9904618ff370";
    public const string DllName = "VRCFaceTracking.Baballonia.dll";

    public static VrcftBaballoniaModuleIdentity Inspect(string? moduleDirectory = null)
    {
        var directory = moduleDirectory ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "VRCFaceTracking", "CustomLibs", ModuleId);
        var manifestPath = Path.Combine(directory, "module.json");
        var dllPath = Path.Combine(directory, DllName);
        if (!Directory.Exists(directory))
        {
            return new VrcftBaballoniaModuleIdentity(
                false, directory, null, null, null, null, null,
                "Baballonia VRCFT module folder was not found.");
        }

        string? version = null;
        bool? isLocal = null;
        string? moduleName = null;
        var notes = new List<string>();
        try
        {
            using var manifest = JsonDocument.Parse(File.ReadAllText(manifestPath));
            var root = manifest.RootElement;
            if (root.TryGetProperty("Version", out var versionElement))
                version = versionElement.GetString();
            if (root.TryGetProperty("IsLocal", out var localElement) &&
                localElement.ValueKind is JsonValueKind.True or JsonValueKind.False)
                isLocal = localElement.GetBoolean();
            if (root.TryGetProperty("ModuleName", out var nameElement))
                moduleName = nameElement.GetString();
        }
        catch (Exception ex)
        {
            notes.Add($"manifest unreadable: {ex.Message}");
        }

        string? productVersion = null;
        string? sha256 = null;
        if (File.Exists(dllPath))
        {
            try
            {
                using var stream = File.OpenRead(dllPath);
                sha256 = Convert.ToHexString(SHA256.HashData(stream));
            }
            catch (Exception ex)
            {
                notes.Add($"DLL hash unreadable: {ex.Message}");
            }

            try
            {
                productVersion = FileVersionInfo.GetVersionInfo(dllPath).ProductVersion;
            }
            catch (Exception ex)
            {
                notes.Add($"DLL version unreadable: {ex.Message}");
            }
        }
        else
        {
            notes.Add($"{DllName} is missing");
        }

        var status = notes.Count == 0
            ? "Installed module identity read from disk. Restart VRCFT after replacing its zip."
            : string.Join("; ", notes);
        return new VrcftBaballoniaModuleIdentity(
            File.Exists(dllPath), directory, version, isLocal, moduleName,
            productVersion, sha256, status);
    }
}
