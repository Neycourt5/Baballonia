using BabblePersonalizer.Core.Models;
using BabblePersonalizer.Core.Storage;
using BabblePersonalizer.Core.Camera;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BabblePersonalizer.Core.Calibration;

public sealed class PersonalProfileStore(PersonalizerDataPaths paths)
{
    public async Task<string> SaveAsync(PersonalCalibrationProfile profile, CancellationToken cancellationToken = default)
    {
        paths.EnsureCreated();
        var path = Path.Combine(paths.Profiles, profile.ProfileId + ".json");
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(profile, Options()), cancellationToken);
        return path;
    }

    public async Task<ProfileLoadResult> LoadAsync(
        string path, ModelContract contract, CameraConfiguration? camera = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            if (!File.Exists(path)) return new(null, "Profile file does not exist.", false);
            var profile = JsonSerializer.Deserialize<PersonalCalibrationProfile>(
                await File.ReadAllTextAsync(path, cancellationToken), Options());
            if (profile == null) return new(null, "Profile JSON was empty.", false);
            if (profile.SchemaVersion != PersonalCalibrationProfile.CurrentSchemaVersion)
                return new(null, $"Unsupported profile schema {profile.SchemaVersion}.", false);
            if (profile.StockModelHash != contract.Sha256) return new(null, "Stock-model hash mismatch.", false);
            if (profile.ExpressionListHash != contract.ExpressionListHash) return new(null, "Expression-list hash mismatch.", false);
            if (!profile.StockOutput.Dimensions.SequenceEqual(contract.Output.Dimensions) ||
                profile.Parameters.Count != contract.Parameters.Count ||
                profile.Parameters.Where((x, i) => x.OriginalOutputIndex != i ||
                    x.CanonicalName != contract.Parameters[i].CanonicalName).Any())
                return new(null, "Output dimensions, names, or ordering are incompatible.", false);
            var cameraMismatch = camera != null && profile.CameraConfigurationHash != PersonalProfileGenerator.HashCamera(camera);
            return new(profile, cameraMismatch ? "Camera/preprocessing settings differ from calibration." : "", cameraMismatch);
        }
        catch (JsonException ex) { return new(null, "Profile JSON is corrupt: " + ex.Message, false); }
        catch (Exception ex) { return new(null, "Profile could not be loaded: " + ex.Message, false); }
    }

    public static JsonSerializerOptions Options() => new()
    {
        WriteIndented = true, PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter() }
    };
}

public sealed record ProfileLoadResult(PersonalCalibrationProfile? Profile, string Warning, bool CameraMismatch)
{
    public bool Success => Profile != null;
}
