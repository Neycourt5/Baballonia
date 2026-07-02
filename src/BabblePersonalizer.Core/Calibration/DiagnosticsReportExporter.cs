using BabblePersonalizer.Core.Storage;
using System.Text.Json;

namespace BabblePersonalizer.Core.Calibration;

public sealed class DiagnosticsReportExporter(PersonalizerDataPaths paths)
{
    public async Task<string> ExportAsync(
        PersonalCalibrationProfile profile, CancellationToken cancellationToken = default)
    {
        paths.EnsureCreated();
        var path = Path.Combine(paths.Reports, $"Diagnostics-{DateTime.Now:yyyyMMdd-HHmmss}.json");
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(new
        {
            formatVersion = 2, exportedUtc = DateTimeOffset.UtcNow,
            profile.ProfileId, profile.SessionId, profile.StockModelHash, profile.ExpressionListHash,
            profile.CameraConfigurationHash, profile.Parameters, profile.CrossActivations, profile.Validation
        }, PersonalProfileStore.Options()), cancellationToken).ConfigureAwait(false);
        return path;
    }
}
