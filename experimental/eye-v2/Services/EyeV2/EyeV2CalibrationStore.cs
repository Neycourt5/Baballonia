using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.Services.EyeV2;

/// <summary>
/// Separate V2 persistence. This class never reads or writes EyeHome_EyeModel, CalibrationParams,
/// ModelData, or any other Default Baballonia calibration state.
/// </summary>
public sealed class EyeV2CalibrationStore
{
    public const string FileName = "EyeV2_Calibration.json";

    private readonly ILogger<EyeV2CalibrationStore> _logger;
    private readonly JsonSerializerOptions _json = new() { WriteIndented = true };

    public string Path { get; }

    public EyeV2CalibrationStore(ILogger<EyeV2CalibrationStore> logger)
        : this(logger, System.IO.Path.Combine(Utils.ModelsDirectory, FileName))
    {
    }

    public EyeV2CalibrationStore(ILogger<EyeV2CalibrationStore> logger, string path)
    {
        _logger = logger;
        Path = path;
    }

    public bool TryLoad(out EyeV2Calibration? calibration, out string error)
    {
        calibration = null;
        error = "";
        try
        {
            if (!File.Exists(Path))
            {
                error = $"No Eye V2 calibration at {Path}.";
                return false;
            }

            var json = File.ReadAllText(Path);
            calibration = JsonSerializer.Deserialize<EyeV2Calibration>(json, _json);
            if (calibration == null)
            {
                error = "Eye V2 calibration file was empty.";
                return false;
            }

            if (calibration.SchemaVersion != EyeV2Calibration.CurrentSchemaVersion)
            {
                error = $"Eye V2 calibration schema {calibration.SchemaVersion} is not supported.";
                calibration = null;
                return false;
            }

            if (!calibration.IsValid())
            {
                error = "Eye V2 calibration anchors or gaze map are invalid.";
                calibration = null;
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            error = $"Could not load Eye V2 calibration: {ex.Message}";
            _logger.LogWarning(ex, "Could not load Eye V2 calibration from {Path}", Path);
            calibration = null;
            return false;
        }
    }

    public async Task SaveAsync(EyeV2Calibration calibration, CancellationToken cancellationToken = default)
    {
        if (!calibration.IsValid())
            throw new InvalidOperationException("Refusing to persist invalid Eye V2 calibration.");

        var directory = System.IO.Path.GetDirectoryName(Path);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
        var temporary = Path + ".tmp";
        var json = JsonSerializer.Serialize(calibration, _json);
        await File.WriteAllTextAsync(temporary, json, cancellationToken);
        File.Move(temporary, Path, overwrite: true);
    }

    /// <summary>Opaque byte-for-byte snapshot used by the calibration commit transaction.</summary>
    internal readonly record struct FileSnapshot(bool Existed, byte[] Contents);

    internal async Task<FileSnapshot> CaptureSnapshotAsync(
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(Path))
            return new FileSnapshot(false, []);
        return new FileSnapshot(true,
            await File.ReadAllBytesAsync(Path, cancellationToken).ConfigureAwait(false));
    }

    internal async Task RestoreSnapshotAsync(FileSnapshot snapshot)
    {
        if (!snapshot.Existed)
        {
            if (File.Exists(Path)) File.Delete(Path);
            return;
        }

        var directory = System.IO.Path.GetDirectoryName(Path);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
        var temporary = $"{Path}.rollback-{Guid.NewGuid():N}";
        try
        {
            await File.WriteAllBytesAsync(temporary, snapshot.Contents).ConfigureAwait(false);
            File.Move(temporary, Path, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary)) File.Delete(temporary);
        }
    }
}
