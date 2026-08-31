using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Finds and removes quick-correction sessions without ever treating an ordinary recording as a
/// correction merely because of its folder name. Deletion is limited to direct, non-link children
/// of the configured dataset root that contain both correction metadata and SessionType=Correction.
/// </summary>
public sealed class CorrectionSessionStore(string datasetRoot)
{
    public sealed record DeleteResult(int Deleted, int Failed)
    {
        public bool Success => Failed == 0;
    }

    public int Count()
    {
        var count = 0;
        foreach (var _ in FindCorrectionDirectories()) count++;
        return count;
    }

    public DeleteResult DeleteAll()
    {
        var deleted = 0;
        var failed = 0;
        foreach (var directory in FindCorrectionDirectories())
        {
            try
            {
                Directory.Delete(directory, recursive: true);
                deleted++;
            }
            catch
            {
                failed++;
            }
        }

        return new DeleteResult(deleted, failed);
    }

    private IEnumerable<string> FindCorrectionDirectories()
    {
        var root = Path.GetFullPath(datasetRoot);
        if (!Directory.Exists(root)) yield break;

        var rootPrefix = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
                         Path.DirectorySeparatorChar;
        foreach (var candidate in Directory.EnumerateDirectories(root))
        {
            string directory;
            try
            {
                directory = Path.GetFullPath(candidate);
                var info = new DirectoryInfo(directory);
                if (!directory.StartsWith(rootPrefix, StringComparison.OrdinalIgnoreCase) ||
                    info.Attributes.HasFlag(FileAttributes.ReparsePoint) ||
                    !File.Exists(Path.Combine(directory, "correction.json")))
                    continue;

                var metadataPath = Path.Combine(directory, "session.json");
                if (!File.Exists(metadataPath)) continue;
                var metadata = JsonSerializer.Deserialize<SessionMetadata>(
                    File.ReadAllText(metadataPath), PersonalizationPaths.Json);
                if (!string.Equals(metadata?.SessionType, SessionType.Correction.ToString(),
                        StringComparison.OrdinalIgnoreCase))
                    continue;
            }
            catch
            {
                // Ambiguous or corrupt folders are preserved. Deletion must fail closed.
                continue;
            }

            yield return directory;
        }
    }
}
