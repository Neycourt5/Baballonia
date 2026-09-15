using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Baballonia.Contracts;
using Baballonia.Services.Personalization.Eye;

namespace Baballonia.Services.Calibration;

/// <summary>One saved Quick Eye Setup result, kept so it can be compared against later ones.</summary>
public sealed record EyeCalibrationEntry(
    string Id,
    string Name,
    string CreatedUtc,
    EyeCalibrationProfile Left,
    EyeCalibrationProfile Right);

/// <summary>
/// Keeps every saved eye calibration, and remembers which one is in use.
/// </summary>
/// <remarks>
/// <para>Calibration used to be a single slot: saving overwrote whatever was there, with no way to
/// tell what changed, no way back to a good result after a bad capture, and no way to run
/// uncalibrated for comparison. Since a capture takes twelve seconds and its quality depends
/// entirely on how well the poses were performed, "I preferred the previous one" is the normal
/// case, not an edge case.</para>
///
/// <para>The pipeline is deliberately untouched by this. Exactly one entry is <em>active</em>, and
/// activating copies its two profiles into the same <c>EyeCalibration_Left</c>/<c>_Right</c> keys
/// the post-processor has always read. The library is a drawer in front of that slot, not a
/// replacement for it — so a build that has never heard of the library still loads the active
/// calibration correctly.</para>
/// </remarks>
public sealed class EyeCalibrationLibrary(ILocalSettingsService settings)
{
    public const string EntriesKey = "EyeCalibration_Saved";
    public const string ActiveIdKey = "EyeCalibration_ActiveId";

    /// <summary>The id meaning "no calibration" — the pipeline runs on identity mappings.</summary>
    public const string DefaultId = "";

    /// <summary>Most recent first, because that is the one being judged.</summary>
    public IReadOnlyList<EyeCalibrationEntry> Entries =>
        (settings.ReadSetting<List<EyeCalibrationEntry>>(EntriesKey) ?? [])
        .Where(entry => entry is { Id: not null, Left: not null, Right: not null })
        .OrderByDescending(entry => entry.CreatedUtc, StringComparer.Ordinal)
        .ToList();

    /// <summary><see cref="DefaultId"/> when running uncalibrated.</summary>
    public string ActiveId => settings.ReadSetting<string>(ActiveIdKey, DefaultId) ?? DefaultId;

    public EyeCalibrationEntry? Active =>
        Entries.FirstOrDefault(entry => string.Equals(entry.Id, ActiveId, StringComparison.Ordinal));

    /// <summary>Saves a capture, makes it active, and returns it.</summary>
    public EyeCalibrationEntry Add(EyeCalibrationProfile left, EyeCalibrationProfile right, string? name = null)
    {
        var now = DateTime.UtcNow;
        var entry = new EyeCalibrationEntry(
            Id: now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture),
            Name: string.IsNullOrWhiteSpace(name)
                ? now.ToLocalTime().ToString("d MMM, HH:mm", CultureInfo.CurrentCulture)
                : name.Trim(),
            CreatedUtc: now.ToString("o", CultureInfo.InvariantCulture),
            Left: left,
            Right: right);

        var entries = Entries.ToList();
        entries.Insert(0, entry);
        settings.SaveSetting(EntriesKey, entries);
        Activate(entry.Id);
        return entry;
    }

    /// <summary>
    /// Copies an entry into the slot the pipeline reads. An unknown id — including
    /// <see cref="DefaultId"/> and any entry that has since been deleted — means uncalibrated.
    /// </summary>
    public void Activate(string id)
    {
        var entry = Entries.FirstOrDefault(e => string.Equals(e.Id, id, StringComparison.Ordinal));

        settings.SaveSetting(ActiveIdKey, entry?.Id ?? DefaultId);
        // Whatever gaze centre is now live was chosen deliberately against the current correction,
        // so the "re-run Quick Eye Setup" prompt has been answered.
        settings.SaveSetting(EyePersonalizationManager.CalibrationStaleSetting, false);
        settings.SaveSetting(EyeCalibrationSettings.LeftKey, entry?.Left ?? EyeCalibrationProfile.Default);
        settings.SaveSetting(EyeCalibrationSettings.RightKey, entry?.Right ?? EyeCalibrationProfile.Default);
    }

    /// <summary>
    /// Removes an entry. Deleting the active one falls back to uncalibrated rather than silently
    /// promoting a neighbour — which calibration you are running should never change by surprise.
    /// </summary>
    public void Delete(string id)
    {
        if (string.IsNullOrEmpty(id))
            return;

        var remaining = Entries
            .Where(entry => !string.Equals(entry.Id, id, StringComparison.Ordinal))
            .ToList();

        settings.SaveSetting(EntriesKey, remaining);

        if (string.Equals(ActiveId, id, StringComparison.Ordinal))
            Activate(DefaultId);
    }

    /// <summary>
    /// Brings a pre-library installation into the drawer: if a calibration is already stored in the
    /// slot but no entry describes it, it is adopted as one so it is not lost the first time
    /// something else is selected.
    /// </summary>
    public void AdoptExistingCalibrationIfUnseen()
    {
        if (Entries.Count > 0)
            return;

        var left = settings.ReadSetting<EyeCalibrationProfile>(EyeCalibrationSettings.LeftKey);
        var right = settings.ReadSetting<EyeCalibrationProfile>(EyeCalibrationSettings.RightKey);
        if (left is null || right is null)
            return;

        if (left == EyeCalibrationProfile.Default && right == EyeCalibrationProfile.Default)
            return;

        Add(left, right, "Previous calibration");
    }
}
