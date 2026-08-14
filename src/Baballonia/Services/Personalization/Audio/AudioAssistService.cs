using System;
using Baballonia.Contracts;
using Microsoft.Extensions.Logging;

namespace Baballonia.Services.Personalization.Audio;

/// <summary>
/// Owns the optional audio path: capture, analysis, and the enhancer stage on the face pipeline.
///
/// The contract this service exists to guarantee is that audio is *strictly* additive. Off, absent,
/// broken, or denied by the OS all resolve to the same thing - no enhancer installed, and a pipeline
/// that produces exactly the values it produced before this feature was written. There is no
/// degraded audio mode, because a half-working microphone must never be able to make tracking worse
/// than not having one.
/// </summary>
public sealed class AudioAssistService : IDisposable
{
    public const string EnabledSetting = "AudioAssist_Enabled";
    public const string StrengthSetting = "AudioAssist_Strength";
    public const string SyncOffsetSetting = "AudioAssist_SyncOffsetMs";

    private readonly Func<IAudioFeatureSource> _sourceFactory;
    private readonly ILocalSettingsService _settings;
    private readonly ILogger<AudioAssistService> _logger;
    private readonly Action<IExpressionEnhancer?> _installEnhancer;

    private IAudioFeatureSource? _source;
    private ProsodyEnhancer? _enhancer;

    public AudioAssistService(
        Func<IAudioFeatureSource> sourceFactory,
        ILocalSettingsService settings,
        Action<IExpressionEnhancer?> installEnhancer,
        ILogger<AudioAssistService> logger)
    {
        _sourceFactory = sourceFactory;
        _settings = settings;
        _installEnhancer = installEnhancer;
        _logger = logger;
    }

    /// <summary>Whether the enhancer is installed and receiving audio.</summary>
    public bool IsActive => _enhancer != null && _source is { IsRunning: true };

    /// <summary>Human-readable state for the settings page.</summary>
    public string StatusMessage { get; private set; } = "Audio assist is off.";

    public bool Enabled => _settings.ReadSetting<bool>(EnabledSetting);

    /// <summary>0..1; 0 makes the enhancer an exact passthrough even while installed.</summary>
    public float Strength
    {
        get
        {
            var value = _settings.ReadSetting<float>(StrengthSetting);
            return value <= 0f ? 0.5f : Math.Clamp(value, 0f, 1f);
        }
        set
        {
            var clamped = Math.Clamp(value, 0f, 1f);
            _settings.SaveSetting(StrengthSetting, clamped);
            if (_enhancer != null)
                _enhancer.Strength = clamped;
        }
    }

    /// <summary>Milliseconds of audio delay applied before the features are used.</summary>
    public int SyncOffsetMs
    {
        get => Math.Clamp(_settings.ReadSetting<int>(SyncOffsetSetting), -200, 500);
        set
        {
            var clamped = Math.Clamp(value, -200, 500);
            _settings.SaveSetting(SyncOffsetSetting, clamped);
            if (_enhancer != null)
                _enhancer.SyncOffset = TimeSpan.FromMilliseconds(clamped);
        }
    }

    /// <summary>Latest analysed window, for the diagnostics readout.</summary>
    public AudioFeatures Features => _source?.Latest ?? AudioFeatures.Silent();

    /// <summary>Gain the enhancer applied most recently. 1.0 means it changed nothing.</summary>
    public float CurrentGain => _enhancer?.LastGain ?? 1f;

    /// <summary>Persists the on/off state and applies it.</summary>
    public void SetEnabled(bool enabled)
    {
        _settings.SaveSetting(EnabledSetting, enabled);
        Apply();
    }

    /// <summary>
    /// Brings the audio path in line with the current settings. Safe to call repeatedly.
    /// </summary>
    public void Apply()
    {
        if (!Enabled)
        {
            Teardown("Audio assist is off.");
            return;
        }

        try
        {
            _source ??= _sourceFactory();

            if (!_source.IsRunning && !_source.Start())
            {
                // The microphone is the optional half. Losing it costs the enhancement and nothing
                // else, so this is a status line rather than an error.
                Teardown($"Audio assist unavailable: {_source.StatusMessage}");
                return;
            }

            _enhancer ??= new ProsodyEnhancer(_source)
            {
                Strength = Strength,
                SyncOffset = TimeSpan.FromMilliseconds(SyncOffsetMs)
            };

            _installEnhancer(_enhancer);
            StatusMessage = "Audio assist active.";
            _logger.LogInformation("Audio expression assist enabled (strength {Strength:P0})", Strength);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not enable audio assist; visual tracking is unaffected");
            Teardown("Audio assist could not start. Visual tracking is unaffected.");
        }
    }

    private void Teardown(string status)
    {
        // Uninstall before stopping capture, so no tick can run the enhancer against a dead source.
        _installEnhancer(null);
        _enhancer = null;

        try
        {
            _source?.Stop();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Audio capture did not stop cleanly");
        }

        StatusMessage = status;
    }

    public void Dispose()
    {
        Teardown("Audio assist is off.");
        _source?.Dispose();
        _source = null;
    }
}
