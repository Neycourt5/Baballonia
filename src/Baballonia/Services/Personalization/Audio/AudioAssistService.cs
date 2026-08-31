using System;
using System.Collections.Generic;
using System.Linq;
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
    public const string DeviceIdSetting = "AudioAssist_DeviceId";
    public const string DeviceNameSetting = "AudioAssist_DeviceName";

    private readonly IAudioInputDeviceCatalog _deviceCatalog;
    private readonly IAudioFeatureSourceFactory _sourceFactory;
    private readonly ILocalSettingsService _settings;
    private readonly ILogger<AudioAssistService> _logger;
    private readonly Action<IExpressionEnhancer?> _installEnhancer;
    private readonly object _lifecycleGate = new();

    private IAudioFeatureSource? _source;
    private ProsodyEnhancer? _enhancer;
    private bool _enhancerInstalled;
    private AudioInputDevice? _activeDevice;
    private bool _isUsingDeviceFallback;
    private string _statusMessage = "Audio assist is off.";

    public AudioAssistService(
        IAudioInputDeviceCatalog deviceCatalog,
        IAudioFeatureSourceFactory sourceFactory,
        ILocalSettingsService settings,
        Action<IExpressionEnhancer?> installEnhancer,
        ILogger<AudioAssistService> logger)
    {
        _deviceCatalog = deviceCatalog;
        _sourceFactory = sourceFactory;
        _settings = settings;
        _installEnhancer = installEnhancer;
        _logger = logger;
    }

    /// <summary>Whether the enhancer is installed and receiving audio.</summary>
    public bool IsActive
    {
        get { lock (_lifecycleGate) return IsActiveCore; }
    }

    private bool IsActiveCore => _enhancer != null && _source is { IsRunning: true };

    /// <summary>Human-readable state for the settings page.</summary>
    public string StatusMessage
    {
        get
        {
            lock (_lifecycleGate)
            {
                var source = _source;
                if (Enabled && source != null && !source.IsRunning &&
                    !string.IsNullOrWhiteSpace(source.StatusMessage))
                {
                    return $"Audio assist unavailable: {source.StatusMessage} Visual tracking is unaffected.";
                }

                return _statusMessage;
            }
        }
    }

    public bool Enabled => _settings.ReadSetting<bool>(EnabledSetting);

    /// <summary>The user's persisted preference. It is retained when that device is unplugged.</summary>
    public string PreferredDeviceId => _settings.ReadSetting<string>(DeviceIdSetting, "") ?? "";

    /// <summary>Last known name, used to recover across a driver identifier change.</summary>
    public string PreferredDeviceName => _settings.ReadSetting<string>(DeviceNameSetting, "") ?? "";

    /// <summary>The device capture actually opened, which may be a temporary fallback.</summary>
    public AudioInputDevice? ActiveDevice
    {
        get { lock (_lifecycleGate) return _activeDevice; }
    }

    /// <summary>Whether <see cref="ActiveDevice"/> is standing in for an unavailable preference.</summary>
    public bool IsUsingDeviceFallback
    {
        get { lock (_lifecycleGate) return _isUsingDeviceFallback; }
    }

    /// <summary>Fresh snapshot for a microphone selector. Enumeration never starts capture.</summary>
    public IReadOnlyList<AudioInputDevice> GetInputDevices()
    {
        try
        {
            return _deviceCatalog.GetDevices();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not enumerate microphone inputs");
            return Array.Empty<AudioInputDevice>();
        }
    }

    /// <summary>0..1; 0 makes the enhancer an exact passthrough even while installed.</summary>
    public float Strength
    {
        get
        {
            // Zero is a meaningful persisted value: it makes the installed enhancer an exact
            // passthrough. Supply the default to the settings store so only a missing value becomes
            // 50%; treating zero as "unset" silently changed the user's choice after a restart.
            var value = _settings.ReadSetting<float>(StrengthSetting, 0.5f);
            return Math.Clamp(value, 0f, 1f);
        }
        set
        {
            var clamped = Math.Clamp(value, 0f, 1f);
            _settings.SaveSetting(StrengthSetting, clamped);
            lock (_lifecycleGate)
            {
                if (_enhancer != null)
                    _enhancer.Strength = clamped;
            }
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
            lock (_lifecycleGate)
            {
                if (_enhancer != null)
                    _enhancer.SyncOffset = TimeSpan.FromMilliseconds(clamped);
            }
        }
    }

    /// <summary>Latest analysed window, for the diagnostics readout.</summary>
    public AudioFeatures Features
    {
        get { lock (_lifecycleGate) return _source?.Latest ?? AudioFeatures.Silent(); }
    }

    /// <summary>
    /// Meter-friendly 0..1 input level. RMS is mapped from -60 dBFS..0 dBFS so normal speech is
    /// visible instead of occupying only the first few pixels of a linear meter.
    /// </summary>
    public float InputLevel
    {
        get
        {
            var rms = Math.Clamp(Features.Rms, 0f, 1f);
            if (rms <= 0.000001f)
                return 0f;

            var db = 20f * MathF.Log10(rms);
            return Math.Clamp((db + 60f) / 60f, 0f, 1f);
        }
    }

    /// <summary>Live analyser decision for a simple quiet/speaking indicator.</summary>
    public bool IsVoiceDetected => Features.IsVoiced;

    /// <summary>Gain the enhancer applied most recently. 1.0 means it changed nothing.</summary>
    public float CurrentGain
    {
        get { lock (_lifecycleGate) return _enhancer?.LastGain ?? 1f; }
    }

    /// <summary>Persists the on/off state and applies it.</summary>
    public void SetEnabled(bool enabled)
    {
        _settings.SaveSetting(EnabledSetting, enabled);
        Apply();
    }

    /// <summary>
    /// Persists an explicit device and, when audio assist is enabled, restarts only microphone
    /// capture and its optional enhancer. Camera capture and visual inference are not involved.
    /// </summary>
    public bool SelectInputDevice(string? deviceId)
    {
        var devices = GetInputDevices();
        var selected = string.IsNullOrWhiteSpace(deviceId)
            ? null
            : devices.FirstOrDefault(device =>
                string.Equals(device.Id, deviceId, StringComparison.Ordinal));

        _settings.SaveSetting(DeviceIdSetting, selected?.Id ?? deviceId ?? "");
        _settings.SaveSetting(DeviceNameSetting, selected?.DisplayName ?? "");

        return RestartAudioCapture();
    }

    /// <summary>
    /// Reopens the preferred microphone without toggling the persisted enabled state. This is safe
    /// after a USB reconnect or permission change and never restarts a camera or inference loop.
    /// </summary>
    public bool RestartAudioCapture()
    {
        lock (_lifecycleGate)
        {
            TeardownCore(Enabled ? "Restarting audio capture..." : "Audio assist is off.");
            if (!Enabled)
                return false;

            ApplyCore();
            return IsActiveCore;
        }
    }

    /// <summary>
    /// Atomic background-lifecycle check. If assist is enabled but capture has stopped or could not
    /// open, retry only the audio source; an already-active or disabled service is left untouched.
    /// </summary>
    public bool TryRecoverAudioCapture()
    {
        lock (_lifecycleGate)
        {
            if (!Enabled)
                return false;
            if (IsActiveCore)
                return true;

            TeardownCore("Retrying audio capture...");
            ApplyCore();
            return IsActiveCore;
        }
    }

    /// <summary>
    /// Brings the audio path in line with the current settings. Safe to call repeatedly.
    /// </summary>
    public void Apply()
    {
        lock (_lifecycleGate)
            ApplyCore();
    }

    private void ApplyCore()
    {
        if (!Enabled)
        {
            TeardownCore("Audio assist is off.");
            return;
        }

        try
        {
            var resolution = AudioInputDeviceResolver.Resolve(
                GetInputDevices(), PreferredDeviceId, PreferredDeviceName);
            if (!resolution.IsAvailable)
            {
                TeardownCore($"Audio assist unavailable: {resolution.Message} Visual tracking is unaffected.");
                return;
            }

            _activeDevice = resolution.Device;
            _isUsingDeviceFallback = resolution.UsedFallback;
            _source ??= _sourceFactory.Create(resolution.Device!);

            if (!_source.IsRunning && !_source.Start())
            {
                // The microphone is the optional half. Losing it costs the enhancement and nothing
                // else, so this is a status line rather than an error.
                TeardownCore(
                    $"Audio assist unavailable: {_source.StatusMessage} Visual tracking is unaffected.");
                return;
            }

            _enhancer ??= new ProsodyEnhancer(_source)
            {
                Strength = Strength,
                SyncOffset = TimeSpan.FromMilliseconds(SyncOffsetMs)
            };

            _installEnhancer(_enhancer);
            _enhancerInstalled = true;
            _statusMessage = resolution.UsedFallback
                ? $"Audio assist active. {resolution.Message}"
                : $"Audio assist active using {resolution.Device!.DisplayName}.";
            _logger.LogInformation(
                "Audio expression assist enabled using {Device} (strength {Strength:P0}, fallback {Fallback})",
                resolution.Device!.DisplayName,
                Strength,
                resolution.UsedFallback);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not enable audio assist; visual tracking is unaffected");
            TeardownCore("Audio assist could not start. Visual tracking is unaffected.");
        }
    }

    private void TeardownCore(string status)
    {
        // Uninstall before stopping capture, so no tick can run the enhancer against a dead source.
        if (_enhancerInstalled)
        {
            _installEnhancer(null);
            _enhancerInstalled = false;
        }
        _enhancer = null;

        var source = _source;
        try
        {
            source?.Stop();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Audio capture did not stop cleanly");
        }

        try
        {
            source?.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Audio capture did not dispose cleanly");
        }

        _source = null;
        _activeDevice = null;
        _isUsingDeviceFallback = false;
        _statusMessage = status;
    }

    public void Dispose()
    {
        lock (_lifecycleGate)
            TeardownCore("Audio assist is off.");
    }
}
