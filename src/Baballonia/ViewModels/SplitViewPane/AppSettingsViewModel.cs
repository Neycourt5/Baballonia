using Avalonia;
using Avalonia.Media;
using Avalonia.Styling;
using Baballonia.Assets;
using Baballonia.Contracts;
using Baballonia.Models;
using Baballonia.Services;
using Baballonia.Services.Inference;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.DependencyInjection;
using CommunityToolkit.Mvvm.Messaging;
using Microsoft.Extensions.Logging;
using OscCore;
using System;
using System.Collections.ObjectModel;
using System.IO;

namespace Baballonia.ViewModels.SplitViewPane;

public partial class AppSettingsViewModel : ViewModelBase
{
    [ObservableProperty]
    [property: SavedSetting("AppSettings_RecalibrateAddress", "/avatar/parameters/etvr_recalibrate")]
    private string _recalibrateAddress;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_RecenterAddress", "/avatar/parameters/etvr_recenter")]
    private string _recenterAddress;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_OSCPrefix", "")]
    private string _oscPrefix;

    [ObservableProperty]
    private IBrush _oscPrefixBackgroundColor;

    private bool _isOscPrefixValid = true;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_OneEuroEnabled", true)]
    private bool _oneEuroMinEnabled;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_OneEuroMinFreqCutoff", 0.5f)]
    private float _oneEuroMinFreqCutoff;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_OneEuroSpeedCutoff", 3f)]
    private float _oneEuroSpeedCutoff;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_UseDFR", false)]
    private bool _useDFR;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_UseGPU", true)]
    private bool _useGPU;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_SteamVRAutoStart", true)]
    private bool _steamvrAutoStart;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_CheckForUpdates", false)]
    private bool _checkForUpdates;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_ShareEyeData", false)]
    private bool _shareEyeData;

    public bool DataSharingAvailable { get; } = DataUploadConfiguration.FromEnvironment().IsConfigured;
    public string DataSharingStatus => DataSharingAvailable
        ? "Optional data sharing is available. Uploads require your opt-in."
        : "Data sharing is unavailable in this build until upload access is configured. Calibration stays on this computer.";

    [ObservableProperty]
    private string _logLevel;

    public ObservableCollection<string> LowestLogLevel { get; } =
    [
        Resources.Settings_LogLevel_Debug,
        Resources.Settings_LogLevel_Information,
        Resources.Settings_LogLevel_Warning,
        Resources.Settings_LogLevel_Error
    ];

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DebugMenuToggleVisible))]
    [property: SavedSetting("AppSettings_AdvancedOptions", false)]
    private bool _advancedOptions;

    /// <summary>
    /// The "Show Debug menu" toggle is desktop-only (the Debug page isn't reachable on mobile), so it
    /// shows only when Advanced is on <em>and</em> we're on a supported desktop OS.
    /// </summary>
    public bool DebugMenuToggleVisible => AdvancedOptions && Utils.IsSupportedDesktopOS;

    // Advanced-only options. Surfaced in Settings beneath the Advanced toggle and only visible while
    // AdvancedOptions is on (see AppSettingsView.axaml).
    [ObservableProperty]
    [property: SavedSetting("AppSettings_SplitEyeVideoSwap", false)]
    private bool _splitEyeVideoSwap;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_ShowDebugMenu", false)]
    private bool _showDebugMenu;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_StabilizeEyes", true)]
    private bool _stabilizeEyes;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_EyePostProcessing", true)]
    private bool _eyePostProcessing;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_EyeLidSyncAmount", 0.75f)]
    private float _eyeLidSyncAmount;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_EyeSquintSyncAmount", 0f)]
    private float _eyeSquintSyncAmount;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_EyeWidenSyncAmount", 0f)]
    private float _eyeWidenSyncAmount;

    /// <summary>
    /// Which eye the others follow: "Average", "Left" or "Right". Stored as a string so the setting
    /// stays readable in the settings file and survives the enum gaining members.
    /// </summary>
    [ObservableProperty]
    [property: SavedSetting("AppSettings_EyeSyncSource", "Average")]
    private string _eyeSyncSource = "Average";

    /// <summary>Index into <see cref="EyeSyncSourceOptions"/>, for the ComboBox to bind to.</summary>
    public int EyeSyncSourceIndex
    {
        get => Array.FindIndex(EyeSyncSourceOptions,
            option => string.Equals(option, EyeSyncSource, StringComparison.OrdinalIgnoreCase)) is var i and >= 0
            ? i
            : 0;
        set
        {
            if (value >= 0 && value < EyeSyncSourceOptions.Length)
                EyeSyncSource = EyeSyncSourceOptions[value];
        }
    }

    public static string[] EyeSyncSourceOptions { get; } = ["Average", "Left", "Right", "Stronger"];

    [ObservableProperty]
    [property: SavedSetting("AppSettings_EyeWinkDetection", true)]
    private bool _eyeWinkDetection;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_EyeGazeSyncAmount", 0f)]
    private float _eyeGazeSyncAmount;

    /// <summary>
    /// Exponent on jaw open. 1 is unchanged; the sender re-reads it once a second, so no pipeline
    /// reload is needed.
    /// </summary>
    [ObservableProperty]
    [property: SavedSetting("AppSettings_JawOpenCurve", 1f)]
    private float _jawOpenCurve;

    /// <summary>
    /// How strongly a detected squint is expressed. 1 is unchanged; the sender re-reads it once a
    /// second, so no pipeline reload is needed.
    /// </summary>
    [ObservableProperty]
    [property: SavedSetting("AppSettings_EyeSquintStrength", 1f)]
    private float _eyeSquintStrength;

    /// <summary>
    /// BlinkGuard's master switch. Off by default: it changes live gaze, so it is opted into.
    /// </summary>
    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardEnabled", false)]
    private bool _blinkGuardEnabled;

    /// <summary>"Subtle", "Balanced", "Strong" or "Custom". Stored as a string so it stays readable.</summary>
    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardPreset", "Balanced")]
    private string _blinkGuardPreset = "Balanced";

    public static string[] BlinkGuardPresetOptions { get; } = ["Subtle", "Balanced", "Strong", "Custom"];

    public int BlinkGuardPresetIndex
    {
        get => Array.FindIndex(BlinkGuardPresetOptions,
            option => string.Equals(option, BlinkGuardPreset, StringComparison.OrdinalIgnoreCase)) is var i and >= 0
            ? i
            : 1;
        set
        {
            if (value >= 0 && value < BlinkGuardPresetOptions.Length)
                BlinkGuardPreset = BlinkGuardPresetOptions[value];
        }
    }

    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardClosedThreshold", 0.60f)]
    private float _blinkGuardClosedThreshold;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardReopenThreshold", 0.30f)]
    private float _blinkGuardReopenThreshold;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardMinimumHoldMs", 30f)]
    private float _blinkGuardMinimumHoldMs;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardStableSamples", 3)]
    private int _blinkGuardStableSamples;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardToleranceDegrees", 5f)]
    private float _blinkGuardToleranceDegrees;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardTimeoutMs", 175f)]
    private float _blinkGuardTimeoutMs;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardTransitionMs", 80f)]
    private float _blinkGuardTransitionMs;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardUseConfidence", false)]
    private bool _blinkGuardUseConfidence;

    [ObservableProperty]
    [property: SavedSetting("AppSettings_BlinkGuardNormalSpikeGuard", false)]
    private bool _blinkGuardNormalSpikeGuard;

    /// <summary>Live BlinkGuard readout, refreshed by the view's timer. Diagnostics only.</summary>
    [ObservableProperty] private string _blinkGuardStatus = "Off";
    [ObservableProperty] private string _blinkGuardCounters = "";
    [ObservableProperty] private bool _blinkGuardCapturing;
    [ObservableProperty] private string _blinkGuardCaptureStatus = "Not recording.";

    /// <summary>Polls the live filter. Cheap: reads a handful of fields and formats two strings.</summary>
    public void RefreshBlinkGuardDiagnostics()
    {
        var guard = _eyePipelineManager.BlinkGuard;
        var c = guard.Diagnostics.Counters;

        BlinkGuardStatus = !guard.Settings.Enabled
            ? "Off"
            : $"L {guard.LeftState}   R {guard.RightState}" +
              (guard.IsIntervening ? $"   reacquiring {guard.LeftReacquiringSeconds * 1000:F0} ms" : "");

        BlinkGuardCounters =
            $"Blinks {c.BlinksDetected}   rejected {c.SamplesRejected}   " +
            $"glitches prevented {c.GlitchesPrevented}   timeouts {c.TimeoutCount}   " +
            $"reacquire avg {c.ReacquisitionAverageSeconds * 1000:F0} ms / max {c.ReacquisitionMaxSeconds * 1000:F0} ms";

        BlinkGuardCapturing = guard.Diagnostics.IsCapturing;
        BlinkGuardCaptureStatus = guard.Diagnostics.IsCapturing
            ? $"Recording… {guard.Diagnostics.RecordCount} / {guard.Diagnostics.Capacity} frames"
            : guard.Diagnostics.RecordCount > 0
                ? $"{guard.Diagnostics.RecordCount} frames captured. Export writes a CSV."
                : "Not recording.";
    }

    [RelayCommand]
    private void StartBlinkGuardCapture()
    {
        _eyePipelineManager.BlinkGuard.Diagnostics.StartCapture();
        RefreshBlinkGuardDiagnostics();
    }

    [RelayCommand]
    private void StopBlinkGuardCapture()
    {
        _eyePipelineManager.BlinkGuard.Diagnostics.StopCapture();
        RefreshBlinkGuardDiagnostics();
    }

    [RelayCommand]
    private void ResetBlinkGuardCounters()
    {
        _eyePipelineManager.BlinkGuard.Diagnostics.Counters.Reset();
        RefreshBlinkGuardDiagnostics();
    }

    /// <summary>Writes the capture beside the app's other persistent data and reports the path.</summary>
    [RelayCommand]
    private void ExportBlinkGuardCapture()
    {
        try
        {
            var directory = Path.Combine(Baballonia.Utils.PersistentDataDirectory, "BlinkGuard");
            Directory.CreateDirectory(directory);

            var path = Path.Combine(directory,
                $"blinkguard_{DateTime.Now:yyyyMMdd_HHmmss}.csv");
            File.WriteAllText(path, _eyePipelineManager.BlinkGuard.Diagnostics.ToCsv());

            BlinkGuardCaptureStatus = $"Exported to {path}";
        }
        catch (Exception ex)
        {
            // An export failure must never take the settings page down with it.
            BlinkGuardCaptureStatus = $"Export failed: {ex.Message}";
        }
    }

    [RelayCommand]
    private void ResetBlinkGuardDefaults()
    {
        BlinkGuardPreset = "Balanced";
        BlinkGuardClosedThreshold = 0.60f;
        BlinkGuardReopenThreshold = 0.30f;
        BlinkGuardMinimumHoldMs = 30f;
        BlinkGuardStableSamples = 3;
        BlinkGuardToleranceDegrees = 5f;
        BlinkGuardTimeoutMs = 175f;
        BlinkGuardTransitionMs = 80f;
        BlinkGuardUseConfidence = false;
        BlinkGuardNormalSpikeGuard = false;
    }

    [ObservableProperty]
    [property: SavedSetting("AppSettings_EyeGazeConjugateAmount", 0f)]
    private float _eyeGazeConjugateAmount;

    [ObservableProperty] private bool _onboardingEnabled;

    public string MachineID => _identityService.GetUniqueUserId();

    public IOscTarget OscTarget { get; }

    private readonly FacePipelineManager _facePipelineManager;
    private readonly EyePipelineManager _eyePipelineManager;
    private readonly IIdentityService _identityService;
    private readonly ILogger<AppSettingsViewModel> _logger;
    private readonly ILocalSettingsService _localSettingsService;
    private readonly OpenVRService? _openVrService;
    // Set once the constructor finishes loading persisted settings, so OnSteamvrAutoStartChanged
    // can tell a user toggle apart from the initial load (which must not touch OpenVR).
    private bool _settingsLoaded;

    public AppSettingsViewModel(
        FacePipelineManager facePipelineManager,
        EyePipelineManager eyePipelineManager,
        ILocalSettingsService localSettingsService,
        IOscTarget oscTarget,
        IIdentityService identityService,
        GithubService githubService,
        ParameterSenderService parameterSenderService,
        ILogger<AppSettingsViewModel> logger,
        IThemeSelectorService themeSelectorService)
    {
        OscTarget = oscTarget;
        _localSettingsService = localSettingsService;
        _facePipelineManager = facePipelineManager;
        _eyePipelineManager = eyePipelineManager;
        _identityService = identityService;
        _logger = logger;
        // OpenVRService is only registered on supported desktop OSes; resolve it optionally
        // so the SteamVR-autostart toggle works there and no-ops elsewhere (see null guard below).
        _openVrService = Ioc.Default.GetService<OpenVRService>();
        _localSettingsService.Load(this);

        LogLevel = _localSettingsService.ReadSetting("AppSettings_LogLevel", "Debug");

        // Handle edge case where OSC port is used and the system freaks out
        if (OscTarget.OutPort == 0)
        {
            const int port = 8888;
            OscTarget.OutPort = port;
            _localSettingsService.SaveSetting("OSCOutPort", port);
        }

        // Edge case: Update the OscPrefix Background color if and only if
        // The theme changes and the previous input WAS valid (IE keep red)
        themeSelectorService.ThemeChanged += variant =>
        {
            if (_isOscPrefixValid)
                SetOscPrefixBackgroundColor(variant);
        };

        OnboardingEnabled = Utils.IsSupportedDesktopOS;

        PropertyChanged += (_, p) =>
        {
            _localSettingsService.Save(this);

            if (p.PropertyName is nameof(OneEuroMinEnabled) or nameof(OneEuroMinFreqCutoff)
                or nameof(OneEuroSpeedCutoff))
            {
                _facePipelineManager.LoadFilter();
                _eyePipelineManager.LoadFilter();
            }

            if (p.PropertyName == nameof(StabilizeEyes))
            {
                _eyePipelineManager.LoadEyeStabilization();
            }

            if (p.PropertyName is nameof(EyePostProcessing) or nameof(EyeLidSyncAmount)
                or nameof(EyeSquintSyncAmount) or nameof(EyeWidenSyncAmount) or nameof(EyeSyncSource)
                or nameof(EyeWinkDetection) or nameof(EyeGazeSyncAmount))
            {
                if (p.PropertyName == nameof(EyeSyncSource))
                    OnPropertyChanged(nameof(EyeSyncSourceIndex));

                _eyePipelineManager.LoadEyePostProcessor();
            }

            if (p.PropertyName == nameof(EyeGazeConjugateAmount))
            {
                _eyePipelineManager.LoadEyeStabilization();
            }

            if (p.PropertyName is not null && p.PropertyName.StartsWith("BlinkGuard", StringComparison.Ordinal))
            {
                if (p.PropertyName == nameof(BlinkGuardPreset))
                    OnPropertyChanged(nameof(BlinkGuardPresetIndex));

                _eyePipelineManager.LoadBlinkGuard();
            }

            if (p.PropertyName == nameof(SplitEyeVideoSwap))
            {
                _eyePipelineManager.LoadSplitEyeSwap();
            }
        };

        _settingsLoaded = true;
    }

    // Tell the navigation sidebar to add/remove the Debug page entry as soon as the toggle flips,
    // instead of only on next launch. The generic PropertyChanged handler above persists the value.
    partial void OnShowDebugMenuChanged(bool value)
    {
        WeakReferenceMessenger.Default.Send(new ShowDebugMenuChangedMessage(value));
    }

    partial void OnLogLevelChanged(string value)
    {
        var prev = _localSettingsService.ReadSetting("AppSettings_LogLevel", value);
        if (prev == value)
            return;

        var newLogLevel = value switch
        {
            var v when v == Resources.Settings_LogLevel_Debug => "Debug",
            var v when v == Resources.Settings_LogLevel_Information => "Information",
            var v when v == Resources.Settings_LogLevel_Warning => "Warning",
            var v when v == Resources.Settings_LogLevel_Error => "Error",
            _ => "Debug"
        };
        _localSettingsService.SaveSetting("AppSettings_LogLevel", newLogLevel);
    }

    partial void OnOscPrefixChanged(string value)
    {
        // 1) A valid OSC prefix is also a valid message itself
        // IE: /foo/bar + /cheekPuffLeft
        // 2) Empty strings are also valid, IE no prefix
        _isOscPrefixValid = OscMessage.TryParse(value, out _) || string.IsNullOrEmpty(value);

        if (_isOscPrefixValid)
        {
            _localSettingsService.SaveSetting("AppSettings_OSCPrefix", value);
            SetOscPrefixBackgroundColor(Application.Current!.ActualThemeVariant);
            return;
        }

        OscPrefixBackgroundColor = new SolidColorBrush(Colors.PaleVioletRed);
    }

    private void SetOscPrefixBackgroundColor(ThemeVariant theme)
    {
        // Workaround to get proper SystemChromeMediumColor color
        OscPrefixBackgroundColor = theme.ToString() switch
        {
            "Light" => new SolidColorBrush(Colors.White),
            "Dark" => SolidColorBrush.Parse("#ff202020"),
            _ => OscPrefixBackgroundColor
        };
    }

    partial void OnSteamvrAutoStartChanged(bool value)
    {
        // Ignore the initial load (the value came from disk); only apply real user toggles.
        // OpenVRService is null on non-desktop platforms where SteamVR isn't available.
        if (!_settingsLoaded || _openVrService == null)
            return;

        try
        {
            // Idempotent at the OpenVR layer; applies both enabling and disabling autostart.
            _openVrService.SteamvrAutoStart = value;
            _localSettingsService.SaveSetting("AppSettings_SteamVRAutoStart", value);
        }
        catch (Exception e)
        {
            _logger.LogError(e, "Failed to update SteamVR AutoStart");
        }
    }

    async partial void OnUseGPUChanged(bool value)
    {
        var prev = _localSettingsService.ReadSetting("AppSettings_UseGPU", value);
        if (prev == value)
            return;

        try
        {
            _localSettingsService.SaveSetting("AppSettings_UseGPU", value);
            var loadFace = _eyePipelineManager.LoadInferenceAsync();
            var loadEye = _facePipelineManager.LoadInferenceAsync();

            await loadEye;
            await loadFace;
        }
        catch (Exception e)
        {
            _logger.LogError("", e);
        }
    }
}
