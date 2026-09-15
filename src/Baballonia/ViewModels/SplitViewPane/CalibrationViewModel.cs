using Avalonia.Threading;
using Baballonia.Contracts;
using Baballonia.Helpers;
using Baballonia.Models;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using Baballonia.Services.events;
using Baballonia.Services.Inference;
using CommunityToolkit.Mvvm.DependencyInjection;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.ViewModels.SplitViewPane;

public partial class CalibrationViewModel : ViewModelBase, IDisposable
{
    public ObservableCollection<SliderBindableSetting> EyeSettings { get; set; }
    public ObservableCollection<SliderBindableSetting> JawSettings { get; set; }
    public ObservableCollection<SliderBindableSetting> MouthSettings { get; set; }
    public ObservableCollection<SliderBindableSetting> TongueSettings { get; set; }
    public ObservableCollection<SliderBindableSetting> NoseSettings { get; set; }
    public ObservableCollection<SliderBindableSetting> CheekSettings { get; set; }

    private ILocalSettingsService _settingsService { get; }
    private readonly ICalibrationService _calibrationService;
    private readonly ParameterSenderService _parameterSenderService;
    private readonly ProcessingLoopService _processingLoopService;
    private readonly EyePipelineManager _eyePipelineManager;
    private readonly EyeCalibrationRecorder _eyeCalibrationRecorder;
    private readonly EyeCalibrationLibrary _eyeCalibrationLibrary;
    private readonly IEyePipelineEventBus _eyePipelineEventBus;
    private bool _applyingCalibrationSelection;
    private readonly CancellationTokenSource _quickEyeSetupCancellation = new();
    private long _lastQuickEyeUiTick;
    private bool _disposed;

    [ObservableProperty] private float _quickLeftOpenness = 1f;
    [ObservableProperty] private float _quickRightOpenness = 1f;
    [ObservableProperty] private bool _eyeCalibrationAvailable;
    [ObservableProperty] private bool _eyeCalibrationBusy;
    [ObservableProperty] private bool _relaxedStepComplete;
    [ObservableProperty] private bool _closedStepComplete;
    [ObservableProperty] private bool _wideStepComplete;
    [ObservableProperty] private string _eyeCalibrationStatus =
        "Start the eye camera, then record each step in order.";

    /// <summary>Saved calibrations, newest first, with an explicit uncalibrated entry at the top.</summary>
    public ObservableCollection<EyeCalibrationChoice> EyeCalibrationChoices { get; } = [];

    [ObservableProperty] private EyeCalibrationChoice? _selectedEyeCalibration;

    /// <summary>What the active calibration actually measured, so saving has visible consequences.</summary>
    [ObservableProperty] private string _activeEyeCalibrationSummary = "No calibration selected.";

    public bool CanDeleteEyeCalibration =>
        SelectedEyeCalibration is { IsDefault: false } && !EyeCalibrationBusy;

    /// <summary>
    /// The deeper eye-personalization workflow, hosted here so everything to do with setting up
    /// eyes lives on one page. Its own view model, because the two share nothing but a page.
    /// </summary>
    public EyePersonalizationViewModel EyePersonalization { get; }

    public bool CanCaptureEyeCalibration => EyeCalibrationAvailable && !EyeCalibrationBusy;
    public bool CanSaveEyeCalibration =>
        !EyeCalibrationBusy && RelaxedStepComplete && ClosedStepComplete && WideStepComplete;

    public CalibrationViewModel(
        EyePipelineManager eyePipelineManager,
        EyeCalibrationRecorder eyeCalibrationRecorder,
        EyeCalibrationLibrary eyeCalibrationLibrary,
        IEyePipelineEventBus eyePipelineEventBus,
        EyePersonalizationViewModel eyePersonalization)
    {
        EyePersonalization = eyePersonalization;
        _eyePipelineManager = eyePipelineManager;
        _eyeCalibrationRecorder = eyeCalibrationRecorder;
        _eyeCalibrationLibrary = eyeCalibrationLibrary;
        _eyePipelineEventBus = eyePipelineEventBus;
        _settingsService = Ioc.Default.GetService<ILocalSettingsService>()!;
        _calibrationService = Ioc.Default.GetService<ICalibrationService>()!;
        _parameterSenderService = Ioc.Default.GetService<ParameterSenderService>()!;
        _processingLoopService = Ioc.Default.GetService<ProcessingLoopService>()!;

        EyeSettings =
        [
            new("LeftEyeLid"),
            new("RightEyeLid"),
            new("LeftEyeWiden"),
            new("LeftEyeSquint"),
            new("LeftEyeBrow"),
            new("RightEyeWiden"),
            new("RightEyeSquint"),
            new("RightEyeBrow"),
        ];

        JawSettings =
        [
            new("JawOpen"),
            new("JawForward"),
            new("JawLeft"),
            new("JawRight")
        ];

        CheekSettings =
        [
            new("CheekPuffLeft"),
            new("CheekPuffRight"),
            new("CheekSuckLeft"),
            new("CheekSuckRight")
        ];

        NoseSettings =
        [
            new("NoseSneerLeft"),
            new("NoseSneerRight")
        ];

        MouthSettings =
        [
            new("MouthFunnel"),
            new("MouthPucker"),
            new("MouthLeft"),
            new("MouthRight"),
            new("MouthRollUpper"),
            new("MouthRollLower"),
            new("MouthShrugUpper"),
            new("MouthShrugLower"),
            new("MouthClose"),
            new("MouthSmileLeft"),
            new("MouthSmileRight"),
            new("MouthFrownLeft"),
            new("MouthFrownRight"),
            new("MouthDimpleLeft"),
            new("MouthDimpleRight"),
            new("MouthUpperUpLeft"),
            new("MouthUpperUpRight"),
            new("MouthLowerDownLeft"),
            new("MouthLowerDownRight"),
            new("MouthPressLeft"),
            new("MouthPressRight"),
            new("MouthStretchLeft"),
            new("MouthStretchRight")
        ];

        TongueSettings =
        [
            new("TongueOut"),
            new("TongueUp"),
            new("TongueDown"),
            new("TongueLeft"),
            new("TongueRight"),
            new("TongueRoll"),
            new("TongueBendDown"),
            new("TongueCurlUp"),
            new("TongueSquish"),
            new("TongueFlat"),
            new("TongueTwistLeft"),
            new("TongueTwistRight")
        ];

        foreach (var setting in EyeSettings.Concat(JawSettings).Concat(CheekSettings)
                     .Concat(NoseSettings).Concat(MouthSettings).Concat(TongueSettings))
        {
            setting.PropertyChanged += OnSettingChanged;
        }

        PropertyChanged += (o, p) =>
        {
            var propertyInfo = GetType().GetProperty(p.PropertyName!);
            object value = propertyInfo?.GetValue(this)!;
            if (value is float floatValue)
            {
                if (p.PropertyName == null) return;
                _calibrationService.SetExpression(DisplayNameToInternalName(p.PropertyName!), floatValue);
            }
        };

        _processingLoopService.ExpressionChangeEvent += ExpressionUpdateHandler;
        _eyePipelineEventBus.Subscribe<EyePipelineEvents.NewFilteredResultEvent>(OnFilteredEyeResult);
        _eyePipelineManager.LeftSlot.StateChanged += EyeCameraStateChanged;
        _eyePipelineManager.RightSlot.StateChanged += EyeCameraStateChanged;
        EyeCalibrationAvailable = AreTargetedEyeCamerasRunning();

        LoadInitialSettings();
        _settingsService.Load(this);

        // Adopt a pre-library calibration so upgrading does not appear to lose it.
        _eyeCalibrationLibrary.AdoptExistingCalibrationIfUnseen();
        RefreshEyeCalibrationChoices();
    }

    partial void OnEyeCalibrationAvailableChanged(bool value) => RefreshQuickEyeSetupActions();
    partial void OnEyeCalibrationBusyChanged(bool value) => RefreshQuickEyeSetupActions();
    partial void OnRelaxedStepCompleteChanged(bool value) => RefreshQuickEyeSetupActions();
    partial void OnClosedStepCompleteChanged(bool value) => RefreshQuickEyeSetupActions();
    partial void OnWideStepCompleteChanged(bool value) => RefreshQuickEyeSetupActions();

    private void RefreshQuickEyeSetupActions()
    {
        OnPropertyChanged(nameof(CanCaptureEyeCalibration));
        OnPropertyChanged(nameof(CanSaveEyeCalibration));
        OnPropertyChanged(nameof(CanDeleteEyeCalibration));
    }

    /// <summary>
    /// Rebuilds the dropdown from the library and selects whatever is actually active.
    /// </summary>
    private void RefreshEyeCalibrationChoices()
    {
        _applyingCalibrationSelection = true;
        try
        {
            EyeCalibrationChoices.Clear();
            EyeCalibrationChoices.Add(EyeCalibrationChoice.Default);
            foreach (var entry in _eyeCalibrationLibrary.Entries)
                EyeCalibrationChoices.Add(EyeCalibrationChoice.For(entry));

            var activeId = _eyeCalibrationLibrary.ActiveId;
            SelectedEyeCalibration =
                EyeCalibrationChoices.FirstOrDefault(
                    choice => string.Equals(choice.Id, activeId, StringComparison.Ordinal))
                ?? EyeCalibrationChoices[0];
        }
        finally
        {
            _applyingCalibrationSelection = false;
        }

        UpdateActiveEyeCalibrationSummary();
        RefreshQuickEyeSetupActions();
    }

    /// <summary>
    /// Spells out what the active calibration measured. Saving otherwise looks like it did nothing:
    /// this is a different system from the Lower/Upper sliders below and never moves them.
    /// </summary>
    private void UpdateActiveEyeCalibrationSummary()
    {
        var active = _eyeCalibrationLibrary.Active;
        if (active is null)
        {
            ActiveEyeCalibrationSummary =
                "Running uncalibrated — openness and gaze pass through unchanged.";
            return;
        }

        ActiveEyeCalibrationSummary =
            $"Left closed {active.Left.OpennessClosed:F2} / relaxed {active.Left.OpennessNeutral:F2} / " +
            $"wide {active.Left.OpennessWide:F2}, center ({active.Left.GazeCenterX:+0.00;-0.00; 0.00}, " +
            $"{active.Left.GazeCenterY:+0.00;-0.00; 0.00}). " +
            $"Right closed {active.Right.OpennessClosed:F2} / relaxed {active.Right.OpennessNeutral:F2} / " +
            $"wide {active.Right.OpennessWide:F2}, center ({active.Right.GazeCenterX:+0.00;-0.00; 0.00}, " +
            $"{active.Right.GazeCenterY:+0.00;-0.00; 0.00}).";
    }

    partial void OnSelectedEyeCalibrationChanged(EyeCalibrationChoice? value)
    {
        RefreshQuickEyeSetupActions();

        // Rebuilding the list assigns this property; only a real user choice should switch profiles.
        if (_applyingCalibrationSelection || value is null)
            return;

        _eyeCalibrationLibrary.Activate(value.Id);
        _eyePipelineManager.LoadEyePostProcessor();
        UpdateActiveEyeCalibrationSummary();

        EyeCalibrationStatus = value.IsDefault
            ? "Using default (uncalibrated) eye mapping."
            : $"Applied \"{value.Name}\" live.";
    }

    [RelayCommand]
    private void DeleteEyeCalibration()
    {
        if (SelectedEyeCalibration is not { IsDefault: false } choice)
            return;

        var wasActive = string.Equals(
            _eyeCalibrationLibrary.ActiveId, choice.Id, StringComparison.Ordinal);

        _eyeCalibrationLibrary.Delete(choice.Id);
        _eyePipelineManager.LoadEyePostProcessor();
        RefreshEyeCalibrationChoices();

        EyeCalibrationStatus = wasActive
            ? $"Deleted \"{choice.Name}\". Now running uncalibrated — pick another or record a new one."
            : $"Deleted \"{choice.Name}\".";
    }

    private bool AreTargetedEyeCamerasRunning()
    {
        var targeted = _eyePipelineManager.Slots.Where(slot => slot.Target is not null).ToArray();
        return targeted.Length > 0 && targeted.All(slot => slot.State == CameraState.Running);
    }

    private void EyeCameraStateChanged(CameraState state)
    {
        _ = state;
        Dispatcher.UIThread.Post(() =>
        {
            if (_disposed) return;
            EyeCalibrationAvailable = AreTargetedEyeCamerasRunning();
            if (!EyeCalibrationAvailable && !EyeCalibrationBusy)
                EyeCalibrationStatus = "Start the eye camera to run Quick Eye Setup.";
        });
    }

    private void OnFilteredEyeResult(EyePipelineEvents.NewFilteredResultEvent update)
    {
        var now = Environment.TickCount64;
        if (now - Interlocked.Read(ref _lastQuickEyeUiTick) < 33)
            return;
        Interlocked.Exchange(ref _lastQuickEyeUiTick, now);

        if (!update.result.TryGetValue("/leftEyeLid", out var left) ||
            !update.result.TryGetValue("/rightEyeLid", out var right))
            return;

        // Copy before returning from the synchronous worker-thread event; the map is reused.
        left = Math.Clamp(left, 0f, 1f);
        right = Math.Clamp(right, 0f, 1f);
        Dispatcher.UIThread.Post(() =>
        {
            if (_disposed) return;
            QuickLeftOpenness = left;
            QuickRightOpenness = right;
        });
    }

    [RelayCommand]
    private Task CaptureRelaxedEyeCalibrationAsync() =>
        CaptureEyeCalibrationStepAsync(EyeCalibrationStep.Relaxed);

    [RelayCommand]
    private Task CaptureClosedEyeCalibrationAsync() =>
        CaptureEyeCalibrationStepAsync(EyeCalibrationStep.Closed);

    [RelayCommand]
    private Task CaptureWideEyeCalibrationAsync() =>
        CaptureEyeCalibrationStepAsync(EyeCalibrationStep.Wide);

    private async Task CaptureEyeCalibrationStepAsync(EyeCalibrationStep step)
    {
        if (!CanCaptureEyeCalibration)
        {
            EyeCalibrationStatus = "Start the eye camera before recording a calibration step.";
            return;
        }

        EyeCalibrationBusy = true;
        EyeCalibrationStatus = step switch
        {
            EyeCalibrationStep.Relaxed => "Relax and look straight ahead for 5 seconds...",
            EyeCalibrationStep.Closed => "Close both eyes gently for 3 seconds...",
            EyeCalibrationStep.Wide => "Open both eyes wide for 4 seconds...",
            _ => "Recording...",
        };

        try
        {
            var count = await _eyeCalibrationRecorder.CaptureStepAsync(
                step, _quickEyeSetupCancellation.Token);
            var complete = count >= EyeCalibrationEstimator.MinimumSamplesPerStep;
            switch (step)
            {
                case EyeCalibrationStep.Relaxed: RelaxedStepComplete = complete; break;
                case EyeCalibrationStep.Closed: ClosedStepComplete = complete; break;
                case EyeCalibrationStep.Wide: WideStepComplete = complete; break;
            }
            EyeCalibrationStatus = complete
                ? $"Recorded {count} samples. Continue to the next step or repeat this one."
                : $"Only {count} valid samples arrived; at least " +
                  $"{EyeCalibrationEstimator.MinimumSamplesPerStep} are required. Repeat this step.";
        }
        catch (OperationCanceledException)
        {
            EyeCalibrationStatus = "Eye calibration stopped.";
        }
        catch (Exception ex)
        {
            EyeCalibrationStatus = $"Could not record eye calibration: {ex.Message}";
        }
        finally
        {
            EyeCalibrationBusy = false;
        }
    }

    [RelayCommand]
    private void SaveEyeCalibration()
    {
        if (!CanSaveEyeCalibration)
            return;

        var estimate = _eyeCalibrationRecorder.Save(_eyeCalibrationLibrary);
        if (!estimate.IsValid)
        {
            EyeCalibrationStatus =
                estimate.Error ?? "Eye calibration samples were invalid; repeat the three steps.";
            return;
        }

        RefreshEyeCalibrationChoices();

        const string saved =
            "Saved and applied live. Neutral now maps to 75% openness. " +
            "(This does not move the Lower/Upper sliders below - they are a separate manual trim.)";
        EyeCalibrationStatus = estimate.Note is { } note ? $"{saved} {note}" : saved;
    }

    private string DisplayNameToInternalName(string displayName)
    {
        if (string.IsNullOrEmpty(displayName))
        {
            return displayName;
        }

        return string.Create(displayName.Length + 1, displayName, (span, state) =>
        {
            span[0] = '/';
            span[1] = char.ToLowerInvariant(state[0]);
            state.AsSpan(1).CopyTo(span.Slice(2));
        });
    }

    private long _lastCalibUiTick;

    private void ExpressionUpdateHandler(ProcessingLoopService.Expressions expressions)
    {
        // Fired from the inference worker thread(s) at the full inference rate (~115 Hz). These slider
        // values are a visual meter only — calibration capture and OSC output go through other paths —
        // so throttle the UI updates to ~30 Hz to avoid pinning the UI thread with per-slider relayout.
        var now = Environment.TickCount64;
        if (now - _lastCalibUiTick < 33)
            return;
        _lastCalibUiTick = now;

        if(expressions.FaceExpression != null)
        {
            var face = expressions.FaceExpression;
            Dispatcher.UIThread.Post(() =>
            {
                ApplyCurrentExpressionValues(face, CheekSettings);
                ApplyCurrentExpressionValues(face, MouthSettings);
                ApplyCurrentExpressionValues(face, JawSettings);
                ApplyCurrentExpressionValues(face, NoseSettings);
                ApplyCurrentExpressionValues(face, TongueSettings);
            });
        }
        if(expressions.EyeExpression != null)
        {
            var eye = expressions.EyeExpression;
            Dispatcher.UIThread.Post(() =>
            {
                ApplyCurrentExpressionValues(eye, EyeSettings);
            });
        }
    }
    private void OnSettingChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (sender is not SliderBindableSetting setting) return;

        if (e.PropertyName is nameof(SliderBindableSetting.Lower))
        {
            _calibrationService.SetExpression(DisplayNameToInternalName(setting.Name) + "Lower", setting.Lower);
        }

        if (e.PropertyName is nameof(SliderBindableSetting.Upper))
        {
            _calibrationService.SetExpression(DisplayNameToInternalName(setting.Name) + "Upper", setting.Upper);
        }
    }

    private void ApplyCurrentExpressionValues(OrderedFloatMap values, IEnumerable<SliderBindableSetting> settings)
    {
        foreach (var setting in settings)
        {
            var settingName = DisplayNameToInternalName(setting.Name);
            if (values.ContainsKey(settingName)) {
                var weight = values[settingName];
                var val = Math.Clamp(
                    weight.Remap(setting.Lower, setting.Upper, setting.Min, setting.Max),
                    setting.Min,
                    setting.Max);
                // Skip the write (and its PropertyChanged + per-slider relayout) when unchanged.
                if (setting.CurrentExpression != val)
                    setting.CurrentExpression = val;
            }
        }
    }

    [RelayCommand]
    public void ResetMinimums()
    {
        _calibrationService.ResetMinimums();
        LoadInitialSettings();
    }

    [RelayCommand]
    public void ResetMaximums()
    {
        _calibrationService.ResetMaximums();
        LoadInitialSettings();
    }

    private void LoadInitialSettings()
    {
        LoadInitialSettings(EyeSettings);
        LoadInitialSettings(CheekSettings);
        LoadInitialSettings(JawSettings);
        LoadInitialSettings(MouthSettings);
        LoadInitialSettings(NoseSettings);
        LoadInitialSettings(TongueSettings);
    }

    private void LoadInitialSettings(IEnumerable<SliderBindableSetting> settings)
    {
        foreach (var setting in settings)
        {
            var legacyVal = _calibrationService.GetExpressionSettings(setting.Name);
            var newVal = _calibrationService.GetNullableExpressionSettings(DisplayNameToInternalName(setting.Name));

            // if we have "new format" parameter in settings, use it. otherwise fall back to legacy (or default values)
            var val = newVal == null ? legacyVal : newVal;
            setting.Lower = val.Lower;
            setting.Upper = val.Upper;
            setting.Min = val.Min;
            setting.Max = val.Max;
        }
    }

    public void Dispose()
    {
        _disposed = true;
        _quickEyeSetupCancellation.Cancel();
        _processingLoopService.ExpressionChangeEvent -= ExpressionUpdateHandler;
        _eyePipelineEventBus.Unsubscribe<EyePipelineEvents.NewFilteredResultEvent>(OnFilteredEyeResult);
        _eyePipelineManager.LeftSlot.StateChanged -= EyeCameraStateChanged;
        _eyePipelineManager.RightSlot.StateChanged -= EyeCameraStateChanged;
        _quickEyeSetupCancellation.Dispose();
        
        foreach (var setting in EyeSettings.Concat(JawSettings).Concat(CheekSettings)
                    .Concat(NoseSettings).Concat(MouthSettings).Concat(TongueSettings))
        {
            setting.PropertyChanged -= OnSettingChanged;
        }
    }
}

/// <summary>One row in the saved-calibration dropdown.</summary>
public sealed record EyeCalibrationChoice(string Id, string Name, bool IsDefault)
{
    public static EyeCalibrationChoice Default { get; } =
        new(EyeCalibrationLibrary.DefaultId, "Default (no calibration)", true);

    public static EyeCalibrationChoice For(EyeCalibrationEntry entry) =>
        new(entry.Id, entry.Name, false);

    public override string ToString() => Name;
}
