using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services.Personalization.Audio;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

[TestClass]
public class AudioInputDeviceResolverTest
{
    private static readonly AudioInputDevice BuiltIn = new("builtin", "Laptop Microphone");
    private static readonly AudioInputDevice Headset = new("headset-v2", "VR Headset Microphone");

    [TestMethod]
    public void ResolvesExactPersistedIdBeforeAutomaticChoice()
    {
        var result = AudioInputDeviceResolver.Resolve(
            [BuiltIn, Headset],
            Headset.Id,
            Headset.DisplayName);

        Assert.AreEqual(Headset, result.Device);
        Assert.AreEqual(AudioInputResolutionKind.Exact, result.Kind);
        Assert.IsFalse(result.UsedFallback);
    }

    [TestMethod]
    public void RecoversDriverIdChangeByExactDeviceName()
    {
        var result = AudioInputDeviceResolver.Resolve(
            [BuiltIn, Headset],
            "headset-old-driver-id",
            Headset.DisplayName);

        Assert.AreEqual(Headset, result.Device);
        Assert.AreEqual(AudioInputResolutionKind.RecoveredByName, result.Kind);
        Assert.IsTrue(result.UsedFallback);
        StringAssert.Contains(result.Message, "matched");
    }

    [TestMethod]
    public void MissingPreferenceUsesFirstAvailableWhenBackendDoesNotKnowTheDefault()
    {
        var fallback = AudioInputDeviceResolver.Resolve(
            [Headset, BuiltIn],
            "unplugged-usb-mic",
            "USB Microphone");

        Assert.AreEqual(Headset, fallback.Device);
        Assert.AreEqual(AudioInputResolutionKind.Automatic, fallback.Kind);
        Assert.IsTrue(fallback.UsedFallback);
        StringAssert.Contains(fallback.Message, "first available input");

        var unavailable = AudioInputDeviceResolver.Resolve([], "gone", "Gone");
        Assert.IsNull(unavailable.Device);
        Assert.AreEqual(AudioInputResolutionKind.Unavailable, unavailable.Kind);
    }

    [TestMethod]
    public void AutomaticChoiceUsesAPlatformDefaultOnlyWhenBackendMarksOneAuthoritatively()
    {
        var knownDefault = new AudioInputDevice("known-default", "Known Default", true);

        var result = AudioInputDeviceResolver.Resolve(
            [Headset, knownDefault],
            preferredId: null,
            preferredName: null);

        Assert.AreEqual(knownDefault, result.Device);
        Assert.AreEqual(AudioInputResolutionKind.Automatic, result.Kind);
        Assert.IsFalse(result.UsedFallback);
        StringAssert.Contains(result.Message, "platform default");
    }
}

[TestClass]
public class AudioAssistDeviceSelectionTest
{
    [TestMethod]
    public void ZeroStrengthSurvivesAServiceRestart()
    {
        var settings = new MemorySettings();
        settings.SaveSetting(AudioAssistService.StrengthSetting, 0f);

        using (var first = CreateService(new FakeCatalog(), new FakeFactory(), settings, _ => { }))
        {
            Assert.AreEqual(0f, first.Strength);
            first.Strength = 0f;
        }

        using var restarted = CreateService(new FakeCatalog(), new FakeFactory(), settings, _ => { });
        Assert.AreEqual(0f, restarted.Strength,
            "zero is exact passthrough, not an alias for the 50% default");
    }

    [TestMethod]
    public async Task HostStartupAppliesPersistedEnabledStateWithoutOpeningSettingsPage()
    {
        var settings = new MemorySettings();
        settings.SaveSetting(AudioAssistService.EnabledSetting, true);
        var device = new AudioInputDevice("startup-mic", "Startup Mic");
        var factory = new FakeFactory();
        var scheduler = new ManualRetryScheduler();
        IExpressionEnhancer? installed = null;
        using var audio = CreateService(
            new FakeCatalog(device),
            factory,
            settings,
            enhancer => installed = enhancer);
        using var startup = new AudioAssistStartupService(audio, scheduler);

        await startup.StartAsync(CancellationToken.None);

        Assert.IsTrue(audio.IsActive);
        Assert.AreEqual(1, factory.Created.Count);
        Assert.IsNotNull(installed);

        await startup.StopAsync(CancellationToken.None);
    }

    [TestMethod]
    public async Task BackgroundLifecycleReconnectsStoppedInputWithoutOpeningSettingsPage()
    {
        var settings = new MemorySettings();
        settings.SaveSetting(AudioAssistService.EnabledSetting, true);
        var catalog = new FakeCatalog(new AudioInputDevice("headset", "VR Headset Microphone"));
        var factory = new FakeFactory();
        var scheduler = new ManualRetryScheduler();
        using var audio = CreateService(catalog, factory, settings, _ => { });
        using var startup = new AudioAssistStartupService(audio, scheduler);
        await startup.StartAsync(CancellationToken.None);

        var first = factory.Created[0];
        var replacementCreated = factory.ExpectNextCreation();
        first.SimulateUnexpectedStop("VR headset was disconnected.");

        scheduler.Advance();
        var replacement = await replacementCreated.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.IsTrue(audio.IsActive);
        Assert.AreEqual(2, factory.Created.Count);
        Assert.AreSame(replacement, factory.Created[1]);
        Assert.AreEqual(1, first.StopCalls);
        Assert.AreEqual(1, first.DisposeCalls);

        await startup.StopAsync(CancellationToken.None);
    }

    [TestMethod]
    public void MissingPersistedDeviceUsesAutomaticFallbackWithoutForgettingPreference()
    {
        var settings = new MemorySettings();
        settings.SaveSetting(AudioAssistService.EnabledSetting, true);
        settings.SaveSetting(AudioAssistService.DeviceIdSetting, "usb-mic-that-is-unplugged");
        settings.SaveSetting(AudioAssistService.DeviceNameSetting, "USB Microphone");

        var fallback = new AudioInputDevice("array", "Laptop Array");
        var catalog = new FakeCatalog(fallback);
        var factory = new FakeFactory();
        IExpressionEnhancer? installed = null;
        using var service = CreateService(catalog, factory, settings, enhancer => installed = enhancer);

        service.Apply();

        Assert.IsTrue(service.IsActive);
        Assert.AreEqual(fallback, service.ActiveDevice);
        Assert.IsTrue(service.IsUsingDeviceFallback);
        Assert.AreEqual("usb-mic-that-is-unplugged", service.PreferredDeviceId,
            "temporary fallback must not erase the user's preferred USB microphone");
        Assert.IsNotNull(installed);
        StringAssert.Contains(service.StatusMessage, "selected microphone is unavailable");
        StringAssert.Contains(service.StatusMessage, "first available input");
    }

    [TestMethod]
    public void SelectingDevicePersistsAndRestartsOnlyTheAudioSource()
    {
        var settings = new MemorySettings();
        settings.SaveSetting(AudioAssistService.EnabledSetting, true);

        var builtIn = new AudioInputDevice("built-in", "Built-in", true);
        var headset = new AudioInputDevice("headset", "Headset");
        var catalog = new FakeCatalog(builtIn, headset);
        var factory = new FakeFactory();
        var enhancerAssignments = new List<IExpressionEnhancer?>();
        using var service = CreateService(
            catalog,
            factory,
            settings,
            enhancer => enhancerAssignments.Add(enhancer));

        service.Apply();
        var first = factory.Created[0];

        var restarted = service.SelectInputDevice(headset.Id);

        Assert.IsTrue(restarted);
        Assert.AreEqual(2, factory.Created.Count);
        Assert.AreEqual(1, first.StopCalls);
        Assert.AreEqual(1, first.DisposeCalls);
        Assert.AreEqual(headset, service.ActiveDevice);
        Assert.AreEqual(headset.Id, settings.ReadSetting<string>(AudioAssistService.DeviceIdSetting));
        Assert.AreEqual(headset.DisplayName,
            settings.ReadSetting<string>(AudioAssistService.DeviceNameSetting));
        Assert.IsTrue(enhancerAssignments.Exists(enhancer => enhancer == null),
            "the old audio enhancer must be detached before capture is reopened");
        Assert.IsNotNull(enhancerAssignments[^1]);
    }

    [TestMethod]
    public void LiveMeterVoiceAndUnexpectedStopStatusAreAvailableToConsumers()
    {
        var settings = new MemorySettings();
        settings.SaveSetting(AudioAssistService.EnabledSetting, true);

        var device = new AudioInputDevice("mic", "Test Mic", true);
        var catalog = new FakeCatalog(device);
        var factory = new FakeFactory
        {
            NextFeatures = new AudioFeatures(
                Rms: 0.1f,
                NoiseFloor: 0.01f,
                IsVoiced: true,
                SpeechEnergy: 0.7f,
                PitchHz: 140f,
                PitchClarity: 0.9f,
                Onset: 0.2f,
                TimestampTicks: DateTime.UtcNow.Ticks)
        };
        using var service = CreateService(catalog, factory, settings, _ => { });

        service.Apply();

        Assert.IsTrue(service.IsVoiceDetected);
        Assert.AreEqual(2f / 3f, service.InputLevel, 0.01f,
            "-20 dBFS should sit two-thirds up a -60..0 dBFS meter");

        factory.Created[0].SimulateUnexpectedStop("Test Mic was unplugged.");

        Assert.IsFalse(service.IsActive);
        StringAssert.Contains(service.StatusMessage, "unplugged");
        StringAssert.Contains(service.StatusMessage, "Visual tracking is unaffected");
    }

    [TestMethod]
    public void CaptureFailureNeverInstallsTheVisualPipelineEnhancer()
    {
        var settings = new MemorySettings();
        settings.SaveSetting(AudioAssistService.EnabledSetting, true);

        var catalog = new FakeCatalog(new AudioInputDevice("busy", "Busy Mic", true));
        var factory = new FakeFactory { CanStart = false };
        IExpressionEnhancer? installed = null;
        using var service = CreateService(catalog, factory, settings, enhancer => installed = enhancer);

        service.Apply();

        Assert.IsFalse(service.IsActive);
        Assert.IsNull(installed);
        StringAssert.Contains(service.StatusMessage, "Visual tracking is unaffected");
    }

    private static AudioAssistService CreateService(
        IAudioInputDeviceCatalog catalog,
        IAudioFeatureSourceFactory factory,
        ILocalSettingsService settings,
        Action<IExpressionEnhancer?> install) =>
        new(catalog, factory, settings, install, NullLogger<AudioAssistService>.Instance);

    private sealed class FakeCatalog(params AudioInputDevice[] devices) : IAudioInputDeviceCatalog
    {
        public IReadOnlyList<AudioInputDevice> Devices { get; set; } = devices;
        public IReadOnlyList<AudioInputDevice> GetDevices() => Devices;
    }

    private sealed class FakeFactory : IAudioFeatureSourceFactory
    {
        private readonly object _gate = new();
        private TaskCompletionSource<FakeSource>? _nextCreation;

        public bool CanStart { get; init; } = true;
        public AudioFeatures NextFeatures { get; init; } = AudioFeatures.Silent();
        public List<FakeSource> Created { get; } = [];

        public Task<FakeSource> ExpectNextCreation()
        {
            lock (_gate)
            {
                if (_nextCreation != null)
                    throw new InvalidOperationException("A creation expectation is already pending.");

                _nextCreation = new TaskCompletionSource<FakeSource>(
                    TaskCreationOptions.RunContinuationsAsynchronously);
                return _nextCreation.Task;
            }
        }

        public IAudioFeatureSource Create(AudioInputDevice device)
        {
            var source = new FakeSource(device, CanStart, NextFeatures);
            TaskCompletionSource<FakeSource>? signal;
            lock (_gate)
            {
                Created.Add(source);
                signal = _nextCreation;
                _nextCreation = null;
            }
            signal?.TrySetResult(source);
            return source;
        }
    }

    private sealed class ManualRetryScheduler : IAudioAssistRetryScheduler
    {
        private readonly Channel<bool> _attempts = Channel.CreateUnbounded<bool>();

        public async ValueTask WaitForNextAttemptAsync(CancellationToken cancellationToken) =>
            await _attempts.Reader.ReadAsync(cancellationToken);

        public void Advance() => _attempts.Writer.TryWrite(true);
    }

    private sealed class FakeSource(
        AudioInputDevice device,
        bool canStart,
        AudioFeatures features) : IAudioFeatureSource
    {
        private AudioFeatures _latest = features;

        public bool IsRunning { get; private set; }
        public string StatusMessage { get; private set; } = "";
        public AudioFeatures Latest => _latest;
        public int StopCalls { get; private set; }
        public int DisposeCalls { get; private set; }

        public bool Start()
        {
            IsRunning = canStart;
            if (!canStart)
                StatusMessage = $"{device.DisplayName} could not be opened.";
            return IsRunning;
        }

        public void Stop()
        {
            StopCalls++;
            IsRunning = false;
            _latest = AudioFeatures.Silent();
        }

        public void SimulateUnexpectedStop(string message)
        {
            IsRunning = false;
            StatusMessage = message;
            _latest = AudioFeatures.Silent();
        }

        public void Dispose() => DisposeCalls++;
    }

    private sealed class MemorySettings : ILocalSettingsService
    {
        private readonly Dictionary<string, object?> _values = [];

        public T ReadSetting<T>(string key, T? defaultValue = default, bool forceLocal = false)
        {
            return _values.TryGetValue(key, out var value) && value is T typed
                ? typed
                : defaultValue!;
        }

        public void SaveSetting<T>(string key, T value, bool forceLocal = false) =>
            _values[key] = value;

        public void Save(object target) { }
        public void Load(object target) { }
        public void ForceSave() { }
    }
}
