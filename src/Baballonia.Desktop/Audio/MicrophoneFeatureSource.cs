#if WINDOWS
using System;
using System.Threading;
using Baballonia.Services.Personalization.Audio;
using Microsoft.Extensions.Logging;
using NAudio.Wave;

namespace Baballonia.Desktop.Audio;

/// <summary>
/// Captures the default microphone and analyses it into <see cref="AudioFeatures"/>.
///
/// Everything here happens on NAudio's own capture thread. The inference tick only ever reads the
/// last published <see cref="AudioFeatures"/>, which is a small immutable struct swapped under a
/// lock held for the length of an assignment - so audio can stall, glitch or die without the face
/// pipeline noticing.
///
/// No audio is stored. Samples are accumulated into a fixed window, analysed, and overwritten. There
/// is no path in this class that writes a sample anywhere, which is a property worth keeping: this
/// is a face tracker with a microphone open, and it should stay obviously incapable of being
/// anything else.
/// </summary>
public sealed class MicrophoneFeatureSource : IAudioFeatureSource
{
    private const int SampleRate = AudioFeatureAnalyzer.DefaultSampleRate;

    /// <summary>20 ms windows: fine enough to catch a syllable, coarse enough to be cheap.</summary>
    private const int WindowSamples = SampleRate / 50;

    private readonly ILogger _logger;
    private readonly AudioFeatureAnalyzer _analyzer = new(SampleRate);
    private readonly float[] _window = new float[WindowSamples];
    private readonly object _gate = new();

    private WaveInEvent? _capture;
    private int _filled;
    private AudioFeatures _latest = AudioFeatures.Silent();
    private volatile bool _running;
    private string _status = "";

    public MicrophoneFeatureSource(ILogger logger) => _logger = logger;

    public bool IsRunning => _running;

    public string StatusMessage => _status;

    public AudioFeatures Latest
    {
        get { lock (_gate) return _latest; }
    }

    public bool Start()
    {
        if (_running)
            return true;

        try
        {
            if (WaveInEvent.DeviceCount == 0)
            {
                _status = "No microphone was found.";
                return false;
            }

            _analyzer.Reset();
            _filled = 0;

            _capture = new WaveInEvent
            {
                WaveFormat = new WaveFormat(SampleRate, 16, 1),
                // Small buffers keep the added latency well under the camera's own, so the sync
                // offset has something sensible to work with.
                BufferMilliseconds = 20,
                NumberOfBuffers = 3
            };

            _capture.DataAvailable += OnDataAvailable;
            _capture.RecordingStopped += OnRecordingStopped;
            _capture.StartRecording();

            _running = true;
            _status = "";
            _logger.LogInformation("Microphone capture started ({Rate} Hz mono)", SampleRate);
            return true;
        }
        catch (Exception ex)
        {
            // Denied permission, a device in exclusive use, a driver problem - all the same from
            // here: the enhancement is unavailable and tracking carries on.
            _logger.LogWarning(ex, "Could not start microphone capture");
            _status = "The microphone could not be opened.";
            Cleanup();
            return false;
        }
    }

    public void Stop()
    {
        if (!_running)
            return;

        try
        {
            _capture?.StopRecording();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Microphone capture did not stop cleanly");
        }
        finally
        {
            Cleanup();
        }
    }

    private void OnRecordingStopped(object? sender, StoppedEventArgs e)
    {
        if (e.Exception != null)
        {
            // Device unplugged mid-session, typically.
            _logger.LogWarning(e.Exception, "Microphone capture stopped unexpectedly");
            _status = "The microphone stopped unexpectedly.";
        }

        _running = false;

        lock (_gate)
            _latest = AudioFeatures.Silent();
    }

    private void OnDataAvailable(object? sender, WaveInEventArgs e)
    {
        try
        {
            // 16-bit PCM to float, accumulating into fixed windows. Any tail shorter than a window
            // waits for the next buffer rather than being analysed short.
            for (var offset = 0; offset + 1 < e.BytesRecorded; offset += 2)
            {
                var sample = (short)(e.Buffer[offset] | (e.Buffer[offset + 1] << 8));
                _window[_filled++] = sample / 32768f;

                if (_filled < _window.Length)
                    continue;

                var features = _analyzer.Analyze(_window, DateTime.UtcNow.Ticks);
                _filled = 0;

                lock (_gate)
                    _latest = features;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Audio analysis failed; stopping capture");
            _status = "Audio analysis failed.";
            _running = false;
        }
    }

    private void Cleanup()
    {
        _running = false;

        if (_capture != null)
        {
            _capture.DataAvailable -= OnDataAvailable;
            _capture.RecordingStopped -= OnRecordingStopped;
            _capture.Dispose();
            _capture = null;
        }

        lock (_gate)
            _latest = AudioFeatures.Silent();
    }

    public void Dispose()
    {
        Stop();
        Cleanup();
    }
}
#endif
