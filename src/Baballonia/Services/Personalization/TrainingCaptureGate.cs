using System;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Mutual exclusion between activities that may command the avatar and activities that write
/// training frames. A preview must never rely on a disabled button to stay out of the corpus: the
/// recorder and preview both acquire this same backend gate, so neither can race the other.
/// </summary>
public sealed class TrainingCaptureGate
{
    private readonly object _gate = new();
    private object? _ownerToken;
    private string? _activity;

    public string? ActiveActivity
    {
        get { lock (_gate) return _activity; }
    }

    /// <summary>Attempts to reserve training capture for one named activity.</summary>
    public bool TryEnter(string activity, out IDisposable? lease, out string message)
    {
        if (string.IsNullOrWhiteSpace(activity))
            throw new ArgumentException("An activity name is required.", nameof(activity));

        lock (_gate)
        {
            if (_ownerToken != null)
            {
                lease = null;
                message = $"Cannot start while {_activity ?? "another training-data activity"} is active.";
                return false;
            }

            var token = new object();
            _ownerToken = token;
            _activity = activity;
            lease = new GateLease(this, token);
            message = "";
            return true;
        }
    }

    private void Exit(object token)
    {
        lock (_gate)
        {
            if (!ReferenceEquals(_ownerToken, token))
                return;

            _ownerToken = null;
            _activity = null;
        }
    }

    private sealed class GateLease(TrainingCaptureGate owner, object token) : IDisposable
    {
        private TrainingCaptureGate? _owner = owner;

        public void Dispose()
        {
            var current = System.Threading.Interlocked.Exchange(ref _owner, null);
            current?.Exit(token);
        }
    }
}
