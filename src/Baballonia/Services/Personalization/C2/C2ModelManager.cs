using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Baballonia.Services.Inference;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Baballonia.Services.Personalization.C2;

public sealed record C2PreparedCandidate(C2CandidateCorrector Corrector, Func<bool> ContextMatches);
public sealed record C2Selection(string? CandidateDirectory);

/// <summary>Owns C2 independently of the page; only an explicit Keep writes a startup selection.</summary>
public sealed class C2ModelManager : IHostedService, IDisposable
{
    private readonly string _selectionPath;
    private readonly Func<string, bool, Task<C2PreparedCandidate>> _prepare;
    private readonly Func<C2CandidateCorrector?, Func<bool>?, C2CandidateCorrector?> _swap;
    private readonly ILogger<C2ModelManager> _logger;
    private readonly object _sync = new();
    private C2CandidateCorrector? _active;
    private bool _kept;
    private bool _disposed;
    private long _generation;
    private long _trialStarted;

    public C2ModelManager(C2RuntimeContext context, FacePipelineManager face, ILogger<C2ModelManager> logger)
        : this(Path.Combine(C2Contract.Root, "selection.json"), context.PrepareCandidateAsync, face.SwapC2, logger) { }

    /// <summary>Lifecycle seam for tests with real small ONNX fixtures and isolated storage.</summary>
    public C2ModelManager(string selectionPath, Func<string, bool, Task<C2PreparedCandidate>> prepare,
        Func<C2CandidateCorrector?, Func<bool>?, C2CandidateCorrector?> swap, ILogger<C2ModelManager> logger)
    {
        _selectionPath = selectionPath; _prepare = prepare; _swap = swap; _logger = logger;
    }

    public string? CandidateDirectory { get; private set; }
    public string? CandidateName => CandidateDirectory == null ? null : Path.GetFileName(CandidateDirectory);
    public string? Failure => _active?.Failure;
    public bool IsActive => _active is { Failure: null };
    public bool IsKept => IsActive && _kept;
    public bool IsTrial => IsActive && !_kept;
    public double TrialSecondsRemaining => Math.Max(0, 60 - Stopwatch.GetElapsedTime(_trialStarted).TotalSeconds);
    public string? RestoreFailure { get; private set; }

    public async Task ActivateAsync(string directory, bool keep)
    {
        long generation;
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!keep && _kept && IsActive)
                throw new InvalidOperationException("Return to Model C before starting another temporary trial.");
            generation = ++_generation;
        }
        directory = Path.GetFullPath(directory);
        var prepared = await _prepare(directory, keep);
        var published = false;
        try
        {
            lock (_sync)
            {
                // Leaving the page, Return to C, or a later activation cancels pending publication.
                if (_disposed || generation != _generation) throw new OperationCanceledException();
                if (!prepared.ContextMatches() || prepared.Corrector.Failure != null)
                    throw new InvalidOperationException(prepared.Corrector.Failure ??
                        "C2 settings changed while loading. Try again under the recorded settings.");
                if (keep) C2Contract.WriteAtomic(_selectionPath, new C2Selection(directory));
                _swap(prepared.Corrector, prepared.ContextMatches)?.Dispose();
                _active = prepared.Corrector;
                _kept = keep;
                CandidateDirectory = directory;
                RestoreFailure = null;
                _trialStarted = Stopwatch.GetTimestamp();
                published = true;
            }
        }
        finally { if (!published) prepared.Corrector.Dispose(); }
    }

    public void EndTrial()
    {
        lock (_sync)
        {
            ++_generation;
            if (_kept) return;
            Unload();
        }
    }

    public void ReturnToModelC()
    {
        lock (_sync)
        {
            // Persist first: a failed write must not claim the startup choice was cleared.
            C2Contract.WriteAtomic(_selectionPath, new C2Selection(null));
            ++_generation;
            Unload();
            RestoreFailure = null;
        }
    }

    private void Unload()
    {
        _swap(null, null)?.Dispose();
        _active = null;
        _kept = false;
        CandidateDirectory = null;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        // Registered after PersonalModelStartupService, so C and its feature runner are ready.
        if (!File.Exists(_selectionPath)) return;
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            var selection = JsonSerializer.Deserialize<C2Selection>(await File.ReadAllTextAsync(_selectionPath, cancellationToken));
            if (!string.IsNullOrWhiteSpace(selection?.CandidateDirectory))
            {
                await ActivateAsync(selection.CandidateDirectory, keep: true);
                _logger.LogInformation("Restored kept C2 candidate {Candidate}", CandidateDirectory);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { throw; }
        catch (Exception ex)
        {
            RestoreFailure = "Saved C2 could not load; using the current base model. " + ex.Message;
            _logger.LogWarning(ex, "Could not restore kept C2 candidate; base model retained");
        }
    }

    public Task StopAsync(CancellationToken cancellationToken) { Dispose(); return Task.CompletedTask; }
    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            ++_generation;
            Unload(); // Shutdown releases runtime ownership but preserves the saved choice.
        }
    }
}
