using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using Baballonia.Contracts;
using Baballonia.Services.Calibration;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.C2;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Baballonia.ViewModels.SplitViewPane;

/// <summary>One task, one review decision, persistent accepted progress. No existing model writes.</summary>
public partial class C2ViewModel : ObservableObject, IDisposable
{
    private readonly DatasetRecorderService _recorder;
    private readonly PersonalModelManager _models;
    private readonly PersonalTrainingService _trainer;
    private readonly TrainingCaptureGate _gate;
    private readonly C2RuntimeContext _context;
    private readonly C2ModelManager _c2Models;
    private readonly IVrCalibrationPresenter? _presenter;
    // A development profile can inspect the home's recordings read-only. Legacy recording and
    // training paths still point at the development profile, never this source-only override.
    private readonly C2Workspace _workspace = new(C2Contract.Root,
        !string.IsNullOrEmpty(Utils.Profile) && !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("C2_LEGACY_DATASET_ROOT"))
            ? Environment.GetEnvironmentVariable("C2_LEGACY_DATASET_ROOT")! : PersonalizationPaths.DatasetRoot);
    private readonly CueStateSource _cue = new() { Source = "c2_instruction" };
    private readonly DispatcherTimer _timer;
    private CancellationTokenSource? _cancellation;
    private bool _disposed;
    private bool _finishing;
    private bool _overlay;
    private long _started;
    private string? _currentId;
    private string? _contractPath;
    private string _origin = "";
    private C2Task? _recordingTask;
    private string _recordingRole = "practice";
    private Func<bool>? _captureContextMatches;

    [ObservableProperty] private string _usingNow = "Checking active model…";
    [ObservableProperty] private string _status = "Examples needed — C2 is experimental and has not been validated.";
    [ObservableProperty] private string _nextAction = "Refresh recordings to assess what can be reused.";
    [ObservableProperty] private string _instruction = "Choose a short task. Read its instruction, then press Record this attempt when ready.";
    [ObservableProperty] private string _coverage = "";
    [ObservableProperty] private string _details = "";
    [ObservableProperty] private string _recordingState = "Not recording";
    [ObservableProperty] private bool _isBusy;
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private bool _checkOnly;
    [ObservableProperty] private bool _useHeadsetInstructions = true;
    [ObservableProperty] private C2Task _selectedTask = C2Task.All[0];
    [ObservableProperty] private C2Recording? _selectedRecording;
    [ObservableProperty] private string _recordingExplanation = "Select a recording to see its role and limitations.";
    [ObservableProperty] private string _candidateStatus = "No candidate has been trained. Better / Similar / Worse: Not checked.";
    [ObservableProperty] private string? _selectedCandidate;
    [ObservableProperty] private bool _isTrial;
    [ObservableProperty] private bool _isKept;

    public bool HasActiveC2 => _c2Models.IsActive;
    public string? ActiveC2Name => _c2Models.CandidateName;

    public IReadOnlyList<C2Task> Tasks => C2Task.All;
    public ObservableCollection<C2Recording> Recordings { get; } = [];
    public ObservableCollection<string> Candidates { get; } = [];

    public C2ViewModel(DatasetRecorderService recorder, PersonalModelManager models,
        PersonalTrainingService trainer,
        TrainingCaptureGate gate, C2RuntimeContext context, C2ModelManager c2Models, IVrCalibrationPresenter? presenter = null)
    {
        _recorder = recorder; _models = models; _trainer = trainer;
        _gate = gate; _presenter = presenter;
        _context = context; _c2Models = c2Models;
        Instruction = SelectedTask.Instruction;
        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
        _timer.Tick += async (_, _) => await TickAsync();
        _timer.Start();
        _ = RefreshAsync();
    }

    partial void OnSelectedTaskChanged(C2Task value)
    {
        if (!IsRecording) Instruction = value.Instruction;
    }
    partial void OnSelectedRecordingChanged(C2Recording? value)
    {
        RecordingExplanation = value == null ? "Select a recording to review." :
            $"{value.State}: {value.Reason}\nRole: {value.Role}; task: {value.TaskId}; saved frames: {value.Frames}.";
        if (value is { Legacy: false })
            RecordingExplanation += "\nRecorded instruction: " + Tasks.FirstOrDefault(t => t.Id == value.TaskId)?.Instruction;
    }
    partial void OnCheckOnlyChanged(bool value) => NextAction = _workspace.NextAction(Recordings.ToArray(), value ? "check" : "practice");

    [RelayCommand]
    private async Task RefreshAsync()
    {
        if (IsBusy || IsRecording || _disposed) return;
        IsBusy = true;
        try
        {
            var inventory = await Task.Run(_workspace.Inventory);
            if (_disposed) return;
            Recordings.Clear();
            foreach (var row in inventory) Recordings.Add(row);
            var accepted = inventory.Where(r => !r.Legacy && r.State == "Accepted").ToArray();
            Coverage = string.Join("\n", Tasks.Select(t => $"{t.Name}: " +
                (accepted.Any(r => r.TaskId == t.Id && r.Role == "practice") ? "practice accepted" : "practice missing") +
                (accepted.Any(r => r.TaskId == t.Id && r.Role == "check") ? "; check accepted" : "; check not recorded") +
                (t.Optional ? " (optional)" : "")));
            NextAction = _workspace.NextAction(inventory, CheckOnly ? "check" : "practice");
            var candidates = Path.Combine(_workspace.Root, "Candidates");
            var summaries = Directory.Exists(candidates) ? Directory.GetFiles(candidates, "summary.json", SearchOption.AllDirectories) : [];
            Candidates.Clear();
            foreach (var summary in summaries.Order()) Candidates.Add(Path.GetFileName(Path.GetDirectoryName(summary))!);
            if (SelectedCandidate == null || !Candidates.Contains(SelectedCandidate))
                SelectedCandidate = Candidates.Contains(_c2Models.CandidateName!) ? _c2Models.CandidateName : Candidates.LastOrDefault();
            CandidateStatus = _c2Models.IsKept ? $"Keeping C2 {_c2Models.CandidateName}. Saved for use across pages and restarts." :
                summaries.Length == 0 ? "No candidate has been trained. Better / Similar / Worse: Not checked." :
                $"{summaries.Length} saved candidate(s). Select one to compare, try, or keep using. Latest report under Details.";
            if (summaries.Length > 0) Details = await File.ReadAllTextAsync(summaries.Order().Last());
        }
        catch (Exception ex) { Status = "Could not assess recordings: " + ex.Message; }
        finally { IsBusy = false; }
    }

    [RelayCommand]
    private void NewWearingSession()
    {
        if (IsRecording || IsBusy) return;
        _origin = _workspace.GetOrCreateOrigin(true);
        Status = "New wearing session saved. Use this only after removing and putting the headset back on. Stop/Start and app restart keep the same session.";
    }

    private Task<string> PrepareContractAsync() => _context.PrepareContractAsync();

    [RelayCommand]
    private async Task RecordAsync()
    {
        if (IsBusy || IsRecording || _disposed) return;
        IsBusy = true;
        try
        {
            if (!_recorder.HasRecentSourceFrame)
                throw new InvalidOperationException("Start the face camera on Home and wait for the preview to move, then record this attempt.");
            _c2Models.EndTrial();
            _contractPath = await PrepareContractAsync();
            if (_disposed) return;
            var contract = JsonSerializer.Deserialize<C2ReferenceContract>(await File.ReadAllTextAsync(_contractPath))!;
            _captureContextMatches = ContextGuard(contract);
            _origin = _workspace.GetOrCreateOrigin();
            _recordingRole = CheckOnly ? "check" : "practice";
            var sameOrigin = _workspace.Inventory().Where(r => !r.Legacy && r.OriginId == _origin);
            if (sameOrigin.Any(r => r.Role != _recordingRole))
                throw new InvalidOperationException("Practice and check must come from different wearing sessions. Reseat the headset, then press I have reseated the headset.");
            _recordingTask = SelectedTask;
            var context = new C2CaptureContext(_origin, _recordingRole, _recordingTask.Id, _contractPath,
                C2Contract.Hash(_contractPath), 1280, Stopwatch.Frequency);
            _currentId = _recorder.StartSession(SessionType.Guided, notes: "C2 " + _recordingTask.Name,
                requireRecentSourceFrame: true, datasetRoot: _workspace.RecordingsRoot, c2: context, cueSource: _cue);
            _started = Stopwatch.GetTimestamp();
            _cue.Clear();
            IsRecording = true;
            Instruction = _recordingTask.Instruction;
            Status = "Prepare, then hold when the countdown ends. You will review this attempt before it teaches C2.";
            if (UseHeadsetInstructions && _presenter?.IsAvailable == true)
                _overlay = _presenter.Begin("C2 — " + _recordingTask.Name).Started;
        }
        catch (Exception ex) { Status = ex.Message; }
        finally { IsBusy = false; }
    }

    private async Task TickAsync()
    {
        if (_disposed) return;
        var wasKept = IsKept;
        IsTrial = _c2Models.IsTrial;
        IsKept = _c2Models.IsKept;
        if (_c2Models.Failure != null)
        {
            CandidateStatus = "C2 stopped; using your base model. " + _c2Models.Failure;
        }
        else if (IsTrial && _c2Models.TrialSecondsRemaining <= 0)
        {
            _c2Models.EndTrial();
            IsTrial = false;
            CandidateStatus = "Temporary trial ended. Select Keep using this C2 to save it for everyday use.";
        }
        else if (IsTrial)
            CandidateStatus = $"TEMPORARY C2 trial — {_c2Models.TrialSecondsRemaining:0}s left. Like it? Choose Keep using this C2.";
        else if (IsKept && !wasKept)
            CandidateStatus = $"Keeping C2 {_c2Models.CandidateName}. Stays active when you leave this page and loads again on restart.";
        else if (_c2Models.RestoreFailure != null) CandidateStatus = _c2Models.RestoreFailure;
        UsingNow = _models.IsActive ? $"Using now: {_models.LoadedMetadata?.DisplayName} — {Path.GetFileName(Path.GetDirectoryName(_models.ActiveModelPath))}. C2 collection does not change it."
            : "Using now: Stock" + (_models.FallbackReason == null ? "" : ". " + _models.FallbackReason);
        if (IsTrial) UsingNow = "Using now: C2 temporary trial, with your working C retained for immediate fallback.";
        if (IsKept) UsingNow = $"Using now: C2 {_c2Models.CandidateName} — saved for everyday use, with Model C underneath.";
        if (!IsRecording || _finishing || _recordingTask == null) return;
        var elapsed = Stopwatch.GetElapsedTime(_started).TotalSeconds;
        if (!_recorder.HasRecentSourceFrame || (_overlay && _presenter?.IsHealthy != true) ||
            _captureContextMatches?.Invoke() != true)
        {
            await FinishAsync(false, "Capture, instructions, or model/output settings changed. This attempt is excluded; repeat it under stable settings.");
            return;
        }
        var phase = elapsed < 3 ? "prepare" : elapsed < 3.75 ? "settling" : "hold";
        var remaining = phase == "prepare" ? 3 - elapsed : Math.Max(0, 3.75 + _recordingTask.Seconds - elapsed);
        RecordingState = phase == "hold" ? $"RECORDING — hold steady ({remaining:0}s); {_recorder.FramesWritten} saved frames" :
            phase == "prepare" ? $"Get ready: {remaining:0}s" : "Settle into the pose — these frames are unlabelled";
        _cue.SetPhase(new CuePhase(_recordingTask.Id, phase, _recordingTask.Dims,
            _recordingTask.Target(), _recordingTask.Target(), 1, Stopwatch.GetTimestamp(), 0, 0, 0));
        if (_overlay)
        {
            _presenter!.Present(new VrCalibrationFrame(_recordingTask.Name, Instruction + "  " + RecordingState,
                phase == "hold" ? VrCalibrationPhase.Hold : VrCalibrationPhase.Preparing, 0,
                CountdownSeconds: remaining, AllowCancel: true).Quantize());
            if (_presenter.ConsumeAction() == VrCalibrationAction.Cancel)
            {
                await FinishAsync(false, "Paused. Completed accepted attempts are saved; repeat this attempt when ready.");
                return;
            }
        }
        if (elapsed >= 3.75 + _recordingTask.Seconds)
            await FinishAsync(true, "Relax. Review the saved attempt. Accept only if the instruction matched what you actually did; otherwise retry or skip.");
    }

    private async Task FinishAsync(bool completed, string message)
    {
        if (_finishing || !IsRecording) return;
        _finishing = true;
        _cue.Clear();
        try
        {
            var summary = await _recorder.StopSessionAsync();
            if (_overlay) _presenter?.End();
            _overlay = false;
            IsRecording = false;
            RecordingState = "Not recording — saved attempts await review";
            if (!completed && summary != null)
                _workspace.Review(new C2Recording(summary.Directory, summary.SessionId, _recordingTask!.Id,
                    _recordingRole, _origin, "Needs review", message, summary.FrameCount, false), false, message);
            Status = message;
            await RefreshAsync();
            SelectedRecording = Recordings.FirstOrDefault(r => r.Id == _currentId);
        }
        catch (Exception ex) { IsRecording = false; Status = "Recording could not finalize: " + ex.Message; }
        finally
        {
            if (_overlay) _presenter?.End();
            _overlay = false;
            _captureContextMatches = null;
            _finishing = false;
        }
    }

    [RelayCommand]
    private Task PauseAsync() => FinishAsync(false, "Paused. Accepted work is saved. This unfinished attempt is excluded; Record this attempt resumes with a fresh retry.");

    [RelayCommand]
    private async Task AcceptAsync()
    {
        if (IsBusy || IsRecording || SelectedRecording == null) return;
        var recording = SelectedRecording;
        IsBusy = true;
        try
        {
            await Task.Run(() => _workspace.Review(recording, true, recording.Legacy
                ? "User selected for local replay/preservation; no new human labels asserted."
                : "User confirmed the displayed instruction and completed steady hold. Values are approximate; uncued channels unknown."));
            Status = recording.Legacy ? "Existing recording selected for reuse as replay/preservation. No all-zero labels imported." : "Accepted and saved. You can continue later.";
            var next = Tasks.FirstOrDefault(t => !t.Optional && !Recordings.Any(r => r.TaskId == t.Id &&
                r.Role == (CheckOnly ? "check" : "practice") && (r.State == "Accepted" || r.Id == recording.Id)));
            if (next != null) SelectedTask = next;
        }
        catch (Exception ex) { Status = ex.Message; }
        finally { IsBusy = false; }
        await RefreshAsync();
    }

    [RelayCommand]
    private async Task ExcludeAsync()
    {
        if (IsBusy || IsRecording || SelectedRecording == null) return;
        var recording = SelectedRecording;
        IsBusy = true;
        try
        {
            await Task.Run(() => _workspace.Review(recording, false, "Missed/not shown correctly/skipped by user. Original samples retained."));
            Status = "Excluded; source recording retained. Record this task again when ready.";
        }
        catch (Exception ex) { Status = "Could not save review: " + ex.Message; }
        finally { IsBusy = false; }
        await RefreshAsync();
    }

    [RelayCommand]
    private async Task TrainAsync()
    {
        if (IsBusy || IsRecording || _disposed) return;
        if (!_gate.TryEnter("C2 preparation and training", out var lease, out var reason)) { Status = reason; return; }
        IsBusy = true;
        _cancellation = new CancellationTokenSource();
        try
        {
            _c2Models.EndTrial();
            var inventory = await Task.Run(_workspace.Inventory);
            if (!inventory.Any(r => !r.Legacy && r.Role == "practice" && r.TaskId == "rest" && r.State == "Accepted") ||
                !inventory.Any(r => !r.Legacy && r.Role == "practice" && r.TaskId is "jaw-small" or "jaw-comfortable" or "smile-gentle" or "smile-natural" && r.State == "Accepted"))
                throw new InvalidOperationException("Accept a relaxed jaw/smile attempt and at least one gentle or comfortable positive practice attempt first. Existing replay recordings alone cannot teach a correction.");
            _contractPath = await PrepareContractAsync();
            _cancellation.Token.ThrowIfCancellationRequested();
            var manifest = await Task.Run(() => _workspace.WriteManifest(_contractPath));
            var progress = new Progress<TrainingProgress>(p => { if (!_disposed) Status = p.Message; });
            var result = await _trainer.TrainC2Async(manifest, Path.Combine(_workspace.Root, "Candidates"), progress, _cancellation.Token);
            if (!_disposed) { Status = result.Message; Details = _trainer.DetailLog; }
        }
        catch (OperationCanceledException) { Status = "Cancelled. Partial candidate files are retained but cannot be selected; active model unchanged."; }
        catch (Exception ex) { Status = "C2 needs attention: " + ex.Message; }
        finally { lease?.Dispose(); _cancellation?.Dispose(); _cancellation = null; IsBusy = false; }
        if (!_disposed) await RefreshAsync();
    }

    [RelayCommand] private void CancelTraining() => _cancellation?.Cancel();

    [RelayCommand]
    private async Task CompareAsync()
    {
        if (SelectedCandidate == null) { CandidateStatus = "No completed candidate is available. Record and accept practice first."; return; }
        var report = Path.Combine(_workspace.Root, "Candidates", SelectedCandidate, "summary.json");
        try { Details = await File.ReadAllTextAsync(report); CandidateStatus = "Report opened in Advanced. Reported checks are exploratory; unmeasured behaviors remain Not checked."; }
        catch (Exception ex) { CandidateStatus = "Could not read candidate report: " + ex.Message; }
    }

    [RelayCommand]
    private async Task TryCandidateAsync()
    {
        if (IsTrial) return;
        await ActivateCandidateAsync(keep: false);
    }

    [RelayCommand]
    private Task KeepCandidateAsync() => ActivateCandidateAsync(keep: true);

    private async Task ActivateCandidateAsync(bool keep)
    {
        if (IsBusy || IsRecording || SelectedCandidate == null || _disposed) return;
        IsBusy = true;
        try
        {
            await _c2Models.ActivateAsync(Path.Combine(_workspace.Root, "Candidates", SelectedCandidate), keep);
            IsTrial = _c2Models.IsTrial;
            IsKept = _c2Models.IsKept;
            Status = CandidateStatus = keep ? "C2 saved for everyday use. You can leave this page; it will also load on restart. Audio Assist can stay on." : "C2 trial started for 60 seconds.";
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { Status = CandidateStatus = "Candidate not activated: " + ex.Message; }
        finally { IsBusy = false; }
    }

    [RelayCommand]
    private void ReturnToCurrent()
    {
        try
        {
            _c2Models.ReturnToModelC();
            IsTrial = false;
            IsKept = false;
            CandidateStatus = "Using Model C. C2 will stay off after restart; saved candidates and recordings are retained.";
        }
        catch (Exception ex) { CandidateStatus = "Could not save the return to Model C: " + ex.Message; }
    }

    private Func<bool> ContextGuard(C2ReferenceContract contract) => _context.ContextGuard(contract);

    public void Dispose()
    {
        _c2Models.EndTrial();
        _disposed = true; _timer.Stop(); _cancellation?.Cancel(); _cue.Clear();
        if (_overlay) _presenter?.End();
        if (IsRecording) _ = FinishAsync(false, "Page closed; unfinished attempt excluded.");
    }
}
