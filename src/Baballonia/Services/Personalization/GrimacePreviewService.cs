using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Baballonia.Contracts;
using Baballonia.Services.Calibration;
using Microsoft.Extensions.Logging;

namespace Baballonia.Services.Personalization;

/// <summary>
/// One schema-valid hypothesis for the HOME user's Grimace / 😬 shape. These are deliberately
/// called candidates: none is a training label until the user has seen it on their own avatar and
/// explicitly confirmed it.
/// </summary>
public sealed class GrimaceCandidate
{
    private readonly IReadOnlyList<int> _dims;
    private readonly IReadOnlyList<int> _suppressedDims;
    private readonly IReadOnlyDictionary<int, float> _components;

    internal GrimaceCandidate(
        string id,
        string displayName,
        string description,
        IReadOnlyDictionary<int, float> components,
        string fingerprint,
        string cueId = "Grimace",
        string cueDisplayName = "Grimace / 😬",
        string cueAction = "GRIMACE — show your teeth and pull your lower lip down")
    {
        CueId = cueId;
        CueDisplayName = cueDisplayName;
        CueAction = cueAction;
        Id = id;
        DisplayName = displayName;
        Description = description;
        _components = components;
        _dims = Array.AsReadOnly(components.Keys.OrderBy(dim => dim).ToArray());
        _suppressedDims = Array.AsReadOnly(components
            .Where(component => component.Value == 0f)
            .Select(component => component.Key)
            .OrderBy(dim => dim)
            .ToArray());
        Fingerprint = fingerprint;
    }

    public string Id { get; }
    public string DisplayName { get; }
    public string Description { get; }
    public IReadOnlyList<int> Dims => _dims;
    public IReadOnlyList<int> SuppressedDims => _suppressedDims;
    public IReadOnlyDictionary<int, float> Components => _components;
    internal string Fingerprint { get; }

    /// <summary>
    /// What the confirmed cue is called and says.
    /// </summary>
    /// <remarks>
    /// Carried on the candidate because a second catalogue exists. A smile candidate that confirmed
    /// itself into a cue telling the user to pull their lower lip down would be worse than useless -
    /// they would perform a grimace and it would be recorded as a smile.
    /// </remarks>
    internal string CueId { get; }
    internal string CueDisplayName { get; }
    internal string CueAction { get; }

    /// <summary>A fresh 45-value vector; callers cannot mutate the catalogue's candidate.</summary>
    public float[] CreateTarget()
    {
        var target = new float[PersonalizationSchema.ExpressionCount];
        foreach (var (dim, value) in _components)
            target[dim] = value;
        return target;
    }

    /// <summary>
    /// The training cue this candidate becomes once confirmed.
    /// </summary>
    /// <remarks>
    /// Public so the pairing of pose and instruction can be asserted. A candidate whose instruction
    /// described a different expression would be a silent, expensive failure: the user performs what
    /// the words say and it is recorded under the label of the pose they picked.
    /// </remarks>
    public GuidedCue CreateConfirmedCue() => new(
        CueId,
        CueDisplayName,
        Dims,
        CueAction,
        Levels: [1.0f],
        DimScale: Components,
        Suppressed: SuppressedDims);

    public override string ToString() => DisplayName;
}

/// <summary>
/// Exactly three restrained experiments over dimensions that exist in the canonical 45-value
/// schema and are routed by the Baballonia VRCFT module: symmetric MouthLowerDown,
/// symmetric MouthStretch, and JawOpen. Funnel/Pucker and every unrelated component stay zero.
/// </summary>
/// <summary>
/// A named set of candidate poses the user can preview and confirm one of.
/// </summary>
/// <remarks>
/// Introduced so a second lab could exist without a second copy of the preview service. The grimace
/// lab asks an identity question - what does this face even look like in the schema - and the smile
/// lab asks a calibration one, how much of each component makes an ordinary smile. The machinery is
/// the same either way: play it on the avatar, let the person judge it, record what they chose.
/// </remarks>
public sealed record PoseCandidateCatalog(
    string Name,
    int Version,
    string ConfirmationSetting,
    IReadOnlyList<GrimaceCandidate> Candidates,
    string ConfirmedRoutineId,
    string ConfirmedRoutineDisplayName,
    string ConfirmedRoutineDescription,
    string Prompt,
    string NotTrainableMessage)
{
    public GrimaceCandidate? ById(string? id) =>
        Candidates.FirstOrDefault(candidate =>
            string.Equals(candidate.Id, id, StringComparison.Ordinal));
}

public static class GrimaceCandidateCatalog
{
    public const int Version = 1;

    public static IReadOnlyList<GrimaceCandidate> All { get; } = Build();

    public static GrimaceCandidate? ById(string? id) =>
        All.FirstOrDefault(candidate =>
            string.Equals(candidate.Id, id, StringComparison.Ordinal));

    /// <summary>This catalogue as the service consumes it.</summary>
    public static PoseCandidateCatalog Catalog { get; } = new(
        "Grimace", Version, "Guided_ConfirmedGrimaceCandidate", All,
        "grimace-confirmed", "Grimace / 😬 (confirmed)",
        "The exact Grimace candidate you previewed and accepted on this avatar.",
        "Preview the candidates on your avatar before enabling Grimace training.",
        "Grimace is not trainable until you preview and explicitly confirm one candidate.");

    private static IReadOnlyList<GrimaceCandidate> Build()
    {
        return Array.AsReadOnly(new[]
        {
            Create(
                "grimace-a-tight",
                "Candidate A — tight / nearly closed",
                "Lower lip 70%, horizontal stretch 78%, jaw 8%. A low-jaw hypothesis.",
                ("MouthLowerDownLeft", 0.70f),
                ("MouthLowerDownRight", 0.70f),
                ("MouthStretchLeft", 0.78f),
                ("MouthStretchRight", 0.78f),
                ("JawOpen", 0.08f)),
            Create(
                "grimace-b-balanced",
                "Candidate B — balanced",
                "Lower lip 82%, horizontal stretch 72%, jaw 16%. A middle hypothesis.",
                ("MouthLowerDownLeft", 0.82f),
                ("MouthLowerDownRight", 0.82f),
                ("MouthStretchLeft", 0.72f),
                ("MouthStretchRight", 0.72f),
                ("JawOpen", 0.16f)),
            Create(
                "grimace-c-open",
                "Candidate C — more jaw",
                "Lower lip 88%, horizontal stretch 66%, jaw 28%. A more-open hypothesis.",
                ("MouthLowerDownLeft", 0.88f),
                ("MouthLowerDownRight", 0.88f),
                ("MouthStretchLeft", 0.66f),
                ("MouthStretchRight", 0.66f),
                ("JawOpen", 0.28f)),
        });
    }

    private static GrimaceCandidate Create(
        string id,
        string displayName,
        string description,
        params (string Name, float Value)[] components) =>
        CreateCandidate(Version, id, displayName, description, cue: null, components);

    /// <summary>
    /// Builds and validates a candidate. Shared so a second catalogue inherits the same checks -
    /// an unknown expression name or an out-of-range component should fail loudly wherever it is
    /// written, not only in the catalogue that happened to be first.
    /// </summary>
    internal static GrimaceCandidate CreateCandidate(
        int version,
        string id,
        string displayName,
        string description,
        (string Id, string Name, string Action)? cue,
        params (string Name, float Value)[] components)
    {
        var byDim = new SortedDictionary<int, float>();
        foreach (var (name, value) in components)
        {
            var dim = PersonalizationSchema.IndexOf(name);
            if (dim < 0)
                throw new InvalidOperationException($"Pose candidate uses unknown expression '{name}'.");
            if (value is < 0f or > 1f)
                throw new InvalidOperationException($"Pose candidate '{id}' has invalid {name}={value}.");
            if (!byDim.TryAdd(dim, value))
                throw new InvalidOperationException($"Pose candidate '{id}' repeats {name}.");
        }

        var readOnly = new ReadOnlyDictionary<int, float>(byDim);
        var canonical = new StringBuilder()
            .Append(version).Append('\n')
            .Append(PersonalizationSchema.Sha256).Append('\n')
            .Append(id).Append('\n');
        foreach (var (dim, value) in byDim)
        {
            canonical.Append(dim).Append(':')
                .Append(value.ToString("R", CultureInfo.InvariantCulture)).Append('\n');
        }

        var fingerprint = Convert.ToHexStringLower(
            SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString())));
        return cue is { } c
            ? new GrimaceCandidate(id, displayName, description, readOnly, fingerprint,
                c.Id, c.Name, c.Action)
            : new GrimaceCandidate(id, displayName, description, readOnly, fingerprint);
    }
}
/// <summary>
/// Candidate shapes for an everyday, teeth-showing smile.
/// </summary>
/// <remarks>
/// A different question from the grimace lab, answered with the same machinery. Grimace asks what an
/// expression *is*; this asks how much of it there should be. The complaint that produced it was an
/// ordinary friendly smile arriving on the avatar as a wide open-mouthed grin - so the candidates
/// vary how far the corners pull and how much the upper lip lifts, and keep the jaw at or near shut
/// throughout.
///
/// Teeth show because the upper lip rises, not because the mouth opens. Only the last candidate lets
/// the jaw move at all, and only slightly, so that somebody whose smile genuinely does part their
/// teeth is not forced to choose between two wrong answers.
/// </remarks>
public static class SmileCandidateCatalog
{
    public const int Version = 1;

    public static IReadOnlyList<GrimaceCandidate> All { get; } = Build();

    public static GrimaceCandidate? ById(string? id) =>
        All.FirstOrDefault(candidate => string.Equals(candidate.Id, id, StringComparison.Ordinal));

    public static PoseCandidateCatalog Catalog { get; } = new(
        "Smile", Version, "Guided_ConfirmedSmileCandidate", All,
        "smile-confirmed", "Smile (confirmed shape)",
        "The exact smile candidate you previewed and accepted on this avatar.",
        "Preview each smile on your avatar and pick the one that looks like your ordinary smile.",
        "Smile shaping is not trained until you preview and confirm one candidate.");

    private static IReadOnlyList<GrimaceCandidate> Build()
    {
        return Array.AsReadOnly(new[]
        {
            Create(
                "smile-a-gentle",
                "Candidate A — gentle",
                "Corners 55%, upper lip 30%, jaw shut. A closed, polite smile.",
                ("MouthSmileLeft", 0.55f),
                ("MouthSmileRight", 0.55f),
                ("MouthUpperUpLeft", 0.30f),
                ("MouthUpperUpRight", 0.30f),
                ("JawOpen", 0f)),
            Create(
                "smile-b-everyday",
                "Candidate B — everyday",
                "Corners 70%, upper lip 45%, jaw shut. The one most people mean by smiling.",
                ("MouthSmileLeft", 0.70f),
                ("MouthSmileRight", 0.70f),
                ("MouthUpperUpLeft", 0.45f),
                ("MouthUpperUpRight", 0.45f),
                ("JawOpen", 0f)),
            Create(
                "smile-c-toothy",
                "Candidate C — toothy",
                "Corners 82%, upper lip 65%, jaw shut. More teeth, mouth still closed.",
                ("MouthSmileLeft", 0.82f),
                ("MouthSmileRight", 0.82f),
                ("MouthUpperUpLeft", 0.65f),
                ("MouthUpperUpRight", 0.65f),
                ("JawOpen", 0f)),
            Create(
                "smile-d-parted",
                "Candidate D — slightly parted",
                "Corners 78%, upper lip 55%, jaw 12%. For a smile that does part the teeth a little.",
                ("MouthSmileLeft", 0.78f),
                ("MouthSmileRight", 0.78f),
                ("MouthUpperUpLeft", 0.55f),
                ("MouthUpperUpRight", 0.55f),
                ("JawOpen", 0.12f)),
        });
    }

    private static GrimaceCandidate Create(
        string id, string displayName, string description,
        params (string Name, float Value)[] components) =>
        GrimaceCandidateCatalog.CreateCandidate(
            Version, id, displayName, description,
            ("SmileShaped", "Smile (confirmed shape)",
             "Smile the way you normally would, showing a little teeth"),
            components);
}


/// <summary>Persisted proof of the exact candidate the user explicitly accepted.</summary>
public sealed record GrimaceCandidateConfirmation(
    string CandidateId,
    int CatalogVersion,
    string SchemaSha256,
    string CandidateFingerprint,
    string ConfirmedUtc);

public sealed record GrimacePreviewResult(bool Success, string Message);

/// <summary>
/// Commands Grimace candidates for empirical inspection without creating a dataset session or
/// publishing a <see cref="CuePhase"/> to <see cref="CueStateSource"/>. The capture gate makes that
/// separation race-safe: no recorder can start while preview owns the avatar, and preview refuses
/// while any training recording owns the gate.
/// </summary>
public sealed class GrimacePreviewService : IDisposable
{
    /// <summary>Kept for callers that named the grimace key directly; new code reads the catalogue.</summary>
    public const string ConfirmationSetting = "Guided_ConfirmedGrimaceCandidate";

    private readonly PoseCandidateCatalog _catalog;

    private readonly ExpressionOverrideService _override;
    private readonly TrainingCaptureGate _captureGate;
    private readonly ILocalSettingsService _settings;
    private readonly IVrCalibrationPresenter _presenter;
    private readonly ILogger<GrimacePreviewService> _logger;

    private IDisposable? _captureLease;
    private int _candidateIndex = -1;
    private bool _candidateWasPresented;
    private bool _ownsOverride;

    public GrimacePreviewService(
        ExpressionOverrideService overrideService,
        TrainingCaptureGate captureGate,
        ILocalSettingsService settings,
        IVrCalibrationPresenter presenter,
        ILogger<GrimacePreviewService> logger,
        PoseCandidateCatalog? catalog = null)
    {
        _catalog = catalog ?? GrimaceCandidateCatalog.Catalog;
        _override = overrideService;
        _captureGate = captureGate;
        _settings = settings;
        _presenter = presenter;
        _logger = logger;
        Status = _catalog.Prompt;
    }

    /// <summary>Which lab this instance is running, for a window that shows more than one.</summary>
    public PoseCandidateCatalog Catalog => _catalog;

    public IReadOnlyList<GrimaceCandidate> Candidates => _catalog.Candidates;
    public bool IsPreviewing { get; private set; }
    public string Status { get; private set; }
    public GrimaceCandidate? CurrentCandidate =>
        _candidateIndex >= 0 && _candidateIndex < Candidates.Count
            ? Candidates[_candidateIndex]
            : null;
    public int CurrentCandidateIndex => _candidateIndex;

    /// <summary>Only a current, schema-matching, fingerprint-matching persisted choice is valid.</summary>
    public GrimaceCandidateConfirmation? Confirmation
    {
        get
        {
            try
            {
                return _settings.ReadSetting<GrimaceCandidateConfirmation?>(
                    _catalog.ConfirmationSetting, null);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not read {CatalogName} candidate confirmation",
                    _catalog.Name);
                return null;
            }
        }
    }

    public GrimaceCandidate? ConfirmedCandidate
    {
        get
        {
            var confirmation = Confirmation;
            if (confirmation == null ||
                confirmation.CatalogVersion != _catalog.Version ||
                !string.Equals(confirmation.SchemaSha256, PersonalizationSchema.Sha256,
                    StringComparison.Ordinal))
            {
                return null;
            }

            var candidate = _catalog.ById(confirmation.CandidateId);
            return candidate != null &&
                   string.Equals(candidate.Fingerprint, confirmation.CandidateFingerprint,
                       StringComparison.Ordinal)
                ? candidate
                : null;
        }
    }

    /// <summary>
    /// Null until explicit confirmation. This is the only API that turns an experimental vector
    /// into a guided cue, so unconfirmed candidates never enter a routine or trainer label.
    /// </summary>
    public GuidedCue? ConfirmedCue => ConfirmedCandidate?.CreateConfirmedCue();

    /// <summary>VM hook: add this choice to the picker only when non-null.</summary>
    public GuidedRoutineChoice? ConfirmedRoutineChoice => ConfirmedCue is not { } cue
        ? null
        : new GuidedRoutineChoice(
            _catalog.ConfirmedRoutineId,
            _catalog.ConfirmedRoutineDisplayName,
            _catalog.ConfirmedRoutineDescription,
            [cue]);

    public GrimacePreviewResult BeginPreview()
    {
        if (IsPreviewing)
            return new GrimacePreviewResult(false, $"A {_catalog.Name} preview is already running.");
        if (_override.IsActive)
            return new GrimacePreviewResult(false,
                $"Another avatar calibration override is active. Stop it before previewing {_catalog.Name}.");

        if (!_captureGate.TryEnter($"the {_catalog.Name} candidate preview", out _captureLease,
                out var gateMessage))
            return new GrimacePreviewResult(false, gateMessage);

        var started = _presenter.Begin(
            $"{_catalog.Name.ToUpperInvariant()} CANDIDATE PREVIEW — NOT TRAINING DATA");
        if (!started.Started)
        {
            ReleaseCaptureGate();
            return new GrimacePreviewResult(false,
                $"{_catalog.Name} preview needs the in-headset presenter. {started.Message}");
        }

        try
        {
            _override.Activate();
            _ownsOverride = true;
            IsPreviewing = true;
            _candidateIndex = 0;
            PresentCurrentCandidate();
            Status = "Previewing Candidate A. Inspect the avatar and try making the same face.";
            return new GrimacePreviewResult(true, Status);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not begin {CatalogName} candidate preview", _catalog.Name);
            EndPreviewInternal($"{_catalog.Name} preview could not start.", null);
            return new GrimacePreviewResult(false, Status);
        }
    }

    /// <summary>Keeps the deadman alive and consumes Previous/Next/Cancel actions from VR.</summary>
    public bool Tick()
    {
        if (!IsPreviewing)
            return false;

        // Fail closed before refreshing the avatar deadman. A lost/unhealthy headset surface means
        // the user can no longer verify what the avatar is showing, so keeping the command alive
        // would turn a blind pose into something that could later be confirmed.
        if (!_presenter.IsPresenting || !_presenter.IsHealthy)
        {
            var detail = string.IsNullOrWhiteSpace(_presenter.Status)
                ? "the in-headset preview is no longer healthy."
                : _presenter.Status;
            EndPreviewInternal(
                $"{_catalog.Name} preview stopped safely: {detail} No training data was recorded.",
                null);
            return false;
        }

        if (!_override.IsCommandHealthy)
        {
            EndPreviewInternal(
                $"{_catalog.Name} preview stopped because the avatar command lapsed. " +
                "No training data was recorded.",
                null);
            return false;
        }

        _override.KeepAlive();
        switch (_presenter.ConsumeAction())
        {
            case VrCalibrationAction.Retry:
                ShowPreviousCandidate();
                break;
            case VrCalibrationAction.Skip:
                ShowNextCandidate();
                break;
            case VrCalibrationAction.Cancel:
                CancelPreview();
                return false;
        }

        return IsPreviewing;
    }

    public GrimacePreviewResult ShowCandidate(string candidateId)
    {
        if (!IsPreviewing)
            return new GrimacePreviewResult(false, $"Start the {_catalog.Name} preview first.");

        var index = Candidates
            .Select((candidate, i) => (candidate, i))
            .FirstOrDefault(item => string.Equals(item.candidate.Id, candidateId, StringComparison.Ordinal)).i;
        if (index < 0 || index >= Candidates.Count ||
            !string.Equals(Candidates[index].Id, candidateId, StringComparison.Ordinal))
        {
            return new GrimacePreviewResult(false,
                $"That {_catalog.Name} candidate does not exist.");
        }

        _candidateIndex = index;
        PresentCurrentCandidate();
        Status = $"Previewing {CurrentCandidate!.DisplayName}. Nothing is being recorded.";
        return new GrimacePreviewResult(true, Status);
    }

    public GrimacePreviewResult ShowPreviousCandidate() =>
        ShowAtOffset(-1);

    public GrimacePreviewResult ShowNextCandidate() =>
        ShowAtOffset(1);

    private GrimacePreviewResult ShowAtOffset(int offset)
    {
        if (!IsPreviewing)
            return new GrimacePreviewResult(false, $"Start the {_catalog.Name} preview first.");

        _candidateIndex = (_candidateIndex + offset + Candidates.Count) % Candidates.Count;
        PresentCurrentCandidate();
        Status = $"Previewing {CurrentCandidate!.DisplayName}. Nothing is being recorded.";
        return new GrimacePreviewResult(true, Status);
    }

    /// <summary>
    /// Explicit user acceptance. It succeeds only for the candidate actively presented in a live
    /// VR preview; selecting a dropdown or merely shipping the catalogue cannot confirm anything.
    /// </summary>
    public GrimacePreviewResult ConfirmCurrentCandidate()
    {
        var candidate = CurrentCandidate;
        if (!IsPreviewing || !_candidateWasPresented || candidate == null ||
            !_presenter.IsPresenting || !_presenter.IsHealthy || !_override.IsCommandHealthy)
        {
            return new GrimacePreviewResult(false,
                "A candidate must be visibly active under a healthy VR avatar command before it can be confirmed.");
        }

        try
        {
            var confirmation = new GrimaceCandidateConfirmation(
                candidate.Id,
                _catalog.Version,
                PersonalizationSchema.Sha256,
                candidate.Fingerprint,
                DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture));
            _settings.SaveSetting(_catalog.ConfirmationSetting, confirmation);

            // LocalSettingsService deliberately swallows serialization/I/O failures after logging.
            // Read the decision back through the same validation path before exposing supervision,
            // otherwise the UI could claim success for a confirmation that will vanish on restart.
            if (!string.Equals(ConfirmedCandidate?.Id, candidate.Id, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    $"The {_catalog.Name} confirmation could not be verified.");

            var status = $"Confirmed {candidate.DisplayName}. {_catalog.Name} is now available " +
                         "as a separate guided routine.";
            EndPreviewInternal(status,
                $"{_catalog.Name} candidate confirmed. You may remove the headset.");
            return new GrimacePreviewResult(true, status);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not persist {CatalogName} candidate confirmation",
                _catalog.Name);
            Status = $"The candidate could not be saved, so {_catalog.Name} training remains unavailable.";
            return new GrimacePreviewResult(false, Status);
        }
    }

    public void ClearConfirmation()
    {
        _settings.SaveSetting<GrimaceCandidateConfirmation?>(_catalog.ConfirmationSetting, null);
        Status = Confirmation == null
            ? $"{_catalog.Name} confirmation cleared. {_catalog.NotTrainableMessage}"
            : $"The {_catalog.Name} confirmation could not be cleared; the saved choice is still active.";
    }

    public void CancelPreview() =>
        EndPreviewInternal($"{_catalog.Name} preview cancelled. No training data was recorded.", null);

    private void PresentCurrentCandidate()
    {
        var candidate = CurrentCandidate ??
            throw new InvalidOperationException($"No {_catalog.Name} candidate is selected.");
        var target = candidate.CreateTarget();

        // Upload the explicit PREVIEW ONLY message before moving the avatar. No CueStateSource is
        // involved, and TrainingCaptureGate prevents a DatasetRecorderService session concurrently.
        _presenter.Present(new VrCalibrationFrame(
            $"{candidate.DisplayName} — PREVIEW ONLY",
            $"NOT TRAINING DATA. {candidate.CueAction}. " +
            "Inspect the avatar, then imitate it. RETRY = previous; SKIP = next. " +
            "Confirm in Baballonia only if this shape looks natural and is easy to repeat.",
            VrCalibrationPhase.Hold,
            (_candidateIndex + 1d) / Candidates.Count,
            PhaseProgress: 1,
            Repetition: _candidateIndex + 1,
            RepetitionCount: Candidates.Count,
            AllowRetry: true,
            AllowSkip: true,
            AllowCancel: true));

        _override.PushPhase(new CuePhase(
            $"{_catalog.Name}Preview-{candidate.Id}",
            "preview-not-recorded",
            candidate.Dims,
            target,
            (float[])target.Clone(),
            DurationSeconds: 60 * 60,
            StartTimestamp: System.Diagnostics.Stopwatch.GetTimestamp(),
            Level: 1f));
        _candidateWasPresented = true;
    }

    private void EndPreviewInternal(string status, string? completionMessage)
    {
        // Neutralize first. Even if overlay teardown fails, the avatar must not remain frozen on a
        // candidate and the recorder gate must eventually be released.
        if (_ownsOverride)
            _override.Deactivate();
        _ownsOverride = false;
        IsPreviewing = false;
        _candidateWasPresented = false;
        _candidateIndex = -1;

        try
        {
            _presenter.End(completionMessage);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "{CatalogName} preview overlay did not close cleanly",
                _catalog.Name);
        }
        finally
        {
            ReleaseCaptureGate();
        }

        Status = status;
    }

    private void ReleaseCaptureGate()
    {
        _captureLease?.Dispose();
        _captureLease = null;
    }

    public void Dispose()
    {
        if (IsPreviewing || _ownsOverride)
            EndPreviewInternal($"{_catalog.Name} preview stopped. No training data was recorded.",
                null);
        else
            ReleaseCaptureGate();
    }
}
