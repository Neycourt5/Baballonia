# ALEXALLONIA_TRACKING_TUNING_PLAN.md

> **Save this file to your home PC repo root as `ALEXALLONIA_TRACKING_TUNING_PLAN.md`.**
> It is written to be handed to a coding model together with the newer home-PC codebase.

---

## Context

Three polish changes to an already-good tracking system. Nothing here is a rewrite; every change is additive, defaults to current behavior, and is designed to be reverted independently.

1. **Eye Gaze Responsiveness** — gaze feels more reserved/slower than Paper Tracker. Add a temporal-responsiveness control.
2. **Eye Squint Strength** — squint detection is good; add an output-side strength scalar without touching detection.
3. **Toothy smile** — a preview-confirmed guided calibration exists, the full capture→train→export→activate loop was completed and the model is active, yet live tracking still cannot produce a convincing toothy smile. Find out why and fix it.

This document was produced by inspecting the **work-PC checkout**, which is **older than the home-PC build**. Read §2 before trusting any specific symbol.

---

## 1. Executive summary

| # | Change | Verdict from this checkout | Risk |
|---|---|---|---|
| 1 | Eye Gaze Responsiveness | **Well-understood.** One Euro is the *only* temporal filter on eyes. A responsiveness scalar multiplied into its `minCutoff`/`beta` at eye-filter construction time is exact, isolated, and identity at default. One prerequisite bug must be fixed first (filter rebuild snaps eyes to center). | Low |
| 2 | Eye Squint Strength | **Blocked on version drift by design.** Squint does not exist in this checkout's eye output at all. Do not design it from here. Trace the working Alpha Expressions path, then apply `clamp(v * strength, 0, 1)` at the last mutable point before OSC. | Low, once the path is traced |
| 3 | Toothy smile | **Root cause identified with high confidence.** Guided calibration is *not* a similarity/threshold matcher — it is dataset capture feeding an ONNX residual adapter. The plain `Smile` cue actively trains the teeth-revealing dimensions toward **zero** on every hold, and it outnumbers the toothy cue. The toothy label is being averaged away in training. | Medium — training-side fix |

**The single most important reframing:** every question in the brief about "activation threshold", "similarity score", "which calibration wins", "blend strength too weak" assumes a template-matching calibration system. **That system does not exist here.** There is no cosine, dot product, euclidean distance, or activation threshold anywhere in the runtime path (verified by exhaustive grep). Calibration produces a *trained neural residual*, and the runtime application is one ONNX inference plus a scalar lerp. The toothy-smile failure is therefore a **training-label** problem, not a matching problem. §6 rewrites the diagnosis in the architecture that actually exists.

---

## 2. Repository / version caveat

**This checkout is a stale work-PC snapshot.** Confirmed drift, stated by you and corroborated by the code:

- This tree is **pre-Alpha Expressions**. Your home build is Alpha Expressions and **does** output `EyeSquintLeft` / `EyeSquintRight`, working well.
- In *this* checkout, eye squint does not exist in any shipped form:
  - `EyeProcessingPipeline.ProjectLegacyEyeOutput` **discards** `rightEyeSquint`/`leftEyeSquint` when a 12-output model is loaded (there is a unit test asserting exactly this).
  - `ParameterSenderService._eyeExpressionMap` has exactly 6 channels, with a comment saying widen/squint/brow "were an experiment that is no longer part of the build".
  - `BabbleVRC.cs` has the `UnifiedExpressions.EyeSquintLeft/Right` assignments **commented out**.
  - An `experimental/eye-v2/` tree exists but is **not compiled** (no project globs it).
- Your recent eye-tracking work, reliability/camera-recovery work, and calibration changes may be newer or absent here.
- The toothy-smile calibration is **not present here at all**.

**Consequences for the implementer:**

- Treat filenames and symbol names below as **landmarks**, not requirements.
- Treat the entire eye squint data path as **unresolved**. Do not add squint support; find the working one.
- Never revert, bypass, or "restore" eye logic that looks unfamiliar. Unfamiliar means newer.
- If something in this document appears missing from the home build, verify twice before concluding it was removed — more likely it moved or was renamed.

Every claim below is tagged:

- **[VERIFIED-HERE]** — read directly in this checkout.
- **[MAY-DIFFER]** — likely changed in the home build.
- **[VERIFY-AT-HOME]** — you must check before implementing.

---

## 3. Existing architecture discovered

### 3.1 Eye processing pipeline **[VERIFIED-HERE]**

Driven by `ProcessingLoopService` — an Avalonia `DispatcherTimer` at `Interval = 10ms`, running **both** pipelines **synchronously on the UI thread**. Effective tick rate degrades under UI load, which directly changes the One Euro `dt`.

`Services/Inference/EyeProcessingPipeline.cs : RunUpdate()`:

```
VideoSource.GetFrame(Gray8)
  → FastCorruptionDetector.IsCorrupted        → drop frame
  → publish NewFrameEvent
  → DualImageTransformer.Apply                 (split L/R, ROI, gamma, rotate/flip, resize 128²,
                                                merge to 2-channel Mat)
  → publish NewTransformedFrameEvent
  → ImageCollector.Apply                       (EqualizeHist, channel reverse, 4-frame temporal
                                                stack → 8-channel Mat; returns null while filling)
  → InferenceService.Run()                     → raw model output
  → publish NewRawModelOutputEvent             ← full native output, incl. squint on 12-ch models
                                                  (NO SUBSCRIBERS in this build)
  → ProjectLegacyEyeOutput()                   ← 12 → 6 projection BY NAME; squint/widen/brow DROPPED
  → publish NewRawExpressionsEvent             ← documented calibration tap (no subscribers here)
  → Filter.Filter(...)                         ← ★ THE ONLY TEMPORAL FILTER (One Euro)
  → ProcessExpressions(ref ...)                ← ★ THE ONLY VALUE POST-PROCESSING
  → publish NewFilteredResultEvent
→ ProcessingLoopService.ExpressionChangeEvent
→ ParameterSenderService.ProcessEyeExpressionData  ← per-channel Remap → OSC
→ VRCFaceTracking.Baballonia / BabbleVrc.Update    ← verbatim copy, no math
```

The 6-slot legacy contract (`Utils.EyeRawExpressions = 6`), pre-`ProcessExpressions`:

```
[0] rightEyeY|Pitch  [1] rightEyeX|Yaw  [2] rightEyeLid
[3] leftEyeY |Pitch  [4] leftEyeX |Yaw  [5] leftEyeLid     — all sigmoid, [0,1]
```

`ProcessExpressions` math (names in the local variables are swapped relative to the model's own naming; two sign errors cancel downstream — **read the math, not the identifiers**):

```csharp
const float mulV = 2.0f, mulY = 2.0f;            // the ONLY gaze gain in the app; not user-settable

pitch = raw * 2 - 1;   yaw = raw * 2 - 1;   lid = 1 - raw;    // lid becomes OPENNESS (1 = open)

eyeY = (leftPitch*leftLid + rightPitch*rightLid) / (leftLid + rightLid);   // fused, single Y for both eyes
                                                                          // ⚠ no epsilon → NaN if both lids = 0

leftYawC  = rightYaw*(1-leftLid)  + leftYaw*leftLid;    // a closed eye BORROWS the open eye's yaw
rightYawC = leftYaw*(1-rightLid)  + rightYaw*rightLid;

if (StabilizeEyes) {                                    // divergence clamp, identity when convergence >= 0
    convergence = max((rightYawC - leftYawC)/2, 0);
    averaged    = (rightYawC + leftYawC)/2;
    leftYawC = averaged - convergence;  rightYawC = averaged + convergence;
}
```

**Load-bearing fact for Change 2:** `lid` is an *input* to the gaze math. Anything that perturbs lid perturbs gaze. If Alpha derives squint from the lid scalar (as the shelved `experimental/eye-v2/EyeV2Mapper` does, disambiguating blink from squint by dwell time), then scaling squint **in place, before this block** would leak into gaze. Scale at the output instead.

### 3.2 The One Euro filter **[VERIFIED-HERE]**

`Services/Inference/Filters/OneEuroFilter.cs`, `IFilter { float[] Filter(float[] input); }`.

```
dt      = (UtcNow - tPrev).TotalSeconds          // wall clock, NOT the tick delta
if dt == 0: xPrev = x; return x                  // passthrough; tPrev NOT advanced
dx      = (x - xPrev) / dt
a_d     = r/(r+1),  r = 2π · dCutoff · dt,   dCutoff = 1.0 hardcoded
dxHat   = a_d·dx + (1-a_d)·dxPrev
cutoff  = minCutoff + beta · |dxHat|             // ← the adaptive term
a       = r/(r+1),  r = 2π · cutoff · dt
xHat    = a·x + (1-a)·xPrev                      // ← the EMA that produces the "reserved" feel
```

Semantics that matter for the responsiveness knob:

- **`minCutoff`** sets smoothing when nearly still. Raising it → snappier on small/slow movement, **more jitter at rest**.
- **`beta`** sets how far the cutoff opens as speed rises. Raising it → snappier on saccades **without** adding rest jitter.
- The "reserved on fast gaze shifts" complaint is predominantly a **`beta`** problem. This is why the recommended mapping in §4 weights `beta` far more heavily than `minCutoff`.

Construction, `Services/Inference/EyePipelineManager.cs : LoadFilter()` (face twin in `FacePipelineManager`):

| Setting key | Ctor param | Default | UI |
|---|---|---|---|
| `AppSettings_OneEuroEnabled` | gate | `true` | ToggleSwitch |
| `AppSettings_OneEuroMinFreqCutoff` | `minCutoff` | `0.5f` | NumericUpDown 0.1–10 + presets |
| `AppSettings_OneEuroSpeedCutoff` | `beta` | `3f` | NumericUpDown 0.1–10 + presets |

**⚠ Face and eyes share these two keys.** Both managers read the same settings. A responsiveness control must therefore be **eye-specific and multiplicative on top**, never a change to these keys.

Three defects in this path that Change 1 must work around or fix:

1. **`LoadFilter()` seeds `xPrev` to zeros** (`new OneEuroFilter(new float[6], ...)`), not to the current value. Every reconstruction pulls the eyes toward center for several frames.
2. **`AppSettingsViewModel.PropertyChanged` calls `LoadFilter()` on *every* property change** — including `OscPrefix` and `LogLevel`. Combined with (1), dragging any new slider on that page will rebuild the filter continuously and make the eyes visibly convulse. **This is a hard prerequisite, not a nice-to-have.**
3. **`if (!enabled) return;`** leaves the previously-installed filter running. Toggling One Euro off does nothing until restart.

Useful seam: `EyePipelineManager.SetFilter(IFilter?)` is public — a wrapping/decorating filter can be installed without editing the pipeline.

### 3.3 Eye range / calibration remap **[VERIFIED-HERE]**

Only two places scale gaze:

1. `ProcessExpressions` — hardcoded `×2 − 1`.
2. `ParameterSenderService.ProcessEyeExpressionData` — per named channel:
   `weight.Remap(settings.Lower, settings.Upper, settings.Min, settings.Max)`
   where `Remap` (`Helpers/FloatExtensions.cs`) is `Min + (v - Lower)*(Max-Min)/(Upper-Lower)`.

`CalibrationParameter { Lower, Upper, Min, Max }` is stored per name by `CalibrationService` under the settings key `CalibrationParams`. Eye gaze defaults are `(-1, 1, -1, 1)` — **identity**.

**⚠ Eye channels are NOT clamped after remap** (face channels are). A narrowed `Lower/Upper` extrapolates without limit.

**Latent opportunity:** `LeftEyeX/Y` and `RightEyeX/Y` already have `CalibrationParameter` entries, are already applied at send time, and are already persisted — they simply have **no UI**. `CalibrationViewModel.EyeSettings` currently exposes only the two lids. Adding four `new SliderBindableSetting("LeftEyeX")`-style rows would give gaze range sliders through the entirely existing path. See §4.7.

Also dead in this build: `AppSettings_RecenterAddress` / `AppSettings_RecalibrateAddress` exist in settings and the VM but **nothing reads them** — no recenter is implemented. **[MAY-DIFFER]** — you may well have implemented this at home.

### 3.4 Settings & persistence **[VERIFIED-HERE]** — reuse exactly this

`Contracts/ILocalSettingsService.cs` — all **synchronous**:

```csharp
T    ReadSetting<T>(string key, T? defaultValue = default, bool forceLocal = false);
void SaveSetting<T>(string key, T value, bool forceLocal = false);
void Save(object target);   // reflection over [SavedSetting] properties
void Load(object target);
void ForceSave();
```

- Store: flat `ConcurrentDictionary<string, JsonElement>` → `%APPDATA%\ProjectBabble\ApplicationData\LocalSettings.json`, `WriteIndented`, **2000 ms debounced** writes, `ForceSave()` on shutdown.
- Backward compatibility: `Load(object)` assigns `SavedSettingAttribute.Default()` whenever the key is absent, null, or fails to deserialize. **A missing new key therefore resolves to the attribute default automatically.** This satisfies the "old settings files must keep working" requirement with no extra code.
- **Trap:** the attribute default is boxed `object?` and assigned via `PropertyInfo.SetValue`, so the literal's CLR type must match the property type *exactly*. Write `1f` for a `float`, never `1.0`. A mismatch throws and is silently swallowed into a log line.
- **Trap:** `ReadSetting<float>("Key")` with no default returns `0f`, not the attribute default. **Always pass the default at every `ReadSetting` call site.**

**Pattern A — persisted VM property (use for both new sliders):**

```csharp
[ObservableProperty]
[property: SavedSetting("AppSettings_EyeGazeResponsiveness", 1f)]
private float _eyeGazeResponsiveness;
```

Persistence and pipeline reload are then free — `AppSettingsViewModel`'s constructor already wires
`PropertyChanged += (_, p) => { Save(this); LoadFilter(); ... }`.

**Pattern B — service-owned knob** (`AudioAssistService.Strength`, `PersonalModelManager.Blend`): a clamping property that writes the setting and pushes the value into the live stage. Use this if the value must be readable/clampable outside the UI.

UI containers to reuse: `Controls/SettingsBlock.axaml` (icon + title + description + arbitrary content on the right), `Controls/SettingsToggle`, `Controls/SettingsExpander`.

Localization: core views use RESX (`Assets/Resources.resx` + `Resources.Designer.cs`, Crowdin-managed, 30 locales). `PersonalizationView.axaml` **deliberately hardcodes English** — its header comment explains that adding RESX keys for a fork-local feature conflicts on every rebase. Follow the file you are editing.

### 3.5 Guided calibration **[VERIFIED-HERE]** — read this before touching Change 3

**There is no similarity matching, no distance metric, and no activation threshold.** Verified by exhaustive grep for `cosine|dot|euclidean|similarity|distance` across the runtime path — zero relevant hits.

What actually happens:

**Capture.** `GuidedCue(Id, DisplayName, Dims, Action, Levels?, DimScale?)` in `Services/Personalization/GuidedCues.cs` defines a pose as a **45-value commanded target vector**. `ExpressionOverrideService` drives your avatar to that vector over OSC (`ParameterSenderService._faceOverrideActive` suppresses live face output while it does). You imitate the avatar. `DatasetRecorderService` taps `FacePipelineEvents.NewRawExpressionsEvent` — **raw, pre-corrector, pre-filter inference output** — and writes one sample per frame:

```
{PersistentDataDirectory}\PersonalDataset\{timestamp}_guided\
    session.json                    ← SessionMetadata, schema SHA-256, camera geometry
    labels.jsonl                    ← FrameLabel { i, t, stock[45], personal[45]?, cue }
    frames\000000.jpg               ← the exact 224² gray image the model consumed, JPEG q95
    guided_quality_overrides.json   ← attempts invalidated by Retry/Skip in-headset
```

`FrameLabel.CueLabel = { id, phase, dims[], target[45], level, rep, attempt, source }`. **The commanded target is the training label.** The routine per attempt is: lead-in 5s → prep 3s → transition 1s → **hold 3s** → transition 1s → rest 2s, with `HoldSettleTrimSeconds = 0.75` trimmed off the front of each hold.

**Training** (`training/babble_personal/`, Python). Labels are built with per-frame, per-dimension weights:

```
W_GUIDED_HOLD        = 1.0     W_GUIDED_RAMP   = 0.5
W_GUIDED_REST        = 1.0     W_UNCUED_DIM    = 0.25   ← ★ the one that breaks toothy smile
W_GUIDED_HOLD_TONGUE = 0.4     W_SPEECH_PSEUDO = 0.3
W_NEUTRAL_SESSION    = 1.0     W_MANUAL_CORRECTION = 2.0
```

`CO_ACTIVATION_EXCLUSIONS` names, per cued dimension, the dimensions that may legitimately co-activate and must therefore **not** be pushed to zero. Everything not cued and not excluded is supervised toward **0 at weight 0.25**.

**Serving.** `PersonalModelCorrector` (or `EmbeddingModelCorrector`) runs the exported adapter and lerps:

```csharp
output[i] = stock[i] + (personal[i] - stock[i]) * blend;    // blend = PersonalModel_Blend, default 1f
```

Inside the exported graph: `personal = clamp(stock + residual, 0, 1)`, residual from a small MLP whose final layer is zero-initialized (identity at init).

**Runtime chain after correction:**

```
Corrector → Enhancer (ProsodyEnhancer) → Filter (One Euro) → ParameterSenderService (Remap+Clamp) → OSC → BabbleVrc (verbatim copy)
```

`ProsodyEnhancer` is the only post-corrector value transform, off by default, and multiplies every `Jaw*`/`Mouth*` dim by up to 1.6× while you speak. It **never adds**, so it cannot manufacture an expression — but it can dilute a calibrated magnitude.

**Confirmed absent** (exhaustive grep): no JawOpen derivation or late clamp, no tongue derivation, no tongue gating on JawOpen at runtime, no smile post-scaling, no expression-conflict-resolution pass. The tongue↔jaw rule exists **only** as a training-label exclusion.

**The preview-confirm mechanism** — which you told me your toothy calibration uses — is `GrimacePreviewService.cs`:

- `GrimaceCandidate` — one hypothesis target vector, with `Fingerprint` = SHA-256 over `Version \n PersonalizationSchema.Sha256 \n id \n {dim:value}…`.
- `GrimaceCandidateCatalog.All` — three restrained hypotheses varying `MouthLowerDown` / `MouthStretch` / `JawOpen`.
- `BeginPreview` → drive the avatar → `ConfirmCurrentCandidate()`, which requires `IsPreviewing && _candidateWasPresented && _presenter.IsHealthy && _override.IsCommandHealthy`, persists a `GrimaceCandidateConfirmation` to `Guided_ConfirmedGrimaceCandidate`, then **reads it back through validation** before reporting success.
- `ConfirmedCue` returns `candidate.CreateConfirmedCue()` — a real `GuidedCue` with `Levels: [1.0f]` and `DimScale: Components`.
- The VM injects a `"grimace-confirmed"` routine into the picker in `PersonalizationViewModel.RefreshGrimaceState()`.

**This is almost certainly the shape of your toothy-smile calibration.** [VERIFY-AT-HOME]

### 3.6 The face model's raw output **[VERIFIED-HERE]**

`src/Baballonia/faceModel.onnx`, opset 17, input `[1,1,224,224]` gray, output `[batch, 45]`, **no embedded metadata** (verified: `blendshape_names` does not appear in the file). Index meaning is pure convention, mirrored in four places that must stay in sync:

- `Services/Personalization/PersonalizationSchema.cs : Names` (canonical)
- `Services/ParameterSenderService.cs : FaceExpressionMap` (insertion order = de-facto source of truth)
- `training/babble_personal/schema.py : EXPRESSION_NAMES`
- `src/BabblePersonalizer.Core/Inventory/LegacyBaballoniaFaceCatalog.cs`

A build-breaking test (`Baballonia.Tests/PersonalizationSchemaTests.cs`) enforces the first two agree.

```
 0 CheekPuffLeft      12 MouthLeft          24 MouthDimpleRight     36 TongueLeft
 1 CheekPuffRight     13 MouthRight         25 MouthUpperUpLeft     37 TongueRight
 2 CheekSuckLeft      14 MouthRollUpper     26 MouthUpperUpRight    38 TongueRoll
 3 CheekSuckRight     15 MouthRollLower     27 MouthLowerDownLeft   39 TongueBendDown
 4 JawOpen            16 MouthShrugUpper    28 MouthLowerDownRight  40 TongueCurlUp
 5 JawForward         17 MouthShrugLower    29 MouthPressLeft       41 TongueSquish
 6 JawLeft            18 MouthClose         30 MouthPressRight      42 TongueFlat
 7 JawRight           19 MouthSmileLeft     31 MouthStretchLeft     43 TongueTwistLeft
 8 NoseSneerLeft      20 MouthSmileRight    32 MouthStretchRight    44 TongueTwistRight
 9 NoseSneerRight     21 MouthFrownLeft     33 TongueOut
10 MouthFunnel        22 MouthFrownRight    34 TongueUp
11 MouthPucker        23 MouthDimpleLeft    35 TongueDown
```

**Can the raw output distinguish a toothy smile from an open mouth? Yes.** Beyond `JawOpen`(4) and `MouthSmileLeft/Right`(19,20), the vector separately carries `MouthUpperUpLeft/Right`(25,26) = upper-lip retraction (upper teeth), `MouthLowerDownLeft/Right`(27,28) = lower-lip depression (lower teeth), `MouthStretchLeft/Right`(31,32), and `MouthClose`(18). A toothy smile is expressible as *{smile high, upperUp moderate-high, lowerDown low-moderate, jawOpen small, mouthClose ~0}* — structurally distinct from *{jawOpen high, rest ~0}*.

**→ Base-model retraining is NOT required.** The axes exist. See §6.7.

Two caveats the repo itself records:
1. There is no `LipsPart` / `TeethVisible` channel. Teeth exposure is only *inferable* from lip channels.
2. `training/babble_personal/labels.py : _select_probe_dim` has a written post-mortem noting that on **your** rig, the stock model reads a *closed* jaw at ≈**0.77** on `JawOpen`, and reads `MouthLowerDown` at ≈**0.01 whether the lip moves or not**, while `MouthStretch` tracks the same expression at **+0.95**. Two of the three most relevant axes are near-blind on your hardware as-shipped. This matters enormously for §6.

### 3.7 Existing diagnostics **[VERIFIED-HERE]** — you already have the debug panel

You do **not** need to build one:

| What | Where |
|---|---|
| **Live all-45 Stock / Personal / Delta / AbsDelta table, sortable by delta** | `Models/ExpressionComparisonRow.cs`; driven by `PersonalizationViewModel.OnRawExpressions` / `OnCorrectedExpressions` on a 250 ms `DispatcherTimer`; UI gated behind the `ShowAdvanced` checkbox in `PersonalizationView.axaml` |
| Live post-remap value beside every Lower/Upper slider, grouped Eye/Jaw/Cheek/Nose/Mouth/Tongue | `CalibrationViewModel.ApplyCurrentFaceExpressionValues` / `CalibrationView.axaml` |
| Audio diagnostics (speaking/energy/pitch/boost) | `PersonalizationViewModel.UpdateAudioDiagnostics`, `ShowAdvanced` only |
| Guided coverage summary, active-model provenance, fallback reason, execution provider | `PersonalizationViewModel`, `PersonalModelManager.LastResult` |
| **Per-attempt cue-lag report: cue, rep, try, frames, lag, correlation, weight, quality** | `training/babble_personal/labels.py : format_cue_lag_report`, written to `guided_quality.json` beside every run |
| Per-dimension MAE, neutral report, cross-talk (`ACTIVATION_THRESHOLD = 0.15`), range retention (`RANGE_RETENTION_FLOOR = 0.8`), false-activation run stats | `training/babble_personal/evaluate.py` |
| "Why did this cue score that way, dimension by dimension" | `training/babble_personal/diagnose_guided.py : attempt_dimension_report` |

---

## 4. Change 1 — Eye Gaze Responsiveness

### 4.1 Current architecture

One Euro on the 6 raw sigmoid values, **before** `ProcessExpressions`. It is the only temporal stage. `minCutoff = 0.5`, `beta = 3` by default, shared with the face pipeline.

### 4.2 Implementation strategy

**Multiply, do not replace.** A responsiveness scalar `r` scales the eye filter's One Euro parameters at construction. `r = 1.0` reproduces the current filter *exactly*, bit for bit. Your existing manual One Euro tuning stays the reference point.

**Where:** `EyePipelineManager.LoadFilter()` only. Do not touch `FacePipelineManager`.

```csharp
// EyePipelineManager.LoadFilter()
var enabled = _localSettings.ReadSetting("AppSettings_OneEuroEnabled", true);
var baseMin  = _localSettings.ReadSetting("AppSettings_OneEuroMinFreqCutoff", 0.5f);
var baseBeta = _localSettings.ReadSetting("AppSettings_OneEuroSpeedCutoff", 3f);
var r        = Math.Clamp(_localSettings.ReadSetting("AppSettings_EyeGazeResponsiveness", 1f), 0.25f, 2.0f);

// beta is weighted far more than minCutoff: beta governs how far the cutoff opens with
// gaze VELOCITY (the "reserved on saccades" complaint) without adding jitter at rest,
// whereas raising minCutoff trades rest-jitter for very little perceived snappiness.
var eyeMin  = Math.Clamp(baseMin  * MathF.Pow(r, 0.5f),  0.05f, 10f);
var eyeBeta = Math.Clamp(baseBeta * MathF.Pow(r, 1.5f),  0.0f,  30f);
```

At `r = 1` both `Pow` terms are exactly 1 → identical construction → **provable no-op at default**.

| `r` | `minCutoff` (from 0.5) | `beta` (from 3.0) | Feel |
|---|---|---|---|
| 0.25 | 0.25 | 0.375 | Heavily smoothed, very reserved |
| 0.50 | 0.354 | 1.06 | Smoother than stock |
| **1.00** | **0.500** | **3.000** | **Exactly today** |
| 1.50 | 0.612 | 5.51 | Noticeably snappier saccades |
| 2.00 | 0.707 | 8.49 | Closest to Paper Tracker |

Both ends stay inside the 0.1–10 bounds the existing manual NumericUpDowns already accept, so no new regime is being entered.

### 4.3 Prerequisite fix (not optional)

`AppSettingsViewModel.PropertyChanged` unconditionally calls `LoadFilter()`, and `LoadFilter()` constructs `new OneEuroFilter(new float[6], ...)` — **`xPrev` seeded to zeros**. Dragging a slider fires `PropertyChanged` per tick; the eyes will snap toward center on every one. Fix both halves:

1. Add a state-preserving reconfigure to `OneEuroFilter` and call it when only the coefficients changed:

```csharp
/// Updates coefficients in place. Deliberately does NOT touch xPrev/dxPrev/tPrev: rebuilding the
/// filter reseeds xPrev to zeros, which yanks the eyes toward center for several frames every
/// time any setting on the page changes.
public void Reconfigure(float minCutoff, float beta)
{
    for (var i = 0; i < this.minCutoff.Length; i++) { this.minCutoff[i] = minCutoff; this.beta[i] = beta; }
}
```

2. Guard the VM so the filter is only reloaded for filter-relevant properties:

```csharp
PropertyChanged += (_, p) =>
{
    _localSettingsService.Save(this);

    if (p.PropertyName is nameof(OneEuroMinEnabled) or nameof(OneEuroMinFreqCutoff)
                       or nameof(OneEuroSpeedCutoff) or nameof(EyeGazeResponsiveness))
    {
        _facePipelineManager.LoadFilter();
        _eyePipelineManager.LoadFilter();
    }

    if (p.PropertyName == nameof(StabilizeEyes)) _eyePipelineManager.LoadEyeStabilization();
};
```

This also removes an existing latent glitch (typing in the OSC prefix box currently rebuilds both filters).

### 4.4 Settings / UI

```csharp
[ObservableProperty]
[property: SavedSetting("AppSettings_EyeGazeResponsiveness", 1f)]   // MUST be `1f`, not `1.0`
private float _eyeGazeResponsiveness;
```

Place it in `AppSettingsView.axaml` **inside the existing One Euro `Expander`**, as a third row in that grid — it is meaningless when One Euro is disabled, and colocating makes that obvious. Use RESX keys (`Settings_EyeResponsiveness_Header` / `_Description`).

```xml
<StackPanel Orientation="Horizontal" Spacing="12" VerticalAlignment="Center">
  <Slider Width="240" Minimum="0.25" Maximum="2.0"
          TickFrequency="0.25" TickPlacement="BottomRight" IsSnapToTickEnabled="False"
          Value="{Binding EyeGazeResponsiveness, Mode=TwoWay}" />
  <TextBlock VerticalAlignment="Center" MinWidth="48"
             Text="{Binding EyeGazeResponsiveness, StringFormat='{}{0:P0}'}" />
</StackPanel>
```

Recommended: **range 0.25–2.00, default 1.00, displayed as 25 %–200 %**. Label the ends "Smoother" / "Snappier". Add a reset affordance or a tick at 1.00 so the default is findable by feel.

### 4.5 Guardrails against jitter and overshoot

- **Overshoot is structurally impossible.** One Euro is a first-order EMA with `a ∈ (0,1)`; `xHat` is a convex combination of `x` and `xPrev`, so the output can never exceed the input range regardless of `minCutoff`/`beta`.
- **Jitter** is the only real risk, and it comes from `minCutoff`. The `Pow(r, 0.5)` exponent means max responsiveness raises `minCutoff` only 1.41× while raising `beta` 2.83× — deliberately buying snappiness from the velocity term.
- Clamp `r` to `[0.25, 2.0]` **in the manager**, not only in the slider, so a hand-edited settings file cannot inject an absurd value.
- Test at max responsiveness with the eyes deliberately still for 30 s. If rest jitter appears, lower the `minCutoff` exponent to `Pow(r, 0.25)` before narrowing the range.
- **Do not** add a deadzone to suppress jitter. There is none today, and adding one would reintroduce exactly the "reserved" feel this change exists to remove.

### 4.6 Risks

| Risk | Mitigation |
|---|---|
| Face tracking regresses | Change confined to `EyePipelineManager.LoadFilter`. Face reads the same two base keys and is untouched. |
| Filter rebuild snaps eyes to center | §4.3 prerequisite. **Do not skip.** |
| Responsiveness silently does nothing | `LoadFilter` early-returns when One Euro is disabled and leaves the stale filter installed. Either place the slider inside the One Euro expander (recommended) or fix the early return to `_pipeline.Filter = null` — the latter is a behavior change; flag it separately. |
| Home build replaced the filter entirely | See §12. If One Euro is gone, apply `r` to whatever the equivalent smoothing coefficient is, keeping "r = 1 ⇒ identity". |

### 4.7 Optional and **separate** — Eye Gaze Range

Only if diagnosis shows the reserved feeling is *amplitude*, not *latency*. **Do not merge this with responsiveness.**

Two candidate implementations, in order of preference:

**(a) Reuse the existing calibration path — zero new mechanism.** `LeftEyeX/Y` and `RightEyeX/Y` already have persisted `CalibrationParameter` entries applied at send time; they just have no UI. Add them to `CalibrationViewModel.EyeSettings` alongside the lids and they appear as range sliders immediately. Fully backward compatible (defaults are identity). ⚠ Eye channels are **not clamped** after remap — widening the range can push past ±1, which becomes >45° on the native VRC path.

**(b) A single scalar gain.** Make `mulV`/`mulY` in `ProcessExpressions` a settable `GazeScale` property loaded by an `EyePipelineManager.LoadGazeScale()` alongside `LoadEyeStabilization()`. Range 0.5–1.5, default 1.0. This lands *before* the calibration remap so existing calibration still composes on top, and before the `StabilizeEyes` convergence math so the whole gaze field scales consistently. **Add a `Math.Clamp(v, -1f, 1f)` on the four gaze outputs when scale > 1.0** to protect the unclamped send path.

Prefer (a) — it adds no code to the hot path.

### 4.8 Validation

1. Set responsiveness to exactly 100 %, restart, confirm gaze is subjectively **identical** to before the change. Ideally capture an OSC trace before and after and diff — at `r = 1` it should match to floating-point equality.
2. Sweep 25 % → 200 % while looking around. Confirm monotonic snappiness, no oscillation, no overshoot past the gaze extremes.
3. Hold the eyes still for 30 s at 200 %. Watch `/LeftEyeX` on an OSC monitor for rest jitter.
4. Drag the slider continuously and confirm the eyes do **not** snap to center (validates §4.3).
5. Confirm face tracking is unchanged at every responsiveness setting.
6. Delete the `AppSettings_EyeGazeResponsiveness` key from `LocalSettings.json`, restart, confirm it resolves to 1.0 and behavior is stock.

---

## 5. Change 2 — Eye Squint Strength

### 5.1 Current architecture — **this checkout cannot tell you**

**[MAY-DIFFER — treat as fully unresolved.]** Stated plainly so no implementer designs from the wrong tree:

- This checkout produces **no eye squint at all**. `ProjectLegacyEyeOutput` drops it, the 6-channel `_eyeExpressionMap` has no slot, and `BabbleVRC.cs` has the `EyeSquintLeft/Right` assignments commented out.
- Your home Alpha Expressions build **does** output `EyeSquintLeft`/`EyeSquintRight` and squint works well.

**Instruction to the implementer: do not add squint support. Do not restore anything from `experimental/eye-v2`. Do not widen the 6-slot contract. Trace the working path that already exists.**

### 5.2 How to trace it at home

Search, in this order:

```
EyeSquintLeft | EyeSquintRight | eyeSquint | /LeftEyeSquint | /RightEyeSquint
UnifiedExpressions.EyeSquint
_eyeExpressionMap | EyeExpressionMap | ExpressionMapping
ProjectLegacyEyeOutput | EyeRawExpressions | ProcessExpressions
```

Then answer these four questions and write the answers down before editing anything:

1. **Where is squint *produced*?** A native model output channel, or derived from the lid scalar (a dwell-timer / hysteresis scheme like the shelved `EyeV2Mapper`), or something new?
2. **Is squint an *input* to anything else?** In this checkout `lid` feeds the gaze fusion and the yaw-borrow blend. If Alpha computes squint from lid, or lid from squint, scaling one in place will perturb gaze.
3. **What is the last point at which the value is mutable?** In this checkout it is `ParameterSenderService.ProcessEyeExpressionData`, immediately before `_vrcftQueue.Enqueue` — `BabbleVrc.Update` downstream is a verbatim copy with no arithmetic.
4. **Does squint pass through a `CalibrationParameter` remap?** If so, scale **after** the remap so the user's calibrated range is preserved and only its expression is scaled.

### 5.3 Implementation strategy — ranked insertion points

**Rank 1 (recommended): the last mutable point before OSC enqueue.** In this checkout's shape, inside `ProcessEyeExpressionData`, applied only to the two squint channels, **after** the `Remap`:

```csharp
var value = weight.Remap(settings.Lower, settings.Upper, settings.Min, settings.Max);

// Output-side expression strength only. Applied after the calibration remap so the user's
// calibrated range is preserved and only how strongly it is EXPRESSED changes. Detection,
// blink, lid and gaze are all upstream and untouched.
if (eyeElement.Key is "LeftEyeSquint" or "RightEyeSquint")
    value = Math.Clamp(value * _eyeSquintStrength, 0f, 1f);
```

Why this is safest: it is provably terminal (nothing downstream transforms it), it is trivially revertible, it cannot feed back into gaze or lid, and it works identically whether squint is a native channel or lid-derived.

**Rank 2 (acceptable):** immediately after squint is produced in the eye post-processing stage — **only if** question 2 above answered "squint is not an input to anything else". Preferred if squint needs scaling before the calibration remap for some Alpha-specific reason.

**Rank 3 (avoid):** anywhere **before** the One Euro filter. Scaling pre-filter changes the filter's own dynamics, because `cutoff = minCutoff + beta·|dxHat|` and `dxHat` scales with the signal — the smoothing behavior would then vary with the strength setting. Also avoid touching `ProjectLegacyEyeOutput` or any model-projection code.

**Math:**

```
processedSquint = clamp(detectedSquint * strength, 0, 1)
```

- `strength = 0` → fully suppressed (`0 × x = 0` for any `x`). ⚠ If Alpha has an additive squint floor or bias, `0` will not fully suppress — check for one.
- `strength = 1` → **bit-identical** to today. Add a `if (strength == 1f)` short-circuit if you want that guaranteed without relying on float multiply-by-one.
- `strength = 2` → exaggerated, saturating early at the top of the range. That compression is expected and acceptable; do not add a soft-knee curve unless you actually dislike the feel.

**Left/right:** apply the **same scalar independently to each channel**. Never average, blend, or couple them — that would destroy the asymmetry Alpha detects.

### 5.4 Blink interaction

The rule that keeps blink safe: **scale the squint channel only; never touch the lid/openness channel.**

- If Alpha derives blink and squint from one lid scalar disambiguated by time (the `EyeV2Mapper` design: closure below a threshold ⇒ blink; a partial closure held past a dwell window ⇒ squint, released with an exponential decay), then scaling the *squint output* affects blink not at all. Ideal case.
- **[VERIFY-AT-HOME]** If Alpha *subtracts* squint from lid, or computes lid as a function of squint, scaling one desynchronizes them — a scaled-down squint would leave the eye reading more open than the face actually is. In that case apply the scalar at Rank 1 (after both have been finalized), never at Rank 2.
- Blink must remain crisp at `strength = 0`. Explicitly test this.

### 5.5 Settings / UI

```csharp
[ObservableProperty]
[property: SavedSetting("AppSettings_EyeSquintStrength", 1f)]   // `1f`, not `1.0`
private float _eyeSquintStrength;
```

Range **0.0–2.0, default 1.00, displayed as 0 %–200 %**, ticks at 0/50/100/150/200. Label: "Eye Squint Strength". Description: *"How strongly a detected squint is expressed on your avatar. Does not change what is detected as a squint."*

If the consumer is `ParameterSenderService` (a hosted background service, not the VM), use **Pattern B**: a `const string` key plus a clamping property, or have the sender re-read the setting in its existing loop — it already re-reads `AppSettings_OSCPrefix`, `VRC_UseNativeTracking`, and `AppSettings_UseDFR` every pass, so adding one more `ReadSetting` there costs nothing and needs no new plumbing.

### 5.6 Risks

| Risk | Mitigation |
|---|---|
| Scaling in the wrong place perturbs gaze or blink | §5.2 questions 1–2 answered **before** editing. Rank 1 placement is immune. |
| Changes what is detected | Placement is strictly post-detection. If you find yourself editing model output, projection, or a threshold, you are in the wrong place — stop. |
| Filter dynamics change with the setting | Never scale pre-filter. Rank 3 is listed only to be forbidden. |
| Old settings file has no key | `SavedSetting` default `1f` → identity. |
| Alpha added a squint calibration range | Scale *after* the remap so both compose predictably. |

### 5.7 Validation

1. At 100 %, confirm squint behavior is **indistinguishable** from before. OSC-trace diff if possible.
2. At 0 %, squint reads 0 on the avatar while **blink still works normally and crisply**. This is the most important test.
3. At 50 %, the same physical squint produces roughly half the avatar deflection.
4. At 200 %, confirm exaggeration with no flicker at the clamp boundary.
5. **Squint left eye only.** Confirm left changes and right does not, at every strength. Repeat mirrored.
6. Confirm **gaze is unaffected** at every strength — particularly at 0 % and 200 %, where a leak into the lid-weighted gaze math would show up.
7. Blink rapidly at 0 % and 200 %; confirm no squint bleed into blink and no lid desync.
8. Delete the key, restart, confirm 1.0 and stock behavior.

---

## 6. Change 3 — Toothy smile

### 6.1 Reframing the problem

You asked whether the toothy calibration fails to cross an activation threshold, whether closed-mouth smile "wins", whether blend strength is too weak, whether similarity is dominated by irrelevant dimensions. **None of those mechanisms exist.** There is no template bank, no similarity score, no threshold, no precedence between calibrations. There is one trained residual adapter and one scalar blend.

Given your answers — the calibration is a **preview-confirmed candidate**, and you completed the **full capture → train → export → activate loop with the model active** — the cheap explanations are eliminated. The remaining explanation is that **the adapter was trained on contradictory labels and learned to average them away.**

### 6.2 Primary hypothesis: the plain `Smile` cue trains teeth *off* **[VERIFIED-HERE]**

This is verifiable in this checkout and is the leading cause.

`CO_ACTIVATION_EXCLUSIONS` for the smile dimensions is:

```python
schema.INDEX_OF["MouthSmileLeft"]: (
    schema.INDEX_OF["MouthSmileRight"],
    schema.INDEX_OF["MouthDimpleLeft"],  schema.INDEX_OF["MouthDimpleRight"],
    schema.INDEX_OF["CheekPuffLeft"],    schema.INDEX_OF["CheekPuffRight"],
),   # ← and the mirror for MouthSmileRight
```

It does **not** include `MouthUpperUpLeft/Right`, `MouthLowerDownLeft/Right`, `MouthStretchLeft/Right`, `JawOpen`, or `MouthClose`.

Therefore **every frame of every plain-`Smile` hold supervises all of those toward 0 at `W_UNCUED_DIM = 0.25`.**

Now consider what actually happens during a `Smile` hold at level 1.0: **a person smiling at full intensity shows teeth.** The captured image looks very much like a toothy smile. That frame is teaching the network:

> "this image → smile high, upperUp **0**, lowerDown **0**, stretch **0**, jawOpen **0**"

Meanwhile the toothy cue teaches, on a nearly identical image:

> "this image → smile high, upperUp **high**, stretch **high**, jawOpen **small**"

These are direct contradictions on visually similar inputs. And the arithmetic is lopsided:

- `Smile` is in `CorePass`, so it appears in **every** core session, at 2 levels × N repetitions.
- The toothy cue is a single opt-in confirmed routine, `Levels: [1.0f]` (binary) — **one level**, far fewer attempts.

Even at weight 0.25, the plain-`Smile` frames outnumber the toothy frames enough to dominate the gradient. The network's least-loss solution is to **regress the toothy prediction toward the plain-smile label** — smile fires, teeth dimensions stay near zero. **That is precisely your reported symptom: the smile is there, the teeth are not.**

Compounding factor: the `masked_residual_loss` adds `shrinkage * residual.pow(2).mean()` (λ = 1e-2), which biases every under-supervised dimension toward "leave stock alone" — and per §3.6, stock is near-blind on `MouthLowerDown` for your rig.

### 6.3 Secondary hypothesis: the toothy attempts are training at weight ≈ 0 **[VERIFIED-HERE]**

`labels.py : _select_probe_dim` picks **one** dimension to grade each attempt, requiring `observed_range >= GUIDED_MIN_OBSERVABLE_RANGE (0.02)`. Its docstring is a written post-mortem of exactly this going wrong on the Grimace cue. Then:

- correlation ≥ `GUIDED_GOOD_CORRELATION (0.60)` → full weight
- correlation < 0.60 → `GUIDED_WEAK_WEIGHT_SCALE = 0.25`
- correlation ≤ `GUIDE_SUPPRESS_CORRELATION (0.10)` **and** a same-session same-cue peer scored ≥ 0.60 → **auto-suppressed to weight 0**

If the probe dim landed on `MouthLowerDown` — which the recorded note says reads ≈0.01 on your rig *whether the lip moves or not* — every toothy attempt scores near-zero correlation. **Your toothy holds may literally have contributed nothing to training.** This is directly checkable in `guided_quality.json`; see §6.5 step 2.

### 6.4 Additional candidates, in descending likelihood

| # | Hypothesis | How to confirm |
|---|---|---|
| 4 | **`MouthClose`(18) stays high during a toothy smile.** Nothing excludes it from the smile cue either, so it is trained toward 0 during Smile holds but is unconstrained during toothy. If the adapter emits high `MouthClose`, `/mouthClose → UnifiedExpressions.MouthClosed` will seal the avatar's lips regardless of how high `MouthUpperUp` goes. | Watch row 18 live in the Advanced comparison table while making a toothy smile. |
| 5 | **Cues captured in different sessions.** The discriminating evidence is *between* toothy and plain-smile. Captured separately, the network never sees them under matched lighting/camera pose, weakening the contrast. | Check `PersonalDataset\` session folders and which sessions the trainer consumed. |
| 6 | **Range retention failure.** `evaluate.py` has `RANGE_RETENTION_FLOOR = 0.8` — the model may be compressing the toothy dims' output range. | `evaluate.py` `dim_report` for dims 25–28, 31–32. |
| 7 | **Confirmed candidate target is too conservative.** The preview-confirm flow guarantees the target *looked right on your avatar*, which largely rules this out — but `JawOpen` too low can leave lips unparted on some avatar rigs. | Re-run `GrimacePreview`-style preview on the confirmed toothy candidate and look hard at the lip line. |
| 8 | **`ProsodyEnhancer` diluting magnitude.** Off by default; multiplies `Jaw*`/`Mouth*` by up to 1.6× while speaking. It multiplies smile and upperUp equally so it cannot cause *selective* failure, but check whether it is on. | `AudioAssist_Strength` setting; audio diagnostics readout. |

**Explicitly ruled out** by grep in this checkout: no late `JawOpen` derivation or re-clamp, no runtime tongue derivation, no runtime tongue gating on `JawOpen`, no smile post-scaling, no expression-conflict-resolution pass, no "generic open mouth" synthesis. Your worry that "JawOpen is recalculated later" or "tongue values are introduced later" has **no basis in this architecture** — the tongue↔jaw rule exists only as a training-label exclusion. **[VERIFY-AT-HOME]** that Alpha did not add such a stage.

### 6.5 Diagnostic procedure — run this **before changing any code**

Everything needed already exists (§3.7). Work top to bottom and stop when you find the cause.

**Step 1 — Is the correction reaching the output at all?**
Open Personalization → **Advanced** → the live Stock / Personal / Delta table. Make a toothy smile and hold it. Record `Stock`, `Personal`, and `Delta` for:

```
19 MouthSmileLeft   20 MouthSmileRight
25 MouthUpperUpLeft 26 MouthUpperUpRight
27 MouthLowerDownLeft 28 MouthLowerDownRight
31 MouthStretchLeft 32 MouthStretchRight
 4 JawOpen          18 MouthClose          33 TongueOut
```

Interpretation:
- **`Delta ≈ 0` on the teeth dims** → the adapter learned nothing for this pose. → hypotheses 2 or 3. Go to step 2.
- **`Personal` is high on the teeth dims but the avatar still looks closed-mouthed** → the problem is *downstream of the adapter*: check `MouthClose`(18) (hypothesis 4), then the calibration `Lower/Upper` for those dims in `CalibrationView`, then the avatar's own blendshape rig.
- **`JawOpen` high / `TongueOut` non-zero** → contradicts §6.4's grep result; something new exists in Alpha. Trace it before proceeding.

**Step 2 — Did the toothy attempts actually train?**
Open `guided_quality.json` beside the training run you exported the active model from (or re-run `format_cue_lag_report`). Find the toothy cue's rows and read `corr`, `weight`, `quality`, and which dimension was chosen as the probe.
- `weight = 0` or `quality = suppressed` → **hypothesis 3 confirmed.** Fix A2 in §6.6.
- `corr < 0.60` with a probe dim of `MouthLowerDown*` → **hypothesis 3 confirmed** via the blind-axis path. Fix A2.
- `corr ≥ 0.60`, full weight, and step 1 showed `Delta ≈ 0` → **hypothesis 2 confirmed.** Fix A1.

**Step 3 — Per-dimension detail.**
`python -m babble_personal.diagnose_guided` → `attempt_dimension_report` for the toothy cue. This tells you dimension by dimension what the stock model saw versus what was commanded.

**Step 4 — Contrast check.**
Confirm the plain `Smile` and toothy cues were captured in the **same session**. Confirm the toothy routine actually ran (frame counts in `session.json`).

**Step 5 — Cross-talk and range.**
`evaluate.py` → `cross_talk` (`ACTIVATION_THRESHOLD = 0.15`) and `dim_report` (`RANGE_RETENTION_FLOOR = 0.8`) for dims 25–28 and 31–32.

### 6.6 Recommended fix — Strategy A (training-side, primary)

This is personalization/feature-weighting, not retraining the base model. Fixes are ordered; apply only what diagnosis indicates.

**A1 — Stop the plain `Smile` cue from training teeth off.** *(Fixes hypothesis 2. Highest value.)*

In `training/babble_personal/labels.py`, extend the smile exclusions:

```python
schema.INDEX_OF["MouthSmileLeft"]: (
    schema.INDEX_OF["MouthSmileRight"],
    schema.INDEX_OF["MouthDimpleLeft"],   schema.INDEX_OF["MouthDimpleRight"],
    schema.INDEX_OF["CheekPuffLeft"],     schema.INDEX_OF["CheekPuffRight"],
    # A real smile parts the lips and shows teeth. Without these, every Smile hold teaches
    # "and the lips were together", which is false for a full smile and directly contradicts
    # the toothy-smile cue on a near-identical image.
    schema.INDEX_OF["MouthUpperUpLeft"],  schema.INDEX_OF["MouthUpperUpRight"],
    schema.INDEX_OF["MouthLowerDownLeft"],schema.INDEX_OF["MouthLowerDownRight"],
    schema.INDEX_OF["MouthStretchLeft"],  schema.INDEX_OF["MouthStretchRight"],
    schema.INDEX_OF["JawOpen"],
    schema.INDEX_OF["MouthClose"],
),   # mirror for MouthSmileRight
```

This does **not** teach teeth *on* during a plain smile. It says "have no opinion" — those dims drop out of the loss for smile frames rather than being pushed to zero. That is exactly the semantics `W_UNCUED_DIM` was designed to allow.

Add a matching entry for the toothy cue's primary dimension so *its* co-activations are protected too.

**A2 — Make the toothy cue gradable.** *(Fixes hypothesis 3.)*
Ensure `_select_probe_dim` cannot select a stock-blind axis for this cue. Prefer `MouthStretch*` (recorded at **+0.95** correlation on your rig) or `MouthSmile*` over `MouthLowerDown*` (≈0.01, blind). Either pin an explicit probe dim for the toothy cue or raise `GUIDED_MIN_OBSERVABLE_RANGE` for it. Do not change the global 0.02 floor — that would affect every other cue.

**A3 — Give the toothy cue proportional evidence.**
It is `Levels: [1.0f]` (binary), so it inherently produces fewer frames than two-level cues. Increase its repetitions in the routine choice, or apply a `--dim-boost` on the teeth dims for the toothy cue. `parse_dim_boost` already exists in the trainer.

**A4 — Capture toothy and closed-mouth smile in the same session.** *(Fixes hypothesis 5.)*
Same lighting, same camera pose, back to back. This is the contrast that makes the distinction learnable.

**Then retrain, export, and re-select the model.** Re-run step 1 of §6.5 to confirm `Delta` is now non-zero on dims 25–28 / 31–32.

### 6.7 On contrast poses — a qualified yes

The architecture **does** benefit from contrast, but not in the way template-matching would. There is no negative-example mechanism; the benefit is that adjacent, visually similar poses with *correct* labels force the network to find the discriminating feature (visible teeth).

**Recommended (worth it):**
- **Closed-mouth smile** — you already have this as `Smile`. Its labels are currently *wrong* for full-intensity smiles; A1 fixes that. **The highest-value contrast is fixing this cue's labels, not adding a new pose.**
- **Toothy smile** — already confirmed.

**Only if diagnosis shows it (hypothesis 6 / cross-talk report):**
- **Open mouth without smiling** — you already have `JawOpen`. Add a dedicated variant only if the cross-talk report shows toothy→JawOpen confusion.

**Do not add:**
- **Tongue out as a negative example.** `TongueOut` already exists with the correct `CO_ACTIVATION_EXCLUSIONS`, and this checkout has no runtime path by which a toothy smile could activate tongue parameters. Adding a pose to solve a problem that has no mechanism just costs capture time and dilutes the corpus.

The repo's own guidance applies: *"a cue nobody can imitate reliably produces confident wrong labels at full weight — worse than no cue at all."*

### 6.8 Fallback — Strategy B (runtime output correction)

Use **only** if Strategy A is exhausted and diagnosis showed the adapter genuinely cannot learn the distinction. This is post-processing/output correction, which you listed as acceptable.

Implement as an `IExpressionEnhancer` — the seam `ProsodyEnhancer` already uses:

```csharp
public interface IExpressionEnhancer { float[] Enhance(float[] expressions); }   // must never throw,
                                                                                 // must return a NEW array
```

Installed via `FaceProcessingPipeline.Enhancer` (a `volatile IExpressionEnhancer?`) through a DI callback in `App.axaml.cs`. It runs **after** the corrector and **before** the One Euro filter — the right place, because the filter then smooths the shaped signal.

**⚠ There is exactly one `Enhancer` slot.** A `ToothySmileShaper` must be **composed** with `ProsodyEnhancer` (a `CompositeExpressionEnhancer` applying each in turn), not assigned over it — otherwise installing one silently disables the other.

Shape of the rule — gated, multiplicative-with-headroom, never manufacturing:

```
if (smileL + smileR)/2 > smileGate  AND  jawOpen < jawCeiling:
    upperUpL/R  = clamp(upperUpL/R  + (1 - upperUpL/R) * lift * smileAmount, 0, 1)
    mouthClose  = mouthClose * (1 - closeSuppress * smileAmount)
    // jaw, tongue, cheeks, eyes: NEVER touched
```

Constraints: off by default; a single strength setting where 0 = exactly today; never write `JawOpen` or any `Tongue*` dim; gate on smile so it cannot fire during a neutral or open-mouth pose; keep the parameter mask to `MouthUpperUp*`, optionally `MouthClose`, and nothing else.

### 6.9 Base-model retraining — **not recommended**

The 45-value output carries `MouthUpperUpLeft/Right`, `MouthLowerDownLeft/Right`, `MouthStretchLeft/Right`, and `MouthClose` as separate channels from `JawOpen` and `MouthSmileLeft/Right`. The information needed to distinguish a toothy smile from an open mouth **is present in the base output**. The failure is in how it is being labelled and weighted, not in what the model can see.

Two of the relevant axes are near-blind on your specific rig (§3.6), but **that is exactly what the personal adapter exists to correct** — and `MouthStretch*` tracks the same expression at +0.95, giving the adapter a working axis to key on.

⚠ Also note: **adding a 46th output would change `PersonalizationSchema.Sha256`, which would reject every previously trained model at load and every previously recorded session in the trainer.** Do not go near the schema.

### 6.10 Validation

1. Neutral → nothing fires. `MouthSmile*`, `MouthUpperUp*`, `JawOpen`, `Tongue*` all near 0.
2. **Closed-mouth smile** → smile fires, `MouthUpperUp*` stays low, `JawOpen` low, lips visually together. **This must not regress** — A1 changes the smile cue's labels, so verify it explicitly.
3. **Toothy smile** → smile stays strong, `MouthUpperUp*` (and/or `MouthStretch*`) rises, `MouthClose` drops, `JawOpen` stays restrained, `Tongue*` stays at 0.
4. **Open mouth without smiling** → `JawOpen` high, `MouthSmile*` low. No smile bleed.
5. **Tongue out** → still works. `JawOpen` co-activates correctly (its exclusion permits it).
6. **Transitions**: neutral → closed smile → toothy → open mouth → back. Smooth throughout, no popping, no flicker at any boundary.
7. Cheek, eye, and squint contribution during a toothy smile looks natural — verify squint especially, since Change 2 also touches it.
8. Re-run `evaluate.py`: `neutral_report`, `cross_talk`, `dim_report` range retention, and the false-activation run stats for `WATCHED_DIMS = (JawOpen, TongueOut)`.
9. Confirm no regression on the other core expressions — frown, pucker, funnel, mouth left/right.

---

## 7. Files, components and symbols found in this checkout

**Landmarks, not requirements.** Paths are relative to the repo root.

### Eye pipeline
| Path | Responsibility |
|---|---|
| `src/Baballonia/Services/Inference/EyeProcessingPipeline.cs` | `RunUpdate`, `ProjectLegacyEyeOutput`, `ProcessExpressions`, `StabilizeEyes`, `ResetTemporalState` |
| `src/Baballonia/Services/Inference/EyePipelineManager.cs` | `InitializePipeline`, `LoadFilter`, `LoadEyeStabilization`, `SetFilter`, `SetLeft/RightTransformation`, camera stop/start |
| `src/Baballonia/Services/Inference/Filters/OneEuroFilter.cs` | The only temporal filter |
| `src/Baballonia/Contracts/IFilter.cs` | `float[] Filter(float[] input)` |
| `src/Baballonia/Services/Inference/ImageCollector.cs` | 4-frame temporal stack, channel reverse, `Reset()` |
| `src/Baballonia/Services/Inference/Transformers/DualImageTransformer.cs` | Per-eye ROI/gamma/rotate/flip/resize |
| `src/Baballonia/Services/ProcessingLoopService.cs` | 10 ms `DispatcherTimer`, `ExpressionChangeEvent` |
| `src/Baballonia/Services/events/PipelineEvents.cs` | `NewRawModelOutputEvent`, `NewRawExpressionsEvent`, `NewFilteredResultEvent` |
| `src/Baballonia/Utils/Utils.cs` | `EyeRawExpressions = 6`, `FaceRawExpressions = 45`, `PersistentDataDirectory` |

### Output
| Path | Responsibility |
|---|---|
| `src/Baballonia/Services/ParameterSenderService.cs` | `_eyeExpressionMap` (6), `FaceExpressionMap` (45), `ProcessEyeExpressionData`, `ProcessFaceExpressionData`, `ProcessNativeVrcEyeTracking`, `EnqueueRawFaceVector` |
| `src/Baballonia/Helpers/FloatExtensions.cs` | `Remap` |
| `src/Baballonia/Services/Calibration/CalibrationParameter.cs` | `Lower/Upper/Min/Max` |
| `src/Baballonia/Services/CalibrationService.cs` | `GetExpressionSettings`, `SetExpression`, `Reset*`, persists to `CalibrationParams` |
| `src/VRCFaceTracking.Baballonia/BabbleOSC.cs` | UDP listen, address → slot |
| `src/VRCFaceTracking.Baballonia/BabbleVRC.cs` | Verbatim copy into `UnifiedTracking.Data`; **`EyeSquintLeft/Right` commented out** |
| `src/VRCFaceTracking.Baballonia/ExpressionMapping.cs` | Slot indices |

### Settings & UI
| Path | Responsibility |
|---|---|
| `src/Baballonia/Contracts/ILocalSettingsService.cs` | `SavedSettingAttribute`, `ReadSetting`/`SaveSetting`/`Load`/`Save`/`ForceSave` |
| `src/Baballonia/Services/LocalSettingsService.cs` | Flat JSON store, 2 s debounce |
| `src/Baballonia/ViewModels/SplitViewPane/AppSettingsViewModel.cs` | One Euro properties, the `PropertyChanged` → `LoadFilter` wiring |
| `src/Baballonia/Views/AppSettingsView.axaml` | One Euro `Expander`, `StabilizeEyes` toggle |
| `src/Baballonia/Controls/SettingsBlock.axaml` | Reusable settings row |
| `src/Baballonia/Assets/Resources.resx` + `Resources.Designer.cs` | RESX strings (Crowdin-managed) |
| `src/Baballonia/ViewModels/SplitViewPane/CalibrationViewModel.cs` | `EyeSettings` (lids only), `_eyeKeyIndexMap`, `ApplyCurrentEyeExpressionValues` |
| `src/Baballonia/Models/SliderBindableSetting.cs` | Range-slider item model |
| `src/Baballonia/Controls/RangeSlider.cs` | Dual-thumb slider |

### Guided calibration & personalization
| Path | Responsibility |
|---|---|
| `src/Baballonia/Services/Personalization/GuidedCues.cs` | `GuidedCue` record; 13 poses; `All`/`CorePass`/`CombinationPass`. **No toothy cue here.** |
| `src/Baballonia/Services/Personalization/GrimacePreviewService.cs` | `GrimaceCandidate`, `GrimaceCandidateCatalog`, preview + confirm + fingerprint + `CreateConfirmedCue`. **The pattern your toothy calibration uses.** |
| `src/Baballonia/Services/Personalization/GuidedRoutineChoice.cs` | Pickable routines |
| `src/Baballonia/Services/Personalization/GuidedCaptureRoutine.cs` | Phase machine, `CueTiming`, `HoldSettleTrimSeconds` |
| `src/Baballonia/Services/Personalization/GuidedCalibrationService.cs` | `Preflight`/`Start`/`Tick`/`StopAsync`, `WriteQualityOverridesAsync` |
| `src/Baballonia/Services/Personalization/DatasetRecorderService.cs` | `CapturedFrame`, session writing, 30 fps cap, checksum dedup |
| `src/Baballonia/Services/Personalization/DatasetSession.cs` | `SessionMetadata`, `FrameLabel`, `CueLabel`, `PersonalizationPaths` |
| `src/Baballonia/Services/Personalization/PersonalizationSchema.cs` | Canonical 45 names, `Sha256` |
| `src/Baballonia/Services/Personalization/PersonalModelManager.cs` | Discovery, `Validate`, `SelectModelAsync`, `ReloadAsync`, `Blend`, `.previous` rollback |
| `src/Baballonia/Services/Personalization/PersonalModelCorrector.cs` | Adapter inference + `stock + (personal-stock)*blend` |
| `src/Baballonia/Services/Personalization/IExpressionCorrector.cs` | Corrector contract |
| `src/Baballonia/Services/Personalization/Audio/ProsodyEnhancer.cs` | **`IExpressionEnhancer` is declared here**; `AffectedDims` = every `Jaw*`/`Mouth*` |
| `src/Baballonia/Services/Personalization/ExpressionOverrideService.cs` | Drives the avatar during capture; keep-alive deadman |
| `src/Baballonia/Services/Personalization/CueStateSource.cs` | Fail-closed label emission |
| `src/Baballonia/Services/Inference/FaceProcessingPipeline.cs` | `Corrector` → `Enhancer` → `Filter` chain; **one `Enhancer` slot** |
| `src/Baballonia/Models/ExpressionComparisonRow.cs` | **The live 45-row Stock/Personal/Delta debug table** |
| `src/Baballonia/ViewModels/SplitViewPane/PersonalizationViewModel.cs` | `ShowAdvanced`, `PersonalStrength`, `RefreshGrimaceState`, comparison buffer |
| `src/Baballonia/Views/PersonalizationView.axaml` | Advanced section; hardcoded English by design |

### Trainer
| Path | Responsibility |
|---|---|
| `training/babble_personal/labels.py` | **Weights, `CO_ACTIVATION_EXCLUSIONS`, `_select_probe_dim`, `format_cue_lag_report`, `guided_quality_artifact`** |
| `training/babble_personal/schema.py` | Python mirror of the 45 names, `schema_sha256`, `assert_compatible` |
| `training/babble_personal/models.py` | Adapters A/B/C, `masked_residual_loss` |
| `training/babble_personal/evaluate.py` | `per_expression_mae`, `neutral_report`, `cross_talk`, `dim_report`, `coverage_report` |
| `training/babble_personal/diagnose_guided.py` | `attempt_dimension_report` |

### Not compiled / not in the solution
- `experimental/eye-v2/**` — `EyeV2Mapper` (lid-derived blink/squint by dwell time), `EyeGazeMap`, `EyeLidAnchors`. **Reference only. Do not restore.**
- `src/BabblePersonalizer*` — a standalone statistical calibration tool, not referenced by the app.

---

## 8. Recommended implementation order

Do them in this order and commit separately, so any one can be reverted alone.

1. **§4.3 prerequisite** — `OneEuroFilter.Reconfigure` + the `PropertyChanged` property-name guard. Behavior-neutral, fixes an existing glitch, unblocks Change 1. Commit alone.
2. **Change 1** — Eye Gaze Responsiveness. Self-contained, immediately gratifying, validates the settings plumbing end to end.
3. **§6.5 diagnostics** — read-only investigation of the toothy failure. No code changes. Do this *before* Change 2 so the findings are fresh; it is the longest-lead item.
4. **Change 2** — Eye Squint Strength. Trace first (§5.2), then implement.
5. **Change 3 fix** — apply only what §6.5 indicated, retrain, export, re-select, re-validate.
6. **Optional extras** (§14) — only if wanted, and only after 1–5 are validated.

Rationale: 1–2 are low-risk and prove the plumbing. 3 is investigation whose results may change 5. 4 is independent. 5 has the longest cycle time (capture + train + export + validate) and should not block anything else.

---

## 9. Settings: defaults, ranges, persistence

| Key | Type | Default | Range | Stored | Consumer |
|---|---|---|---|---|---|
| `AppSettings_EyeGazeResponsiveness` | `float` | **`1f`** | 0.25–2.00 (clamped in the manager too) | `[SavedSetting]` on `AppSettingsViewModel` | `EyePipelineManager.LoadFilter` |
| `AppSettings_EyeSquintStrength` | `float` | **`1f`** | 0.00–2.00 (clamped at the consumer too) | `[SavedSetting]`, or Pattern B if the consumer is `ParameterSenderService` | Last mutable point before OSC |
| `AppSettings_EyeGazeRange` *(optional, §4.7b)* | `float` | **`1f`** | 0.50–1.50 | `[SavedSetting]` | `EyePipelineManager.LoadGazeScale` → `ProcessExpressions` |

Rules:

- **`1f`, never `1.0`.** The attribute default is boxed and assigned via `PropertyInfo.SetValue`; a type mismatch throws and is swallowed into a log line.
- **Always pass the default at every `ReadSetting` call site.** `ReadSetting<float>("Key")` with no default returns `0f`, not the attribute default. This is a live bug in the existing `LoadFilter` methods.
- **Backward compatibility is automatic.** `Load(object)` assigns the attribute default when a key is absent, null, or fails to deserialize. Old settings files need no migration.
- **Clamp in the consumer, not only in the UI**, so a hand-edited JSON file cannot inject an out-of-range value.
- Do **not** add these to the seeded `src/Baballonia/LocalSettings.json`. Letting them be absent proves the default path works. Add them only after that is verified.
- Do **not** repurpose `AppSettings_OneEuroMinFreqCutoff` / `AppSettings_OneEuroSpeedCutoff` — face and eyes share them.

---

## 10. Regression risks

| Area | Risk | Guard |
|---|---|---|
| **Face tracking** | Change 1 touches a manager whose face twin reads the same keys | Edit `EyePipelineManager` only. Diff `FacePipelineManager` at the end to confirm it is untouched. |
| **Eye gaze** | Change 2 scaling in the wrong place leaks into the lid-weighted gaze math | Scale at the last mutable point (§5.3 Rank 1). Test gaze at squint strength 0 % and 200 %. |
| **Blink** | Squint and blink may share a lid scalar in Alpha | Never scale the lid channel. Test blink crispness at strength 0 %. |
| **Closed-mouth smile** | **A1 changes the plain `Smile` cue's training labels** | Explicitly re-validate closed-mouth smile after retraining. This is the most likely regression in the whole plan. |
| **Jaw / tongue** | A1 adds `JawOpen`/`MouthClose` to the smile exclusions | Exclusion means "no opinion", not "train on". Confirm via `cross_talk` and the `WATCHED_DIMS` false-activation run stats. |
| **Existing personal model** | Any schema change invalidates every trained model and recorded session | **Do not touch `PersonalizationSchema.Names` or its SHA-256.** Nothing in this plan requires it. |
| **Prosody enhancer** | A `ToothySmileShaper` assigned to `Enhancer` would silently disable it | Only one `Enhancer` slot. Use a composite if Strategy B is ever needed. |
| **Camera recovery / reliability** | Nothing here touches camera lifecycle | Do not edit `Stop*Camera`, `ResetTemporalState`, or `FastCorruptionDetector`. |
| **Guided calibration** | Change 3 edits trainer weights, not the capture flow | Do not change `HoldSettleTrimSeconds` — it is manually mirrored in C# and Python and must move together. |
| **Old settings files** | New keys absent | Attribute defaults resolve to current behavior. Test by deleting the keys. |
| **UI thread** | Both pipelines run synchronously on the `DispatcherTimer` | Keep all added per-frame work to a couple of multiplies. Add no allocation to the tick path. |

---

## 11. Testing checklist

**Regression baseline — capture before any change**
- [ ] OSC trace of gaze, lid, and squint during a scripted eye routine (look L/R/U/D, blink, squint, hold still 30 s)
- [ ] OSC trace of neutral, closed smile, toothy smile, open mouth, tongue out
- [ ] Screenshot of the Advanced Stock/Personal/Delta table for each of those poses
- [ ] Note the active model path, `PersonalModel_Blend`, and `AudioAssist_Strength`

**Change 1**
- [ ] 100 % is byte-identical to baseline
- [ ] Monotonic snappiness across 25 % → 200 %
- [ ] No overshoot at any setting
- [ ] No rest jitter at 200 % over 30 s
- [ ] Dragging the slider does not snap the eyes to center
- [ ] Face tracking unchanged at every setting
- [ ] Key deleted → resolves to 1.0 → stock behavior

**Change 2**
- [ ] 100 % identical to baseline
- [ ] 0 % suppresses squint fully **and blink still works crisply**
- [ ] 50 % ≈ half deflection
- [ ] 200 % exaggerates without flicker at the clamp
- [ ] Left-only squint moves left only, at every strength; mirrored
- [ ] Gaze unaffected at 0 % and 200 %
- [ ] Rapid blinking at 0 % and 200 % — no bleed, no lid desync
- [ ] Key deleted → resolves to 1.0

**Change 3**
- [ ] Neutral fires nothing
- [ ] **Closed-mouth smile unregressed** (lips together, `MouthUpperUp*` low)
- [ ] Toothy smile: smile strong, teeth dims up, `MouthClose` down, `JawOpen` restrained, `Tongue*` at 0
- [ ] Open mouth without smiling: `JawOpen` high, no smile bleed
- [ ] Tongue out still works
- [ ] Transitions neutral → closed → toothy → open → back are smooth, no popping
- [ ] Cheek/eye/squint contribution natural during a toothy smile
- [ ] `evaluate.py`: `neutral_report`, `cross_talk`, `dim_report` range retention, `WATCHED_DIMS` false-activation runs
- [ ] Frown, pucker, funnel, mouth left/right unregressed

**Cross-cutting**
- [ ] `dotnet test` green (`PersonalizationSchemaTests` especially)
- [ ] Camera hot-unplug/replug recovery still works
- [ ] Settings survive a restart; the file is still valid JSON
- [ ] No new per-frame allocation in `RunUpdate`

---

## 12. Home-PC reconciliation checklist

For each change: the subsystem, this checkout's symbols, their responsibility, what to search for if they moved, and what tells you the home build has materially diverged.

### Change 1 — Eye responsiveness

| | |
|---|---|
| **Subsystem** | Eye temporal filtering |
| **Symbols here** | `EyePipelineManager.LoadFilter()`, `OneEuroFilter`, `IFilter`, `EyeProcessingPipeline.Filter`, `EyePipelineManager.SetFilter` |
| **Responsibility** | Construct the eye filter from settings and install it on the pipeline |
| **If renamed, search for** | `IFilter`, `OneEuroFilter`, `minCutoff`, `beta`, `SmoothingFactor`, `ExponentialSmoothing`, `LoadFilter`, `SetFilter`, `OneEuroMinFreqCutoff`, `OneEuroSpeedCutoff`, `xPrev`, `dxPrev` |
| **Expected data flow** | settings → manager → `new OneEuroFilter(...)` → `pipeline.Filter` → applied per tick before `ProcessExpressions` |
| **Divergence warning signs** | ⚠ A second smoothing stage exists (a post-`ProcessExpressions` EMA, a velocity limiter, a deadzone, output interpolation) — then responsiveness must target **the dominant** stage, or both. ⚠ Eye and face filters no longer share settings keys — then some of the §4.2 reasoning is already handled. ⚠ The filter is fed *after* `ProcessExpressions` rather than before — coefficient meaning changes because the value range is now [-1,1] not [0,1]. ⚠ `dt` now comes from the tick rather than `DateTime.UtcNow`. ⚠ One Euro was replaced entirely — then apply `r` to whatever the smoothing coefficient is, preserving "r = 1 ⇒ identity". |
| **Verify before editing** | Does `LoadFilter` still reseed `xPrev` to zeros? Is it still called on every `PropertyChanged`? If you already fixed either, skip that part of §4.3. |

### Change 2 — Squint strength

| | |
|---|---|
| **Subsystem** | Eye expression output |
| **Symbols here** | **None — squint does not exist in this checkout.** Nearest landmarks: `ParameterSenderService._eyeExpressionMap`, `ProcessEyeExpressionData`, `EyeProcessingPipeline.ProjectLegacyEyeOutput`, `BabbleVRC.Update`, `ExpressionMapping` |
| **Responsibility (in Alpha)** | Produce `EyeSquintLeft/Right`, remap, enqueue to OSC, copy into `UnifiedExpressions` |
| **Search for** | `EyeSquintLeft`, `EyeSquintRight`, `eyeSquint`, `/LeftEyeSquint`, `UnifiedExpressions.EyeSquint`, `_eyeExpressionMap`, `EyeRawExpressions`, `ProjectLegacyEyeOutput`, `ProcessExpressions` |
| **Expected data flow** | model or derivation → eye post-processing → per-channel remap → OSC → verbatim copy in the VRCFT module |
| **Divergence warning signs** | ⚠ Squint is derived from the lid scalar rather than a model channel — then check whether lid is also modified as a side effect. ⚠ Squint feeds *back* into gaze or lid computation — then Rank 1 placement is mandatory. ⚠ Squint already has a calibration `CalibrationParameter` — scale after its remap. ⚠ The VRCFT module now transforms rather than copies — then the last mutable point is inside the module, not the app. ⚠ `Utils.EyeRawExpressions` is no longer 6 — the whole 6-slot contract moved; re-derive the channel indices. |
| **Verify before editing** | Answer all four questions in §5.2 in writing. Do not edit until you can name the exact last mutable point. |

### Change 3 — Toothy smile

| | |
|---|---|
| **Subsystem** | Guided calibration + personal adapter training |
| **Symbols here** | `GuidedCues`, `GuidedCue`, `GrimaceCandidate`, `GrimaceCandidateCatalog`, `GrimacePreviewService`, `GuidedRoutineChoice`, `DatasetRecorderService`, `PersonalModelManager`, `PersonalModelCorrector`, `labels.py`, `evaluate.py`, `diagnose_guided.py` |
| **Responsibility** | Define poses → drive avatar → record (image, stock, target) → weight labels → train residual → export → load → lerp at runtime |
| **Search for** | `toothy`, `teeth`, `grin`, `smile with teeth`, `CandidateCatalog`, `ConfirmedCue`, `Guided_Confirmed`, `CO_ACTIVATION_EXCLUSIONS`, `W_UNCUED_DIM`, `_select_probe_dim`, `GUIDED_GOOD_CORRELATION`, `guided_quality` |
| **Expected data flow** | See §3.5 |
| **Divergence warning signs** | ⚠ A *runtime* similarity/threshold/blend system now exists — then the §6.2 diagnosis does not apply and you must re-derive it from that system. ⚠ A post-corrector stage now writes `JawOpen` or `Tongue*` — this checkout has none; if Alpha added one, it becomes hypothesis #1. ⚠ A second `IExpressionEnhancer` is installed — check for a composite; a bare assignment would have disabled `ProsodyEnhancer`. ⚠ `PersonalizationSchema.Names` changed — everything about model compatibility changes; stop and reassess. ⚠ `CO_ACTIVATION_EXCLUSIONS` already has smile→teeth entries — then A1 is already done and you should go straight to hypothesis 3. |
| **Verify before editing** | Confirm the toothy calibration is a preview-confirmed candidate in the `GrimaceCandidate` shape. Confirm which training run produced the currently active model. Confirm `PersonalModel_Blend` > 0 and which adapter type (A/B/C) is loaded. |

### Settings & UI (all changes)

| | |
|---|---|
| **Symbols here** | `ILocalSettingsService`, `SavedSettingAttribute`, `LocalSettingsService`, `AppSettingsViewModel`, `AppSettingsView.axaml`, `SettingsBlock`, `Assets/Resources.resx` |
| **Search for** | `SavedSetting`, `ReadSetting`, `SaveSetting`, `ObservableProperty`, `SettingsBlock`, `LocalSettings.json` |
| **Divergence warning signs** | ⚠ Settings became async or typed-POCO-backed — adjust the call shape but keep "missing key ⇒ default". ⚠ A settings-migration/versioning layer now exists — register the new keys with it. ⚠ `AppSettingsView` was reorganized — place the sliders by *meaning* (next to One Euro / next to eye settings), not by line number. |

---

## 13. Git / version reconciliation

**Goal: understand the drift. Protect the home working tree. Nothing here discards anything.**

Every command below is read-only. **Do not run `git reset`, `git checkout --`, `git clean`, `git stash drop`, or any `push --force`.** If the implementation model proposes any of those, stop it.

**Step 1 — Establish the home state before touching anything.**

```bash
cd <home repo>
git status                      # uncommitted work — THIS IS THE THING TO PROTECT
git branch --show-current
git log --oneline -30
git log --oneline --graph --decorate --all -30
```

**Step 2 — Where is home relative to the remote?**

```bash
git fetch --all --prune         # read-only; updates remote-tracking refs, touches nothing local
git status -sb                  # the [ahead N, behind M] line
git log --oneline @{u}..HEAD    # local commits not pushed
git log --oneline HEAD..@{u}    # remote commits not pulled
```

**Step 3 — What does the work-PC snapshot correspond to?**
The work PC is on `main` at `6553057` ("Merge branch 'main' of https://github.com/Neycourt5/Baballonia"), clean. Recent: `746b792 docs: record v14`, `cce52b6 eye: return eye tracking to stock behaviour`, `3b1c940`, `2902c34`.

```bash
git log --oneline 6553057 -1                 # does home even have this commit?
git log --oneline 6553057..HEAD              # what home has that the work PC does not
git diff --stat 6553057..HEAD                # scale of the drift
git diff --stat 6553057..HEAD -- src/Baballonia/Services/Inference/ \
                                  src/Baballonia/Services/Personalization/ \
                                  src/Baballonia/Services/ParameterSenderService.cs \
                                  src/VRCFaceTracking.Baballonia/ \
                                  training/babble_personal/
```

That last one is the money command — it tells you exactly how stale each subsystem in this plan is.

**Step 4 — Snapshot the uncommitted work before any edits.** Non-destructive, keeps the working tree intact:

```bash
git stash list                                        # see what is already stashed
git diff > ../home-uncommitted-$(date +%Y%m%d).patch  # plain backup of unstaged changes
git diff --cached > ../home-staged-$(date +%Y%m%d).patch
```

Or, preferred — commit the work-in-progress on a branch so nothing can be lost:

```bash
git switch -c wip/pre-tuning-snapshot
git add -A && git commit -m "wip: snapshot before tracking tuning work"
git switch -                                          # back to where you were
```

**Step 5 — Work on a branch.**

```bash
git switch -c feature/tracking-tuning
```

Commit each change from §8 separately so any one can be reverted alone.

**Step 6 — Targeted drift checks against this plan.**

```bash
git log --oneline -- src/Baballonia/Services/Inference/EyeProcessingPipeline.cs
git log --oneline -- src/Baballonia/Services/Inference/EyePipelineManager.cs
git log --oneline -- src/Baballonia/Services/ParameterSenderService.cs
git log --oneline -- src/Baballonia/Services/Personalization/GuidedCues.cs
git log --oneline -- training/babble_personal/labels.py
git grep -n "EyeSquint"                       # find the Alpha squint path fast
git grep -rn "toothy\|teeth\|CandidateCatalog"
```

**Explicitly not recommended:** resetting to remote, force-pushing, discarding uncommitted changes, or "cleaning up" unfamiliar code. The home working tree is the newest and most valuable copy in this whole picture.

---

## 14. Optional, directly related — **not part of the required work**

Each is independent. Do none, some, or all, and only after §8 items 1–5 are validated.

1. **Eye Gaze Range** (§4.7). Only if diagnosis shows amplitude, not latency, drives the reserved feeling. Prefer reusing the existing `CalibrationParameter` path by adding the four gaze channels to `CalibrationViewModel.EyeSettings`.
2. **Clamp eye output after remap.** Eye channels are unclamped while face channels are clamped — a narrowed `Lower/Upper` extrapolates without limit and becomes >45° on the native VRC path. Small, defensive. Behavior change if any user relies on the overshoot.
3. **Fix `LoadFilter`'s disabled early-return.** Set `_pipeline.Filter = null` when One Euro is disabled, so toggling it off takes effect immediately instead of at restart. **This is a behavior change** — ship it separately and flag it.
4. **Guard `eyeY` against NaN.** `(leftPitch*leftLid + rightPitch*rightLid) / (leftLid + rightLid)` has no epsilon; both eyes fully closed gives `0/0` → NaN propagated to OSC. Add an epsilon or fall back to the unweighted mean.
5. **Verify the native-VRC `LeftRightPitchYaw` argument order.** Currently enqueued as `(leftEyeY, rightEyeX, rightEyeY, leftEyeX)`, which interleaves eyes oddly versus the expected `(leftPitch, leftYaw, rightPitch, rightYaw)`. Only matters if you use `VRC_UseNativeTracking` or DFR. Verify against VRChat's spec before changing.
6. **A pinned watch-list in the Advanced comparison table.** The 45-row table already exists; a small "pin these dims" affordance for `{19,20,25,26,27,28,31,32,4,18,33}` would make toothy-smile debugging much faster. Keep it behind `ShowAdvanced` so it never clutters the normal UI.
7. **Subscribe something to `NewRawModelOutputEvent`.** It publishes the full native model output including channels the legacy projection discards, and has **no subscribers**. A temporary debug logger there would show exactly what the eye model emits — useful for confirming the Alpha squint path. Remove or hide when done.

---

## 15. Implementation handoff — read this first

**To the coding model working on the home PC:**

You have been given this plan plus a codebase that is **newer** than the one the plan was written against. The plan's job is to tell you what to look for; the codebase is the authority on what is actually there.

**Before modifying anything:**

1. **Run §13 steps 1–4.** Establish what is uncommitted, what is unpushed, and how far the tree has drifted from commit `6553057`. Snapshot the uncommitted work. **Never reset, discard, clean, or force-push.** The home working tree is the newest copy of this project that exists.
2. **Read §2.** This plan was written from a stale snapshot. Symbol names are landmarks. If a symbol is missing, it moved or was renamed — it was not deleted, and you should not recreate it.
3. **Work through §12** for each of the three changes. For each: locate the equivalent subsystem, confirm the responsibility matches, and check every divergence warning sign. **Write down what you found before writing code.**
4. **Change 2 is deliberately unspecified.** Eye squint does not exist in the snapshot this plan was written from. Do not add squint support, do not restore `experimental/eye-v2`, do not widen the 6-slot eye contract. **Trace the working `EyeSquintLeft`/`EyeSquintRight` path that already exists**, answer the four questions in §5.2 in writing, and only then place the scalar at the last mutable point before OSC.
5. **Change 3 is diagnosis-first.** Run §6.5 completely before changing a line. The user has already completed the capture → train → export → activate loop with the model active, so the cheap explanations are ruled out. The leading hypothesis (§6.2) is a training-label conflict that is verifiable in `training/babble_personal/labels.py`. Do not add calibration poses, do not touch `PersonalizationSchema`, and do not propose retraining the base model — §6.9 explains why the base output already carries the needed information.
6. **Do not "improve" unfamiliar code.** Newer eye processing, camera recovery, reliability work, and calibration changes are deliberate and working well. If something looks odd, assume it is intentional and ask.
7. **Preserve default behavior exactly.** Every new setting defaults to a value that reproduces current behavior. Verify this by deleting the key from `LocalSettings.json` and confirming stock behavior, not just by setting the slider to its default.
8. **Follow §8's order and commit each item separately** so any single change can be reverted without touching the others.

**The three hard constraints:**

- **Never** modify `PersonalizationSchema.Names` or anything that changes its SHA-256. Doing so invalidates every trained model and every recorded session.
- **Never** scale a value before the One Euro filter to achieve an output-strength effect. It changes the filter's dynamics, because the adaptive cutoff depends on the signal's own derivative.
- **Never** assign to `FaceProcessingPipeline.Enhancer` without composing with what is already there. There is exactly one slot, and a bare assignment silently disables `ProsodyEnhancer`.

---

## Verification summary

End-to-end validation for the whole plan:

1. **Build and test:** `dotnet build` and `dotnet test` — `Baballonia.Tests/PersonalizationSchemaTests.cs` must stay green; it fails the build if the 45-name schema drifts.
2. **Default-behavior proof:** delete `AppSettings_EyeGazeResponsiveness` and `AppSettings_EyeSquintStrength` from `%APPDATA%\ProjectBabble\ApplicationData\LocalSettings.json`, restart, and confirm tracking is indistinguishable from the pre-change baseline captured in §11.
3. **Live app validation:** run the app with a camera attached, use the Personalization → Advanced live Stock/Personal/Delta table as the instrument, and an external OSC monitor on `127.0.0.1:8888` for the output values.
4. **Per-change checklists:** §4.8, §5.7, §6.10, consolidated in §11.
5. **Trainer validation for Change 3:** `evaluate.py` (`neutral_report`, `cross_talk`, `dim_report`, `WATCHED_DIMS` false-activation runs) and `diagnose_guided.py` on the new run, compared against the run that produced the currently active model.
6. **Reliability regression:** hot-unplug and replug each camera; confirm recovery still works and `ResetTemporalState` still fires on camera change.
