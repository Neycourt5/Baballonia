# Personalized Face-Tracking for Baballonia — Implementation Plan

## Context

The user's face tracking in VR is inconsistent: Baballonia's general-purpose face model misclassifies some of their expressions, bleeds expressions into each other, and false-activates near neutral. A previous slider-calibration fork proved that post-hoc output remapping cannot recover information the stock model already misread. The goal is a **learned, user-specific correction model** that sees the actual camera imagery, trained locally (PyTorch → ONNX), loaded by Baballonia at runtime with graceful fallback to stock behavior. Eye tracking is explicitly out of scope (Paper Tracker will handle it). This plan is the deliverable; implementation will be done by another coding agent.

**Amended 2026-08-13 (same day, before implementation started): Avatar-Guided Personal Calibration** (§14) — the user's VRChat avatar becomes the interactive visual calibration target: Baballonia temporarily replaces its tracked output with independently generated cue target vectors (driving the avatar through the existing OSC→VRCFT chain), the user imitates the avatar, and the commanded vector is the training label. This primarily upgrades supervision quality (the plan's #1 risk); the residual-adapter architecture is unchanged.

---

## 1. Existing Architecture (verified facts, all file:line-cited)

### The face pipeline, end to end

```
Capture plugin DLL (own thread, latest-frame-wins Mat)
  → SingleCameraSource.GetFrame(ColorType.Gray8)          [native res, CV_8UC1]
  → ImageTransformer.Apply()                              [user ROI crop → gamma multiply → rot/flip+resize WarpAffine → 224×224 gray]
  → MatToFloatTensorConverter.Convert()                   [/255 only, NCHW DenseTensor [1,1,224,224]]
  → DefaultInferenceRunner.Run()                          [ONNX Runtime → float[45]]
  → OneEuroFilter.Filter()                                [if enabled; minCutoff .5, beta 3]
  → NewFilteredResultEvent (currently ZERO subscribers)
  → ProcessingLoopService.ExpressionChangeEvent
  → ParameterSenderService: per-expression Remap(Lower,Upper→Min,Max)+Clamp   [the "calibration"]
  → OSC UDP 127.0.0.1:8888 → VRCFT module → UnifiedExpressions
```

- Driver: `ProcessingLoopService` ([src/Baballonia/Services/ProcessingLoopService.cs](src/Baballonia/Services/ProcessingLoopService.cs)) — a **10 ms Avalonia DispatcherTimer on the UI thread** (≈100 Hz ceiling; actual rate = camera-gated, latest-frame-wins).
- Pipeline body: `FaceProcessingPipeline.RunUpdate()` ([src/Baballonia/Services/Inference/FaceProcessingPipeline.cs:9-41](src/Baballonia/Services/Inference/FaceProcessingPipeline.cs#L9-L41)). The raw pre-filter float[45] exists only as a local variable (lines 30–34); the published event is post-filter.
- Calibration+send: `ParameterSenderService.ProcessFaceExpressionData` ([src/Baballonia/Services/ParameterSenderService.cs:191-209](src/Baballonia/Services/ParameterSenderService.cs#L191-L209)); calibration ranges from `CalibrationService` (name-keyed, persisted in `LocalSettings.json`). No gain/offset/deadzone exists anywhere; the only smoothing is the One Euro filter (`Services/Inference/Filters/OneEuroFilter.cs`), applied to **raw model-space output, before calibration**.

### The face model (inspected the actual ONNX graph)

- `src/Baballonia/faceModel.onnx` — 23.7 MB, committed to git, also EmbeddedResource. PyTorch 2.6.0 export, opset 17.
- Input `x.1`: float32 **[1, 1, 224, 224]** (fixed batch 1), grayscale, /255 normalization only.
- Output `1210`: **[N, 45]**.
- Architecture: timm **EfficientNet-B0-class** (~5.89 M params, SE blocks, SiLU) ending in `conv_head → GlobalAveragePool → Flatten → Gemm[45×1280]`. The penultimate tensor `/model/global_pool/flatten/Flatten_output_0` is a clean **1280-d embedding** — exposable by adding one graph output to a *derived copy* (no weight changes, zero extra compute).
- Model path is **hardcoded** in `FacePipelineManager.CreateInference()` ([src/Baballonia/Services/Inference/FacePipelineManager.cs:57-61](src/Baballonia/Services/Inference/FacePipelineManager.cs#L57-L61)) — unlike the eye model, which has a settings key (`EyeHome_EyeModel`), file picker, and hot reload. `FacePipelineManager.LoadInferenceAsync()` already exists and is invoked on GPU toggle.
- ORT: DirectML 1.24.3 on Windows (Desktop csproj), plain ORT 1.22.0 elsewhere; `AppSettings_UseGPU` defaults **false** → CPU EP.

### Expression schema — positional, no enum

The canonical 45-name list is the **insertion order** of `ParameterSenderService.FaceExpressionMap` ([src/Baballonia/Services/ParameterSenderService.cs:46-93](src/Baballonia/Services/ParameterSenderService.cs#L46-L93)): 0–3 CheekPuff/Suck L/R, 4 JawOpen, 5–7 Jaw, 8–9 NoseSneer, 10–32 Mouth (19/20 SmileL/R, 21/22 FrownL/R), 33–44 Tongue. The ONNX file carries **no** name metadata (verified: `CustomMetadataMap` empty; a `blendshape_names` metadata hook exists in `DefaultInferenceRunner` but its result is never consumed). UnifiedExpressions mapping lives entirely on the VRCFT-module side (`src/VRCFaceTracking.Baballonia/BabbleExpressions.cs`, some 1→4 fan-out).

### The eye pipeline already has the loop we want (reusable patterns, face has none)

- `Baballonia.CaptureBin.IO`: `.bin` capture format — packed 100-byte header (label floats, timestamps, `RoutineState` flag word incl. 4 version bits) + L/R JPEG payloads; `ReadAll/WriteAll/Concatenate`.
- `FrameCollector.cs` (Desktop/Calibration): JPEG q85 encode + buffered writes to `%APPDATA%\ProjectBabble\ModelData`.
- Guided VR calibration: external Godot overlay over TCP + step framework (`Desktop/Calibration/ICalibrationRoutine.cs`) that stamps label fields into frame headers, then spawns external `BabbleTrainer.exe`, swaps the tuned eye ONNX in via settings + hot reload.
- **Nothing equivalent exists for the face model, and no face-model training code exists anywhere upstream** (BabbleTrainer is eye-only; no HuggingFace/checkpoint links in the repo). The original face training checkpoint must be assumed unavailable.

### App infrastructure relevant to us

- Avalonia + CommunityToolkit.Mvvm; DI in `App.axaml.cs`; adding a page = VM+View + 3 registrations (`App.axaml.cs`, `ViewLocator.cs`, `MainViewModel._desktopTemplates`). RESX localization.
- Settings: flat JSON `%APPDATA%\ProjectBabble\ApplicationData\LocalSettings.json` via `LocalSettingsService` + `[SavedSetting]` attributes.
- Data dirs (`Utils.cs`): `Documents\ProjectBabble` (user-accessible), `%APPDATA%\ProjectBabble\Models`, `...\ModelData`.
- Tests: MSTest+Moq (`Baballonia.Tests`); CI is workflow_dispatch-only, no test gate.
- `.gitignore` has **no** dataset/onnx ignores (models are deliberately committed) — must add entries for personal data.
- Known upstream quirks to not trip over: disabling One Euro doesn't clear the installed filter; `FaceProcessingPipeline` leaks the source Mat; `RunUpdate` is shadowed not virtual; UI-thread inference.

---

## 2. Recommended Architecture

A nullable **corrector stage** inserted into `FaceProcessingPipeline` between stock inference and the One Euro filter, running a second tiny ONNX session; a **recorder service** tapping a new raw-result event; a **Python package** under `training/` for labels/training/export. Everything is new files except ~15 additive lines across 3 upstream files.

```
camera frame ──► ImageTransformer (224×224 gray) ──► /255 tensor [1,1,224,224]
                                                          │
                                             DefaultInferenceRunner (stock, untouched)
                                                          │  raw float[45]
                       ┌──────────────────────────────────┤
                       │ NEW: NewRawExpressionsEvent      │
                       │ (frame + raw values + ticks)     ▼
                       │                     NEW: PersonalModelCorrector (nullable)
              DatasetRecorderService          personal = clamp(stock + f(image, stock), 0, 1)
              (background writer)             out = lerp(stock, personal, Blend)
                       │                                  │
                 session folders              NEW: NewCorrectedExpressionsEvent ──► debug view
                       │                                  ▼
                       ▼                        OneEuroFilter (unchanged, smooths final signal)
              training/ (PyTorch)                         ▼
              labels → train → export      ExpressionChangeEvent → calibration remap → OSC → VRCFT
                       │                                  ▲
                       └── personalFaceModel.onnx ────────┘ (loaded, validated, hot-reloadable)
```

**Why corrector before the filter:** the corrector trains on raw pre-filter stock outputs (that's what recording captures), so it must consume raw stock at runtime — train/serve parity. The One Euro filter then smooths the *final* corrected signal, and its state simply tracks whatever signal flows through, so toggling personalization causes only a brief transient. This also answers the temporal question: **no temporal model in MVP** — One Euro already provides output smoothing downstream, and per-frame correction is the hypothesis to validate first. Temporal context (e.g., stacking the last N stock vectors as extra MLP inputs — cheap, no image history needed) is a Phase-3+ option only if per-frame metrics show temporal confusion (e.g., speech cross-talk) that static correction can't fix.

**Blend math is well-defined for this architecture:** `lerp(stock, clamp(stock+r), α) = stock + α·(clamped residual)` — 0 % is exactly stock, 100 % is full personalization, and it's a pure evaluation control in C#, independent of training.

## 3. Alternatives Considered

| Approach | Verdict | Reason |
|---|---|---|
| **A. Output-only learned correction** (45 → MLP → Δ45) | **Build first as baseline** (~29 k params) | Can un-mix systematic cross-talk and remap ranges, but cannot recover information the stock model misread from the image. Cheap to build; quantifies how much of the problem is "output relationships" vs "visual". If A ≈ B on metrics, ship A. |
| **B. Image-conditioned residual adapter** (image + stock → Δ45) | **MVP recommendation** (~48 k params) | Sees the pixels, so it can fix visual misclassification (smile/frown confusion, false jaw). Tiny (<1 ms CPU). Trained from scratch on user data only — feasible because the residual-shrinkage prior makes "no change" the default. |
| **C. Personalized head on the stock 1280-d embedding** | **Phase 5** | Verified cheap to expose: add one graph output (`/model/global_pool/flatten/Flatten_output_0`) to a *derived copy* of faceModel.onnx — no weights change, zero extra compute, stock output bit-identical. Best features-per-FLOP, but it swaps the stock inference session (bigger blast radius, DirectML multi-output to validate) and ties the adapter to the exact stock model file. Do it once B's value is proven. |
| **D. Partial/full backbone fine-tuning** | **Phase 6, only if evidence demands** | No upstream training code or checkpoint exists (verified — BabbleTrainer is eye-only); would require reconstructing training from the ONNX weights, risks catastrophic forgetting on thin personal data, and produces a divergent 24 MB private model that breaks upstream rebasing. Only justified if C shows the *feature extractor itself* is the bottleneck. |

Phase ordering A → B → C → D is confirmed sensible after inspection: each step adds capability only where the previous step's measured ceiling demands it.

## 4. Data Collection Design

**Format — new session-folder layout** (CaptureBin rejected: its 100-byte header is eye-semantic with L/R JPEG pairs; video rejected: inter-frame compression + fragile frame↔label alignment):

```
%APPDATA%\ProjectBabble\PersonalDataset\20260813_193000_guided\
  session.json     schema_version, app version, ordered 45-name list + SHA-256,
                   camera backend/address/native res, ROI+rotation/flip/gamma,
                   session type (neutral|guided|speech), cue-routine version
  frames\000000.jpg …   224×224 Gray8, JPEG q95 (~10–15 KB each)
  labels.jsonl     {"i":123, "t_ticks":…, "stock":[45 raw pre-filter floats],
                    "cue":{"id":"SmileHold50","dims":[19,20],"target":[45 floats],
                           "level":0.5,"phase":"hold","rep":1,
                           "source":"avatar"|"bar"} | null}
  avatar\000000.png …   OPTIONAL sidecar (Spout captures + timestamps, P4+; review/eval only,
                        never supervision)
```

`session.json` additionally records (when avatar-guided): `cue_source`, avatar/VRChat notes, and the cue-preview flags (per-cue "avatar renders this wrong" → bar-only/reduced weight).

- **Store the transformed 224×224 frame, not raw+crop metadata**: it is exactly the tensor the stock prediction was computed from, and it avoids re-implementing `ImageTransformer`'s ROI/gamma/WarpAffine bit-exactly in Python (silent-skew trap). ROI settings are recorded in `session.json` and embedded in exported model metadata; a load-time warning fires if current camera settings differ.
- **Tap point:** new `FacePipelineEvents.NewRawExpressionsEvent(Mat transformedFrame, float[] rawResult, long ticks)` published in `RunUpdate()` after inference, before the corrector/filter. Verified constraint: `GenericEventBus.Publish` runs handlers **synchronously under the bus lock** — the recorder handler must only `Clone()` the Mat + copy floats into a bounded `Channel<CapturedSample>` (capacity 256, DropOldest); JPEG encoding and file I/O happen on a background consumer task.
- **Duplicate suppression:** the 100 Hz tick re-serves latest-frame-wins Mats from a ~30–60 fps camera. Dedupe via a cheap 64-sample pixel checksum vs previous frame; cap at 30 fps; log effective unique-fps.
- **Location — APPDATA, deliberately not Documents:** this machine's Documents is **OneDrive-synced**; thousands of small JPEGs would cause sync churn/locks. A "reveal dataset folder" button keeps it user-discoverable for backup/deletion (privacy requirement: everything local, clearly identifiable, easily deletable).
- **Disk budget:** ~23 MB/min at 30 fps → full guided routine (~8 min) ≈ 190 MB; realistic corpus (3 guided + 2 speech + 2 neutral sessions) ≈ 600–800 MB.
- **.gitignore additions:** `training/.venv/`, `training/runs/`, `training/data/`, `**/PersonalDataset/`, `*.personal.onnx`.

**Guided capture UX (user-confirmed: desktop visuals + audio):** new `PersonalizationPage` (standard 3-touchpoint page registration). Cue = `prep 2 s → ramp-up 3 s (animated 0→1 target bar + huge expression name) → hold 1.5 s → ramp-down 3 s → rest 1.5 s`, ×2 reps, with audio ticks/tones at phase transitions so it's runnable while wearing the HMD. Godot overlay explicitly **not** used in MVP — it exists for world-space gaze targets; face cues are purely temporal. Cue coverage: ~20 practically trainable outputs (jaw ×4, cheeks ×4, sneer, smile both/L/R, frown, pucker, funnel, upper-raise/lower-depress, press, stretch) plus **basic tongue cues** (TongueOut + 4 directions, near-binary, low confidence — user-confirmed); fine tongue shapes and untrainable dims stay stock passthrough via the residual default. Session types: **neutral** (45 s relax), **guided**, **speech** (60–90 s reading on-screen text). Multi-session across days/headset repositions is just "record again" — session folders are the natural unit for variation and for the train/val split.

## 5. Labeling Strategy

Per-frame targets `y[i,j]` with per-frame-per-dim weights `m[i,j]` (masked loss — no frame needs all 45 dims labeled):

| Source | y | m |
|---|---|---|
| Neutral session, all dims | 0 | 1.0 |
| Guided (avatar or bar), cued dims, **step-hold** at level L (0/.25/.50/.75/1), after settle | commanded L | 1.0 (tongue 0.4) |
| Guided, first 0.5 s of each hold ("settle trim") | commanded L | 0 |
| Guided, step transitions between levels | commanded value | 0 (masked — avatar animator smoothing + mimicry lag make transitions unreliable) |
| Guided, continuous ramps (secondary) | commanded value | 0.5 |
| Combo cues (e.g., smile+jaw), all cued dims | commanded 45-vector | 1.0 on holds, same trims |
| Guided, non-cued dims ("rest of face relaxed") | 0 | 0.25, **except** known co-activators of the cue (per-cue exclusion list) |
| Speech, jaw/mouth dims | stock value (pseudo-label) | 0.3 |
| Speech, tongue/cheek/nose dims | — | 0 (ignore) |
| **Override (avatar-guided) sessions, stock pseudo-labels** | — | **0 for all dims** — stock is model *input* only; reintroducing it as a label would re-create the circularity the avatar path removes |
| Manual corrections from review tool | corrected value | 2.0 |

Cue records store both the full 45-float `target` vector **and** `dims` (the cued set) — the vector alone cannot distinguish "commanded 0" from "not commanded". **Step-holds are primary, ramps secondary** (amended, §14): holds give calibrated mid-intensity anchors at full weight and are robust to avatar animator smoothing; ramps only ever carried w=0.5. Intensity semantics: **1.0 = the user's own maximum** — the avatar teaches shape identity and trajectory; the commanded fraction defines the label (consistent with blendshape conventions and avatar-agnostic).

- **Human lag correction:** per cue, cross-correlate the stock model's primary output channel with the cue signal, shift by `argmax` lag (0–700 ms window); auto-drop reps with correlation < 0.3 (user missed the cue). This is the main "no manual labeling" guard.
- **Pseudo-label circularity defense** (stock as weak label must not teach stock's mistakes): pseudo-labels capped at w=0.3, and the **residual-shrinkage term** `λ_r·mean(r²)` makes r=0 (stock passthrough) the zero-cost default anywhere labels are absent/weak — corrections are only learned where confident priors (guided/neutral/manual) demand them.
- **Review/correction tool = Python-side, not C#** (`training/babble_personal/review.py`): dumps an HTML grid of worst frames per cue (largest |stock − prior| on trusted phases, plus top-loss frames after a first training round = hard-example mining), user edits a `corrections.jsonl` (`frame_id, dim, value` — also supports range marks and "ignore" marks); ingested at weight 2.0. No per-frame-all-45 labeling ever required.

## 6. Model Design

- **Baseline A — `OutputMlpAdapter`** (~29 k params): `stock[45] → 128 → 128 → 45 = r`, SiLU. ~0.05 ms CPU.
- **Model B — `ImageResidualAdapter`** (~48 k params): in-graph `AveragePool 2×2` (224→112, so C# feeds the *exact tensor it already has* — zero new preprocessing), then Conv3×3-s2 stack 1→8→16→32→64 (+BN+SiLU), GAP → 64-d ⊕ stock 45 → 128 → 45 = r. ≈6 MFLOPs → **est. 0.3–0.8 ms single-thread CPU** (measured in P1 with a random-weight export before real training).
- Both export `personal = Clip(stock + r, 0, 1)` **in-graph**; blend stays in C#. Inputs `image [1,1,224,224]`, `stock [1,45]`; output `personal [1,45]`; fixed batch 1; opset 17.
- **ONNX metadata** (validated by the C# loader): `personal_adapter_version`, `adapter_type`, `expression_names` (JSON array), `expression_schema_sha256`, `input_normalization=gray_div255`, `input_size=224`, `base_model_md5`, `roi_settings`, `trained_utc`.

## 7. Training Pipeline (Python, `training/` top-level dir)

```
training/babble_personal/
  schema.py     ordered 45 names (mirrors FaceExpressionMap insertion order) + sha256
  dataset.py    session discovery, loading, SESSION-LEVEL train/val split
  labels.py     target/weight construction, lag alignment, rep quality filter
  models.py     OutputMlpAdapter, ImageResidualAdapter
  train.py      CLI: python -m babble_personal.train --data … --model a|b --val-sessions …
  evaluate.py   metrics for stock vs A vs B side-by-side (prints + CSV)
  export.py     ONNX export + metadata + ORT-vs-torch parity assert (|Δ|<1e-4, 32 samples)
  review.py     HTML worst-frame grid + corrections.jsonl ingestion
  derive_embedding_model.py   Phase 5: derived faceModelWithEmbedding.onnx
```

- **Split: by session, never by frame** — hold out ≥1 guided + 1 neutral + 1 speech session entirely (adjacent frames are near-duplicates; frame-shuffled splits would fake validation).
- Loss: masked Huber(δ=0.05) + `λ_r=1e-2` residual shrinkage; temporal-smoothness term deferred (One Euro smooths downstream). AdamW 1e-3 cosine, ~30 epochs, early stop patience 5, checkpoints under gitignored `runs/`.
- **Metrics beyond MSE:** per-expression MAE on held-out hold-phase frames; **neutral false-activation rate** (fraction of held-out neutral frames with output > 0.15, per dim); **cross-talk matrix** (guided cue × 45 dims, mean non-cued activation during holds, Δ vs stock). These directly target the user's neutral-stability and cross-talk complaints without dead-zoning (the model corrects toward 0 only where labels say the face was neutral — subtle motion elsewhere is untouched by the shrinkage default).
- Environment: Python 3.13 venv **outside the OneDrive repo** (`%LOCALAPPDATA%\babble-train-venv`); pinned `torch==2.7.1` (CPU wheels, cp313 supported), `onnx==1.18.0`, `onnxruntime==1.22.0`, `numpy`, `opencv-python-headless`, `tqdm`. CPU training is fine at this scale.

## 8. Baballonia Integration (exact files)

**Modified (additive, ~15 lines total — rebase-friendly):**
1. [src/Baballonia/Services/events/PipelineEvents.cs](src/Baballonia/Services/events/PipelineEvents.cs) — add `NewRawExpressionsEvent(Mat, float[], long)` + `NewCorrectedExpressionsEvent(float[] raw, float[] corrected)` to `FacePipelineEvents`.
2. [src/Baballonia/Services/Inference/FaceProcessingPipeline.cs:27-37](src/Baballonia/Services/Inference/FaceProcessingPipeline.cs#L27-L37) — move `transformed.Dispose()` after a new raw-event publish; insert nullable `public volatile IExpressionCorrector? Corrector` stage between `Run()` and `Filter` (verified against current source; eye pipeline untouched). Corrector must return a **new** array (OneEuroFilter holds internal buffers).
3. [src/Baballonia/Services/Inference/FacePipelineManager.cs](src/Baballonia/Services/Inference/FacePipelineManager.cs) — one-line `SetCorrector(IExpressionCorrector?)`.
4. [src/Baballonia/Services/ParameterSenderService.cs](src/Baballonia/Services/ParameterSenderService.cs) — **avatar-guided override hook (§14)**: constructor param `IExpressionOverrideSource?`, a `_faceOverrideActive` early-return at the top of `ProcessFaceExpressionData` (face branch only — eye messages keep flowing), and `EnqueueRawFaceVector` called from the `ExecuteAsync` loop (raw `Clamp(v,0,1)`, **no calibration remap**, `FaceExpressionMap` addresses). ~10 additive lines.
5. `App.axaml.cs` / `ViewLocator.cs` / `MainViewModel.cs` — standard 3-touchpoint page + service registration.
6. `.gitignore` — dataset/venv/runs entries.

**New (isolated in `src/Baballonia/Services/Personalization/` + ViewModels/Views):**
- `IExpressionCorrector` — `float[] Correct(DenseTensor<float> image, float[] stock); float Blend {get;set;}`
- `PersonalModelCorrector` — own `InferenceSession`, **CPU EP unconditionally** (sub-ms model; sidesteps DirectML two-session interplay); wraps the stock runner's already-filled input tensor zero-copy (safe: single-threaded tick); try/catch around `Run()` self-disables on first exception → stock behavior.
- `PersonalModelManager` — settings (`PersonalModel_Enabled` default false, `PersonalModel_Path` default `%APPDATA%\ProjectBabble\Models\personalFaceModel.onnx`, `PersonalModel_Blend` default 1.0), background load, **validation-on-load** (file exists → session opens → metadata `expression_schema_sha256` == `PersonalizationSchema.Sha256` → shapes `[1,1,224,224]`/`[1,45]` → warn on ROI mismatch), hard failure ⇒ log + `SetCorrector(null)`; `ReloadAsync()` for hot reload (mirrors the eye model's `EyeHome_EyeModel` picker pattern). **On disable, explicitly `SetCorrector(null)`** — do not replicate the known One Euro can't-clear bug.
- `PersonalizationSchema` — the 45 names + SHA-256, *duplicated* rather than editing `ParameterSenderService`, guarded by a new MSTest asserting it matches `FaceExpressionMap.Keys` order (upstream drift breaks a test, not users).
- `DatasetRecorderService`, `CueStateProvider` (volatile snapshot record), `GuidedCaptureRoutine` (Stopwatch-driven phase machine).
- `PersonalizationPage` View/VM — session controls + live 224×224 preview (`NewTransformedFrameEvent`, existing pattern), cue display, Stock/Personal toggle + blend slider + model picker/reload, and the **debug comparison panel**: 45 fixed rows (name, stock bar, personal bar, colored Δ), fed by `NewCorrectedExpressionsEvent` into latest-value buffers, pushed to bindings by a 10 Hz timer (no 100 Hz binding storms); optional sort-by-|Δ|.

**Also required by the user:** create and maintain `WORK_PROGRESS.md` at repo root (investigated facts, decisions, files changed, tests run, next steps) — first commit, updated every phase.

## 9. UI / User Workflow

```
Collect  PersonalizationPage → Record Neutral / Guided / Speech → session folders
Review   training/: python -m babble_personal.review → HTML grid → corrections.jsonl
Train    python -m babble_personal.train --data %APPDATA%\...\PersonalDataset --model b
Install  copy runs/…/personalFaceModel.onnx → %APPDATA%\ProjectBabble\Models (or picker)
Test     toggle Personal on, watch debug Δ panel
Compare  blend slider 0↔100 %, in-VR A/B via toggle; delete/disable ⇒ instant stock behavior
```

## 10. Evaluation

- **Session holdout** (train A/B/C-sessions, test D) — primary; **controlled expression validation** on unseen held/ramped recordings; **neutral test** (false-activation rate); **cross-talk test** (JawOpen while measuring Smile/Frown et al.); **camera-placement robustness** (train on several normal placements, test another); **natural speech review** via review.py sampling; **real VR A/B** via the toggle.
- **Avatar replay A/B (§14, P4):** replay the *same recorded real performance's* raw stock vs personal output sequences through the override path and watch the avatar render each — direct visual answer to "does my avatar reproduce my performance better?", the user's actual success condition. Sequential eyeballing (mirror/OBS capture) suffices; Spout only if frame-accurate side-by-side artifacts are wanted.
- **Success criteria for shipping B as default:** on held-out sessions, B beats stock on neutral false-activation rate and cross-talk *without* degrading MAE on intentional expressions; B beats A by a margin that justifies the image branch (else ship A). "Feels better" in VR is the final gate, informed — not replaced — by the metrics.

## 11. Development Phases

- **P0 — Hooks + recording:** pipeline events, `RunUpdate` restructure, `SetCorrector`, **`ParameterSenderService` override hook (§14 — bundled here so all upstream-file edits land in one commit)**, `PersonalizationSchema` + drift test, `DatasetRecorderService` + session format, minimal page (Record Neutral/Speech + preview), `.gitignore`, `WORK_PROGRESS.md`. *Exit:* sessions on disk, unique-fps logged, stock behavior verifiably unchanged, override-path MSTest green.
- **P1 — End-to-end loop with baseline A:** `training/` package, neutral+speech labels, train/evaluate/export A, C# corrector + manager + toggle/blend/hot-reload + debug panel; random-weight **model-B latency benchmark on the UI thread**. *Exit:* Python→ONNX→C# round-trip with validation; tick-time p50/p95 known.
- **P2 — Guided capture + labels, avatar-first (§14):** `ExpressionOverrideService` + cue engine (step-holds primary, ramps secondary) + audio + cue stamping; **cue-preview pass with step-response probe** (avatar quirk flagging); bar-only fallback (works with VRChat closed — override with no listener is harmless UDP fire-and-forget); lag alignment + priors; retrain A; first real neutral-FAR/cross-talk numbers. *Exit:* go/no-go on hold-prior learnability (notebook check first — no model needed).
- **P3 — Model B:** train, A-vs-B on identical splits, ship winner.
- **P4 — Review/hard-example + replay A/B:** review.py grid, corrections at w=2.0, retrain loop; **replay evaluation** — feed recorded raw stock vs personal vectors through the same override path and watch the avatar (mirror/OBS window capture; no Spout code). Optional `AvatarViewService` (Spout capture into `avatar/` sidecar) only if synchronized recording proves needed. (Ring-buffer "save last 10 s": architecture supports it — optional.)
- **P5 — Shared-embedding (optional):** `derive_embedding_model.py` appends the 1280-d `Flatten_output_0` as a second graph output → new file `faceModelWithEmbedding.onnx` in `%APPDATA%\ProjectBabble\Models` (stock file untouched); C# surfaces `results[1]` when personalization is on; adapter B′ = MLP(1280 ⊕ 45) drops the conv trunk. Validate DirectML multi-output + stock-output bit-parity first.
- **P6 — Active learning (future):** |personal − stock| > threshold frames auto-queued for review — recorder + review.py already provide the pieces.
- **P7 — Cross-domain visual matching (research, gated):** real-face ↔ avatar-face computational comparison. Gate: only if avatar-guided labels prove insufficient. First (and possibly only) step: offline notebook on synchronized Spout+real frames comparing simple geometric proxies (e.g., mouth-opening area proxy on both). Standard landmark models won't work on a lower-face-only IR mouth cam — this is genuine research, never an MVP dependency.

## 12. Risks / Unknowns (ranked, with cheapest resolving experiment)

1. **Guided-intensity priors too noisy** — *the strongest reason the whole approach might fail*: if the user can't track ramps and the lag-corrected cue↔stock correlation is weak, the primary label source collapses, leaving only pseudo-labels (circular) and manual work. **Cheapest exposure experiment (do at P2 start, before any model):** record 3 cues ×2 reps, plot stock channel vs cue signal in a notebook; require lag-corrected correlation ≳0.6 and monotonic ramps. Fallback already designed: 3-level step cues (0/0.5/1 holds) instead of continuous ramps — holds carry w=1.0 regardless.
2. **UI-thread budget** (10 ms tick already runs stock face+eye inference): P1 Stopwatch benchmark with random-weight B, CPU and GPU-toggle variants.
3. **Pseudo-label circularity:** structurally mitigated (w=0.3 + shrinkage); if held-out neutral FAR/cross-talk don't beat stock, drop speech pseudo-labels to w=0.
4. **OneDrive:** datasets → APPDATA, venv → LOCALAPPDATA, runs/ gitignored; README warns.
5. **Python 3.13 + torch on Windows:** 5-min venv smoke test in P1.
6. **DirectML two-session interplay:** eliminated in MVP (personal session pinned CPU); revisit only in P5.
7. **Schema drift on upstream rebase:** drift test + load-time schema-hash rejection.
8. **JPEG q95 domain gap:** negligible expected; A/B one PNG session if metrics look odd.

## 13. Exact First Implementation Step

Smallest safe first task (pure additive, zero behavior change, immediately testable):

> Create `src/Baballonia/Services/Personalization/PersonalizationSchema.cs` (ordered 45 names + SHA-256) and `src/Baballonia.Tests/PersonalizationSchemaTests.cs` asserting it exactly matches `ParameterSenderService.FaceExpressionMap.Keys` order; add the `.gitignore` entries; create `WORK_PROGRESS.md` recording this plan's findings. Run `dotnet test`.

This locks the most dangerous invariant (the positional schema) before any pipeline code depends on it. Next commit: the two pipeline events + `RunUpdate` restructure + a no-op assertion that behavior is unchanged.

---

## 14. Avatar-Guided Personal Calibration (Amendment)

### 14.1 Verdict and mechanism

**Feasible, and the right upgrade to the plan's weakest link (supervision quality).** The avatar serves as **visual teacher**; the independently commanded target vector remains the ML label. The avatar image is never model input or supervision in MVP.

**Do not talk to VRChat directly.** Baballonia already owns a complete, verified path to the avatar's face: 45 named OSC floats → UDP 8888 → VRCFT module (verified: exact bare-address matching, **stores floats verbatim — no smoothing/clamping/filtering** — copied into `UnifiedTracking` at ~100 Hz) → VRCFT.Core avatar-specific parameter encoding → VRChat (OSC 9000). Driving avatar parameters ourselves would mean replicating VRCFT's per-avatar binary parameter encoding — fragile and pointless. Instead, during calibration `ParameterSenderService` **replaces its outgoing tracked face values with the cue target vector**:

```
GuidedCaptureRoutine (UI-thread Stopwatch state machine)
      │ publishes immutable phase snapshot {cueId, dims, from/to vectors,
      │ phaseStart, duration, phase} via volatile swap
      ▼
ExpressionOverrideService : IExpressionOverrideSource
      │ SampleTarget() — PULL model: interpolates the vector at call time
      ▼
ParameterSenderService.ExecuteAsync loop (~64 Hz real — Task.Delay(10) quantizes
      │ to Windows 15.6 ms timer; fine, VRChat consumes stepwise)
      │ ovr != null ⇒ EnqueueRawFaceVector(ovr)  [Clamp(0,1), NO calibration remap,
      │ FaceExpressionMap addresses]; ProcessFaceExpressionData early-returns
      │ (face branch only — eye messages keep flowing)
      ▼
existing queue → UDP 8888 → VRCFT → avatar shows the target
```

Meanwhile the camera, stock inference, and `DatasetRecorderService` run untouched — only *sent* values are replaced. `CueStateProvider` reads the same snapshot the sampler uses, so sent targets and recorded labels share one source (no drift). **Non-circular by construction**: injected values originate in the cue generator, never in inference; and stock pseudo-labels get weight 0 in override sessions (§5).

**Constraints (assert at session start):** `AppSettings_OSCPrefix` must be `""` (module matching is exact/bare — refuse to start with a visible warning otherwise); VRChat OSC enabled; avatar VRCFT-compatible. In-headset the user watches their avatar via a VRChat mirror or selfie/stream camera — better visibility than the desktop bar for an HMD wearer; the desktop bar+audio remain as fallback and run alongside.

**Stop-safety (never leave the avatar frozen):** (1) deadman — cue engine refreshes a keep-alive each UI tick; `SampleTarget()` returns null if stale >1 s → live tracking resumes next iteration; (2) per-phase hard cap — neutral vector if elapsed exceeds `phaseDuration + 2 s`; (3) `Deactivate()` (normal end, abort, exception, page navigation) serves all-zeros ~200 ms before returning null; (4) process kill is unrecoverable from our side — every cue ends in a rest phase so a crash statistically lands near neutral; verify VRCFT's stale-param behavior once and document.

### 14.2 Cue sequence and reaction delay

- **Step-holds primary** (levels 0/.25/.50/.75/1, ~1.5–2 s each, smooth 0.5 s commanded moves between levels), **short ramps secondary**. Holds are robust to the two lag sources — avatar-side animator smoothing (many VRCFT avatars ship 100–500 ms FX-layer parameter smoothing) and human mimicry delay — which cross-correlation absorbs as one constant. Labels: holds w=1.0 after a 0.5 s settle-trim; transitions masked w=0; ramps w=0.5 (§5 table).
- **Cue-preview pass with step-response probe** before the first real session: command a 0→1 step per cue; the user judges shape correctness while the stock channel's time-to-plateau is auto-measured. Cues flagged "avatar renders this wrong" or slow-settling → bar-only or reduced weight, recorded in `session.json`. This is the containment for avatar blendshape quirks (§14.4).
- **Combination cues** (smile+jaw open, frown+jaw, pucker+jaw, asymmetric smile) become unambiguous with an avatar teacher — start with 3–4 pairs in one section and compare the validation cross-talk matrix trained with vs without before expanding. Natural-speech trajectories played by the avatar: defer — speech sessions already cover it more cheaply.

### 14.3 Visual matching (real face ↔ avatar pixels) — rejected for MVP

The domains are radically different (IR/gray close-up lower face vs stylized rendered color; no landmark model works on a mouth-cam crop). The commanded-vector supervision removes the *need* for it. It stays as **gated P7 research** with a defined cheap probe (geometric proxies on synchronized Spout+real frames, offline notebook). The honest expectation: low value unless avatar-guided labels prove insufficient.

### 14.4 Avatar quirks vs canonical correctness

The base personal model must stay **avatar-agnostic**. Defenses: (1) intensity semantics = fraction of the *user's own* maximum — the avatar teaches shape identity and trajectory, not absolute amplitude; (2) numeric/bar target always displayed alongside; (3) cue-preview flagging removes badly-authored cues from avatar teaching; (4) an **avatar-specific second-stage mapper** (canonical corrected output → this-avatar-adjusted output) is a legitimate concept but a separate, optional, post-P5 layer — never folded into the base model, and only built if the user perceives avatar-specific visual issues after the base model works.

### 14.5 New components

- `src/Baballonia/Services/Personalization/ExpressionOverrideService.cs` (+ `IExpressionOverrideSource { float[]? SampleTarget(); }`), DI singleton, injected into `ParameterSenderService` (constructor param — part of the P0 upstream-edit commit).
- Cue engine additions to `GuidedCaptureRoutine`/`CueStateProvider` (phase snapshots, keep-alive).
- Optional P4+: `AvatarViewService` (Spout2 receive via LightjamsSpout COM wrapper or Spout2 SDK P/Invoke, Windows-only) writing timestamped `avatar/` sidecar frames. OBS manual window capture is the zero-code intermediate; VRChat's stream camera has a native Spout output toggle (desktop app, world-mask for clean background).
- New MSTest: override path enqueues raw values at correct addresses, no remap applied, eye path unaffected.

### 14.6 Delta-specific risks (each with cheapest experiment)

1. **Avatar animator smoothing / wrong blendshapes distort displayed targets** (highest): the cue-preview step-response probe (built into preview, not a separate tool).
2. **User can't hold intermediate intensities (25/75)**: one JawOpen+Smile session with 5-level holds; check stock plateaus are monotone/separated; else collapse to 0/50/100.
3. **Override-path leaks** (remap applied, prefix non-empty, eye muted): MSTest + one manual session with an OSC monitor on 8888; prefix assert closes the silent-failure mode.
4. **Frozen avatar on abnormal termination**: deadman/hard-cap/neutral-flush; kill-test once, document VRCFT residual behavior.
5. **Combo cues teach unproducible co-activations**: keep combos small; cross-talk matrix with-vs-without comparison.
6. **In-headset cue comprehension**: dry-run a full routine in HMD; distinct per-phase audio + spoken cue names if unclear.

---

## Appendix — direct answers to the 20 required questions

1. **Insertion point:** inside `FaceProcessingPipeline.RunUpdate()`, after stock inference, **before** the One Euro filter (§2).
2. **Inputs:** both camera image (the existing 224×224 input tensor, zero-copy) and stock 45-vector; stock 1280-d embedding in P5 (§2, §6).
3. **MVP architecture:** residual adapter `clamp(stock + f(image, stock), 0, 1)`; tiny conv trunk + MLP (§6).
4. **Size:** ~48 k params (baseline A ~29 k); vs the stock model's 5.89 M (§6).
5. **Temporal context in MVP:** no — One Euro already smooths output; stacked stock-vector history is a cheap later option if metrics demand (§2).
6. **Labels:** guided-cue priors + neutral anchors + weighted stock pseudo-labels + sparse manual corrections, combined via per-frame-per-dim weight masks (§5).
7. **Minimizing manual labeling:** lag alignment + rep auto-rejection + masked loss + shrinkage default + worst-frame HTML review with range/ignore marks (§5).
8. **Guided expression representation:** cue records (`id`, driven dims, phase, target) stamped per frame in `labels.jsonl` (§4).
9. **Continuous intensity:** on-screen animated 0→1→0 target; prior = cue value at frame time, lag-corrected; ramps down-weighted vs holds (§4–5).
10. **Natural speech:** own session type; jaw/mouth dims get low-weight stock pseudo-labels, rest ignored (§5).
11. **Leakage prevention:** split by session, never by frame (§7).
12. **Better-than-stock determination:** session-holdout metrics (per-expression MAE, neutral FAR, cross-talk Δ) + in-VR blend/toggle A/B (§10).
13. **Performance cost:** ≈0.05 ms (A) / 0.3–0.8 ms (B) CPU per frame; measured on the UI thread in P1 (§6, §12.2).
14. **Upstream compatibility:** ~15 additive lines in 3 files, all else new/isolated; schema drift test + load-time schema-hash rejection (§8).
15. **C# vs Python:** C# = capture, runtime inference, UI/toggle/debug; Python = labels, training, export, review tool (§7–9).
16. **Dataset format:** session folders — JPEG q95 224×224 frames + JSONL labels + session.json (§4).
17. **Metadata/versioning:** ONNX `metadata_props` (adapter version/type, schema hash + names, normalization, input size, base-model MD5, ROI, date), validated on load (§6, §8).
18. **Reuse stock embedding?** Yes, practical and cheap (verified graph): derived-copy model with one extra output — deliberately P5, not MVP (§3-C, §11).
19. **Better architecture than the residual proposal?** No — residual + shrinkage is the right MVP; the P5 embedding head is the promising refinement, not a replacement (§3).
20. **Smallest proving experiment:** the no-training notebook check that stock outputs correlate with guided cues (label-source validation) (§12.1, self-critique).

## Appendix B — answers to the 17 avatar-amendment questions

1. **Feasible?** Yes — verified end to end: override at `ParameterSenderService` → VRCFT module passes floats verbatim → avatar renders targets faithfully (modulo per-avatar animator smoothing, handled by step-holds + settle trim) (§14.1).
2. **Improves supervision?** Yes, directly at the plan's #1 risk: mimicking a rendered face resolves shape-identity ambiguity that "smile 63%" text cannot; holds give full-weight mid-intensity anchors (§14.2).
3. **What the avatar does:** plays commanded step-hold sequences (with smooth transitions) per cue, plus a small combo section; preview pass first (§14.2).
4. **How to imitate:** match the avatar's shape; scale intensity to your own maximum (1.0 = your max, not the avatar's rendered extremity); settle during holds — transitions are masked anyway (§14.2, §14.4).
5. **Reaction delay:** settle-trimmed holds carry the weight; transitions masked; residual constant lag (render + human) estimated per cue by cross-correlation, as in the base plan (§14.2).
6. **Holds vs ramps:** both — holds primary (w=1.0), short ramps secondary (w=0.5) (§5, §14.2).
7. **Avatar image = input/supervision?** Neither in MVP — calibration reference and evaluation aid only; optional recorded sidecar for review (§14.3).
8. **Automatic visual matching realistic?** Not near-term — extreme domain gap, no landmarks on a mouth-cam IR crop; gated P7 research (§14.3).
9. **If attempted, representation:** simple geometric proxies first (mouth-opening area, lip-corner displacement measured per-domain), never pixels; learned cross-domain embeddings only after proxies show correspondence (§14.3).
10. **Cheapest matching experiment:** offline notebook correlating per-domain geometric proxies on synchronized Spout+real frames from one session (§14.3, P7).
11. **Preventing avatar-quirk corruption:** own-max intensity semantics + numeric target alongside + cue-preview flagging (bar-only fallback per cue) + avatar-agnostic base model (§14.4).
12. **Avatar-specific second-stage mapper?** Worthwhile concept, kept strictly separate and optional, post-P5, perception-driven (§14.4).
13. **Dataset format changes:** `cue.target` → 45-float vector, keep `cue.dims`, add `level`/`source`; session.json gains avatar metadata + preview flags; optional `avatar/` sidecar (§4).
14. **UI/workflow changes:** cue presenter becomes avatar-first with bar+audio fallback; adds preview pass, prefix/OSC preflight, abort with neutral flush (§14.1–14.2).
15. **Phase changes:** P0 +override hook; P2 avatar-first capture + preview probe; P4 +replay A/B (+optional Spout); P7 visual-matching research (§11).
16. **Before P1/P2?** Only the override hook moves into P0 (to keep upstream edits in one commit). P1 is unchanged; nothing else front-runs (§11).
17. **What the implementing agent does differently:** include the `ParameterSenderService` constructor param/early-return/enqueue in the P0 upstream commit + its MSTest; build the cue engine as snapshot-publisher with pull-based `SampleTarget()` and the three-layer stop-safety; zero stock-pseudo-label weight in override sessions; treat ~64 Hz (not 100 Hz) as the real send cadence (§14.1, §5).

## Verification (for the implementing agent, per phase)

- **P0:** `dotnet test` green (schema drift test + override-path test: raw values, correct addresses, no remap, eye path unaffected); run app with feature off → confirm identical behavior (OSC values unchanged, no new logs); record a 30 s neutral session → inspect `session.json`/`labels.jsonl`/frames by eye; confirm unique-fps log ≈ camera fps.
- **P2 (avatar mode):** with VRChat running + OSC on, start a cue routine → avatar visibly performs the commanded sequence while an OSC monitor on 8888 shows raw target floats; abort mid-hold → avatar returns to neutral within ~1 s; run once with VRChat closed → no errors, bar fallback works. Post-approval: update `WORK_PROGRESS.md` decisions log with the §14 amendment.
- **P1:** `export.py` parity assert passes; C# rejects a schema-hash-tampered model with a logged warning and stock fallback; blend 0 % ⇒ outputs byte-equal to stock; tick p95 within budget.
- **P2+:** evaluate.py metric tables (stock vs A vs B) on held-out sessions; in-VR A/B via toggle.

## Self-critique — strongest failure mode & the cheap experiment that exposes it

The architecture's weakest link is **supervision, not modeling**: a 48 k-param adapter will happily fit whatever labels it's given; if guided-intensity priors are noisy and pseudo-labels are circular, it will learn either nothing (shrinkage wins ⇒ no improvement over stock) or the wrong thing (over-suppression near neutral ⇒ dead subtle expressions — the exact failure the user forbade). The cheapest exposure is **the P2-start notebook experiment — no training required**: record a handful of guided ramps and check whether the lag-corrected stock-output channels correlate with the cues at all. If they do, labels exist and the approach is sound; if they don't, we learn it for the cost of one 10-minute recording session and pivot to step-cues + manual correction before writing any model code. This is also the answer to "smallest experiment to prove/disprove the idea" — it validates the label source, which everything else depends on.
