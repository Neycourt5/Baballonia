# WORK_PROGRESS — Personalized Face-Tracking Fork

Handoff log. Read this plus `you-are-working-inside-harmonic-taco.md` (the approved plan) and you
should be able to continue without any prior conversation.

---

## Current Status

```
Current phase: PHASE 2 IMPLEMENTED (M1-M5, M7 complete and committed).
               PHASE 3 P3-1 THROUGH P3-5 IMPLEMENTED. Eye V2-A remains the
               unchanged selectable control; V2-B Geometry Hybrid is separate.
               The user intentionally overrode the original P3-4 home validation
               gate on 2026-08-14. Hardware/VRChat validation is still pending.
Branch: main (29 commits ahead of upstream 84eca8c, not pushed)
Build: OK (core, Desktop, tests).
Suite: 258 passed / 9 failed / 2 skipped.  The 9 are the same pre-existing
       hardware failures (serial board, firmware JSON, missing BabbleTrainer.exe);
       the 2 skips need PyTorch state or real recordings.
Python: 102 checks across 8 suites, all passing (unchanged by P3-4).
Recordings (HOME PC): 4+ sessions (2 neutral, 2 speech + later additions), ~6,375 frames.
```

**What changed on 2026-08-14 (this session).** Six milestones implemented on the work PC, each
committed separately with its reasoning. Both toolchains turned out to be available here (.NET SDK
10.0.302 at `%LOCALAPPDATA%\Microsoft\dotnet\dotnet.exe`, and the training venv), so everything is
built *and* tested rather than merely written.

| Milestone | Commit | What it adds |
|---|---|---|
| M1 | `92065ad` | JawOpen metrics that measure persistence, not just rate; settle-trim bug fix; opt-in targeted loss options; summary.json v2 |
| M2 | `faf90cf` | 10-second in-memory ring buffer and the "My mouth was closed" button; correction sessions at weight 2.0 |
| M3 | `9f546f7` | Model C offline tooling: derived embedding graph, backfill, the C head, and a fair-comparison harness |
| M4 | `c2efe40` | Guided jaw calibration — the first non-zero supervision in the project |
| M5 | `26467e1` | Model C runtime (secondary output, staleness check, fallback chain), feature-flagged **off** |
| M7 | `7dc366e` | Optional audio expression assist, **off** by default |

Three real bugs were found by the new tests rather than by inspection, and are worth recording
because each would have been silent in production:

1. **The settle-trim aliasing** (M1). A `rest` phase shares its cue id and repetition with the
   `hold` before it, so keying the trim on those alone made the rest look like a continuation and
   skipped its trim entirely — labelling the frames where the user is *relaxing out of* an
   expression as "face at rest" at full confidence.
2. **The embedding output name** (M5). `derive_embedding` was exposing the tensor under its internal
   PyTorch-export name, so the C# runtime looked for `embedding` and found nothing. It now taps it
   through an Identity node with a stable name.
3. **Pitch had no support window** (M7). Pitch was estimated over a 20 ms window, which cannot
   contain two periods of a 60 Hz fundamental, so it silently returned nothing and voice was never
   detected at all.

**REAL-WORLD VALIDATION (2026-08-13/14, home PC — authoritative):**

- **Personalization works on the user's actual face.** The personalized tracker produces a very
  large subjective improvement over stock Baballonia in real VRChat use. This is no longer a
  speculative proof of concept.
- **Model B (image-conditioned) noticeably beats Model A (output-only) in actual VR use.** This is
  direct evidence that visual information in the camera image corrects errors that cannot be
  recovered from the stock 45-vector alone. Preserve this finding: B is the baseline to beat.
- **Remaining problem #1 — intermittent false JawOpen:** the avatar's jaw/mouth sometimes hangs
  slightly open when the real mouth is closed/neutral. Intermittent, and now the single most
  noticeable tracking problem.
- **Remaining problem #2 — rare false TongueOut:** brief, rare, much less annoying. Fix naturally
  if the same architecture covers it; do not let it distract from JawOpen.
- The old flood of calibration/crop/slider problems is greatly reduced by personalization.

**Model naming (standardized):** **A** = output-only residual adapter (stock 45 → MLP). **B** =
image-conditioned residual adapter (camera image + stock 45, own small CNN). **C** =
shared-embedding adapter (Baballonia's internal 1280-d visual embedding + stock 45). **D** =
partial/full fine-tuning of the stock visual network — documented endgame research only, not
scheduled.

Two machines are in play: the **home PC** holds the real dataset, trained models, camera and
VRChat; the **work PC** is code/planning only. Items below marked **HOME-PC VALIDATION REQUIRED**
need the user physically present at the home PC; everything else is agent-safe.

P0 delivered capture and interception: schema lock, pipeline hooks, calibration override path,
dataset recorder, capture page.

P1 delivered the full training and inference loop: a Python package that turns recorded sessions
into a trained ONNX adapter, and a C# runtime that loads, validates, blends and displays it.

---

# PHASE 2 PLAN — approved 2026-08-14 (IMPLEMENTED; see Implementation Record below)

This section is the implementation handoff. It was produced after a full repo audit at HEAD
`84d71c8` plus a verified in-memory ONNX experiment. Everything below the "Approved milestone
ordering" is the work queue; the exact first task is at the end of this section.

## Session-verified facts (2026-08-14, work PC)

**Model C feasibility — CONFIRMED BY EXPERIMENT.** `src/Baballonia/faceModel.onnx`: input
`x.1 [1,1,224,224]`, output `1210 [N,45]`. The tensor
`/model/global_pool/flatten/Flatten_output_0` is a clean **[N,1280]** embedding feeding the final
`Gemm(model.classifier.weight [45,1280])`. Appending it as a second graph output (in memory,
onnx + ORT-CPU): checker passes, stock output **bit-identical**, embedding `[1,1280]` emitted,
**no measurable overhead** (stock p50 9.05 ms vs derived 8.64 ms — within noise; the stock forward
itself dominates any budget). Recomputing expressions as `embedding @ W.T + b` matches ≤1.4e-6.
The ONE unvalidated runtime assumption is **DirectML multi-output** (needs home PC GPU).

**C# integration points (audited):**
- `DefaultInferenceRunner.Run()` is hardcoded single-output (`results[0]`,
  `src/Baballonia/Services/DefaultInferenceRunner.cs:182-194`); `IInferenceRunner` has no
  multi-output concept. This is Model C's main touch point.
- The corrector already borrows the exact input `DenseTensor` the stock session consumed
  (`FaceProcessingPipeline.cs:58`) — zero-copy pattern to preserve.
- Face model path is hardcoded in `FacePipelineManager.CreateInference()`; the eye side
  (`EyeHome_EyeModel` + `EyePipelineManager.LoadInferenceAsync`) is the hot-reload pattern to
  imitate for the derived-model swap.
- `DatasetRecorderService` has **no ring buffer**; `IReadOnlyCueStateSource` has **no production
  implementation** (DI resolves null); **no cue engine exists** (only the tested override
  plumbing); the OSC-prefix preflight is **not implemented**; `StartRecording` passes
  `camera: null` so camera geometry never lands in session.json.
- No audio/microphone code exists anywhere in `src/`.
- `PersonalTrainingService.TrainAsync` passes only `--data --model --out` (trainer defaults fill
  the rest); export is never given `--roi`.

**Python package (audited):**
- Param counts: A `output_mlp_v1` = 28,205; B `image_residual_v1` = 44,389 (docstring "~48k" is
  wrong; README's ~44k is right).
- **BUG (fix in M1):** `labels.py` settle-trim key is `(session_id, cue_id, rep)` — omits phase —
  so a `rest` phase sharing a hold's cue id+rep inherits the hold's start time and is never
  trimmed, and holds at different levels under one id/rep alias each other.
- Dead code to bring alive: `evaluate.cross_talk` (implemented, never called),
  `labels.apply_corrections` / `W_MANUAL_CORRECTION=2.0` (no ingestion path),
  `labels.estimate_cue_lag_seconds` (never called). `evaluate.py` has no CLI (README claims one).
- `per_expression_mae` scores only weight≥0.9 cells — today that is exclusively zero-target
  neutral frames; no JawOpen-specific or persistence metrics exist anywhere.
- Exports are fixed batch 1, opset 17, inputs `image`+`stock` → `personal`, in-graph clip.
- Schema: JawOpen = index 4, TongueOut = 33, tongue = 33–44; sha `c35805d0…392e` pinned
  identically in C# and Python.

## Approved milestone ordering (user-confirmed 2026-08-14)

```
M1  JawOpen metrics + labels fixes + JawOpen-targeted loss options    [agent-safe]
M2  Hard-example ring buffer + "My mouth was closed" desktop button   [agent-safe code]
M3  Model C offline spike (Python-only fair B-vs-C comparison)        [agent-safe code]
M4  Avatar-guided JawOpen calibration MVP (cue engine)                [agent-safe code]
M5  Model C runtime in C#   — ONLY IF the M3 spike wins (or ties with a clearly
                              better JawOpen block)
M6  B capacity (B-Medium, maybe B-Large) — ONLY IF C loses/unclear
M7  Audio Expressiveness Assist — architecture documented below; build only after
    M1–M5 are stable
M8+ Expand guided cues (Smile/Frown/Pucker/Funnel/MouthL-R/TongueOut, then
    cross-talk-driven combos); active-learning polish
D   Stock-model fine-tuning: OPTIONAL ENDGAME RESEARCH only — no upstream training
    code/checkpoint exists; catastrophic-forgetting + rebase risk; revisit only if
    A/B/C + hard examples + guided supervision all demonstrably ceiling.
```

Rationale for deviations from the earlier phase list: hard-example capture **compounds with
elapsed time** (every VR session passively accumulates labeled real failures — the rarest data in
the project) and is model-agnostic, so it ships early. Model C is split into a ~30-minute
zero-C#-risk offline spike before any runtime investment. B-capacity is demoted to conditional:
growing B's scratch CNN (trained on ~6k highly correlated frames) competes with a free pretrained
1280-d embedding; run it only if C disappoints. The JawOpen-specific B improvements that attack
the actual complaint (loss/weighting) still land first, in M1.

## M1 — Evaluation upgrade + labels fixes + JawOpen loss treatment

Files: `training/babble_personal/labels.py`, `evaluate.py`, `train.py`, `models.py`; C# tail in
`PersonalTrainingService.cs` + `PersonalizationView.axaml`(+VM).

1. **labels.py trim fix**: replace the `hold_started_at` dict with sequential segment tracking —
   track `current_segment = (session_id, cue_id, rep, phase, round(level,4))` +
   `segment_start_ticks`, reset on any change; trim the first 0.5 s of both `hold` and `rest`
   segments. New test `training/tests/test_labels_trim.py` (standalone `main()` runner like the
   existing tests): a rest sharing a hold's cue id+rep must have its first 0.5 s at weight 0 —
   must FAIL against current code first. Add a `dim_boost` kwarg to `build_labels`.
2. **evaluate.py new metrics**:
   - `activation_runs(values, ticks, threshold=0.15)` → per-session false-activation run
     durations; stats `runs_per_minute, mean_run_s, p95_run_s, max_run_s`. Persistence is the
     real complaint: one 2-second jaw hang ≫ twenty 1-frame blips at equal FP rate.
   - `DimReport` for JawOpen(4) and TongueOut(33), stock-vs-personal paired:
     *closed set* (neutral sessions ∪ correction windows ∪ guided rest post-trim): `fp_rate`,
     `mean_closed`, `p95_closed`, `max_closed` + run stats.
     *Sensitivity guardrail* (anti-dead-zone): guided-hold `mean_at_level[]`, `open_separation`
     (mean@1.0 − mean_closed); speech `std`, `range = p95−p05`, and
     `range_retention = personal_range / stock_range` (flag < 0.8).
     *Monotonicity* on guided data: Spearman ρ of commanded level vs per-hold-segment mean +
     pairwise ordering accuracy.
   - Wire the existing `cross_talk` into train.py's evaluation when guided sessions exist.
   - New CLI: `python -m babble_personal.evaluate --data ROOT [--sessions …] [--onnx PATH]
     [--json OUT]` — scores any exported model (or stock passthrough) on chosen sessions;
     detects v1 (image+stock) vs v2 (stock+embedding) signatures from the ONNX inputs.
3. **train.py / summary.json v2**: `summary_version: 2` + blocks `jaw_open`, `tongue_out`,
   `cross_talk|null`, `hard_examples|null`. New CLI options, all default-off/conservative, none of
   which are dead zones (desired behavior stays: closed → ≈0, slightly open → small, open →
   strong):
   - `--dim-boost "JawOpen=2.0,TongueOut=1.5"` — weight (never target) emphasis on those dims in
     zero-target rows.
   - `--fp-penalty 0.25 --fp-dims "JawOpen,TongueOut"` — asymmetric false-positive term over
     supervised zero-target cells: `fp_penalty * mean(relu(predicted)^2)`. Guardrail: the summary
     must print `range_retention` beside `fp_rate` so any sensitivity trade is visible.
   - `--hard-negative-boost` (default 0) — oversample neutral/correction rows whose *stock*
     JawOpen > 0.3 (decision-boundary frames) via `WeightedRandomSampler`.
   - Add an `fp` component to the loss metrics dict.
4. **C# tail**: `TrainingSummary`/`ReadSummary` gain nullable v2 fields (all `TryGetProperty`,
   v1-tolerant); Compare section gains two headline lines: "False jaw-open at rest: X% → Y%" and
   "Longest false open: Xs → Ys".
5. **HOME recording guidance** (diversity over volume, ~90 s each). **Wear the headset normally —
   do not deliberately reposition it.** A headset cannot be re-seated identically to the precision
   a 224×224 mouth crop resolves, so recording across separate sittings covers real placement drift
   by itself; deliberately odd angles spend model capacity on situations that never occur. What to
   record: neutral ×3 in different room lighting (attacks the measured brightness↔JawOpen
   correlation of −0.56, the one variable that is genuinely worth varying on purpose); **silent-hold
   probe** (jaw silently held open 10 s / closed 10 s, ×3 — tests the predicted context-gate failure
   from the first-training analysis); speech variety (reading vs conversation vs exaggerated).
   Note the real hazard is changing the **ROI/crop settings** after recording — that is an actual
   domain shift and requires re-recording, unlike normal headset wear.

**HOME-PC VALIDATION REQUIRED:** retrain on the real corpus; record the JawOpen block (fp_rate,
p95 run duration, range_retention) stock vs current B — these are the baseline numbers every
later milestone is judged against.

## M2 — Hard-example capture ("My mouth was closed")

New files in `src/Baballonia/Services/Personalization/`:

1. **`HardExampleBuffer.cs`** — in-memory ring, 300 entries (10 s @ the existing 30 fps cap).
   Entry = preallocated `byte[50176]` raw Gray8 224×224 (copy via `Marshal.Copy`; assert
   `Mat.IsContinuous()`) + ticks + `float[45]` stock + `float[45]` personal + checksum.
   **≈15.2 MB total, allocated once, zero steady-state GC.** Raw bytes, not JPEG — encoding
   happens only at persist time, never near the tick. Subscribes to `NewRawExpressionsEvent` +
   `NewCorrectedExpressionsEvent` (same-tick ordered pair — attach corrected to the newest raw
   entry; no pairing key needed). Reuse the 30 fps cap + sparse FNV dedupe logic (duplicate the
   ~30 lines privately from the recorder; comment the duplication — cheaper to rebase than a
   shared refactor). `Snapshot(TimeSpan window)` deep-copies matching entries under a short lock;
   persistence runs entirely off the lock. Setting `PersonalModel_HardExampleBuffer`, default ON.
   The buffer is never written to disk except on an explicit user flag.
2. **`HardExampleService.cs`** — `FlagAsync(CorrectionKind kind, TimeSpan window)` (default
   **5 s**; UI offers 2/5/10 s): snapshot → background-write a standard session directory
   `<yyyyMMdd_HHmmss>_correction` (new `SessionType.Correction` in `DatasetSession.cs`) with the
   normal layout (frames JPEG q95; labels.jsonl rows also carrying the model's answer via a new
   null-suppressed `FrameLabel.Personal` field) **plus `correction.json`**:
   ```json
   { "version": 1, "kind": "mouth_closed", "corrected_dims": [4], "target": 0.0,
     "window_seconds": 5.0, "flagged_utc": "…", "source": "hard_example",
     "model": { "adapter_type": "image_residual_v1", "trained_utc": "…", "blend": 1.0 } }
   ```
   Model provenance stamped from `PersonalModelManager.LoadedMetadata` at flag time. 3 s debounce
   between flags. The data architecture is dimension-generic — a future "Tongue was not out" /
   "I was neutral" / "This was a smile" flag is just another `kind` + `corrected_dims`/`target`;
   only the first UI is JawOpen-specific.
3. **Trainer ingestion**: `dataset.py` reads `correction.json` → `Session.is_correction`;
   `labels.py` correction branch: corrected dims get the correction target at
   `W_MANUAL_CORRECTION` (2.0 — the dead constant becomes live); **all other dims weight 0
   (fully masked)** — the user asserted only "mouth closed"; they may have been emoting otherwise
   in VR, so weak zero-priors on other dims would be actively wrong. Correction sessions are
   auto-discovered by the existing session discovery; the default last-per-type holdout gives a
   genuine hard-example holdout once ≥2 exist, scored by M1's `hard_examples` summary block.
4. **UX (user-decided 2026-08-14)**: **desktop window button only for MVP** — "My mouth was
   closed" button + 2/5/10 s window selector in the main Personalization section (it is a primary
   loop, not Advanced). OSC avatar-toggle and a global hotkey are deferred follow-ups, documented
   here so they are not re-litigated. Retraining stays the explicit **[Improve My Model]** action
   (= the existing Train button) — never automatic.

**HOME-PC VALIDATION REQUIRED:** flag real false-JawOpen moments (brief headset-peek to click is
acceptable for MVP), verify the persisted window brackets the actual failure, retrain via
[Improve My Model], and record the hard_examples holdout numbers before/after.

## M3 — Model C offline spike (Python only; zero C# risk)

Purpose: decide whether the pretrained 1280-d embedding beats B's scratch CNN **before** paying
for any runtime work.

1. **`training/babble_personal/derive_embedding.py`**: load the stock model → append
   `/model/global_pool/flatten/Flatten_output_0` as a second graph output named `embedding`
   (after the existing output so ordering is stable; `--tensor` overridable) → `onnx.checker` →
   ORT-CPU parity (stock output must be bit-identical — verified achievable this session) →
   write `faceModelWithEmbedding.onnx` + sidecar `faceModelWithEmbedding.json`
   `{base_model_md5, embedding_source_tensor, tool_version}`. **Never modifies the stock file.**
   Staleness policy: the derived model is invalid unless sidecar md5 == md5 of the current stock
   file → refuse + regenerate. Never silently use an embedding from a mismatched base model.
2. **`compute_embeddings.py`** (backfill for existing recordings): per session, decode frames in
   labels order, batch through the derived model (CPU) → `embeddings.bin` (little-endian
   **float16**, 1280 per row ≈ 2.5 KB/frame ≈ 7.7 MB per 3000-frame session) + `embeddings.json`
   `{dim, dtype, count, source: "backfill", model_md5}`. Known bounded skew: backfill uses
   JPEG-decoded frames while recorded stock vectors came from the live pre-JPEG tensor; q95 keeps
   it small; a `--verify-against-live` mode quantifies it once live capture exists (M5). The M5
   live recorder writes the same two files with `source: "live"` — one convention, one loader.
3. **`models.py` C head** — `EmbeddingHeadAdapter`, `adapter_type = "embedding_head_v1"`,
   `uses_embedding=True`:
   `LayerNorm(1280) → Linear(1280,256) → SiLU → Dropout(0.1) → concat stock45 →
   Linear(301,128) → SiLU → Linear(128,45)` zero-init → residual → clamp. ≈374k params; offer
   `--c-width 128` (≈190k) for the grid. AdamW weight decay applies; the temporal penalty works
   unchanged (previous-row lookup already exists); photometric consistency is N/A (no pixels) —
   replace with `--embedding-noise 0.01` (Gaussian noise relative to per-dim std, same
   residual-invariance idea). Small uniform refactor: `residual(image, stock, embedding=None)`
   across all models; `_Batch` loads embeddings when `uses_embedding`; `--model c` errors with
   the backfill command line if a train session lacks embeddings.
4. **`export.py` v2**: embedding models export inputs `stock`+`embedding` → `personal` (no
   vestigial image input); `personal_adapter_version = 2` for this type only (A/B stay v1 — old
   builds cleanly refuse C via the existing version check); metadata adds `embedding_dim`,
   `requires_embedding`, `embedding_source_model_md5`. Parity check feeds random stock+embedding.
5. **`experiment.py`** — fair-comparison harness (also serves M6):
   `python -m babble_personal.experiment --data ROOT --models b c --seeds 0 1 2
   --val-sessions <pinned ids>` → identical splits + label config per run → markdown/CSV table:
   params, analytic FLOPs, train wall-time, ONNX size, val fit, neutral FAR, resting jitter, the
   full JawOpen block, cross-talk, MAE — mean±spread over seeds.
6. **Decision gate (record the outcome here):** C wins → M5. C ties but the JawOpen block is
   clearly better → M5. C loses across seeds → skip M5, run M6. Subjective VR remains the final
   arbiter after whichever runtime ships. Architecture elegance is not a success metric — if C
   does not beat B, keep B.

**HOME-PC VALIDATION REQUIRED:** the spike run itself (~30 min: derive + backfill + experiment) —
the dataset lives there. All code is written and synthetic-tested on the work PC first.

## M4 — Avatar-guided JawOpen calibration MVP

The core loop: the app commands a known JawOpen target → the existing override path drives the
avatar through OSC→VRCFT → the user imitates what they SEE (no guessing what "50% jaw" feels
like) → the commanded vector is the label. Holds matter more than transitions. Both known-open
AND known-closed supervision are produced — exactly the "closed but Babble thinks open" vs
"actually open" discrimination the JawOpen problem needs.

1. **`GuidedCaptureRoutine.cs`** — table-driven state machine over the existing tested
   `ExpressionOverrideService`. JawOpen routine (3 reps): 5 s lead-in (override active at neutral
   so the avatar settles) → per level in {0.5, 1.0}: prep 1 s → transition 0.75 s → **hold 3 s** →
   transition 0.75 s → rest 2 s. **Distinct cue ids per level** (`JawOpen50`, `JawOpen100`),
   `Dims=[4]` — belt-and-braces with the M1 trim fix. UI-thread DispatcherTimer: `KeepAlive()`
   each tick, `PushPhase` on change; all three existing stop-safety layers apply unchanged. The
   routine starts/stops the recorder session itself (`SessionType.Guided`). **Implements the
   OSC-prefix preflight** (refuse to start if `AppSettings_OSCPrefix` is non-empty — the standing
   watch item) + requires an active video source. Start with 0/0.5/1.0 levels (the safer first
   experiment); expand to finer levels only if the monotonicity metrics justify it.
2. **`CueStateSource.cs`** — the missing production `IReadOnlyCueStateSource`: holds the current
   `CuePhase` (volatile); `CurrentCue()` interpolates the commanded vector at call time (same
   math as `ExpressionOverrideService`) and returns a `FrameLabel.CueLabel` with
   `source: "avatar"`. Register in DI (`App.axaml.cs`) and pass into `DatasetRecorderService`.
3. **Trainer**: wire the dead `estimate_cue_lag_seconds` as a per-rep diagnostic printed for
   guided sessions — correlation ≳0.6 with monotone plateaus validates the whole guided design
   (this IS the original plan-§12.1 experiment, now run on the first real session). Optional
   `--min-cue-corr` rep gate (default off). **Ordinal supervision experiment** (optional, default
   off): `--ordinal` margin ranking loss over hold-frame pairs at different commanded levels —
   teaches `1.0 attempt > 0.5 attempt > neutral` ordering without trusting exact human
   intensities. Add only if it demonstrably improves supervision.
4. VM/View: "Guided: Jaw calibration (~1 min)" button + live instruction text in Recordings.
   Expansion path (M8+): Smile, Frown, Pucker, Funnel, MouthLeft/Right, TongueOut, then combos
   chosen by measured cross-talk — new routine-table rows only.

**HOME-PC VALIDATION REQUIRED:** first guided session watching the avatar in a VRChat mirror;
check per-rep cue-lag correlation ≥0.6 and monotone stock plateaus; then retrain and read the new
monotonicity/sensitivity metrics. Fallback if correlation is poor: two-level 0/1.0 holds only.

## M5 — Model C runtime in C# (conditional on the M3 gate)

Minimal blast radius: eye path untouched, `IInferenceRunner` untouched.

1. **`DefaultInferenceRunner`** (additive, ~35 lines): a `SecondaryOutputName` property; when set
   and present in `OutputMetadata`, allocate a `_secondaryTensor` and identify the primary output
   by name; in `Run()`, iterate results by name — copy secondary, return primary exactly as
   today. **When `SecondaryOutputName` is null the behavior is byte-for-byte current** (the eye
   path cannot regress). Implements new `IEmbeddingSource { DenseTensor<float>? GetEmbedding(); }`
   (interface lives in `Services/Personalization/`, keeping upstream `Contracts/` untouched).
   `InferenceFactory.CreateWithSecondaryOutput(path, name)`.
2. **Model store + fallback chain**: setting `PersonalModel_UseEmbeddingRunner` (default false
   until DirectML validation passes, then flip ON so every future recording is C-trainable).
   `FacePipelineManager.CreateInference()`: if enabled → `EmbeddingModelStore.TryGetValid()`
   (`%APPDATA%\ProjectBabble\Models\faceModelWithEmbedding.onnx` exists AND sidecar md5 == md5 of
   the current stock model) → load with the secondary output; any miss/mismatch/exception → log +
   plain stock. Regeneration via `PersonalTrainingService.RegenerateEmbeddingModelAsync()`
   (child-process `derive_embedding` — C requires training tools anyway). GUI: Advanced →
   "Enable embedding runner (Model C)" (regenerates if stale, hot-reloads via the existing
   `LoadInferenceAsync` pattern).
   **Fallback chain C→B→stock**: `PersonalModelManager.ReloadAsync` gains one rung — if the
   primary personal model fails validation/load, try `.previous.onnx` (typically the last B) with
   a logged warning; if that fails, corrector = null (stock). `SupportedAdapterVersion` → 2; v2
   validation requires inputs `stock[1,45]` + `embedding[1,1280]` AND an active embedding runner
   (exposed as `FacePipelineManager.EmbeddingActive`).
3. **Corrector**: new `IEmbeddingAwareCorrector : IExpressionCorrector` + `EmbeddingModelCorrector`
   (mirror of `PersonalModelCorrector`: own CPU session, blend lerp, self-disable on exception;
   null embedding → stock passthrough + one-time warning). Pipeline block:
   `var emb = (InferenceService as IEmbeddingSource)?.GetEmbedding();` then dispatch on the
   interface. `PersonalModelCorrector` (A/B) untouched.
4. **Live embedding recording**: `NewRawExpressionsEvent` gains a defaulted `float[]? embedding =
   null` parameter (additive, no caller breaks); the pipeline attaches a 1280-float copy only when
   the runner is active; the recorder writes `embeddings.bin` float16 + `embeddings.json`
   (`source: "live"`), row-aligned with labels.jsonl by construction (same writer iteration).
5. **DirectML validation** (**HOME-PC VALIDATION REQUIRED** — the one unvalidated assumption):
   guarded MSTest (`Inconclusive` without DML) asserting derived-model DML outputs match CPU
   ≤1e-3 for both outputs; manual checklist: UseGPU on → derived model → 10 min live tracking →
   tick p50/p95 vs stock baseline. Any DML failure ⇒ automatic fallback to plain stock on DML
   (C disabled while GPU on, clearly logged). Never run a second CPU backbone pass just for
   embeddings — one forward pass is the whole point.

## M6 — B capacity experiment (only if C loses/unclear)

`ImageResidualAdapter` gains width parameters → `--model b-medium` (`image_residual_m_v1`,
channels 1→12→24→48→96, hidden 192, ≈100–130k params) and `--model b-large`
(`image_residual_l_v1`, 1→16→32→64→128, hidden 256, ≈250–300k) — distinct adapter_type strings
for provenance; `TrainingModelChoice` extended additively. Protocol: the same `experiment.py`
harness — identical pinned splits, seeds {0,1,2}, full M1 metric table + params/FLOPs/train-time/
ONNX-size/C#-latency-bench. B-Medium first; B-Large only on a monotone improvement trend. Honest
expectation to record: the corpus, not capacity, is the likely bottleneck — this experiment exists
to prove that cheaply if C hasn't already made it moot. Prefer the smallest model with the best
generalization.

## M7 — Audio Expressiveness Assist (architecture decided; build later)

**Placement: a second nullable stage in `FaceProcessingPipeline`, after the corrector, BEFORE the
One Euro filter** — `IExpressionEnhancer? Enhancer`, identical volatile/fail-safe/null-means-off
pattern as `Corrector`. Reasoning: (a) the filter smooths gain steps and audio-buffer zipper
noise before anything reaches VRChat; (b) the recording tap (`NewRawExpressionsEvent`) is
upstream, so training data is never audio-contaminated; (c) the calibration remap stays last so
the user's range mapping still applies; (d) a null slot = bit-identical output — "off equals the
exact visual path" holds by construction. Rejected alternative (after filter, in
`ParameterSenderService`): puts unsmoothed gain modulation on the wire and grows an upstream file.

Mode 1 (first): local WASAPI/NAudio capture on its own thread → `IAudioFeatureSource` ring of
`{rms, voicedProb, ticks}` — transient features only, no audio ever stored, no cloud, no speech
recognition → `ProsodyEnhancer`: `out_i = corrected_i * (1 + k·smoothedLoudness)` on jaw∪mouth
dims only — amplifies movement already away from neutral, never invents it (Smile = 0 stays 0
regardless of shouting), clamp [0,1]. Sync: sample the newest audio feature older than a
configurable ~50 ms offset; features stale >250 ms ⇒ gain decays to 1 (fail-safe to visual-only);
mic unavailable ⇒ visual tracking unaffected. GUI: one toggle "Enhance expressions while
speaking" (default OFF); strength/sync-offset under Advanced. Mode 2 (viseme-class hints that
mildly reinforce visually-supported shapes, e.g. strong "oo" evidence nudging Funnel/Pucker the
tracker already sees) is explicitly deferred until Mode 1 is proven; multimodal model training is
documented as a distant experiment, not the initial implementation. Audio must never re-create
canned viseme behavior that fights the tracked face.

## Unified supervision (the full label-source table after M1–M4)

| Source | target | weight | notes |
|---|---|---|---|
| Neutral session, all dims | 0 | 1.0 | |
| Speech pseudo (jaw∪mouth) | stock | 0.3 | drop to 0 via flag if circularity shows |
| Guided hold, cued dims | commanded | 1.0 (tongue 0.4) | after 0.5 s settle trim (M1 fix) |
| Guided rest | 0 | 1.0 | now also trimmed (the M1 bug fix) |
| Guided transitions | — | 0 | masked |
| Guided uncued dims | 0 | 0.25 | minus co-activation exclusions |
| **Hard example, corrected dims** | correction target | **2.0** | all other dims weight 0 |
| Ordinal (optional) | ranking pairs | `--ordinal` | ordering only, no magnitudes |

Every sample records provenance (`SessionType`, `correction.json`, cue `source`); the trainer
never flattens sources into equally trusted targets.

## GUI integration (keep the current philosophy)

Main page additions only: M2 "My mouth was closed" button + 2/5/10 s selector (main section);
M1 two JawOpen headline lines in Compare; M4 "Guided: Jaw calibration (~1 min)" in Recordings;
M5 "Enable embedding runner (Model C)" under Advanced; the Train dropdown gains C (and B-Medium
if M6 runs) with plain-language descriptions. [Improve My Model] = the existing Train button.
Normal users never see ONNX/embedding/parameter-count jargon outside Advanced.

## Performance budget (measured, not estimated)

Stock forward ~9 ms CPU (work-PC measurement; home GPU differs) dominates the tick; A 0.03 ms,
B 0.21 ms (home CPU, warm); C head ≈0.4 MFLOP ⇒ expected ≪1 ms CPU (bench in M3's harness and via
the existing C# latency-test pattern in M5); embedding extraction is free (same forward pass —
verified). Ring buffer: one ≤50 KB memcpy at ≤30 fps (~1.5 MB/s), 15.2 MB fixed allocation.
Still unmeasured: whole-tick p50/p95 with a live camera (HOME checklist). Audio (M7) runs on its
own thread and never blocks the visual loop.

## Risks & rollback

| Risk | Mitigation / cheapest resolver |
|---|---|
| DirectML multi-output misbehaves | M5 guarded test + 10-min home checklist; auto-fallback to stock-on-DML designed in |
| Embedding lacks info the stock model discarded (C ceiling) | the M3 spike answers this for ~30 min of home compute, zero C# risk |
| FP penalty suppresses genuine opens (dead face) | range_retention + sensitivity metrics printed beside fp_rate in every summary; silent-hold VR check |
| Backfill vs live embedding skew | `compute_embeddings --verify-against-live` on the first live-embedding session |
| Guided labels garbage (user can't track cues) | per-rep cue-lag correlation diagnostic; `--min-cue-corr` gate; fallback to 0/1.0 levels |
| Ring-buffer tick cost | fixed prealloc; disable setting; latency test |
| Stale derived model after an upstream model update | sidecar md5 check ⇒ refuse + regenerate; never silently mismatch |
| Rebase surface growth | only additive edits to `DefaultInferenceRunner` + one defaulted event param; everything else is new files under `Services/Personalization/` or `training/` |
| Training RAM as the corpus grows | eager loading is fine to ~10k frames (~2 GB); lazy loading is noted future work, not built now |

Every feature fails safe to the previous known-good path: buffer off ⇒ no behavioral change;
C invalid ⇒ `.previous.onnx` (B) ⇒ stock; guided inactive ⇒ normal tracking; audio off or mic
dead ⇒ exact visual path; personalization off ⇒ stock Baballonia.

## IMPLEMENTATION RECORD — what M1–M7 actually built (2026-08-14)

Read this before touching any of it; the reasoning matters more than the file list.

### M1 — measuring the actual complaint (`92065ad`)

`activation_runs` measures false activations **in time**. This exists because a false-activation
rate cannot distinguish one two-second jaw hang from forty single-frame flickers, and only the first
is what the user notices. `DimReport` pairs that with the guardrail in the other direction: any
model can win every false-positive metric by refusing to move the expression at all, so range
retention and open/closed separation are computed from the same run and printed beside it. The
warning only fires where there was real movement to lose — flagging the tongue during ordinary
speech, which is mostly sensor noise, would train the reader to ignore it.

Three opt-in knobs, all default-off: `--dim-boost` (symmetric weight emphasis, never touches a
target), `--fp-penalty` (asymmetric, charges only overshoot above a *confident* zero, on named
dimensions only), `--hard-negative-boost` (oversamples confidently-zero frames the stock model is
already firing on). The asymmetric one is the one capable of producing a dead face, which is exactly
why range retention is printed next to it.

`TrainingSummaryReader` was extracted so the C#/Python contract is testable against literal JSON.
That seam fails silently: a renamed key does not crash, it just makes a number vanish from the
results screen.

### M2 — turning real mistakes into labels (`faf90cf`)

`HardExampleBuffer` holds 10 seconds as a fixed ring of preallocated frames (~15 MB, no
steady-state garbage, ~50 KB memcpy per tick, measured under 1 ms). Raw bytes rather than JPEG
because encoding belongs at save time on a background thread.

The subtle part is pairing. The pipeline publishes the raw event then the corrected event on the
same tick, so "attach the correction to the newest entry" is only right when that entry came from
*this* tick — and it did not, whenever the frame was rejected as a duplicate or over-rate. Without
the guard, tick N+1's prediction silently overwrites tick N's, producing evidence that looks valid
and blames the wrong picture. Tested directly.

**Only the named dimension is supervised, at weight 2.0.** The button says "my mouth was closed" —
a claim about the jaw and nothing else. The user may have been mid-sentence or smiling; labelling
the other 44 expressions as neutral would manufacture supervision they never gave, at the highest
weight in the system, on frames chosen precisely because the model was confused. An unlabelled cell
costs nothing; a confidently wrong one costs a lot.

No auto-retrain: corrections are worth most in batches, and swapping the model after every press
would make it impossible to tell what helped.

### M3 — Model C offline (`9f546f7`)

Verified against the real `faceModel.onnx`: exposing the 1280-d embedding leaves the 45 stock values
**bit-identical**, and `embedding @ classifier.weight.T + bias` reproduces them to 1e-4 — which
confirms the exposed tensor really is the classifier's input rather than some other layer of the
right shape. Parity is asserted at exactly zero, because adding an output computes nothing new, so
any drift would mean every recording describes a different model than the one running.

Embeddings are a float16 binary sidecar (~2.5 KB/frame). JSON would add ~25 KB per line to
labels.jsonl and destroy its readability for debugging.

`EmbeddingHeadAdapter` is ~375k params with LayerNorm (the embedding's scale is uncontrolled) and
dropout (1280 inputs against a small personal corpus is the one place here where overfitting is a
real risk). Zero-initialised, so an untrained adapter is an exact passthrough like A and B.

`experiment.py` exists because "B versus C" is only a real question if nothing else differs, and by
default nothing is held fixed — `split_sessions` picks the holdout itself, so two runs can silently
score against different data.

### M4 — the first non-zero supervision (`c2efe40`)

Every high-confidence label before this was zero. Neutral sessions say "all expressions at rest";
speech pseudo-labels echo the stock model at low weight. So the adapter could learn to quieten a
resting face and nothing else — the ceiling P1 hit.

The jaw routine alternates commanded openings with rests, producing known-open **and** known-closed
frames in one session under identical lighting. Neutral recordings supply only the second kind.

Each level gets its own cue id (`JawOpen50`, `JawOpen100`), which is belt-and-braces with M1's trim
fix. `CueStateSource` and the override service read the same immutable snapshot on their own clocks,
so the value sent to the avatar and the value written to the dataset are the same number by
construction — tested mid-transition, where a separately-computed target would diverge silently.

The **OSC-prefix preflight** is finally implemented: with a prefix set, the VRCFT module never
matches, the avatar never moves, and the recorder fills with neutral faces labelled as expressions.
Nothing about that fails at the time, so it refuses to start.

### M5 — Model C runtime, flag off (`26467e1`)

`DefaultInferenceRunner.SecondaryOutputName` is additive: null takes a fast path that is byte for
byte what it always did, so the eye pipeline and the plain stock path are untouched.

`EmbeddingModelStore` refuses a derived model whose sidecar MD5 does not match the current stock
file. This is the failure worth engineering against: a derived graph from a different network still
loads, still runs, still emits 1280 numbers — describing a different feature space. No exception, no
visibly broken output, just a face that is subtly wrong.

`PersonalModelManager` gained one rung of fallback to `.previous.onnx`, so the likely mistake —
installing a C adapter with the runner off — lands on the model that was working an hour ago rather
than dropping to stock with no explanation.

### M7 — audio, off by default (`7dc366e`)

It multiplies, it never adds. `MouthSmileLeft = 0` stays 0 no matter how loudly the user shouts.
That is the difference between this and the viseme-driven mouth animation it must not become.

Placed after the corrector and before the One Euro filter: the filter smooths gain changes, the
recording tap upstream stays audio-free so training data can never be contaminated, and the
calibration remap stays last.

Old-fashioned DSP rather than a model — microseconds, predictable on untested audio, and
transparently incapable of transcription. Voice detection requires loudness **and** periodicity;
either alone is wrong too often, and a false positive moves the avatar's mouth while the user is
silent.

Measured: analysis 0.06 ms per 20 ms window on its own thread; enhancement under 0.001 ms per tick.

---

## What is agent-safe now vs HOME-PC

**All agent-safe work is done.** M1–M5 and M7 are implemented, tested and committed. There is no
remaining task that can be meaningfully advanced without the real camera, dataset, headset or GPU.

---

# NEXT EXACT TASK — HOME PC, in this order

Every item below needs the home machine. They are ordered by how much they unblock: 1 and 2 are
prerequisites for judging anything else, 3 is the architecture decision, and 4–6 are validation of
features that are already built and switched off.

Record the outcome of each **in this file** as you go. The numbers are the point; without them the
next session is guessing again.

### 1. Get a build with all of this in it

A complete Release build already exists at **`bin\Baballonia-v5\`** (built and verified on the work
PC — see BUILD CONVENTION below). If the repo is synced, just run it; `bin\Baballonia-v4` stays as
the fallback.

To rebuild from scratch on the home PC, note the two gotchas documented under BUILD CONVENTION:
initialize submodules first, and do not pass `-o` (the comma in the repo path breaks MSBuild):

```
git submodule update --init --recursive
cd src\Baballonia.Desktop
%LOCALAPPDATA%\Microsoft\dotnet\dotnet.exe publish -c Release -r win-x64 --self-contained true
```

then copy `bin\Release\net10.0\win-x64\publish` to a new `bin\Baballonia-v6`.

### 2. Establish the JawOpen baseline (M1)

Train on the existing corpus and read the new report. This is the first time the project has a
number for the actual complaint.

* Press **Train My Face Model** (model B).
* Record here: `jaw_open.stock_fp_rate` vs `personal_fp_rate`, **`stock_runs.max_seconds` vs
  `personal_runs.max_seconds`** (the longest single false opening — the number that best matches
  what you notice), and `range_retention`.
* The Compare card now states these in plain language; `metrics.txt` in the run folder has the
  full block.

**Then record the sessions M1's guidance asks for**, which are what the later milestones need:
neutral ×3 in *different room lighting* (this attacks the measured brightness↔JawOpen correlation
of −0.56, and lighting is the one variable genuinely worth varying on purpose), a **silent-hold
probe** (jaw held open silently 10 s, then closed 10 s, ×3 — this tests the context-gate weakness
predicted after the first training run and never checked), and speech variety. Wear the headset
normally; do not reposition it deliberately.

### 3. The Model C decision (M3) — the architecture gate

About 30 minutes, entirely offline, no VR needed.

```
set PY=%LOCALAPPDATA%\babble-train-venv\Scripts\python
cd /d <repo>\training

%PY% -m babble_personal.derive_embedding --stock ..\src\Baballonia\faceModel.onnx
%PY% -m babble_personal.compute_embeddings --data "%APPDATA%\ProjectBabble\PersonalDataset" ^
     --model ..\src\Baballonia\faceModelWithEmbedding.onnx
%PY% -m babble_personal.experiment --data "%APPDATA%\ProjectBabble\PersonalDataset" ^
     --models b c --seeds 0 1 2 --out "%APPDATA%\ProjectBabble\PersonalTraining\experiment"
```

**Paste the resulting table into this file.** The gate, decided in advance so the result cannot be
rationalised after the fact:

* C wins, or ties with a clearly better JawOpen block → adopt C (continue to step 6).
* C loses across seeds → **keep B**, and run the B-capacity experiment (M6) instead. Architecture
  elegance is not a success metric.

### 4. Hard examples (M2)

Use VRChat normally. When the avatar's jaw hangs open while your mouth is closed, press
**My mouth was closed** on the Personalization page straight afterwards (a brief headset peek is
fine for now — in-VR triggers were deferred by your decision). Collect a handful over a session or
two, then retrain and record the `hard_examples` block before and after.

### 5. Guided calibration (M4)

Start VRChat with a VRCFT avatar, clear `AppSettings_OSCPrefix` if set (the app refuses to start
otherwise, and explains why), and press **Start jaw calibration**. Watch your avatar in a mirror
and copy it for about a minute.

Then train and read the **cue-tracking table printed before training starts**:

* Mean correlation ≳0.6 → the guided design holds; expand to more expressions later.
* Much lower → either the avatar is not rendering the cue or the intensities are not reproducible.
  Fall back to two-level 0/1.0 holds and record that decision here.

This is the experiment the entire guided-capture design rests on, and it is cheap.

### 6. Model C runtime + DirectML (M5) — only if step 3 adopted C

The one runtime assumption never verified: everything was tested on CPU.

* Advanced → tick **"Use the face model's visual features"**. It builds the derived model itself.
* Train with model C, confirm it loads, and check `EmbeddingRunnerDirectMlTest`-style behaviour by
  turning **UseGPU on** and running 10 minutes of live tracking.
* Record whole-tick p50/p95 with a real camera, GPU on and off. If DirectML misbehaves, the
  designed response is automatic fallback to the plain stock model — confirm that happens rather
  than a crash.

### 7. Audio assist (M7) — independent of everything above

Personalization → Audio → **Enhance expressions while speaking**. Talk normally and A/B the
toggle. Watch for the boost leading or lagging your voice; if it does, adjust
`AudioAssist_SyncOffsetMs` (default 50 ms). Confirm that with it off, tracking is exactly what it
was.

---

## Deferred by decision, not by omission

* **In-VR correction triggers** (OSC avatar parameter, global hotkey). You chose desktop-button-only
  for the MVP. The OSC route needs one synced bool on your avatar and a menu toggle; the service is
  already dimension-generic, so adding it is a trigger, not a redesign.
* **M6, the B-capacity ladder** (B-Medium/B-Large). Conditional on step 3: only worth running if C
  loses, since growing B's scratch CNN competes with a pretrained embedding that is free.
* **Audio phoneme/viseme assist.** Interfaces are shaped for a second feature source, but it stays a
  later experiment until prosody is validated in real use.
* **Model D.** Reassessed in the Phase 3 plan below: it joins a unified experiment with Model E
  rather than remaining standalone endgame research.

---

# PHASE 3 PLAN — approved 2026-08-14 (next implementation handoff for Opus 5 Medium)

Planned on the work PC after a full eye-subsystem audit, direct ONNX graph inspection, and
external research. The home-PC checklist above remains the immediate queue; the first Phase 3
milestones are agent-safe and independent of it. Two subjects: **(I)** the face-model program
after the B-vs-C gate, including the new Model E; **(II)** Eye Tracking V2 against the Paper
Tracker benchmark.

## Session-verified facts (2026-08-14)

**Eye model, graph parsed directly.** `src/Baballonia/eyeModel.onnx`: 3.8 MB, 956,474 params.
Input `[b,8,128,128]` = 4 temporal frames × 2 eyes, stacked in C# by `ImageCollector`
(`src/Baballonia/Services/Inference/ImageCollector.cs`, queue depth 5, channels swapped L/R at
line 20). Two Gather ops split the 8 channels into two fully independent per-eye conv towers
(`left.*`, `right.*`), each 6×(Conv+ReLU+MaxPool) → Gemm → Sigmoid → 3 values; Concat → `[b,6]`.
**The model outputs exactly X, Y, Lid per eye. Widen/squint/brow do not exist in the model, the
OSC sends, or the module path on `main`** — all commented scaffolding
(`ParameterSenderService.cs:47-55`, `BabbleVRC.cs:80-104`, `ICalibrationRoutine.cs:357-409`,
dead `OutputIndexMap` in `DefaultInferenceRunner.cs:298-310`). Raw output layout is
right-eye-first (`/rightEyeY,-X,-Lid,/leftEyeY,-X,-Lid`, per `expr-dev`'s named map); `main`'s
local variable names are mislabeled but two errors cancel and the wiring is correct.

**Eye pipeline audit (key points, file:line).**
- `EyeProcessingPipeline.RunUpdate()` (`src/Baballonia/Services/Inference/EyeProcessingPipeline.cs:14-58`)
  mirrors the face pipeline *minus* every personalization hook: no corrector/enhancer stage, no
  raw-output event (the tap the face recorder uses), One Euro applied to the **raw** 6-vector
  *before* post-processing (line 45) — opposite of the face order.
- Post-processing (`ProcessExpressions`, lines 60-105): [0,1]→[−1,1] remap; lid inversion; **both
  eyes share one Y** (lid-weighted average); closed-eye yaw borrowing; convergence clamp under
  `AppSettings_StabilizeEyes` (default true).
- `ParameterSenderService` sends exactly 6 eye addresses; eye values are `Remap`ped but **never
  clamped** (line 177), unlike face values.
- Model hot-swap already exists (`EyeHome_EyeModel` setting + `EyePipelineManager.LoadInferenceAsync`,
  lines 55-74) — but the swap write is non-volatile and the old runner is not disposed.
- Mat leaks on every eye tick: the 8-channel `collected` Mat and `ImageCollector`'s `Split()`
  arrays are never disposed; `transformed` is disposed twice (lines 32, 55).
- `MatToFloatTensorConverter` pins the eye input to 8×128×128 — any different model shape needs a
  new converter/collector. `ProcessExpressions` truncates any longer output to 6 (lines 93-102).
- Face + eye share the same 10 ms UI-thread `DispatcherTimer` (`ProcessingLoopService.cs:23-79`).

**Eye calibration today.** Godot overlay (TCP JSON via OverlaySDK, port 2425) + step framework
(`src/Baballonia.Desktop/Calibration/ICalibrationRoutine.cs`); Basic routine ≈ 30 s tutorial +
**120 s gaze reticle** + 30 s blink → CaptureBin (100-byte header already has
Widen/Squint/Brow/Dilate fields `main` never populates) → spawns **BabbleTrainer.exe** (closed
binary; fine-tunes the in-repo per-eye checkpoints `baseline_L/R.pth`, ~480k params each) →
installs `tuned_temporal_eye_tracking_<ts>.onnx` → hot reload. **Both Overlay and Trainer
binaries are absent from this tree** (fetched by `download_dependencies.ps1`) — calibration
cannot currently run here. `GazeOnly`/`BlinkOnly` share a stale-bin merge bug
(`ICalibrationRoutine.cs:426,448`). `AppSettings_ShareEyeData` upload exists, default off — keep
off.

**`origin/expr-dev` IS the "Expressions Alpha"** (13 commits, head `4a69127`, in this fork's
remotes): `OrderedFloatMap` named outputs end-to-end; widen/squint/brow calibration passes live
(20 s each); new `GazeExpressionCaptureStep` (gaze truth *while* holding an expression — fixes
"eye looks up while squinting"); VRCFT module rewritten to a flat address switch handling
Squint/Widen/Brow; trainer assets swapped to a two-headed `model_best.pt` (8.0 MB) +
`gaze_model_best.pt` (0.5 MB) + 60 MB unlabeled footage. Its trainer binary ("qpro trainer") is
external and unverified-obtainable. `origin/expr-dev-dbg` (commit `554f3e2`) adds a per-stage
profiler/debug panel worth porting regardless.

**Paper Tracker benchmark (verified vs inferred).** VERIFIED: Paper's PC eye software is
**EyeTrackVR v2.0 BETA 14** (hands-on review; kit auto-installs VRCFT + ETVR module), so the
benchmark's methods are inspectable: ETVR's `EyeTrackApp` publicly contains `AHSF.py`,
`haar_surround_feature.py`, `ransac.py`, `leap.py`, `blink.py`, `intensity_based_openness.py`,
`osc_calibrate_filter.py` — a **hybrid** of classic pupil localization and a small learned
eyelid+pupil landmark model (LEAP), with **mapping** calibration ("look to all extremes → look
straight → Recenter", tens of seconds, optional 9-point overlay; no retraining). INFERRED ONLY:
the polygon+pupil debug view ≈ LEAP landmarks — the P3-2 spike verifies by reading `leap.py`.
LICENSING: newer ETVR is restrictively licensed; reimplement published ideas (Haar-surround,
RANSAC ellipse are academic), copy no code without a version-specific license review.

## PART I — Face architectures A–E

### The decomposition that answers the Model E hypothesis

| Burden on the universal model | Removed by personalization? |
|---|---|
| Population facial diversity | **Yes** — one face |
| Camera / lens / mount diversity | **Yes** — one geometry |
| Lighting diversity | Mostly — one room, few conditions |
| Label noise at scale | **Yes** — replaced by better-than-corpus guided/hard/neutral labels |
| **Expression coverage** | **No** — a personal model must still *see* every expression it will ever emit; it has no prior to fall back on |
| Behavioral/multi-day diversity | Partially |

**E's binding constraint is coverage, not capacity.** B/C inherit stock behavior on every dim the
personal data never covers (residual default = "leave stock alone"). A from-scratch E has no
prior anywhere: an unlabeled dim is not "stock quality", it is *undefined*. Twelve tongue dims,
asymmetries and NoseSneer are barely cueable — so E-strict's whole-face ceiling sits **below**
B/C regardless of network size.

### The unification: E-strict / E-distilled / D are one experiment

```
              init           architecture     prior on uncovered dims
E-strict      random         free (small)     none            <- the strict scientific claim
E-distilled   random         free (small)     stock-as-teacher (training-time only; runtime independent)
D             stock weights  stock B0         the weights + self-distillation (anti-forgetting)
```

One harness (`--model e-s|e-m|d1 --teacher stock|none --init random|stock`), identical data,
identical **day-level** splits, identical eval — three tracks for the price of one, fair by
construction.

**D prerequisite [AGENT-SAFE spike]:** ONNX→PyTorch weight transplant (EfficientNet-B0-class,
`in_chans=1`, 45 out; initializer names map back — the ONNX was exported from PyTorch). **Assert
torch-vs-ONNX forward parity ≤1e-5 before anything trains; failure ⇒ D gated off, reported.**
D ladder: D1 = last stage + head (~1.3M trainable) + self-distillation on natural footage;
D2 = two stages only if D1 beats best-of(B,C) on the JawOpen block with no range-retention loss
on day-holdout; D3 (full) likely never reached at realistic data volume.

### Data math and capacity ladders

A realistic 2–4-week corpus (~10 min/day): guided holds ≈ 25–50k raw frames but only **~2–5k
effective independent samples** (a 3 s hold ≈ 90 correlated frames); ~15k neutral; 10–20 min
speech; dozens of corrections. Sensible capacity: **hundreds of k params; low millions only with
heavy augmentation** — matching the in-repo precedent (the per-user fine-tuned eye model is 956k
and works).

E ladder (grayscale, in-graph AvgPool→112 so C# feeds its existing tensor):
**E-S ~0.3M** (B's trunk widened 16→32→64→128, hidden 256 — feasibility probe);
**E-M ~1.2M** (hand-written MobileNetV3-small-style inverted residuals + SE — main candidate);
**E-L ~3M** only if E-M improves monotonically on day-holdout.
Runtime: an adopted E **replaces** the ~9 ms stock CPU forward with ~60–120 MFLOPs → likely 3–6×
tick speedup — the only architecture that makes tracking cheaper. Staged rollout: phase 1 = E as
a corrector that ignores its inputs (zero new runtime surface, safe A/B); phase 2 (post-adoption
only) = model swap via the pattern the eye side already has.

### Supervision deltas for E/D

- **Geometric augmentation unlocks (E/D only; B/C cannot warp** — it breaks pairing with the
  recorded stock vector). Pose-level labels are invariant under small affine (±4 % translate,
  ±5° rotate, ±5 % scale) — exactly what a reseat does. Highest-value robustness lever. For
  teacher-supervised speech frames, re-run the teacher on the warped image.
- **Ordinal loss defaults ON for guided data** (`--ordinal` exists): intensity *ordering* is
  trustworthy even when magnitudes aren't; E has no shrinkage prior keeping it calibrated.
- **Day-level holdout is the headline metric** — a guided-pose memorizer aces same-day
  session-holdout.
- **Speech labeling:** E-strict none; E-distilled/D teacher-at-low-weight with personal labels
  overriding (stock is deliberately the *floor*).
- **New metrics:** per-dim coverage-vs-quality table; **uncovered-dim drift** (output vs stock on
  label-free dims — the dead/erratic-dim detector).

### Comparison table

| | A | B | C | D1 | E-strict | E-distilled |
|---|---|---|---|---|---|---|
| Architecture | 45→MLP→Δ | img+45→CNN→Δ | emb1280+45→MLP→Δ | stock B0, last stage tuned | small CNN→45 | small CNN→45 + teacher |
| New runtime params | 28k | 44k | 375k (embed free) | 5.9M (replaces stock) | 0.3–1.2M (replaces) | 0.3–1.2M (replaces) |
| Inference | +0.03 ms | +0.21 ms | +~0.3 ms | ≈ stock | **2–6× faster than stock** | same |
| Prior inherited | stock out | stock out | stock features | full weights | **none** | teacher floor |
| Data / supervision need | tiny/low | small/med | small/med | large/high | **largest/highest** | large/high |
| Ceiling (this user) | low | med | med-high | **highest** | med (coverage-capped) | **highest** |
| Realistic near-term | proven | **proven best** | ≥B pending gate | data-limited | below B whole-face | at/below B initially |
| Overfit risk | low | managed | med | high | **very high** | high |
| Signature failure | can't see | illum. leak (fixed) | stale derived model | forgets uncovered dims | dead/erratic uncovered dims | inherits teacher errors |
| Rebase risk | none | none | low | high (24 MB divergent) | low | low |
| Needs guided/hard labels | little | some | some | heavily | **cannot exist without** | heavily |

### Three rankings (re-derived; the old informal D>C>E>B>A does not survive)

- **Ceiling:** `D ≈ E-distilled > C ≥ B > E-strict > A` — E splits in two; E-strict drops below
  B/C *even at its ceiling* (coverage caps it).
- **Realistic near-term:** `best-of(B,C) > A > D1 ≈ E-distilled > E-strict`.
- **Per engineering effort:** `B (done) > C (validation only) > E-distilled/D1 (one shared
  harness) > E-strict (cheap add-on) > D2+`.
- They disagree because ceiling rewards capacity+prior, realism discounts by the slow-growing
  corpus, and effort discounts by code+risk surface. The disagreement dictates the ordering:
  validate the proven cheap things, grow the corpus (helps every row), let E/D wait for the data
  that is their actual bottleneck.

### E/D experiment (concrete, kill-cheap)

- **Gate E0 [HOME-PC prerequisite for the real run]:** ≥8 guided expressions × ≥2 levels across
  ≥3 days (≥2 reseats); ≥10 neutral sessions across ≥3 lighting conditions; ≥15 min speech; ≥20
  corrections.
- **Harness [AGENT-SAFE, synthetic-tested]:** `models.py` + `StandaloneFace` (E-S/E-M) +
  transplanted `StockBackbone`; `train.py` + `--teacher/--distill`, `--geo-aug`, `--val-days`;
  `export.py` adapter **v3** (input `image` → `personal`); coverage + drift metrics;
  `experiment.py` rows.
- **Runs:** B, C(if adopted), E-S-strict, E-M-strict, E-M-distilled, D1 — pinned day-splits,
  seeds {0,1,2}.
- **Precommitted gates:** *Kill E-strict* if E-M-strict loses to B on overall MAE **and** the
  JawOpen block on day-holdout (record the result either way — the science is the point).
  *Continue E-distilled* only if within ~10 % of best-of(B,C) overall *and* better on JawOpen,
  else park until the corpus doubles. *Adopt* only on beating best-of(B,C) (JawOpen block + MAE,
  no suppression warnings) + in-VR A/B win. *D ladder* gates as above; transplant parity failure
  ⇒ D off.

## PART II — Eye Tracking V2

### Design conclusion

The benchmark wins on calibration because it calibrates a **mapping**, not a **model**: rich
per-frame eye state + a per-user anchor/range fit in tens of seconds. This fork calibrates by
*retraining a CNN* (minutes; currently impossible in this tree — binaries absent). Eye V2 applies
the face-personalization philosophy to eyes: keep a fixed extractor, add a tiny personal layer,
calibrate the layer. **User decision (recorded):** `origin/expr-dev` is one candidate/reference,
**never the presumed basis**; `main` stays the stable base unless evidence strongly justifies
otherwise; Eye V2 is designed for the best achievable gaze/squint/wide/blink + fast calibration,
not merely an improved alpha.

### Tiers

**V2-A — personal mapping calibration over the existing model [build first].** New
`IEyeStateMapper` stage (face-`Corrector` pattern: volatile field in `EyeProcessingPipeline`
after `ProcessExpressions`, `SetMapper` on the manager, null = exact current behavior) + a
~35–45 s anchor calibration producing per-eye: gaze map (offset+gain or 2nd-order poly), lid
anchors (closed / relaxed-neutral / squint / wide), and derived outputs: calibrated gaze,
openness, **Wide** (aperture above the personal neutral envelope), **Squint** (sustained partial
closure — temporal discrimination from blinks, which are <~300 ms transients). Extend
`_eyeExpressionMap` to send Widen/Squint (addresses exist commented) and fix the module-side
Squint mapping in our fork. Honest ceiling: squint from one lid scalar + time is partial; V2-A
buys calibration UX, personal ranges, wide-eye, recenter and measurably better gaze mapping —
not benchmark-grade squint.

**V2-B — pupil/aperture geometry features [likely second].** Per-eye 128×128 crops (tap:
`NewTransformedFrameEvent`) → classic pupil center + aperture extraction (reimplemented
Haar-surround + ellipse/RANSAC — academic, license-safe) → features feed the same mapper: gaze
from pupil position (the responsive path ETVR proves), squint from aperture + pupil-visibility,
debug overlay (pupil point + lid line — the Paper-like view). LEAP-style learned landmarks are
the fallback if classic CV fights this IR imagery.

**V2-C — personal eye model training [conditional].** Extend `babble_personal` to eyes:
fine-tune from `baseline_L/R.pth` or `expr-dev`'s two-headed checkpoints; dot-target gaze labels +
anchor prompts; replaces the closed BabbleTrainer with our own. Only if A+B measurably can't
reach Paper on gaze/squint.

### Safety architecture (hard requirement)

Selectable **Eye Tracking Mode: Default / Experimental (V2)**. V2 state in separate `EyeV2_*`
settings keys and files; Default's calibration/model never written by V2; mapper=null restores
Default instantly; any V2 init/calibration failure logs and falls back; A/B is a radio toggle.
Additive files in a new `Services/EyeV2/` + one volatile stage + one manager hook.

### V2-A calibration protocol (first experiment, exact)

~35–45 s guided flow (UI page; no Godot dependency for the anchor part):
1. **Relax** 5 s → per-eye neutral lid distribution (median+spread ⇒ personal neutral envelope).
2. **Slow blinks ×3** ~8 s → closed anchor + blink transient duration stats.
3. **Squint hold** 5 s → squint aperture band (overlap with blink band accepted; discrimination
   is temporal + band).
4. **Wide hold** 5 s → wide anchor.
5. **Gaze:** center dot 2 s → recenter offset; then 5-dot (center/L/R/U/D, 2 s each) ⇒ per-eye
   offset+gain (+cross-term if needed). 9-point deferred unless corner error demands it. An
   ETVR-style free "look to extremes" sweep is the recorded alternative.
6. Persist per-eye anchors + map + capture metadata (`EyeV2_Calibration.json`).
**Recenter** = center-dot step alone (~2 s), anytime. **Reseat validity:** at V2 start compare
2 s of relaxed stats vs stored anchors; drift ⇒ prompt one-press Recenter; full recal only if lid
anchors also drifted. Continuous micro-adaptation deferred; log the drift signal now.

### Benchmark protocol vs Paper Tracker [HOME-PC]

Same-session ABAB; record each system's OSC stream + screen-record debug views.
**Calibration:** wall-clock, #actions, repeat-twice anchor consistency, post-reseat recovery
time. **Gaze:** fixation jitter (std over 5 s × 5 known points), A→B step response
(latency/overshoot from OSC log), corner error, 15-min drift. **Blink:** 20 natural + 10
deliberate vs annotated video (missed/false/latency). **Squint:** 5×5 s holds — detection,
hold stability, half-blink false positives during conversation. **Wide:** 5 deliberate — 
detection above personal neutral, false-Wide count during normal gaze shifts. Subjective VRChat
mirror A/B last, as arbiter.

### Debug tooling

Port the `expr-dev-dbg` profiler/debug panel idea; eye V2 panel: per-eye crop with pupil/aperture
overlay (V2-B), raw vs calibrated gaze dots, lid/openness/squint/wide bars, per-eye confidence,
blink flag, anchor values, live fixation-jitter + step-latency meters. Purpose: "is calibration
wrong, or did the tracker misread the image?" at a glance. Requires the missing eye raw-output
event (P3-3).

## Ordered Phase 3 milestones (decision gates; tags explicit)

- **F0 [HOME-PC VALIDATION REQUIRED]** — the checklist above (items 1–7), unchanged. Gates the
  face experiment decisions; does not block the agent-safe items below.
- **P3-1 Guided routine expansion [AGENT-SAFE]** — routines for Smile L/R, Frown, Pucker, Funnel,
  MouthLeft/Right, TongueOut(binary) + 3 combos (smile+jaw, frown+jaw, pucker+jaw); per-dim
  coverage table in evaluate. Highest-leverage face work: feeds B/C/E/D and Gate E0.
- **P3-2 Eye comparative spike [AGENT-SAFE, RESEARCH SPIKE]** — inspect `expr-dev` end-to-end
  (code, checkpoints via torch, trainer obtainability) + ETVR sources (method + license). Compare
  three candidate designs on equal footing: (a) expr-dev's approach, (b) hybrid
  geometry/personal-mapping (V2-A→B), (c) new lightweight learned (V2-C family) — scored on
  achievable quality, calibration speed, risk, maintenance. Reuse expr-dev pieces only where
  genuinely best. Written comparison + recommendation appended here; no production code.
- **P3-3 Eye observability [AGENT-SAFE]** — `NewRawEyeExpressionsEvent(Mat, float[6], ticks)`;
  debug panel v1 (raw/calibrated, jitter + step-latency meters); fix the eye-tick Mat leaks (each
  pinned by a test — justified now that we build on this path); volatile+dispose hygiene on
  runner swap. Measurement before change.
- **P3-4 Eye V2-A [AGENT-SAFE code + HOME-PC VALIDATION]** — mapper stage + manager hook +
  `EyeV2_*` settings; calibration flow + Recenter + validity check; Widen/Squint sends + module
  fix; Mode selector. **Gate:** clearly better than current Baballonia and within sight of Paper
  on gaze ⇒ P3-5; can't beat current Baballonia ⇒ stop eye work, keep Paper.
- **P3-5 Eye V2-B geometry [CONDITIONAL on P3-4 gate]** — pupil/aperture extraction, overlay,
  features into mapper; re-benchmark. Gate decides if V2-C is ever needed.
- **P3-6 E/D harness [AGENT-SAFE]** — harness + transplant spike (parity gate) + export v3 +
  coverage/drift metrics; synthetic-tested; waits for Gate E0.
- **P3-7 E/D experiment [HOME-PC + CONDITIONAL on Gate E0 and F0's B-vs-C outcome]** — run it,
  apply the precommitted gates, record verdicts here.
- **P3-8 [CONDITIONAL]** — Eye V2-C / D2+ / E adoption runtime, each only via its gate.
- Audio phoneme/viseme assist stays below all of the above.

## Phase 3 risks and rollback

| Risk | Containment / rollback |
|---|---|
| E/D overfit guided poses | day-holdout headline; capacity ladder; kill gates |
| E uncovered dims dead/erratic | drift metric; teacher floor (E-distilled); adoption gate blocks suppression |
| D transplant imparity | parity ≤1e-5 asserted first; failure ⇒ D off entirely |
| D forgetting / divergent model | self-distillation; artifacts outside repo; stock file never modified |
| Eye V2 breaks eyes | parallel selectable path; Default untouched; failure ⇒ auto-fallback; mapper=null = current behavior |
| V2 calibration corrupts Default's | separate `EyeV2_*` keys/files; Default state never written |
| Stale V2 calibration after reseat | startup validity check; one-press Recenter; full recal only on anchor drift |
| Classic pupil CV fails on this IR imagery | V2-B gate; LEAP-style learned fallback; V2-A value already shipped |
| expr-dev trainer unobtainable / ETVR license | P3-2 establishes before any dependence; no ETVR code copied without version license review |
| DirectML surprises | CPU-first for all personal/eye-V2 models; DML validation stays a home-PC item with auto-fallback |
| Upstream updates | additive-files discipline; derived/personal models regenerate; schema drift tests |

## Phase 3 GUI plan

```
FACE                                          EYES
  Tracker: Stock / Personal                     Eye Tracking Mode: Default / Experimental (V2)
  [ Improve My Model ]                          [ Calibrate Eyes ]  (~40 s guided)
  Quick correction: [ My mouth was closed ]     [ Recenter ]        (2 s, anytime)
  Guided calibration: Jaw / Smile / Frown / …   Eyebrows: Off (default; not a priority)
  Audio: [ ] Enhance while speaking
Advanced: model choice, strength, metrics,    Advanced: eye debug panel (pupil/aperture overlay,
  embedding runner, training log                raw vs calibrated, jitter/latency meters,
                                                anchors, thresholds, per-eye confidence)
```

## P3-2 — EYE RESEARCH SPIKE: FINDINGS (2026-08-14, work PC)

Research/reporting milestone, no production code. Everything below is measured on this machine or
read out of the actual artifacts; inferences are labelled.

### 1. The shipped eye model, exactly

`eyeModel.onnx`: 956,474 params, input `[b,8,128,128]`, output `[b,6]`. Two independent per-eye
towers of six `Conv3x3 → ReLU → MaxPool` stages then `Flatten → Gemm → Sigmoid`, channel ladder
**28, 42, 63, 94, 141, 212**, 3 outputs per eye.

**It is exactly two copies of `baseline_L/R.pth`** — the per-eye trainer checkpoints in the repo
measure **478,237 params each** with precisely that ladder (`conv1..conv6` + `fc(212→3)`), and
2 × 478,237 = 956,474. So the shipped model *is* the baseline architecture, and the calibration
trainer fine-tunes those two checkpoints per user.

### 2. expr-dev's actual architecture (measured from its checkpoints)

| checkpoint | contents | params | notes |
|---|---|---|---|
| `gaze_model_best.pt` | `arch: "microchad"`, `step 12000`, `val_gaze_mse 0.0679` | **119,628** | conv ladder **14,21,32,47,70,106** — literally half-width baseline; 4ch in, `fc_gaze → 2` |
| `model_best.pt` | `backbone: "mobilenetv4_conv_small_050.e3000_r224_in1k"`, `val_mse 0.0783`, keys `student`/`teacher` | **977,266** (each) | `conv_stem (32,4,3,3)` → 4 temporal channels, **one eye**; `classifier (4,1280)` → 4 outputs |

Three conclusions that matter more than the numbers:

1. **expr-dev splits gaze from expression into separate specialised networks**, and made gaze
   *four times smaller* (119k vs 478k) while making expression *twice as large* (977k) and
   ImageNet-pretrained. Their authors concluded, independently, that **gaze needs far less capacity
   than expression discrimination**. That is the single most useful architectural finding here.
2. **The expression net is ImageNet-pretrained** (`e3000_r224_in1k`) — the alpha did not train eye
   expressions from scratch either.
3. `student`/`teacher` + the 60 MB `footage_unlabeled.npz` means **mean-teacher semi-supervised
   training**: labelled calibration frames plus a large pool of unlabelled footage. That is how
   they got expression quality out of a small labelled set.

The 4 expression outputs are **lid, widen, squint, brow** per eye — matching the four new OSC
addresses the branch's rewritten module handles, and the five label fields its capture steps stamp
(`lid, browRaise, browAngry, widen, squint`). Runtime total would be gaze(2) + expression(4) = **6
values per eye, 12 total**, versus main's 3 per eye / 6 total.

### 3. The finding that changes the plan: expr-dev ships main's eye model

`git rev-parse origin/expr-dev:src/Baballonia/eyeModel.onnx main:src/Baballonia/eyeModel.onnx`
returns the **same blob hash** (`0d1c998d…`). The alpha branch ships the *identical* 956k eye ONNX.
Its entire improvement is produced by an external trainer binary ("qpro trainer") that exists in
neither branch, from checkpoints that are only *starting weights*.

**Therefore merging expr-dev would deliver none of its eye improvements.** What it delivers is a
blueprint: architecture, output schema, training regime, and usable pretrained starting weights —
all of which we now have measured. Reimplementing the trainer is tractable; adopting the branch is
not a shortcut.

### 4. Capacity, measured (ONNX Runtime, CPU EP, app session options, 128×128, both eyes)

Reconstructing the shipped architecture and re-measuring it validates the method: the rebuild
(956,474 params) lands at **3.34 ms** against the real artifact's **3.44 ms**.

| dense tower (shipped family) | params | p50 ms | | separable (MobileNet-style) | params | p50 ms |
|---|---|---|---|---|---|---|
| shipped | 956,474 | **3.44** | | w0.5 | 332,012 | 3.19 |
| ×1.5 | 2,151,018 | 5.84 | | w0.75 | 723,524 | 3.82 |
| ×2 | 3,816,982 | 7.49 | | w1.0 | 1,273,012 | 4.16 |
| ×3 | 8,581,530 | 17.65 | | w1.5 | 2,824,828 | 5.98 |
| ×4 | 15,250,118 | 22.16 | | w2.0 | 4,999,004 | 7.05 |
| | | | | w3.0 | 11,177,988 | **10.32** |

At matched latency the separable family buys **~1.3× the parameters at ~6 ms and ~2.5× at ~10 ms**,
and the gap widens with size — the dense tower's cost grows with the square of width. It is also
deeper (8 blocks vs 6 convs) with residuals and squeeze-excite, so its *usable* capacity advantage
is larger than the parameter ratio suggests. expr-dev reached the same conclusion by choosing
MobileNetV4.

### 5. The budget denominator — read these as a *ladder*, not as absolute limits

Measured on the **work PC** (a modest laptop-class CPU), one tick's inference:

```
face only    p50 14.95 ms      eye only  p50 3.37 ms
face + eye   p50 17.26 ms      eye is 20% of the pair; the face model dominates
```

**Do not read these as the home budget.** The home machine is a **Ryzen 9 7950X3D** — 16 fast
cores with a large V-cache, and ORT's intra-op parallelism is left at default (only
`InterOpNumThreads` is pinned to 1), so it parallelises across cores. Realistically that is several
times faster than the numbers above, before DirectML is even considered. The work PC simply cannot
hold 100 Hz on CPU; the home PC very likely can, and runs with `AppSettings_UseGPU` true anyway.

What transfers is the **shape of the curve**, not the milliseconds: relative cost between
architectures and capacity tiers, measured under identical conditions. Three consequences:

- **Capacity is not the binding constraint at home.** On a 7950X3D plus DirectML, a 1–5 M-param
  separable eye model is comfortably real-time. The plan should be limited by *training data and
  overfitting*, not by inference cost. **HOME-PC VALIDATION REQUIRED: re-run this ladder there,
  CPU and DirectML, to fix the absolute numbers.**
- The eye stage is only ~20 % of the pair even today, so **growing it is cheap in relative terms** —
  doubling the eye model costs far less than the face model already does.
- There is a real **face↔eye interaction**: if Model E ever replaces the 5.9 M face model with a
  ~1 M net, it frees the majority of the tick and directly funds a larger eye model. The two tracks
  are not independent.

### 6. Paper Tracker / ETVR (verified vs inferred)

VERIFIED: Paper's PC eye software is **EyeTrackVR v2.0 BETA 14**; its public `EyeTrackApp` contains
`AHSF.py`, `haar_surround_feature.py`, `ransac.py`, `leap.py`, `blink.py`,
`intensity_based_openness.py`, `osc_calibrate_filter.py` — a hybrid of classic pupil localisation
and a small learned eyelid/pupil model, with **mapping-only calibration** ("look to all extremes →
look straight → Recenter", tens of seconds, optional 9-point overlay). No per-user retraining.
INFERRED: the polygon+pupil debug overlay corresponds to LEAP landmarks.
LICENSING: post-v2.0-beta ETVR is restrictively licensed — reimplement the published algorithms
(Haar-surround and RANSAC ellipse are academic), copy no code without a version-specific review.

### 7. Answer: is the planned Eye V2 too conservative on capacity?

**Yes on architecture family and output schema; no on the staging order.**

- The plan's V2-C wording ("lightweight personalised eye model") anchors on the wrong axis.
  Evidence says the axis that matters is **architecture efficiency and task separation**, not
  parameter minimisation. A separable/inverted-residual trunk at **1–5 M params** costs 4–7 ms on
  the *work* CPU — and the home machine is a 7950X3D with DirectML available, so this range is not
  close to the limit there. Inference cost should not be what caps the eye model; **training data
  and overfitting should be**.
- **Split gaze from expression.** Both this measurement and expr-dev's independent choice support
  a small gaze head (~120–500 k) and a larger expression trunk (~1–5 M). Squint/wide/blink
  discrimination is the hard visual problem; gaze is a smooth 2-DOF regression.
- **Expect pretraining to matter.** expr-dev did not train expressions from scratch, and neither
  should we by default — though a from-scratch arm is worth keeping for the same reason
  E-strict is: it answers the question cheaply.
- Unchanged: **V2-A (fast mapping calibration) still goes first.** It is days not weeks, it
  delivers the calibration UX that is half the user's complaint, and it builds the measurement rig
  every later comparison needs. Nothing here argues for skipping it — only for not stopping there.

**Revised recommended ladder** (built once the V2-A rig exists, compared on identical
sitting-level splits):

| tier | trunk | params | why |
|---|---|---|---|
| gaze head | half-width dense tower (microchad-class) | ~0.12–0.5 M | proven sufficient for 2-DOF by expr-dev |
| V2-L Balanced | separable w0.75–1.0 | ~0.7–1.3 M | ≈ shipped latency, far more capable |
| V2-L High Accuracy | separable w1.5–2.0 | ~2.8–5.0 M | 6–7 ms on the work CPU; a non-issue on a 7950X3D / DirectML |
| V2-L Research | separable w3.0+ | ~11 M+ | worth trying — the constraint here is data, not latency |

Keep the tiers as **distinct, selectable, separately-versioned models** (own architecture id,
metadata, provenance, benchmark row) rather than overwriting one file — the point of a ladder is
comparing rungs.

### 8. Recommendation

Build **V2-A first** (mapping calibration over the existing model), because it is cheap, it fixes
the calibration complaint, and it creates the benchmark rig. Then **V2-B/L as one learned track**
with the split-head design above and a real capacity ladder, reusing expr-dev's *findings* (task
split, pretraining, mean-teacher semi-supervision, its pretrained checkpoints as starting weights)
without adopting its branch. Keep `main` as the base.

---

## PHASE 3 PROGRESS — P3-1 THROUGH P3-5 COMPLETE (2026-08-14, work PC)

> **Gate override recorded 2026-08-14:** the user explicitly authorized P3-5 before performing
> the original P3-4 home-PC V2-A validation. This is an intentional override, not evidence that
> V2-A passed its hardware gate. Preserve V2-A exactly as a selectable control. The first serious
> home comparison must test **Default / V2-A / V2-B / Paper Tracker in the same sitting**.

| Milestone | Commit | Result |
|---|---|---|
| P3-1 guided expansion | `e0e1b66` | 13 cues (8 core + 2 asymmetric + 3 combos), routine picker, coverage census, extended exclusions |
| P3-2 eye research spike | `0b5ed4e` | findings above; expr-dev measured, capacity ladder benchmarked |
| P3-3 eye observability | `969aa49` | raw eye event, per-tick leak fixes, temporal reset, swap hygiene |
| P3-4 Eye V2-A | `ad0846d` | selectable personal affine mapping, ~43 s anchors, 2 s recenter/validity, Wide/Squint OSC + module fix |
| P3-5 Eye V2-B | working tree | separate Geometry Hybrid mode; classic pupil/iris + lid/aperture extractor, replaceable extractor seam, annotated eye-crop debug view, exact V2-A fallback |

### P3-5 implementation record (gate override build)

- The Home selector now exposes three independent choices: **Default Baballonia**, **V2-A —
  Personal mapping**, and **V2-B — Geometry Hybrid**. Existing persisted V2-A keeps enum value 1,
  the same calibration file, and the unchanged `EyeV2Mapper`; it was not replaced or silently
  upgraded.
- V2-B wraps that exact V2-A mapper and adds `IEyeGeometryExtractor`, a replaceable seam whose
  implementations can be classic CV or a learned IR landmark model. The first implementation fits
  a dark pupil/iris ellipse, estimates upper/lower lid lines from image gradients, and reports
  normalized pupil center/radius, lid aperture, pupil visibility/occlusion evidence, and per-eye
  confidence.
- Only confident aperture/visibility evidence augments Squint and Wide. Gaze remains V2-A's
  calibrated output for this first hardware experiment. Missing, stale, low-confidence, or failed
  extraction produces the exact V2-A output for that tick; V2-B initialization failure selects
  V2-A, and invalid/missing calibration still selects Default.
- Advanced Eye V2 debug shows the real left/right 128x128 crops side-by-side with pupil cross/circle
  and upper/lower lid lines, plus normalized geometry and confidence. Debug images are capped at
  10 Hz.
- The large learned V2-L capacity ladder was not built. The extractor interface is the only shared
  infrastructure added for a later learned implementation.
- Focused Release tests: **65 passed / 0 failed / 0 skipped**, covering Eye V2/pipeline behavior,
  geometry extraction and fallback, mode selection, A/B/C face labels and persistent slots, and the
  Model C train guard. Desktop self-contained Release publish succeeded. Packaged as
  `bin/Baballonia-v8` and `bin/Baballonia-v8.zip`. The companion VRCFT module also rebuilt with
  0 errors at `src/VRCFaceTracking.Baballonia/bin/Release/net10.0/VRCFaceTracking.Baballonia.zip`.

**Test totals after P3-4:** C# **258 passed / 9 failed / 2 skipped** — the 9 are the unchanged
pre-existing hardware failures (8 ESP32 serial + `TrainerServiceTest` needing `BabbleTrainer.exe`).
Python **102 checks** across 8 suites, all passing (`test_coverage.py` is new: 11).

**Home-PC hardware, recorded for future capacity decisions:** **RTX 4090 + Ryzen 9 7950X3D.** This
settles the eye-capacity question from the inference side — a 1–11 M-param eye model is nowhere
near the limit on that machine, with or without DirectML. Every capacity decision from here should
be bounded by **training data and overfitting**, not latency. The work-PC timings in the P3-2
findings are a *relative ladder only*.

**New defects found and fixed in passing** (none shipped in behaviour):
- `BuildRoutine` iterated the raw nullable `Levels` instead of `EffectiveLevels` → every cue on the
  default ladder null-referenced (P3-1, caught by new tests).
- Eye tick leaked the 8-channel temporal stack and `ImageCollector`'s `Split()` Mats every tick;
  transformed frame disposed twice; six early-return paths released nothing (P3-3).
- Eye model hot-swap did not dispose the outgoing session (file lock on Windows) and did not clear
  the temporal queue (frames spliced across a camera/model change).

---

## P3-4 — EYE V2-A IMPLEMENTATION RECORD (2026-08-14, work PC)

Implementation commit: **`ad0846d`** (`eye: add selectable personal Eye V2 mapping calibration`).

### What shipped

- Added `IEyeStateMapper` as an optional final eye stage after stock `ProcessExpressions` and a
  manager hot-swap hook. `Mapper == null` takes the exact old six-value path with no V2 copy or
  allocation. A mapper exception clears V2 and returns the already-computed stock result on that
  same tick.
- Added a deterministic per-eye **2-D affine** gaze fit (gain, offset, and cross-axis terms) from
  center/left/right/up/down robust medians. Five points support this model well; a polynomial would
  add poorly constrained coefficients. Left and right eyes fit independently.
- Added the exact guided protocol: relax 5 s, slow blinks 8 s, squint 5 s, wide 5 s, then five gaze
  targets at 2 s each, with 750 ms preparation between steps. UI wall-clock is about **42.75 s**.
  The gaze steps display an actual moving target, not text alone.
- Added personal closed/neutral-envelope/squint/wide anchors, blink-duration estimation, calibrated
  openness, personal **Wide**, and temporal **Squint**. A partial closure must outlast 75% of the
  personal typical blink duration (clamped to 300–700 ms); reaching the closed region marks a
  blink and suppresses Squint until recovery.
- Added `EyeV2_Calibration.json`, schema version 1, capture metadata, atomic save, and one setting
  key (`EyeV2_Mode`). V2 never reads or writes `EyeHome_EyeModel`, `CalibrationParams`, or the
  Default model/calibration state.
- Added one-press **Recenter** (~2 s; offsets only) and a 2 s relaxed validity check. Gaze-center
  drift with plausible lid geometry recommends Recenter; material lid-anchor drift recommends a
  full calibration. Saved V2 mode restores before the first processing tick and runs the check
  when the Home page opens.
- Appended four outputs after the exact legacy six-value prefix:
  `LeftEyeWiden`, `LeftEyeSquint`, `RightEyeWiden`, `RightEyeSquint`. V2 outputs bypass Default's
  slider calibration because they are already canonical and must remain separate. Legacy six-value
  sends retain their existing remap exactly.
- Corrected the forked VRCFT module to consume the two explicit Squint OSC addresses and drive
  `EyeSquintLeft/Right`; Wide remains connected. The Release zip is produced at
  `src/VRCFaceTracking.Baballonia/bin/Release/net10.0/VRCFaceTracking.Baballonia.zip` (ignored
  build output; rebuild it after checkout).
- Added a normal-mode selector and three obvious actions (Calibrate, Recenter, Check headset
  position). Advanced users get raw→mapped gaze, raw/normalized lids, anchors, Wide, Squint, blink,
  and live fixation jitter. Processing-thread diagnostics are marshalled to Avalonia's UI thread.

### Safety, tests, and builds

- Focused Eye V2/pipeline/OSC suite: **45 passed / 0 failed / 0 skipped**.
- Full C# suite: **258 passed / 9 failed / 2 skipped**. This is exactly the documented hardware
  baseline plus 23 new passing tests; there is no new failure.
- Desktop Release build: **succeeded, 0 errors**. VRCFT module Release build and zip: **succeeded,
  0 errors**. Existing NuGet vulnerability/compatibility and nullability warnings remain.
- Synthetic mapper guard: 20,000 steady-state mappings after JIT warm-up must average **<0.20 ms
  per tick**; passed on the work PC. The mapper adds only its required ten-float output allocation;
  diagnostics are capped at 10 Hz.
- Tests cover five-point/per-eye affine recovery, anchor robustness, degenerate rejection, recenter
  invariants, validity classification, normal/wide/asymmetric states, short and slow blinks,
  sustained Squint and recovery, missing/stale/malformed calibration fallback, mode A/B/A swaps,
  separation from Default keys, exact null-pipeline behavior, mapper same-tick/failure fallback,
  and legacy versus V2 OSC mapping.
- No Python changed; the previously passing **102 checks** were therefore not rerun.

### Honest limitation

The shipped model exposes only one lid scalar per eye. Time separates ordinary blinks from a
sustained partial closure, but cannot fully distinguish a tensed squint from a long half-blink at
the same aperture. V2-A is expected to buy calibration UX, per-user ranges, Wide, recenter, and
better gaze mapping. Benchmark-grade Squint remains the principal reason for P3-5 geometry or the
learned split-head track — but only if the home gate authorizes it.

---

## PHASE 3 — NEXT EXACT TASK

### For the HOME PC (unchanged priority — the F0 checklist above still comes first)

Run the F0 checklist items 1–7 as written. Two additions from this session:

8. **Run a full guided pass** with the new routine picker ("Full pass", ~4 min), then train and
   read (a) the **cue-tracking correlation table** printed before training — the new cues are only
   as good as your ability to imitate them, and ≳0.6 mean correlation is the gate — and (b) the
   **coverage census**, which should jump from 1/45 to roughly 12–14/45 taught. Record both here.
9. **Re-run the eye capacity ladder on the 4090/7950X3D**, CPU and DirectML, to fix the absolute
   numbers (script: `scratchpad/eye_capacity_bench_ort.py`, or reconstruct from the P3-2 section).

### P3-4/P3-5 HOME-PC HARDWARE / VRCHAT VALIDATION — STILL REQUIRED

P3-5 was intentionally authorized before this validation. Do not interpret implementation as a
hardware verdict, and do not begin the large learned V2-L capacity ladder yet. Run this on the real
cameras and headset, using the Release Desktop build and the newly rebuilt/reinstalled VRCFT module
zip:

1. **Default safety A/B:** launch with Default selected; confirm tracking is unchanged. Switch
   Default → V2 → Default repeatedly and restart in each saved mode. Missing/invalid V2 calibration
   must fall back to Default, and Default calibration/model state must remain intact.
2. **Calibration UX/repeatability:** run full V2 calibration twice. Record wall-clock, actions,
   whether every prompt/target is visible and practical with the headset on, and the two sets of
   left/right gaze coefficients plus closed/neutral/squint/wide anchors. Note retries or confusing
   poses.
3. **Gaze ABAB versus current Baballonia and Paper Tracker:** at center/L/R/U/D, hold 5 s and
   record fixation jitter; perform center↔edge steps for response latency/overshoot; inspect corner
   error; then run 15 minutes for drift. Use the same sitting/headset fit and record OSC where
   possible.
4. **Reseat/recovery:** deliberately reseat the headset. Run Check headset position: a center shift
   with stable lids should recommend the ~2 s Recenter; a meaningful geometry/lid shift should
   recommend full calibration. Time recovery and verify Recenter changes offsets only.
5. **Blink/Squint:** annotate 20 natural and 10 deliberate blinks, including at least 5 slow
   blinks. Then perform 5 × 5 s deliberate Squint holds and conversational half-blinks. Record
   missed/false blinks, Squint detection/hold stability, release behavior, and half-blink false
   positives per eye.
6. **Wide:** perform 5 deliberate Wide holds; confirm per-eye VRCFT/VRChat Wide output and count
   false Wide activations during ordinary gaze shifts.
7. **OSC/module:** verify `LeftEyeWiden`, `LeftEyeSquint`, `RightEyeWiden`, and `RightEyeSquint`
   independently in VRCFT/VRChat; make sure the updated module zip, rather than a cached older
   module, is installed.
8. **Subjective mirror verdict:** finish with a same-sitting comparison in a VRChat mirror:
   **Default, Eye V2-A, Eye V2-B, Paper Tracker**, then repeat the most important transitions.
   Record which wins gaze precision/responsiveness, calibration effort, blink stability, Squint,
   and Wide. Use V2-A as the unchanged control for isolating V2-B's geometry contribution.

**Precommitted gate:** if V2-A is clearly better than current Baballonia on gaze and calibration UX
and is within sight of Paper, preserve it and authorize P3-5; Squint is expected to be its principal
gap. If V2-A cannot beat current Baballonia, stop eye work, reassess the raw/model assumptions, and
keep Paper Tracker. Record the measurements and verdict in this file before any next milestone.

### For the next agent session

P3-5 is authorized and implemented in the working tree under the explicit override above. The next
required work is the same-sitting Default / V2-A / V2-B / Paper hardware benchmark. The large
learned V2-L capacity ladder remains unauthorized unless infrastructure genuinely shared with
P3-5 requires it.

---

Learn a **user-specific correction model** for the face pipeline: it sees the camera image plus the
stock model's 45 raw outputs and emits corrected outputs. Trained locally (PyTorch → ONNX), loaded
by Baballonia via ONNX Runtime, with instant fallback to stock when disabled/missing/invalid.
Eye tracking was out of scope for Phases 0–2 (Paper Tracker handles it) — **superseded by the
Phase 3 Eye V2 plan above**, which adds eyes as a separate, selectable, fail-safe subsystem while
the default eye path stays untouched. Slider calibration is explicitly *not* the solution.

**Avatar-guided calibration** (plan §14): during guided capture the app drives the user's own VRChat
avatar to independently generated target expressions, the user imitates the avatar, and the
**commanded vector is the training label**. Non-circular by construction — targets come from the cue
generator, never from inference. The avatar image is never model input or supervision in MVP.

---

## ENVIRONMENT — read this before running anything

These cost real time to rediscover.

1. **The `dotnet` on PATH has no SDK.** `C:\Program Files\dotnet\dotnet.exe` is runtime-only
   (`No SDKs were found`). The working SDK 10.0.302 is user-local:
   ```
   D:\Users\acourtney\AppData\Local\Microsoft\dotnet\dotnet.exe
   ```
   Use that absolute path for every build/test command.

2. **Submodules must be initialized** or `Baballonia` will not build (`HyperText.Avalonia` supplies
   the `Hyperlink` control used by AboutPageView/OnboardingView):
   ```
   git submodule update --init --recursive
   ```
   Done already in this working copy.

3. **Build/test the test project directly** — it does not reference the VRCFaceTracking submodule:
   ```
   <sdk>\dotnet.exe test src/Baballonia.Tests/Baballonia.Tests.csproj
   ```

4. **Datasets are written outside the repo** to `%APPDATA%\ProjectBabble\PersonalDataset\`.
   Never copy them into the repo; `.gitignore` has a safety net but the real protection is location.

---

## Completed Work (P0)

### P0-1 — Expression schema lock  (commit `38a0680`)
`PersonalizationSchema` mirrors the positional 45-expression order that
`ParameterSenderService.FaceExpressionMap` defines, and exposes a canonical SHA-256 over it
(names joined by `\n`, UTF-8, no trailing newline, lowercase hex). Python must reproduce that exact
recipe. `PersonalizationSchemaTests` constructs the *real* `ParameterSenderService` so drift is a
loud test failure rather than silently repointing every trained model.
`.gitignore` gained personal-data protections (datasets, runs, venvs, personal ONNX, corrections).

### P0-2 — Face pipeline hooks  (commit `49ea7f1`)
`FacePipelineEvents.NewRawExpressionsEvent(Mat, float[], long)` publishes the raw pre-filter vector
together with the exact 224×224 Mat that produced it. `FacePipelineEvents.NewCorrectedExpressionsEvent`
carries stock+corrected for the future debug panel. A nullable `IExpressionCorrector` runs
**between inference and the One Euro filter**; `FacePipelineManager.SetCorrector(null)` clears it
immediately. With no corrector installed, output is unchanged from stock.

### P0-3 — Avatar-guided calibration override  (commit `d20df3b`)
`IExpressionOverrideSource` / `ExpressionOverrideService` + a hook in `ParameterSenderService`.
The existing ~10 ms sender loop **pulls** `SampleTarget()` and enqueues the commanded vector, so the
avatar animates on the transmitting clock. Commanded targets bypass the calibration remap (clamped
only). Face channel only — eye output keeps flowing. Three stop-safety layers: keep-alive deadman,
per-phase overrun cap → neutral, neutral flush on deactivate.

### P0-4 — Dataset recorder  (commit `e6340d2`)
`DatasetRecorderService` writes one directory per session under
`%APPDATA%\ProjectBabble\PersonalDataset\<yyyyMMdd_HHmmss>_<type>\`:
```
session.json          schema names + sha256, geometry, camera settings, frame count, measured fps
frames/000000.jpg     224x224 Gray8, JPEG q95
labels.jsonl          {"i":N,"t":ticks,"stock":[45 floats],"cue":{...}|null}
```
Event handler only clones + enqueues (bounded channel, 256, drop-oldest); JPEG encode and disk I/O
run on a background writer, so a slow disk drops frames instead of stalling tracking. Duplicate
frames (the ~100 Hz tick re-serving a slower camera) are rejected by a sparse FNV checksum, and
capture is capped at 30 fps.

### P0-5 — Minimal capture page  (commit `ebfae7f`)
`PersonalizationViewModel` / `PersonalizationView`: live post-transform preview, Neutral/Speech
session recording with notes, status line, open-dataset-folder. Registered via the usual three
touchpoints. Guided sessions are intentionally not offered yet.

### Prerequisite — build repair  (commit `d9bdb61`)
The test project could not restore or compile before this work (see Decisions).

### P1-1 — Python training package  (commit `76e33bc`)
`training/babble_personal/`: `schema.py` (45 names + the SHA-256 shared with C#), `dataset.py`
(session discovery, **session-level** splitting), `labels.py` (weighted targets, cue-lag
cross-correlation), `models.py` (`OutputMlpAdapter` ~29k, `ImageResidualAdapter` ~44k, masked Huber +
residual shrinkage), `train.py`, `evaluate.py` (per-expression MAE, neutral false activation,
cross-talk), `export.py` (ONNX + metadata + torch/ORT parity assert).

Both adapters initialise to exact identity and predict a residual, so an untrained or unsupervised
adapter is a no-op.

`training/tests/test_pipeline_smoke.py` runs the real code path on synthetic sessions with
deliberately injected defects and asserts they get fixed.

### P1-2 — Personal model runtime  (commit `cfa00a6`)
`PersonalModelCorrector` (own CPU session, borrows the stock input tensor, `lerp(stock, personal,
blend)`, self-disables on failure) and `PersonalModelManager` (settings, background load,
validation, hot reload, graceful fallback on every rejection path).

Page gained: enable toggle, strength slider, reload button, and a live stock/personal/delta table
sorted by largest change (fed from the corrected event, drained at 4 Hz).

Test fixtures in `src/Baballonia.Tests/Assets/PersonalModels/` are real ONNX files exported by the
trainer — valid, schema-mismatched, and metadata-stripped — so the load/validate/infer path is
tested for real rather than mocked. Regenerate them with the snippet in Decisions #9.

### P1-3 — Guided workflow / UX layer  (commit `563d674`)
Thin orchestration over the existing tooling; **no ML changes**. The CLI remains fully usable.

* `PersonalizationEnvironment` — pure detection: locates `training/` by walking up from the exe,
  finds the venv and host Python, counts sessions by type, and turns all of it into friendly
  `SetupItem`s. Two sessions per type is the recommended threshold because validation holds out a
  whole session.
* `PersonalTrainingService` — runs train → export → install → reload as child processes, streaming
  both stdout and stderr into a detail log. Also `SetUpTrainingToolsAsync` (create venv, install
  pinned requirements). Failures are translated into actionable messages with a suggested remedy;
  raw process output stays in the log.
* `train.py` additions: `[stage] …` markers (so progress text is not inferred from log shape) and
  **`summary.json`**, the machine-readable outcome the app reads. Its `verdict` is deliberately
  conservative — `"unclear"` whenever there was no held-out session, since numbers describing
  training data cannot support a claim.
* Page restructured to Setup / Recordings / Train / Compare, with the strength slider and the
  45-row delta table moved under **Advanced** rather than removed.

**Install unloads the current model first.** An active `InferenceSession` holds the file open on
Windows, so retraining over a live model would fail with a sharing violation. A `.previous.onnx`
rollback copy is kept.

---

## Files Changed

**New (all personalization is isolated here):**
```
src/Baballonia/Services/Personalization/PersonalizationSchema.cs
src/Baballonia/Services/Personalization/IExpressionCorrector.cs
src/Baballonia/Services/Personalization/IExpressionOverrideSource.cs
src/Baballonia/Services/Personalization/ExpressionOverrideService.cs
src/Baballonia/Services/Personalization/CuePhase.cs
src/Baballonia/Services/Personalization/DatasetSession.cs        (metadata/label records + paths)
src/Baballonia/Services/Personalization/DatasetRecorderService.cs
src/Baballonia/ViewModels/SplitViewPane/PersonalizationViewModel.cs
src/Baballonia/Views/PersonalizationView.axaml(.cs)
src/Baballonia.Tests/PersonalizationSchemaTests.cs
src/Baballonia.Tests/Services/Inference/FaceProcessingPipelineCorrectorTest.cs
src/Baballonia.Tests/Services/Personalization/ExpressionOverrideServiceTest.cs
src/Baballonia.Tests/Services/Personalization/ParameterSenderOverrideTest.cs
src/Baballonia.Tests/Services/Personalization/DatasetRecorderServiceTest.cs
```

**Modified upstream files (kept deliberately small for rebasing):**
```
src/Baballonia/Services/events/PipelineEvents.cs          +2 event records
src/Baballonia/Services/Inference/FaceProcessingPipeline.cs  corrector stage + raw event
src/Baballonia/Services/Inference/FacePipelineManager.cs   +SetCorrector
src/Baballonia/Services/ParameterSenderService.cs          override hook (~15 lines)
src/Baballonia/App.axaml.cs                                DI + page registration
src/Baballonia/ViewLocator.cs                              page registration
src/Baballonia/ViewModels/MainViewModel.cs                 nav entry
.gitignore                                                 personal-data rules
src/Baballonia.Tests/Baballonia.Tests.csproj               version alignment (build repair)
src/Baballonia.Tests/IpCameraCaptureFactoryTest.cs         stale using (build repair)
```

---

## Tests Run

```
<sdk>\dotnet.exe test src/Baballonia.Tests/Baballonia.Tests.csproj
  Total: 106  Passed: 97   Failed: 9      (was 70/61/9 before this work)

<sdk>\dotnet.exe test ... --filter "FullyQualifiedName~Personalization|FullyQualifiedName~FaceProcessingPipelineCorrector"
  Total: 81   Passed: 81   Failed: 0

<sdk>\dotnet.exe build src/Baballonia.Desktop/Baballonia.Desktop.csproj
  Build succeeded. 0 Errors, 132 Warnings (all pre-existing).

%LOCALAPPDATA%\babble-train-venv\Scripts\python training\tests\test_pipeline_smoke.py
  8/8 passed
```

`PersonalTrainingOrchestrationTest.FullPipeline_TrainsExportsInstallsAndLoads` runs the **real**
Python chain on synthetic recordings (~10 s) and asserts the model is trained, exported, installed
and live on the pipeline. It skips (Inconclusive) when PyTorch is absent or when real recordings
exist, and restores the machine's prior state afterwards.

The Python smoke test is the meaningful one for the ML side: it fabricates sessions containing known
defects and asserts the trained adapter removes them.

```
  false JawOpen while neutral      stock 0.299  ->  personal 0.007
  underestimated MouthSmileLeft    MAE   0.126  ->  personal 0.032
  torch/ORT parity                 max abs diff 2.98e-07
  both adapters export             image + stock -> personal
```

**The 9 failures are all pre-existing and unrelated to personalization** — firmware/hardware and
external-binary tests:
`AllCommandsIntegrationTest`, `BoardIntegrationTest`, `FindAndConnectWifiFail`,
`FindAndConnectWifiSuccess`, `GetModeTest`, `TestBoard`, `TestSendCommand`, `TestSendGeneric`
(all need a real ESP32 over serial; the two `TestSend*` also hit an upstream JSON bug in
`FirmwareSessionV1.WaitForHeartbeat`, which cannot parse `{"heartbeat":{}}`), plus
`TrainerServiceTest.Test` (needs `BabbleTrainer.exe`, obtained via `download_dependencies.ps1`,
not present).
Note there was **no green baseline to regress from**: the test project did not compile at all before
commit `d9bdb61`.

---

**BOM fix (this session):**

```
%USERPROFILE%\.dotnet\dotnet.exe test src/Baballonia.Tests/Baballonia.Tests.csproj \
    --filter "FullyQualifiedName~Personalization"
  -> 72 passed, 0 failed, 2 skipped

<venv>\python.exe training/tests/test_json_encoding.py   -> 8/8 pass (incl. the real 4-session corpus)
<venv>\python.exe training/tests/test_pipeline_smoke.py  -> 8/8 pass (no regression)

full suite -> 97 passed / 9 failed; the 9 are the same pre-existing hardware/fixture failures
              (serial board, firmware JSON models, missing test.bin), re-confirmed by stashing the
              fix and re-running them on a clean tree.
```

Note: `pytest` is not installed in the training venv; the Python tests carry a standalone `main()`
runner and were executed directly.

---

## Benchmarks

Measured on this machine, CPU execution provider, after warmup, via
`PersonalModelCorrectorTest` (500 iterations for A, 200 for B). Reproduce with:
`dotnet test --filter "FullyQualifiedName~Latency" --logger "console;verbosity=detailed"`.

| Adapter | params | p50 | p95 |
|---|---|---|---|
| `output_mlp_v1` (baseline A) | 29k | 0.033 ms | 0.065 ms |
| `image_residual_v1` (model B) | 44k | 0.211 ms | 0.335 ms |

Both sit comfortably inside the 10 ms processing tick, and both matched the plan's estimates
(≈0.05 ms / 0.3–0.8 ms). **Still unmeasured:** whole-tick p50/p95 with a corrector installed while a
real camera is running, and the same with `AppSettings_UseGPU` on. Those need hardware.

One further measured fact: the sender loop's real cadence is **~64 Hz, not 100 Hz** —
`Task.Delay(10)` quantizes to the Windows ~15.6 ms timer. Harmless (VRChat consumes values
stepwise) but do not design anything assuming true 100 Hz.

---

## Decisions / Deviations

**1. Test project was broken before any of this work.**
- Plan expected: `dotnet test` runnable as the P0 gate.
- Actual: NuGet restore failed (NU1605, an error by default on this SDK) because the test project
  pinned `Microsoft.Extensions.Logging{,.Console,.Debug}` 10.0.2 and `OpenCvSharp4.runtime.win`
  4.11.0.20250507 *below* what its own project references require; and
  `IpCameraCaptureFactoryTest` imported `Baballonia.Android.Captures` for a class that now lives in
  `Baballonia.IPCameraCapture`.
- Decision: align the versions, fix the import (commit `d9bdb61`).
- Reason: minimal changes required solely to make the repository testable; no behavior change.

**2. Corrector placement confirmed as planned** — between inference and the One Euro filter, so the
corrector sees the same raw distribution training recorded, and the filter smooths the signal that
actually reaches VRChat.

**3. `transformed.Dispose()` moved after inference** in `FaceProcessingPipeline`. The tensor copy has
already happened, so holding the Mat marginally longer is harmless, and recorders need it alive
during the raw event.

**4. Cue-phase sampling is pull-based**, per the plan's refinement: the cue engine publishes
immutable `CuePhase` snapshots; the sender loop interpolates at call time. Keeps sent values and
recorded labels derived from one source.

**5. UI strings are literal, not RESX.**
- Plan expected: nothing specific.
- Actual: `Assets/Resources.resx` is Crowdin-managed upstream.
- Decision: literal strings in the new page only.
- Reason: adding keys would conflict on every upstream rebase. Revisit if upstreamed.

**6. Dataset root is APPDATA, not Documents** (as planned) — this machine's Documents is
OneDrive-synced and thousands of small JPEGs would cause sync churn and file locks.

**8. Baseline A carries a zero-weighted image tap.**
- Plan expected: both adapters export the same graph signature.
- Actual: `torch.onnx.export` prunes inputs nothing consumes, so the output-only model lost its
  `image` input and would have forced the C# runtime to branch per adapter type.
- Decision: `OutputMlpAdapter` multiplies one pixel by a permanently-zero buffer.
- Reason: keeps the input live at O(1) cost, contributes exactly 0.0, and preserves one runtime code
  path. A test pins that both models export `image + stock -> personal`.

**9. C# tests use real ONNX fixtures**, checked in at
`src/Baballonia.Tests/Assets/PersonalModels/` (~400 KB total, random weights - not personal data).
Mocks would not have caught schema/shape/metadata handling, which is the whole point of that code.
Regenerate after a schema change:

```python
# with the training venv, from the repo root
import torch, tempfile, onnx, sys
from pathlib import Path
sys.path.insert(0, "training")
from babble_personal import models, schema, export
out = Path("src/Baballonia.Tests/Assets/PersonalModels")
m = models.build_model("a")
with torch.no_grad():
    m.net[-1].bias[schema.INDEX_OF["JawOpen"]] = -1.0        # fixtures assert these effects
    m.net[-1].bias[schema.INDEX_OF["MouthSmileLeft"]] = 1.0
ck = Path(tempfile.mkdtemp()) / "m.pt"
torch.save({"state_dict": m.state_dict(), "adapter_type": m.adapter_type}, ck)
export.export(ck, out / "validAdapter.onnx", parity_samples=4)
# then: copy validAdapter.onnx twice, setting schema hash to "0"*64 in one
# (schemaMismatchAdapter.onnx) and clearing metadata_props in the other (noMetadataAdapter.onnx);
# and export a build_model("b") the same way as imageAdapter.onnx
```

**10. Fixed a publish bug that would have shipped a camera-less app.**
- Plan expected: nothing about publishing.
- Actual: `CopyModulesToFolderPublish` hardcoded `$(OutputPath)\publish\`, so
  `dotnet publish -o <dir>` left the capture backends in the publish root. The module loader only
  scans `Modules\`, so the published app started with **no camera backends at all**.
- Decision: use `$(PublishDir)`, which is correct both with and without `-o`.
- Reason: found while producing a build; the default-path CI publish was unaffected, which is why it
  went unnoticed upstream.

Also added `CopyPersonalizationTrainingScripts`, which copies `training/` into the publish output so
"Train My Face Model" works from an installed copy (the scripts are located by walking up from the
executable). A few small text files; PyTorch is still installed on demand into the user's own venv.

**7. Upstream bugs observed and deliberately NOT fixed** (out of scope, documented so nobody
"rediscovers" them): One Euro filter is not cleared when disabled (`FacePipelineManager.LoadFilter`
returns early); `FaceProcessingPipeline` never disposes the source `frame` Mat; `RunUpdate` shadows
rather than overrides the base method; `OscQueryServiceWrapper` is dead code that would repoint
`OSCOutPort` to 9000 and break the VRCFT path if revived.

---

## BUGFIX — UTF-8 BOM in recorded datasets (first real training run)

The first real training attempt died immediately in `training/babble_personal/dataset.py`:

```
json.decoder.JSONDecodeError: Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1
```

### Root cause

`DatasetRecorderService` opened its labels writer as

```csharp
new StreamWriter(labelsPath, append: true, Encoding.UTF8)
```

`Encoding.UTF8` is `UTF8Encoding(encoderShouldEmitUTF8Identifier: true)` — **its preamble is the
BOM**. StreamWriter emits the preamble when it opens a file that is empty, so every `labels.jsonl`
begins with `EF BB BF`. Python's `json.loads` rejects a BOM outright, and the first label line is
the first thing the loader parses, so training could never start.

Only `labels.jsonl` was affected. `session.json` goes through `File.WriteAllTextAsync`, whose
default UTF-8 encoding is BOM-less. Verified on disk across all four recordings:

```
session.json  -> 7b 0d 0a   ('{' CR LF)     no BOM
labels.jsonl  -> ef bb bf                   BOM
```

Exactly one BOM per file, all at byte 0 — `grep -c` over each `labels.jsonl` found one occurrence.

### Could a BOM ever land mid-file?

Not from the current code: the writer is opened once per session, on a fresh directory. Appending
to a *non-empty* file would not add one either, because StreamWriter marks the preamble written when
it can seek and finds `Position > 0`. But that is implicit framework behavior, and a reopen of a
still-empty session file would have produced a mid-stream BOM. The fix removes the hazard by
construction rather than relying on that behavior: the encoding's preamble is now empty, so no code
path can emit a BOM anywhere. `Utf8NoBom_NeverEmitsBom_AcrossReopenAndAppend` pins this down across
three opens (missing file, empty file, non-empty file).

### Fix

**C# (all future recordings, no BOM anywhere)**
- `PersonalizationPaths.Utf8NoBom` — a shared `UTF8Encoding(encoderShouldEmitUTF8Identifier: false)`
  for every personalization JSON file, documented with the reason.
- `DatasetRecorderService` uses it for the `labels.jsonl` StreamWriter, and passes it explicitly to
  the `session.json` write too (already BOM-less by default, now stated rather than assumed).

**Python (backward compatible with recordings already made)**
- `dataset.JSON_ENCODING = "utf-8-sig"` and `dataset.read_json_text()`, used for both `session.json`
  and `labels.jsonl`. `utf-8-sig` consumes a *leading* BOM and is otherwise identical to `utf-8`.
- Deliberately narrow: a BOM anywhere other than byte 0 still raises, so this cannot degrade into a
  general "strip whatever looks malformed" mask. `test_mid_file_bom_is_still_rejected` locks that in.

**Existing recordings were not modified or deleted** — they are read-only in every test and were
only ever inspected.

### Tests

`training/tests/test_json_encoding.py` (new, 8 tests, all pass):

| test | covers |
| --- | --- |
| `test_plain_session_json_loads` | normal UTF-8 session.json |
| `test_bom_prefixed_session_json_loads` | BOM-prefixed session.json |
| `test_plain_labels_jsonl_loads` | normal UTF-8 labels.jsonl |
| `test_bom_prefixed_labels_jsonl_loads` | BOM-prefixed labels.jsonl (the actual failure) |
| `test_both_files_bom_prefixed_load` | both files BOM-prefixed |
| `test_discovery_over_mixed_bom_and_plain_sessions` | discovery across old + new recorder output |
| `test_mid_file_bom_is_still_rejected` | corruption is not masked |
| `test_existing_recordings_load` | **the real corpus**: 4 sessions, 6,375 frames, 4 BOM-prefixed |

BOM fixtures are written as raw bytes so the test asserts on `EF BB BF` itself, not on whatever an
encoder chooses to do.

`DatasetRecorderServiceTest` (2 new tests):
- `RecordedJsonFiles_ContainNoByteOrderMark` — records a real session and scans both output files
  for a BOM at *any* offset.
- `Utf8NoBom_NeverEmitsBom_AcrossReopenAndAppend` — asserts the encoding's preamble is empty and
  that three reopen/append cycles produce no BOM and strictly parseable JSON.

### Build

Rebuilt because the C# recorder changed:

```
%USERPROFILE%\.dotnet\dotnet.exe publish src/Baballonia.Desktop/Baballonia.Desktop.csproj ^
    -c Release -r win-x64 --self-contained true -o bin\Baballonia-bomfix
```

**New build: `<repo>\bin\Baballonia-bomfix\Baballonia.Desktop.exe`** — inside the repository,
under the already-gitignored `bin/`, so nothing existing was overwritten and nothing is committed.

Verified: `Modules\` has all 4 capture DLLs, `training\babble_personal\` is present, and the
published `dataset.py` carries the `utf-8-sig` fix.

The previous install at `%LOCALAPPDATA%\Baballonia` was left untouched and still contains the
**unfixed** trainer, and it still owns the Desktop shortcut. Launch the new path or training will
fail the same way again.

An identical build also sits at `%LOCALAPPDATA%\Baballonia-bomfix` from earlier in the same
session (a new folder, nothing overwritten); harmless, and safe to delete.

### No ML behavior changed

This was purely an encoding/serialization fix. No labels, weights, losses, model code, thresholds or
training hyperparameters were touched. P2 was not started.

---

## FIRST REAL TRAINING RUN — result and analysis

Run `20260813_185338_output_mlp_v1` (model A, 28,205 params, 30 epochs, best epoch 28).
Train: `232210_neutral` + `232525_speech`. Held out: `232328_neutral` + `232647_speech`.

**Verdict: `better`. 45/45 expressions improved, 0 regressed.** Mean MAE 0.0962 → 0.0068.
Neutral false-activation 17.9% → 0.31%.

### What the headline numbers actually measure

`evaluate.per_expression_mae` scores only cells with `weight >= 0.9`. In this corpus that is
**exclusively neutral-session frames, whose target is zero on all 45 dims** — hence the identical
1655-frame count on every row of the table. The speech pseudo-labels (weight 0.3) are excluded by
that threshold, and they were the *only* source of non-zero targets.

Counted directly over the built labels:

```
total supervised cells                       79.8%
expressions with any high-confidence label   45/45
expressions with any NON-ZERO target         27/45
high-confidence cells whose target is 0      142,335 of 142,335   <-- all of them
tongue expressions with any non-zero target  0/12
```

So both the training objective and the report are, in practice, **"how close to zero is the output
while the face is at rest"**. That is a real and worthwhile thing to fix, but at 0.31% false
activation it is also nearly maxed out — there is little headroom left on the only axis measured.

Note for reading the results screen: the "worst offenders" column shows **firing rates**, not
values. `JawOpen 1.000 -> 0.008` means it fired above 0.15 on 100% of resting frames, not that its
value was 1.0. Actual resting values were JawOpen 0.767, MouthClose 0.695, JawForward 0.641 — a
large constant per-user bias, which is exactly what model A is good at removing.

### The model did NOT flatten the face (checked directly)

Running the exported ONNX over held-out frames:

```
                     stock    personal
neutral, fires >0.15  0.179     0.003
speech,  fires >0.15  0.169     0.164     <-- expression activity preserved

rest -> speech separation      stock    personal
  JawOpen                      +0.053    +0.772
  MouthClose                   -0.102    +0.555
  JawForward                   -0.030    +0.565
speech dynamic range (std)     stock    personal
  JawOpen                       0.072     0.229   (2.65x the p05-p95 range)
```

Stock jaw motion only moved the number ~0.05 on top of a 0.767 offset; the adapter removed the
offset *and* restored the range. This is a genuine win, not suppression.

### How it achieves that — and the weak spot it implies

It is a **context gate, not a calibration**. During speech `corr(stock, personal)` on JawOpen is
only **+0.27**, and neutral frames whose stock JawOpen is highest (mean 0.832) sit right on top of
the speech distribution (p50 0.828) yet get very different outputs. Model A cannot see the face, so
it decides "is this an expression?" from the *other* channels — CheekPuff, MouthDimple, MouthSmile
and NoseSneer separate rest from speech with d' ≈ 1.7–2.0.

**Predicted failure mode, not yet tested:** a *silent* expression hold — jaw open with no
speech-correlated cheek/dimple signature — has no context to trigger the gate and may be suppressed.
In the same probe, the personal JawOpen on those overlapping neutral frames was bimodal
(mean 0.146, max 1.000), which is the signature of an unstable decision boundary. Ten seconds of
holding a silent open mouth in front of the Stock/Personal switch would confirm or refute it.

### Why more of the same data will not fix this

Nothing in the corpus says what a *correct* non-zero value looks like. Neutral says "all zeros";
speech copies the stock model's own opinion at weight 0.3. Recording more neutral and speech
sessions improves gate robustness and rest stability, but can never teach magnitude — no label
anywhere states "this is a 0.8 smile". That is precisely what guided capture (P2) produces, and it
is the gating dependency for a meaningful A-vs-B comparison.

---

## MODEL SELECTION — A or B, exposed in the UI

The trainer always accepted `--model b`; only the UI hardcoded `"a"`. The Train card now has a
**Model** dropdown:

- **A — expressions only (recommended)** `output_mlp_v1`, 28,205 params. Sees the stock 45 values
  and nothing else.
- **B — expressions + camera image (experimental)** `image_residual_v1`, ~48k params. Also consumes
  the 224x224 frame, average-pooled to 112 inside the graph, so C# feeds the tensor it already has.

`PersonalizationViewModel.SelectedModelIndex` maps to the `--model` flag; the description text under
the dropdown changes with the selection. Default is unchanged (A), so the existing flow is identical
for anyone who does not touch it. No ML code, loss, label or hyperparameter was modified.

**Expectation-setting for B on the current corpus:** B is scored by the same rest-only metric that A
already nearly saturates, so a large headline improvement is unlikely. Where B *can* help today is
the context-gate weakness above — it can see whether the mouth is actually open instead of inferring
it from other channels. It also loads every frame into RAM (~1.3 GB for 6,375 frames, more at peak)
and trains considerably slower. If it does not clearly win, ship A.

---

## BUILD CONVENTION

Every build goes to a **new** versioned folder under the repo's gitignored `bin/`:
`bin\Baballonia-v2`, `bin\Baballonia-v3`, ... Never overwrite a previous build — the last known-good
one must stay available to fall back to.

Current: **`bin\Baballonia-v10\Baballonia.Desktop.exe`** — v9 with the former face-model
"Compare" section renamed and clarified as **Face model selection**. The current `Running now`
status is prominent, the dropdown is labelled `Model to use`, and comparison terminology is
reserved for the Advanced live expression table. Built on the work PC 2026-08-14,
Release/win-x64/self-contained, 453,108,091 bytes across 395 files. Verified: executable,
`faceModel.onnx`, training scripts, and all 4 capture DLLs present. The last focused suite remains
**83 passed / 0 failed / 0 skipped**; the v10 XAML Release publish succeeded. Zip SHA-256:
`F0CCA2A31F8494CCA1AD2DBE4E01C83F001DA92871F20FC41880EA2F168CD69E`.
Previous: `bin\Baballonia-v9` (guarded deletion of saved quick corrections).
Previous: `bin\Baballonia-v8` (quick-correction toggle, A/B/C selection, and P3-5 V2-B Geometry Hybrid).
Previous: `bin\Baballonia-v7` (quick-correction and early Model C UI work before P3-5).
Previous: `bin\Baballonia-v4` (model B regularizers + resting-jitter reporting).
Previous: `bin\Baballonia-v3` (BOM fix, model A/B selector, active-model display, test no longer deletes the installed model).
Previous: `bin\Baballonia-v2` (BOM fix + selector).
Previous: `bin\Baballonia-bomfix` (BOM fix only), and the pre-fix install at `%LOCALAPPDATA%\Baballonia`.

**Two build gotchas found on the work PC**, both environmental rather than code:

1. **Submodules must be initialized before a Release publish.** A Debug build succeeds without them,
   but Avalonia compiles XAML during Release and fails with `AVLN2000: Unable to resolve ... Url` on
   `OnboardingView.axaml` — the `Hyperlink` control lives in `HyperText.Avalonia`. Fix:
   `git submodule update --init --recursive`.
2. **`dotnet publish -o <path>` breaks on this repo's path.** The absolute path contains a comma
   ("Monitor Technologies, LLC") and MSBuild parses it as a property separator
   (`MSB1006: Property is not valid`). Publish from inside `src/Baballonia.Desktop` with no `-o`,
   then copy `bin/Release/net10.0/win-x64/publish` to the versioned folder.

---

## BUGFIX — the test suite deleted the installed personal model

`PersonalTrainingOrchestrationTest.Initialize()` snapshotted the training-runs directory but **never
assigned `_modelBackup`**. `Cleanup()` branches on that field:

```csharp
if (_modelBackup != null)  { restore it }
else if (File.Exists(model)) { File.Delete(model); }   // <-- always taken
```

So every run of that test class deleted whatever was at
`%APPDATA%\ProjectBabble\Models\personalFaceModel.onnx`, plus the `.previous.onnx` rollback copy.
`[TestCleanup]` runs after `Assert.Inconclusive` too, so the two tests skipping on this machine did
not prevent it. Observed live: the app logged

```
[19:29:17] Personal model active: image_residual_v1 ...
[19:37:15] Warning: Personal model not loaded: No personal model at ...personalFaceModel.onnx
[19:38:44] Personal model active: output_mlp_v1 ...
```

— a trained model B disappeared between runs of the suite, and the next Train produced model A,
which is why "which model am I on?" had no good answer.

**Fix:** `Initialize()` now copies the installed model (and its `.previous.onnx`) to a temp backup
and records the paths; `Cleanup()` restores them. The delete branch now only runs when nothing was
installed beforehand — i.e. when the file really is this test's own residue.

**Regression test:** `Cleanup_RestoresAnAlreadyInstalledModel_RatherThanDeletingIt` plants a
sentinel file at the real install path, runs an Initialize/Cleanup cycle over it, and asserts the
bytes come back unchanged. It skips when a real model is installed, so it never risks one.

**Verified on the real machine:** hashed the installed model, ran the full Personalization filter,
re-hashed — `9FEBDCFB...F98BA`, 115,991 bytes, mtime unchanged. It survives the suite now.

---

## MODEL A vs MODEL B — first real comparison

Five runs exist under `%APPDATA%\ProjectBabble\PersonalTraining\`. Both architectures trained
successfully from the UI dropdown, both scored `better` on held-out sessions:

| run | model | mean personal MAE | neutral false activation |
| --- | --- | --- | --- |
| `185338_output_mlp_v1` | A | 0.00680 | 0.54% |
| `190626_output_mlp_v1` | A | 0.00680 | 0.54% |
| `190701_image_residual_v1` | **B** | **0.00246** | **0.39%** |
| `192821_image_residual_v1` | **B** | **0.00246** | **0.39%** |
| `193840_output_mlp_v1` | A | — | — |

B is ~2.8x lower error on the scored cells and lower false activation. Read that with the caveat
from the analysis above: **both numbers still only describe the resting face**, which is the one
axis with no headroom left. B winning here is encouraging but is not yet evidence it feels better in
use — that needs either the silent-hold check or guided data.

Model B trained in roughly 70 s on this machine (19:07:01 -> 19:07:43 train, export by 19:07:45),
so trying it costs very little.

---

## MODEL B IMPROVEMENTS — resting wiggle and the "slightly open mouth"

User report after living with model B: it works well, but at rest the jaw sometimes wiggles and the
mouth reads as very slightly open - intermittently, and suspected to involve room lighting or a
shifted camera.

### Diagnosis (held-out neutral session `20260814_002926_neutral`, 1555 frames)

The complaint was reproducible and had two independent causes:

```
JawOpen mean at rest                0.117      (should be ~0)
JawOpen frames above 0.15           23.1%
lag-1 autocorrelation               0.805      -> not white noise
lag-30 autocorrelation              0.520      -> slow drift, over a second long
frame-to-frame |delta|, stock       0.0130
frame-to-frame |delta|, personal    0.0228     -> model B amplified stock jitter 1.76x
corr(frame brightness, JawOpen)    -0.558      (stock: -0.356)
```

1. **Illumination leakage.** Model B is the only model that sees pixels, and with one lighting
   condition in the corpus it learned to read brightness as expression - *worse* than stock, which
   is the "my mouth looks slightly open sometimes" symptom. The user's own guess was right.
2. **Jitter amplification.** It made the correction itself twitchy, which is the "wiggle".

### Fix — two constraints, both on the residual, never on the output

That distinction is what lets them be pushed hard without making the face feel laggy: fast movement
keeps coming from the stock model, while the *correction* is required to be smooth and
illumination-invariant.

* `augment.photometric_jitter` - brightness / contrast / gamma / sensor-noise perturbation.
  Deliberately no geometric component: the recorded frame is the exact post-transform tensor the
  stock model consumed, so moving it would break correspondence with the stock vector it is paired
  with.
* `models.consistency_penalty` - the residual must not move when only appearance changed. The
  augmented view is fed **with the same stock vector**, so the model cannot satisfy it by leaning on
  stock; it has to make the image branch itself invariant.
* `models.temporal_penalty` - the residual must not jump between consecutive frames of the same
  session. Session boundaries are masked (`_previous_frame_index`), since pairing across a cut would
  ask the model to make an edit look continuous.

`train.py` now batches **row indices** rather than tensors, which is what lets a shuffled batch look
up each row's predecessor. Defaults: `--consistency 0.5` (model B only; model A ignores images so it
is forced to 0) and `--temporal 0.5`. Both have escape hatches at 0.

### Result — measured on the held-out session, versus the model that was shipped

```
                            neither    temporal    consistency        both
JawOpen mean at rest         0.1173      0.0988         0.0637      0.0261
JawOpen frames >0.15         23.09%      17.11%          0.32%       0.19%
JawOpen jitter               0.0228      0.0171         0.0219      0.0141
all-45 mean activation       0.0137      0.0090         0.0085      0.0046
all-45 jitter                0.0048      0.0036         0.0040      0.0030
re-light sensitivity         0.0118      0.0087         0.0028      0.0028
```

The ablation attributes each symptom to its own term, which is the useful part:

* **consistency** kills the false open mouth - 23.1% -> 0.32% of frames above threshold, and 4.2x
  less output movement when the frame is artificially darkened by 6%.
* **temporal** is what damps the wiggle - jitter 0.0228 -> 0.0171.
* Together: **-99.2%** frames above threshold, **-38%** jitter, **-78%** mean resting JawOpen.

One honest caveat: the brightness *correlation* got slightly worse (-0.558 -> -0.633) even as the
amplitude fell sharply. Correlation is scale-invariant, so a small clean residual that still tracks
brightness scores high on it. The slope (-15.5 -> -10.7) and the direct re-light probe are the
numbers that matter, and both improved.

### The metric that was missing

`evaluate.resting_jitter` is now part of the neutral report and `summary.json`, and the app shows
"Twitchiness while resting: X before, Y after". A model can hold a perfect false-activation rate
while shivering just under the threshold - that is precisely what happened here, and no existing
number would have shown it.

### Tests

`training/tests/test_regularizers.py`, 9 tests: augmentation stays in range / only re-lights (an
ascending ramp must come back ascending) / is seed-reproducible; consistency is exactly zero for
model A and positive for model B; temporal masks session boundaries and - the important one -
**costs nothing when the output moves fast but the residual does not**; resting jitter arithmetic
and its appearance in the report.

---

## Known Problems / Watch Items

- **Supervision is still entirely "be zero".** Model B now behaves well at rest, but every
  high-confidence label in the corpus is still 0 and every non-zero target is the stock model's own
  guess. Nothing has yet taught the adapter what a *correct* smile or jaw-open magnitude looks like,
  and no metric measures it. This is the ceiling, and P2 (guided capture) is the thing that lifts it.
- **The silent-hold prediction is still untested.** Model A inferred "is this an expression?" from
  context; whether model B - which can see the mouth - handles a silent held-open jaw correctly has
  not been checked. Ten seconds with the Stock/Personal switch would settle it.
- **[historical] No model had been trained on real data.** Recording works (4 real sessions, 6,375 frames)
  and the corpus loads, but the training run itself has not completed once, so all ML evidence is
  still from synthetic data with injected defects. That proves the machinery works and says nothing
  about whether it helps this user's actual tracking. The first successful run should confirm:
  logged unique-fps matches the camera, the preview shows the expected crop, and a model trained on
  real neutral sessions reduces neutral false activation.
- **More than one build now exists.** `%LOCALAPPDATA%\Baballonia` is the old install with the
  **unfixed** trainer, and it owns the Desktop shortcut. The current build is
  `<repo>\bin\Baballonia-v2` (earlier builds kept alongside it). Repoint
  or delete the old one once the new build is confirmed, so a stale shortcut cannot reintroduce
  the BOM failure.
- **`AppSettings_OSCPrefix` must be empty** for avatar-guided calibration: the VRCFT module matches
  bare addresses (`/cheekPuffLeft`). The planned preflight assert is **not implemented yet** — add it
  with the cue engine in P2.
- **Model B's value is still unproven, and the current metric cannot settle it.** It is implemented,
  exported, benchmarked and now selectable in the UI, but the only scored axis (resting-face
  accuracy) is already saturated by A at 0.31% false activation. A fair A-vs-B comparison needs
  guided data. Do not assume the image branch
  is worth its cost until measured on held-out real sessions.
- The debug comparison table re-sorts 45 rows at 4 Hz via `ObservableCollection.Move`. Fine in
  practice; if it ever feels janky, sort a view rather than the collection.
- Recorder `Dispose()` blocks on the writer drain via `GetAwaiter().GetResult()`. Fine at shutdown,
  but do not call it on a UI-critical path.
- 132 build warnings are pre-existing upstream noise (nullability, CA rules).

---

## Git State

```
84eca8c  (upstream base)
d9bdb61  build: make Baballonia.Tests restore and compile again
38a0680  personalization: lock the 45-expression schema
49ea7f1  personalization: add face pipeline hooks for capture and correction
d20df3b  personalization: add avatar-guided calibration override path
e6340d2  personalization: add dataset recorder
ebfae7f  personalization: add minimal capture page
15455e6  docs: add approved plan and P0 handoff log
76e33bc  personalization: add local training package
cfa00a6  personalization: add personal model runtime, blend and debug view
235255b  docs: add user guide and record P1 state
563d674  personalization: add guided workflow over the existing tooling
```
Nothing pushed. No history rewritten.

A self-contained Windows build was produced with:

```
<sdk>\dotnet.exe publish src/Baballonia.Desktop/Baballonia.Desktop.csproj ^
    -c Release -r win-x64 --self-contained true -o <output>
```

Verify any build has `Modules\` populated (4 capture DLLs) and a `training\babble_personal` folder —
without the former the app has no cameras, without the latter the Train button cannot run.

User-facing guide: `PERSONALIZATION_GUIDE.md` (record → train → export → install → compare,
plus troubleshooting and a privacy summary).

---

## [SUPERSEDED 2026-08-14] Former next task — kept for history

**Superseded by the PHASE 2 PLAN section near the top of this file.** Step 1 below was completed
(models A and B trained on real data; results recorded in the sections above and validated in VR).
Steps 2–3 are absorbed into milestone M4 of the Phase 2 plan (which also fixes the labels.py
settle-trim bug first, in M1). Do not work from this section.

### Step 1 — First real-data run (validates everything built so far)

**Recording is already done** — 4 sessions (2 neutral, 2 speech), 6,375 frames, under
`%APPDATA%\ProjectBabble\PersonalDataset\`. They load cleanly since the BOM fix. Do **not**
re-record; just train on them.

1. Launch **`<repo>\bin\Baballonia-v4\Baballonia.Desktop.exe`** (the newest build — the old
   Desktop shortcut points at the unfixed one).
2. Open **Personalization**. The four existing recordings are picked up automatically from the
   Roaming dataset folder; nothing needs importing.
3. Press **Train My Face Model**, then compare with the **Stock / Personal** switch.

**Record the outcome in this file**, especially the neutral false-activation rate before and after,
which the results screen reports directly. That single number is the first genuine evidence the
project works on a real face.

Watch for two things the synthetic tests cannot cover: whether the recorded crop is actually usable,
and whether unique-fps matches the camera (a large mismatch means the dedupe or rate cap is
misbehaving on real hardware).

### Step 2 — Supervision experiment, before writing any cue code

Plan §12.1. With the cue mechanism already built (`ExpressionOverrideService`), drive a few
step-holds manually, record simultaneously, and check whether the stock output channel for the cued
expression actually tracks the commanded level after lag correction
(`labels.estimate_cue_lag_seconds` returns the correlation).

* Correlation ≳0.6 and monotonic across levels → build the full guided cue engine as planned.
* Otherwise → fall back to 0/50/100 three-level holds, and record that decision here.

This costs one ~10-minute recording session and determines whether the guided-capture design is
sound. Do not skip it and train on cue data that may be meaningless.

### Step 3 — P2 guided capture (only after Step 2)

Build `GuidedCaptureRoutine` on top of the existing override service: cue list, phase state machine
publishing `CuePhase` snapshots, `KeepAlive()` each UI tick, audio cues, and a
`IReadOnlyCueStateSource` implementation so the recorder stamps commanded targets into
`labels.jsonl`. Add the **`AppSettings_OSCPrefix` must be empty** preflight assert here.

Then compare model A vs model B on real held-out sessions and ship whichever wins.
