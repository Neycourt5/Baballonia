# Phase 3 Research Plan — Face Models A–E + Eye Tracking V2

Planning only; no implementation this session. Deliverable = this document + a WORK_PROGRESS.md
handoff for Opus 5 Medium (written after approval; that update is the only "execution").

## Context

Phase 2 (M1–M5, M7) is implemented, tested, committed; the project is blocked on home-PC
validation, not code. This session answers two new questions: **(1)** where does the face-model
program go after the B-vs-C gate — including the new Model E (standalone personal tracker) and a
re-derived A–E ranking; **(2)** can Baballonia's eye tracking be made competitive with Paper
Tracker (gaze, calibration UX, squint, wide), built as a selectable, fail-safe Eye V2 alongside
the untouched default. The existing home-PC checklist (WORK_PROGRESS §NEXT EXACT TASK items 1–7)
is preserved unchanged and remains the immediate queue.

---

# 1. Executive conclusion

- **Is Model E worth a controlled experiment? Yes — but as one arm of a unified experiment, not a
  separate track.** E-from-scratch (E-strict), E-with-stock-as-teacher (E-distilled), and staged
  fine-tuning (D) collapse into one training harness with different init/architecture. E-strict is
  cheap to run and answers the scientific question, but analysis says its *whole-face* ceiling is
  capped by expression **coverage**, not capacity: dims never labeled (most tongue shapes,
  asymmetries) are undefined for a from-scratch model, while B/C inherit stock behavior there for
  free. E-distilled/D are the versions that could actually win, and an adopted E is the only
  architecture that makes the tick *cheaper* (replaces the 5.9M-param stock forward with a
  ~0.3–1.2M net).
- **When?** After the corpus supports it (Gate E0: multi-day guided coverage of ≥8 expressions —
  does not exist yet). The harness is agent-safe to build now; the experiment runs later. Guided
  routine expansion is therefore the highest-leverage next face work: it feeds B, C, E, and D.
- **Most promising face architecture today:** best-of(**B**, **C**) per the already-precommitted
  gate — B is the proven daily driver; C is built and awaits its fair test. E-distilled/D are the
  plausible next step *after* that gate and after the corpus grows. E-strict is science, run
  cheaply alongside.
- **What needs home-PC evidence first:** checklist items 1–7 (JawOpen baseline, new sessions,
  B-vs-C, hard examples, guided JawOpen validation, DirectML-if-C-wins, audio) — every Phase-3
  face decision keys off those numbers, especially the guided cue-lag correlation (validates the
  entire avatar-supervision premise that E depends on).
- **Can Eye V2 be competitive with Paper Tracker? Plausibly yes, and cheaper than expected** — for
  two newly-established reasons. First, Paper's software is **EyeTrackVR v2.0 beta**, so the
  benchmark's methods are inspectable (hybrid: classic pupil localization + LEAP learned
  eyelid/pupil landmarks + lightweight *mapping* calibration — tens of seconds, no retraining).
  Second, **the "Expressions Alpha" work exists in this fork's own remotes as `origin/expr-dev`**
  (13 commits: live widen/squint/brow calibration passes, rewritten VRCFT module handling
  Squint/Widen/Brow, a two-headed gaze+expression trainer with PyTorch checkpoints on the branch).
  The gap between current calibration (120 s reticle + minutes of retraining) and Paper's (~30 s
  mapping) is primarily a *calibration-architecture* gap, not a camera-hardware gap.
- **Which Eye V2 first: V2-A** (fast personal *mapping* calibration over the existing 6-output
  model — anchors + recenter + wide + temporal squint heuristic) — as the working recommendation,
  subject to the P3-2 comparative spike, which evaluates `expr-dev`, the hybrid geometry design,
  and a new learned design **on equal footing** (user decision: expr-dev is a candidate/reference,
  never the presumed basis; `main` stays the stable base absent strong evidence). V2-B
  (pupil/aperture geometry features) is the likely second step and the closest match to the
  benchmark's design; V2-C (personal eye training) stays conditional.

---

# 2. Session-verified current state (what this plan stands on)

**Face (carried from Phase 2, authoritative in WORK_PROGRESS.md):** B > A > stock proven in VR;
false JawOpen is defect #1; full metric suite (persistence, range-retention, cross-talk), hard
examples, guided JawOpen, Model C offline+runtime (flag off), audio Mode 1 (off) all built and
tested; B-vs-C gate precommitted. Training infra: masked per-dim labels, session splits, ordinal
flag, correction ingestion, evaluate CLI, pinned-split experiment harness.

**Eye model (graph parsed directly):** `eyeModel.onnx`, 3.8 MB, **956,474 params**; input
`[b,8,128,128]` = 4 temporal frames × 2 eyes (stacked in C# by `ImageCollector`, queue depth 5);
two independent per-eye conv towers; output `[b,6]` = X, Y, Lid per eye (sigmoid). **No
widen/squint/brow exists in the model, the OSC sends, or the module path on `main`** — all
commented scaffolding. Raw layout is right-eye-first (`/rightEyeY,-X,-Lid,/leftEyeY,-X,-Lid` per
`expr-dev`'s named map); `main`'s variable names are mislabeled but the wiring cancels out.

**Eye pipeline (audited, file:line in agent report → to be preserved in WORK_PROGRESS):**
`EyeProcessingPipeline` mirrors the face pipeline *minus* the personalization hooks: no corrector
stage, no raw-output event (the tap the face recorder uses), One Euro applied to the **raw**
6-vector *before* post-processing. Post-processing: [0,1]→[−1,1] remap, lid inversion, both eyes
share one Y, closed-eye yaw borrowing, convergence clamp (`AppSettings_StabilizeEyes`).
`ParameterSenderService` sends exactly 6 addresses; eye values are remapped but (unlike face)
never clamped. Model hot-swap already exists (`EyeHome_EyeModel` + `LoadInferenceAsync`), though
the swap write is non-volatile and the old runner isn't disposed. Known leaks on the eye tick:
`collected` 8-ch Mat and `Split()` arrays never disposed; `transformed` double-disposed. The
converter pins the input to 8×128×128 — any different model shape needs a new converter/collector.
Face + eye share the 10 ms UI-thread DispatcherTimer.

**Eye calibration today:** Godot overlay (TCP JSON, OverlaySDK) + step framework; Basic routine ≈
30 s tutorial + **120 s gaze reticle** + 30 s blink → CaptureBin (100-byte header + JPEG pairs;
header already has Widen/Squint/Brow/Dilate fields that `main` never populates) → spawns
**BabbleTrainer.exe** (closed binary, fine-tunes `baseline_L/R.pth`, per-eye ~480k-param
checkpoints in-repo) → installs `tuned_temporal_eye_tracking_<ts>.onnx` → hot reload. **Both the
Overlay and Trainer binaries are absent from this working tree** (fetched by
`download_dependencies.ps1`) — calibration cannot run here at all. `GazeOnly`/`BlinkOnly` routines
have a stale-bin merge bug. Optional encrypted data upload exists (`AppSettings_ShareEyeData`,
default off — keep off; privacy principle).

**`origin/expr-dev` = Expressions Alpha (in this repo's remotes, head `4a69127`):**
`OrderedFloatMap` named outputs end-to-end; widen/squint/brow calibration passes live (20 s each) +
new `GazeExpressionCaptureStep` (gaze truth *while* holding an expression — fixes "eye looks up
while squinting"); VRCFT module rewritten to a flat address switch that handles
Squint/Widen/Brow; trainer assets swapped to a **two-headed** `model_best.pt` (8.0 MB) +
`gaze_model_best.pt` (0.5 MB) + 60 MB unlabeled footage — the branch's trainer binary ("qpro
trainer") is external and its availability is unverified. `origin/expr-dev-dbg` adds a
per-stage profiler/debug panel (worth porting regardless).

**Benchmark (verified vs inferred):** Paper Tracker ships ETVR v2.0 BETA 14 + VRCFT module
[verified, hands-on review]. ETVR `EyeTrackApp` sources are public: `AHSF.py`,
`haar_surround_feature.py`, `ransac.py`, `leap.py`, `blink.py`, `intensity_based_openness.py`,
`osc_calibrate_filter.py` [verified]. ETVR calibration UX: "look to all extremes a few seconds →
look straight → Recenter", per eye; optional 9-point overlay [verified docs]. The polygon+pupil
debug view ≈ LEAP landmarks [inferred — the spike reads `leap.py`]. **Licensing:** newer ETVR is
restrictively licensed; reimplement published ideas (Haar-surround, RANSAC ellipse are academic),
copy no code without version-specific license review.

---

# PART I — Face architectures A–E

## 3. The decomposition that answers the Model E hypothesis

| Burden on the universal model | Removed by personalization? |
|---|---|
| Population facial diversity | **Yes** — one face |
| Camera / lens / mount diversity | **Yes** — one geometry |
| Lighting diversity | Mostly — one room, few conditions |
| Label noise at scale | **Yes — replaced by better-than-corpus guided/hard/neutral labels** |
| **Expression coverage** | **No.** A personal model must still *see* every expression it will ever emit, at multiple intensities — it has no prior to fall back on |
| Behavioral/multi-day diversity | Partially — still needs multi-day data |

**The binding constraint on E is coverage, not capacity.** Structural consequence: B/C inherit
stock behavior on every dim personal data never covers (residual default = "leave stock alone");
a from-scratch E has no prior anywhere — an unlabeled dim isn't "stock-quality", it's *undefined*.
Twelve tongue dims, asymmetries, and NoseSneer are barely cueable. Hence:

## 4. The unification: E-strict / E-distilled / D are one experiment

```
              init           architecture     prior on uncovered dims
E-strict      random         free (small)     none            ← the strict scientific claim
E-distilled   random         free (small)     stock-as-teacher (training-time only; runtime independent)
D             stock weights  stock B0         the weights + self-distillation (anti-forgetting)
```

One harness (`--model e-s|e-m|d1 --teacher stock|none --init random|stock`), identical data,
identical day-level splits, identical eval → fair by construction, and 3 tracks for the price of 1.

**D prerequisite [AGENT-SAFE spike]:** ONNX→PyTorch weight transplant (hand-written or timm
EfficientNet-B0, `in_chans=1`, 45 out; initializer names map back — it was exported from PyTorch).
**Assert torch-vs-ONNX parity ≤1e-5 before anything trains; failure ⇒ D gated off, reported.**

**D ladder:** D1 = last stage + head (~1.3M trainable) + self-distillation on natural footage;
advance to D2 (two stages) only if D1 beats best-of(B,C) on the JawOpen block with no
range-retention loss on *day*-holdout; D3 (full) likely never reached at realistic data volume.

## 5. Honest data math and the capacity ladder

Realistic 2–4-week corpus (~10 min/day): guided holds ≈ 25–50k raw frames but only **~2–5k
effective independent samples** (a 3 s hold ≈ 90 correlated frames); 15k neutral; 10–20 min
speech; dozens of corrections. That bounds sensible capacity at **hundreds of k params, low
millions only with heavy augmentation** — matching the in-repo precedent (the per-user fine-tuned
eye model is 956k and works).

**E ladder** (grayscale, in-graph AvgPool→112 so C# feeds its existing tensor):
- **E-S ~0.3M** — B's trunk widened (16→32→64→128, hidden 256). Feasibility probe, trains in minutes.
- **E-M ~1.2M** — hand-written MobileNetV3-small-style inverted residuals + SE. Main candidate.
- **E-L ~3M** — built only if E-M improves monotonically on day-holdout.

Runtime: adopted E **replaces** the ~9 ms stock CPU forward with ~60–120 MFLOPs → likely 3–6×
tick speedup; the only architecture that makes tracking cheaper. Staged: phase 1 runs E as a
corrector that ignores its inputs (zero new runtime surface, safe A/B); phase 2 (post-adoption
only) loads it via the model-swap pattern the eye side already has.

## 6. Supervision deltas for E/D (beyond what's built)

- **Geometric augmentation unlocks** — new capability, E/D only. B/C can't warp (breaks pairing
  with the recorded stock vector); E/D pose-level labels are invariant under small affine (±4 %
  translate, ±5° rotate, ±5 % scale) — exactly what a reseat does. Highest-value robustness lever.
  For teacher-supervised speech frames, re-run the teacher on the warped image (training-time cost
  only).
- **Ordinal loss defaults ON for guided data** (`--ordinal` exists): intensity *ordering* is
  trustworthy even when magnitudes aren't, and E has no residual-shrinkage prior keeping it sane.
- **Anti-memorization:** **day-level holdout is the headline metric** (a guided-pose memorizer
  aces same-day session-holdout); capacity ladder discipline; photometric+geometric aug.
- **Speech labeling:** E-strict — none (motion realism from guided transitions + corrections
  only). E-distilled/D — teacher at low weight, personal labels overriding: stock is deliberately
  the *floor*, corrected where we know better; that is the point, not a leak.
- **New metrics:** per-dim **coverage-vs-quality table**; **uncovered-dim drift** (E vs stock on
  label-free dims — the dead/erratic-dim detector). Existing evaluate CLI already scores any ONNX.

## 7. Comparison table

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

## 8. Three rankings (re-derived; the old informal D>C>E>B>A does not survive)

- **R1 ceiling:** `D ≈ E-distilled > C ≥ B > E-strict > A`. E splits in two and E-strict *drops
  below B/C even at its ceiling* — coverage caps it, capacity doesn't save it.
- **R2 realistic near-term:** `best-of(B,C) > A > D1 ≈ E-distilled > E-strict`.
- **R3 per engineering effort:** `B (done) > C (validation only) > E-distilled/D1 (one shared
  harness) > E-strict (cheap add-on) > D2+`.
- **Why they disagree:** R1 rewards capacity+prior; R2 discounts by the slow-growing guided
  corpus; R3 by code+risk surface (D's transplant/divergence, E's new runtime path). The
  disagreement dictates ordering: validate the proven cheap things, grow the corpus (helps every
  row), let E/D wait for the data that is their real bottleneck.

## 9. E/D experiment design (concrete, kill-cheap)

- **Gate E0 [HOME-PC prerequisite]:** ≥8 guided expressions × ≥2 levels across ≥3 days
  (≥2 reseats), ≥10 neutral sessions across ≥3 lighting conditions, ≥15 min speech, ≥20
  corrections. Harness development does not wait for this; the *real run* does.
- **Harness [AGENT-SAFE]:** `models.py` + `StandaloneFace` (E-S/E-M) + transplanted
  `StockBackbone` (D); `train.py` + `--teacher/--distill`, `--geo-aug` (pose-labeled frames;
  teacher re-run on warps), `--val-days`; `export.py` adapter **v3** (input `image` → `personal`);
  evaluate + coverage table + uncovered-dim drift; experiment.py rows. All synthetic-tested.
- **Runs:** B, C(if adopted), E-S-strict, E-M-strict, E-M-distilled, D1 — identical pinned
  day-splits, seeds {0,1,2}; full metric table + runtime bench.
- **Precommitted gates:** *Kill E-strict* if E-M-strict loses to B on overall MAE **and** the
  JawOpen block on day-holdout (result recorded either way — the science is the point). *Continue
  E-distilled* only if within ~10 % of best-of(B,C) overall *and* better on JawOpen; else park
  until corpus doubles. *Adopt* only on beating best-of(B,C) (JawOpen block + MAE, no suppression
  warnings) + in-VR A/B win. *D ladder* per §4; transplant parity failure ⇒ D off.

---

# PART II — Eye Tracking V2

## 10. Design conclusion from the audit + benchmark research

The benchmark (ETVR under Paper's branding) wins on calibration because it calibrates a
**mapping**, not a **model**: rich per-frame eye state (pupil position, eyelid landmarks/aperture)
+ a per-user anchor/range fit done in tens of seconds. This fork calibrates by *retraining a CNN*
(minutes, tedious, currently impossible in this tree — binaries absent). The fastest route to
parity is the face-personalization philosophy applied to eyes: **keep a fixed extractor, add a
tiny personal layer, calibrate the layer.** And this fork already contains most of the raw
material: hot-swappable eye models, an events bus, the `expr-dev` branch's expression heads and
routines, and per-eye baseline checkpoints.

## 11. Eye V2 tiers

**V2-A — personal mapping calibration over the existing model [build first].**
Keep whatever eye ONNX is installed. Add an `IEyeStateMapper` stage (face-`Corrector` pattern:
volatile field in `EyeProcessingPipeline` after `ProcessExpressions`, `SetMapper` on the manager,
null = exact current behavior) plus a **~30–45 s anchor calibration** (§13) producing per-eye:
gaze map (offset + gain or 2nd-order poly), personal lid anchors (closed / relaxed-neutral /
squint / wide), and derived outputs: calibrated gaze, openness, **Wide** (aperture above neutral
envelope), **Squint** (sustained partial closure — temporal discrimination from blinks, which are
<~300 ms transients; ETVR's `blink.py` uses the same family of logic). Extend
`_eyeExpressionMap` to send `/LeftEyeWiden` etc. (addresses already exist commented) and fix the
module-side Squint mapping in our fork's VRCFT module. **Ceiling honestly stated:** squint from a
single lid scalar + time is partial — a tensed squint at half-blink aperture is ambiguous; V2-A
buys calibration UX, personal ranges, wide-eye, recenter, and *measurably* better gaze mapping,
not benchmark-grade squint.
**V2-B — pupil/aperture geometry features [likely second].**
Per-eye 128×128 crops (tap exists: `NewTransformedFrameEvent`, 2-channel) → classic pupil center
+ aperture extraction (reimplemented Haar-surround + ellipse/RANSAC family — academic, license-
safe) → features feed the same mapper: gaze from pupil position (the responsive path ETVR
proves), squint from aperture geometry + pupil-visibility, one-frame debug overlay (pupil point +
lid line — the Paper-like view). Learned LEAP-style landmarks are a fallback if classic methods
fight the IR imagery; that needs labels and enters V2-C territory.
**V2-C — personal eye model training [conditional].**
Extend `babble_personal` to eyes: fine-tune from `baseline_L/R.pth` (in-repo, per-eye ~480k) or
`expr-dev`'s two-headed checkpoints; supervision from dot-target gaze (the existing overlay
routines or a simpler in-headset dot flow) + anchor prompts for lid states, replacing the closed
BabbleTrainer with our own trainer. Only if A+B measurably can't reach Paper on gaze/squint.

**Ranking (quality-likelihood / risk / calibration burden):** start V2-A (low risk, low burden,
immediate UX win, creates the measurement rig); V2-B adds the discriminative signal squint needs
(medium risk — classic CV tuning on IR imagery); V2-C highest power, highest cost, gated.

## 12. Eye V2 safety architecture (hard requirement honored)

Selectable **Eye Tracking Mode: Default / Experimental (V2)**; V2 state in separate settings keys
(`EyeV2_*`) and files; Default's calibration/model never written by V2; mapper=null restores
Default instantly; any V2 init/calibration failure logs + falls back; A/B is a radio toggle. Same
additive-files discipline: new `Services/EyeV2/` + one volatile stage + one manager hook.

## 13. V2-A calibration protocol (first experiment, exact)

Guided ~35–45 s flow (UI page or overlay text; no Godot dependency required for the anchor part):
1. **Relax** 5 s → neutral lid distribution per eye (median + spread ⇒ personal neutral envelope).
2. **Slow blinks ×3** ~8 s → closed anchor + blink transient duration stats.
3. **Squint hold** 5 s → squint aperture band (accept overlap with blink band; discrimination is
   temporal + band).
4. **Wide hold** 5 s → wide anchor.
5. **Gaze**: center dot 2 s → recenter offset; then either (a) 5-dot (center/L/R/U/D, 2 s each)
   or (b) ETVR-style free "look to extremes" sweep 8 s capturing the raw-gaze envelope. Start
   with (a): 5 dots ⇒ per-eye offset+gain (+cross-term if needed); 9-point deferred unless corner
   error demands it.
6. Persist per-eye anchors + map + capture-day metadata (`EyeV2_Calibration.json`).
**Recenter** = step 5a-center alone (~2 s), available any time. **Reseat validity check:** at V2
start (and on demand) compare 2 s of relaxed stats against stored anchors; drift beyond
threshold ⇒ prompt one-press Recenter first, full recal only if lid anchors also drifted.
Continuous micro-adaptation is deferred (drift risk > benefit for v1); log the drift signal now
to design it later.

## 14. Benchmark protocol vs Paper Tracker [HOME-PC]

Same headset session, ABAB where practical; record OSC output of each system (existing OSC
tooling; both emit VRCFT-compatible streams) + screen-record the debug views:
- **Calibration:** wall-clock, #actions, repeat-twice consistency (anchor deltas), post-reseat
  recovery time to usable.
- **Gaze:** fixation jitter (std over 5 s staring at each of 5 known points), A→B step response
  (latency + overshoot from OSC log), corner error (subjective grid + logged extremes), drift
  over 15 min.
- **Blink:** 20 natural + 10 deliberate blinks vs annotated video — missed/false/latency.
- **Squint:** 5×5 s holds — detection, stability (std during hold), half-blink false-positive
  count during natural conversation.
- **Wide:** 5 deliberate widenings — detection above personal neutral; false-Wide count during
  normal gaze shifts.
- Subjective VRChat mirror A/B last, as arbiter.

## 15. Eye debug tooling

Port `expr-dev-dbg`'s profiler/debug panel idea; add an eye V2 panel: per-eye crop with pupil
point + aperture line overlay (V2-B), raw vs calibrated gaze dots, lid/openness/squint/wide bars,
per-eye confidence, blink-state flag, anchor values, and the two headline metrics (fixation
jitter, step latency) computed live. Purpose: "is calibration wrong, or did the tracker misread
the image?" answerable at a glance. Requires the missing eye raw-output event (Eye-2 below).

---

# 16. Ordered next-phase milestones (information per effort; decision gates)

**F0 [HOME-PC VALIDATION REQUIRED — unchanged prerequisite]:** WORK_PROGRESS checklist items 1–7.
Gates most of what follows; agent-safe items below do not block on it.

**P3-1 Guided routine expansion [AGENT-SAFE]** — highest-leverage face work; feeds B/C/E/D and
Gate E0. Add table-driven routines for Smile L/R, Frown, Pucker, Funnel, MouthLeft/Right,
TongueOut (binary), + 3 combination cues (smile+jaw, frown+jaw, pucker+jaw); per-dim coverage
table in evaluate; cue-preview/step-probe note in UI copy. (Cue engine already table-driven —
this is routine tables + tests, not architecture.)

**P3-2 Eye spike [AGENT-SAFE, RESEARCH SPIKE]** — comparative evaluation, per user decision:
`expr-dev` is **one candidate/reference, not the presumed basis**. Inspect it end-to-end (code,
checkpoints via torch load, routine changes, module rewrite; establish whether its trainer binary
is obtainable) and read ETVR `leap.py`/`blink.py`/`AHSF.py`/`osc_calibrate_filter.py` (method +
license notes). Then compare **three candidate designs on equal footing**: (a) expr-dev's
model+routines approach, (b) the hybrid geometry/personal-mapping design (V2-A→B), (c) a new
lightweight learned design (V2-C family) — scored on achievable gaze/squint/wide/blink quality,
calibration speed, risk, and maintenance. Reuse expr-dev pieces only where they are genuinely the
best solution; **`main` remains the stable base unless evidence strongly justifies otherwise.**
Output: written comparison + recommendation appended to WORK_PROGRESS; no production code.

**P3-3 Eye observability [AGENT-SAFE]** — `NewRawEyeExpressionsEvent(Mat, float[6], ticks)` (the
missing tap), debug panel v1 (raw/calibrated values, fixation-jitter + step-latency meters), fix
the eye-tick Mat leaks (justified now: we build on this path; each fix pinned by a test),
volatile+dispose hygiene on runner swap. Measurement before change.

**P3-4 Eye V2-A [AGENT-SAFE code + HOME-PC VALIDATION]** — `IEyeStateMapper` stage + manager hook
+ `EyeV2_*` settings; §13 calibration flow + Recenter + validity check; Widen/Squint OSC sends +
module fix in our fork; Mode selector UI. Synthetic/unit tests for mapper math, anchor fitting,
blink-vs-squint temporal logic; home-PC benchmark vs Paper per §14. **Gate:** if V2-A gaze +
calibration UX ≥ "clearly better than current Baballonia" and within sight of Paper on gaze, and
squint remains the main gap ⇒ proceed P3-5; if V2-A can't even beat current Baballonia ⇒ stop eye
work, keep Paper.

**P3-5 Eye V2-B geometry features [CONDITIONAL on P3-4 gate]** — pupil center + aperture
extraction (classic first), overlay in debug panel, features into the mapper; re-run §14.
**Gate:** squint/wide robustness vs Paper decides whether V2-C is ever needed.

**P3-6 E/D harness [AGENT-SAFE]** — §9 harness + transplant spike (parity gate) + export v3 +
coverage/drift metrics; synthetic-tested; sits ready for Gate E0.

**P3-7 E/D experiment [HOME-PC + CONDITIONAL on Gate E0 and on F0's B-vs-C outcome]** — run §9,
apply precommitted gates, record verdicts in WORK_PROGRESS.

**P3-8 [CONDITIONAL] Eye V2-C / D2+ / E adoption runtime** — each opens only via its gate.

Audio phoneme/viseme assist stays below all of the above (unchanged from Phase 2 position).

---

# 17. Risks and rollback

| Risk | Containment / rollback |
|---|---|
| E/D overfit guided poses | day-holdout headline; capacity ladder; §9 kill gates |
| E uncovered dims dead/erratic | drift metric; teacher floor (E-distilled); adoption gate blocks suppression |
| D transplant imparity | parity ≤1e-5 asserted first; failure ⇒ D off entirely |
| D forgetting / divergent 24 MB model | self-distillation; artifacts live outside repo; stock file never modified |
| Eye V2 breaks eyes | parallel selectable path; Default untouched; failure ⇒ auto-fallback; mapper=null = current behavior |
| V2 calibration corrupts Default's | separate `EyeV2_*` keys/files; Default state never written |
| Stale V2 calibration after reseat | startup validity check; one-press Recenter; full recal only on anchor drift |
| Classic pupil CV fails on this IR imagery | V2-B gate; LEAP-style learned fallback documented; V2-A already shipped value |
| `expr-dev` trainer unobtainable / license issues | spike establishes before any dependence; ETVR code never copied without version license review |
| DirectML surprises | CPU-first for all personal/eye-V2 models (proven pattern); DML validation stays a home-PC checklist item with auto-fallback |
| Upstream updates | additive-files discipline; derived/personal models regenerate; schema drift tests |

# 18. GUI plan

```
FACE                                          EYES
  Tracker: Stock / Personal                     Eye Tracking Mode: Default / Experimental (V2)
  [ Improve My Model ]                          [ Calibrate Eyes ]  (~40 s guided)
  Quick correction: [ My mouth was closed ]     [ Recenter ]        (2 s, anytime)
  Guided calibration: [ Improve accuracy ]      Eyebrows: Off (default; not a priority)
    → now offers Jaw / Smile / Frown / …
  Audio: [ ] Enhance while speaking
Advanced: model A/B/C(/E) choice, strength,   Advanced: eye debug panel (pupil/aperture overlay,
  metrics, embedding runner, training log       raw vs calibrated, jitter/latency meters,
                                                anchors, thresholds, per-eye confidence)
```
No ONNX/CNN/optimizer vocabulary outside Advanced. Eyebrows default-off; only surfaced if V2-B
geometry yields them ~free.

# 19. Exact implementation handoff for Opus 5 Medium

**Ordering note (explicit):** the *face* experiment decisions (P3-7, and whether C or B is the
baseline E must beat) genuinely depend on F0 home-PC results. The handoff therefore starts with
work that is rational *regardless* of F0's outcome. **User-confirmed order: P3-1 (guided
expansion) + P3-2 (eye comparative spike) in the first session**, P3-3 next, then P3-4 code —
with P3-2 run as the candidate comparison described above, not an expr-dev adoption.

**First milestone = P3-1 + P3-2 (+P3-3 if capacity remains):**
- P3-1 files: `src/Baballonia/Services/Personalization/GuidedCaptureRoutine.cs` (routine tables
  for the new expressions + combos; reuse `CO_ACTIVATION_EXCLUSIONS` dims), `PersonalizationViewModel/View`
  (routine picker), `training/babble_personal/evaluate.py` (coverage table),
  tests: `GuidedCaptureRoutineTest` extensions + a labels-side coverage test. Feature-flag: none
  needed (guided is already opt-in). Commit boundary: one commit, all tests green
  (`--filter Personalization` + Python standalone runners).
- P3-2 output: a written report appended to WORK_PROGRESS (`expr-dev` findings incl. checkpoint
  architectures via torch inspection, trainer availability, ETVR method/license notes, and the
  V2-A-on-main vs adopt-expr-dev decision with reasons). No production code.
- P3-3 files: `Services/events/PipelineEvents.cs` (+`NewRawEyeExpressionsEvent`),
  `EyeProcessingPipeline.cs` (publish + leak fixes, each pinned by a test),
  `EyePipelineManager.cs` (volatile/dispose hygiene), new debug panel (port pattern from
  `expr-dev-dbg` commit `554f3e2`), tests mirroring `FaceProcessingPipelineCorrectorTest`.
- Acceptance: suite stays at 9 known failures; new tests green; WORK_PROGRESS updated per commit
  (files-changed table + decisions + next task), preserving the F0 checklist at top.

**Then:** P3-4 (Eye V2-A) as its own milestone with the §13 protocol and §12 safety rules;
P3-6 harness once P3-1 ships; everything else opens by gate.

# 20. Verification

- Agent-safe milestones: full C# suite (expect 205+/9-known-fail baseline), Python standalone
  runners (91+ checks), synthetic end-to-end runs of any new trainer paths, latency tests for any
  new tick-path code (<1 ms budget each).
- Home-PC items: §14 eye benchmark, F0 checklist, Gate E0 corpus audit, §9 experiment — all
  results recorded in WORK_PROGRESS with the precommitted gates quoted verbatim before the
  numbers.
