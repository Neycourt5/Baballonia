# WORK_PROGRESS — Personalized Face-Tracking Fork

Handoff log. Read this plus `you-are-working-inside-harmonic-taco.md` (the approved plan) and you
should be able to continue without any prior conversation.

---

## Current Status

```
Current phase: P0 COMPLETE. P1 COMPLETE. Guided workflow (UX layer) COMPLETE.
               Still blocked on real-data validation.
Branch: main (10 commits ahead of upstream 84eca8c, not pushed)
Build: OK.  Suite: 97 passed / 9 failed (all 9 pre-existing, see Tests Run).
Personalization tests: 81/81 pass.  Python: 8/8 pass.
```

P0 delivered capture and interception: schema lock, pipeline hooks, calibration override path,
dataset recorder, capture page.

P1 delivered the full training and inference loop: a Python package that turns recorded sessions
into a trained ONNX adapter, and a C# runtime that loads, validates, blends and displays it.

**The one thing not yet done is running it on real footage.** Every piece is tested — including
end-to-end on synthetic data with injected defects — but no session has been recorded from an actual
face camera, so no model has been trained on a real face. That is the next milestone and it needs
the user, not the agent.

---

## Project goal (unchanged)

Learn a **user-specific correction model** for the face pipeline: it sees the camera image plus the
stock model's 45 raw outputs and emits corrected outputs. Trained locally (PyTorch → ONNX), loaded
by Baballonia via ONNX Runtime, with instant fallback to stock when disabled/missing/invalid.
Eye tracking is out of scope (Paper Tracker handles it). Slider calibration is explicitly *not* the
solution.

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

## Known Problems / Watch Items

- **Nothing has been run against a real face camera.** This is the single biggest gap. All ML
  evidence so far is from synthetic data with injected defects, which proves the machinery works but
  says nothing about whether it helps this user's actual tracking. First real run should confirm:
  logged unique-fps matches the camera, the preview shows the expected crop, and a model trained on
  real neutral sessions reduces neutral false activation.
- **`AppSettings_OSCPrefix` must be empty** for avatar-guided calibration: the VRCFT module matches
  bare addresses (`/cheekPuffLeft`). The planned preflight assert is **not implemented yet** — add it
  with the cue engine in P2.
- **Model B's value is unproven.** It is implemented, exported and benchmarked, but nothing has yet
  shown it beats baseline A, because that comparison needs real data. Do not assume the image branch
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

## NEXT EXACT TASK

**This step needs the user and a face camera; it is not agent work.** Do not start P2 before it,
because P2's design depends on the answer.

### Step 1 — First real-data run (validates everything built so far)

Now a UI flow, not a CLI one:

1. Start the face camera, open **Personalization**, confirm the preview shows a sensible mouth crop
   and Setup reports the camera and training tools ready (press **Set Up Training Tools** if not).
2. **Record Neutral** ×2 (~45 s each, relaxed) and **Record Speech** ×2 (60–90 s), ideally with a
   small headset reposition between them.
3. Sanity-check `%APPDATA%\ProjectBabble\PersonalDataset\`: does `EffectiveFps` in a `session.json`
   match the camera? Do the frames look right?
4. Press **Train My Face Model**, then compare with the **Stock / Personal** switch.

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
