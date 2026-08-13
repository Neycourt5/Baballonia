# WORK_PROGRESS — Personalized Face-Tracking Fork

Handoff log. Read this plus `you-are-working-inside-harmonic-taco.md` (the approved plan) and you
should be able to continue without any prior conversation.

---

## Current Status

```
Current phase: P0 COMPLETE. P1 not started.
Branch: main (5 commits ahead of upstream 84eca8c, not pushed)
Build: OK.  Personalization tests: 45/45 pass.
```

P0 delivered the capture and interception infrastructure: schema lock, pipeline hooks, calibration
override path, dataset recorder, and a minimal capture page. No machine learning yet — that is P1.

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
  Total: 70   Passed: 61   Failed: 9

<sdk>\dotnet.exe test ... --filter "FullyQualifiedName~Personalization|FullyQualifiedName~FaceProcessingPipelineCorrector"
  Total: 45   Passed: 45   Failed: 0

<sdk>\dotnet.exe build src/Baballonia.Desktop/Baballonia.Desktop.csproj
  Build succeeded. 0 Errors, 132 Warnings (all pre-existing).
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

None measured yet. **Do not record estimates as measurements.** The plan's latency figures
(≈0.05 ms baseline A, ≈0.3–0.8 ms model B) are estimates awaiting the P1 benchmark.

One measured behavioral fact: the sender loop's real cadence is **~64 Hz, not 100 Hz** —
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

**7. Upstream bugs observed and deliberately NOT fixed** (out of scope, documented so nobody
"rediscovers" them): One Euro filter is not cleared when disabled (`FacePipelineManager.LoadFilter`
returns early); `FaceProcessingPipeline` never disposes the source `frame` Mat; `RunUpdate` shadows
rather than overrides the base method; `OscQueryServiceWrapper` is dead code that would repoint
`OSCOutPort` to 9000 and break the VRCFT path if revived.

---

## Known Problems / Watch Items

- **Not yet exercised against real hardware.** Everything is unit-tested; no session has been
  recorded from an actual face camera. First real run should confirm the logged unique-fps matches
  the camera and that the preview shows the expected crop.
- **`AppSettings_OSCPrefix` must be empty** for avatar-guided calibration: the VRCFT module matches
  bare addresses (`/cheekPuffLeft`). The planned preflight assert is **not implemented yet** — add it
  with the cue engine in P2.
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
```
Nothing pushed. No history rewritten.

---

## NEXT EXACT TASK

Begin **P1** (plan §7, §11). In order:

1. Create the Python package skeleton at `training/` per plan §7:
   `babble_personal/{schema.py,dataset.py,labels.py,models.py,train.py,evaluate.py,export.py}`
   plus a pinned `requirements.txt` (torch 2.7.1 CPU, onnx 1.18.0, onnxruntime 1.22.0, numpy,
   opencv-python-headless, tqdm) and a README.
   - `schema.py` must reproduce the C# hash **exactly**: the 45 names joined by `"\n"`, UTF-8
     encoded, no trailing newline, lowercase hex SHA-256. Add a test asserting the literal hash
     value matches the one `PersonalizationSchema.Sha256` produces.
   - Create the venv **outside** the OneDrive-synced repo, e.g.
     `py -3.13 -m venv %LOCALAPPDATA%\babble-train-venv`.

2. `dataset.py`: discover session folders, load `session.json` / `labels.jsonl` / frames, and
   **split by session, never by frame** (adjacent frames are near-duplicates and would fake
   validation).

3. `models.py` + `train.py`: baseline A only — `OutputMlpAdapter` (stock[45] → 128 → 128 → 45
   residual, ~29k params), masked Huber loss plus the residual-shrinkage term (λ_r = 1e-2) that
   makes stock passthrough the zero-cost default.

4. `export.py`: emit ONNX with inputs `image [1,1,224,224]` + `stock [1,45]`, output
   `personal [1,45]`, graph-level `Clip(stock + r, 0, 1)`, and metadata
   (`personal_adapter_version`, `adapter_type`, `expression_names`, `expression_schema_sha256`,
   `input_normalization`, `input_size`, `base_model_md5`, `trained_utc`). Assert torch↔ORT parity
   within 1e-4 on ~32 random samples.
   (Baseline A ignores the image input; keep the input present so the C# runtime path is identical
   for both model types.)

5. C# side: `PersonalModelCorrector` (own `InferenceSession`, **CPU EP**, self-disables on first
   exception) and `PersonalModelManager` (settings `PersonalModel_Enabled` / `PersonalModel_Path` /
   `PersonalModel_Blend`, background load, reject on schema-hash or tensor-shape mismatch → log +
   `SetCorrector(null)`).

6. **Measure and record** in this file: tick p50/p95 with a random-weight model B installed, CPU and
   with `AppSettings_UseGPU` toggled. Real numbers only.

Before P2 guided capture, run the plan's cheap supervision experiment (§12.1) **before** writing cue
code: record a few guided ramps and check the stock output channels actually correlate with the cue
signal after lag correction. If they do not, switch to the 0/50/100 step-hold fallback.
