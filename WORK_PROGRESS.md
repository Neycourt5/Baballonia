# WORK_PROGRESS — Personalized Face-Tracking Fork

Handoff log. Read this plus `you-are-working-inside-harmonic-taco.md` (the approved plan) and you
should be able to continue without any prior conversation.

---

## Current Status

```
Current phase: P0 COMPLETE. P1 COMPLETE. Guided workflow (UX layer) COMPLETE.
               First real recordings made; first training attempt hit a dataset
               encoding bug, now FIXED. Re-run of training is the next step.
Branch: main (10 commits ahead of upstream 84eca8c, not pushed)
Build: OK.  Suite: 97 passed / 9 failed (all 9 pre-existing, see Tests Run).
Personalization tests: 74/74 pass (filter run).  Python: 8/8 smoke + 8/8 encoding.
Recordings on disk: 4 sessions (2 neutral, 2 speech), 6,375 frames.
```

P0 delivered capture and interception: schema lock, pipeline hooks, calibration override path,
dataset recorder, capture page.

P1 delivered the full training and inference loop: a Python package that turns recorded sessions
into a trained ONNX adapter, and a C# runtime that loads, validates, blends and displays it.

**Real footage now exists; no model has been trained on it yet.** Four sessions were recorded
successfully, and the first training attempt failed on a *data encoding* bug (UTF-8 BOM), not on
anything ML-related. That bug is fixed and covered by tests (see **BUGFIX — UTF-8 BOM** below), and
the recordings load cleanly. Training on a real face is still the next milestone and it needs the
user, not the agent.

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

Current: **`bin\Baballonia-v4\Baballonia.Desktop.exe`** (model B regularizers + resting-jitter reporting).
Previous: **`bin\Baballonia-v3\Baballonia.Desktop.exe`** (BOM fix, model A/B selector, active-model display, test no longer deletes the installed model).
Previous: `bin\Baballonia-v2` (BOM fix + selector).
Previous: `bin\Baballonia-bomfix` (BOM fix only), and the pre-fix install at `%LOCALAPPDATA%\Baballonia`.

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

## NEXT EXACT TASK

**This step needs the user and a face camera; it is not agent work.** Do not start P2 before it,
because P2's design depends on the answer.

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
