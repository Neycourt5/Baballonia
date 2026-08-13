# Personalized Face Tracking — How To Use

A practical guide to training Baballonia to recognise **your** face on **your** camera.

This is not slider calibration. You record real footage of your face, a small model learns where the
stock model gets your expressions wrong, and that correction runs live in front of VRChat.

> **Status:** capture, training and runtime all work. Avatar-guided calibration is built at the
> plumbing level but the on-screen cue sequence is not written yet, so today you record Neutral and
> Speech sessions. See [What Doesn't Exist Yet](#what-doesnt-exist-yet).

---

## Contents

1. [How it works](#1-how-it-works)
2. [One-time setup](#2-one-time-setup)
3. [Recording your data](#3-recording-your-data)
4. [Training a model](#4-training-a-model)
5. [Installing and testing it](#5-installing-and-testing-it)
6. [Reading the numbers](#6-reading-the-numbers)
7. [Improving it](#7-improving-it)
8. [Troubleshooting](#8-troubleshooting)
9. [Privacy and your data](#9-privacy-and-your-data)
10. [What doesn't exist yet](#10-what-doesnt-exist-yet)

---

## 1. How it works

```
   your face camera
          ↓
   stock Baballonia model          ← unchanged, never modified
          ↓
   45 raw expression values
          ↓
   your personal model             ← the new part: learns your face's corrections
          ↓
   corrected values
          ↓
   One Euro smoothing → calibration → OSC → VRCFaceTracking → VRChat
```

The personal model predicts a **correction**, not a replacement. It starts life as an exact
passthrough and only learns to move an expression where your recordings give it a reason to. That is
deliberate: it means a model trained on a modest amount of footage cannot wreck the expressions it
never saw evidence about.

Turning it off, or deleting the file, returns you to stock behaviour instantly.

---

## 2. One-time setup

### Build the app

```bat
git submodule update --init --recursive
cd src\Baballonia.Desktop
dotnet run
```

If `dotnet` reports "No SDKs were found", your PATH has the runtime but not the SDK. On this machine
the SDK is at `%LOCALAPPDATA%\Microsoft\dotnet\dotnet.exe` — use that full path.

### Set up training (only needed when you train)

```bat
py -3.13 -m venv %LOCALAPPDATA%\babble-train-venv
%LOCALAPPDATA%\babble-train-venv\Scripts\python -m pip install --upgrade pip
%LOCALAPPDATA%\babble-train-venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
%LOCALAPPDATA%\babble-train-venv\Scripts\python -m pip install -r training\requirements.txt
```

Keep the virtualenv **outside** the repo — this project folder is inside OneDrive, and a venv there
causes constant sync churn.

Verify it works:

```bat
%LOCALAPPDATA%\babble-train-venv\Scripts\python training\tests\test_pipeline_smoke.py
```

You should see `7/7 passed`. This fabricates data with known faults and checks the trainer actually
fixes them, so it proves the whole pipeline, not just that Python imports.

**Python is only needed for training.** Baballonia itself never runs Python.

---

## 3. Recording your data

Open Baballonia and go to the **Personalization** page in the sidebar.

Start your face camera on the Home page first. The Personalization page shows a small square preview
— that is the exact image the model sees. If it is blank or badly cropped, fix that on the Home page
before recording anything; every frame you record inherits that crop.

### The two session types

| Type | How long | What to do |
|---|---|---|
| **Neutral** | ~45 seconds | Let your face rest. Breathe normally, swallow once or twice. Do not "hold still" tensely — just relax. |
| **Speech** | 60–90 seconds | Talk naturally. Read something aloud, or just ramble. Smile and frown while talking. |

Neutral sessions are the most valuable thing you can record. They are what teach the model what your
face looks like when it should be doing *nothing*, which is what fixes expressions firing at rest.

### Recording

1. Pick the session type.
2. Optionally add a note (e.g. `headset a bit high today`) — useful later when comparing sessions.
3. Press **Start recording**, do the thing, press **Stop**.
4. The status line reports frames saved and the measured frame rate.

### How much to record

Minimum useful set:

- 2 × Neutral
- 2 × Speech

You need **at least two of each type** so one can be held out to honestly test the model.

Better, over several days:

| Session | Why |
|---|---|
| Neutral, normal headset position | Baseline |
| Neutral, headset slightly higher | Teaches tolerance for real placement drift |
| Neutral, another day | Different lighting, different face |
| Speech ×2–3 | Natural expression combinations |

Do not contort the camera into positions you would never actually use. The goal is the variation of
normal daily wear, nothing more.

Roughly 23 MB per minute; a full set is a few hundred MB.

---

## 4. Training a model

```bat
set VENV=%LOCALAPPDATA%\babble-train-venv\Scripts\python
cd /d <repo>\training

%VENV% -m babble_personal.train ^
    --data "%APPDATA%\ProjectBabble\PersonalDataset" ^
    --model a ^
    --out D:\babble-runs
```

**Start with `--model a`.** It only sees the stock model's output numbers, not the picture, and it
answers a question worth answering cheaply: how much of your problem is fixable just from the
relationships between expressions? It trains in seconds.

Then try `--model b`, which also sees the camera image and can fix mistakes that need visual
information:

```bat
%VENV% -m babble_personal.train --data "%APPDATA%\ProjectBabble\PersonalDataset" --model b --out D:\babble-runs
```

**If A and B score about the same, use A.** It is smaller, faster, and has less to overfit.

Useful options:

| Option | Effect |
|---|---|
| `--val-sessions <id> ...` | Choose which sessions are held out |
| `--epochs 60` | Train longer |
| `--shrinkage 0.05` | Stay closer to stock (more conservative) |
| `--no-speech-pseudo-labels` | Ignore the weak speech labels entirely |

Point `--out` somewhere outside OneDrive.

### Export it

```bat
%VENV% -m babble_personal.export ^
    --checkpoint D:\babble-runs\<run-folder>\model.pt ^
    --base-model ..\src\Baballonia\faceModel.onnx
```

This writes `personalFaceModel.onnx` next to the checkpoint and refuses to finish unless ONNX
Runtime reproduces PyTorch exactly — so a model that exports successfully will behave in the app the
way it behaved in training.

---

## 5. Installing and testing it

1. Copy `personalFaceModel.onnx` to:
   ```
   %APPDATA%\ProjectBabble\Models\personalFaceModel.onnx
   ```
2. In the app: **Personalization → Use personal model**.
3. The status line should read `Active: output_mlp_v1 (trained ...)`.
4. Press **Reload model** after replacing the file with a newer one.

### The A/B test

The **Strength** slider is your comparison tool:

- **0%** — byte-for-byte the stock model
- **100%** — fully personalized
- anywhere between — a blend

Slide it while pulling faces and watch the **Stock vs personal** table, which lists expressions by
how much the model is changing them. That table is the fastest way to see what your model actually
learned — and to catch it moving something it shouldn't.

Then try it in VRChat. Slide between 0% and 100% while talking to someone and see whether your
avatar tracks you better.

**Trust the numbers over the vibe when they disagree**, especially early on. A model that suppresses
everything feels "stable" for about ten minutes and then feels dead.

---

## 6. Reading the numbers

Training prints three things, for stock and personal side by side.

### Per-expression MAE

Average error on frames where we genuinely know the right answer. A positive `delta` means the
personal model is closer to the truth.

```
  expression                frames     stock  personal     delta
  JawOpen                       60    0.2991    0.2582   +0.0409
```

The report also lists any expression that got **worse**. Pay attention to those.

### Neutral stability

```
Neutral stability (60 resting frames, threshold 0.15)
  false activation rate  stock 0.0222   personal 0.0071
```

How often an expression fires above 0.15 while your face is at rest. This is the "my avatar smiles
when I'm not smiling" complaint, as a number. Lower is better.

### Cross-talk

How much unrelated expressions light up during a deliberate one — "I opened my jaw and my avatar
smiled".

### What "better" means

A personal model is genuinely better when it improves neutral stability and cross-talk **without**
raising MAE on intentional expressions. Improving one by destroying the other is not a win — you can
get a perfectly stable neutral by outputting nothing at all.

---

## 7. Improving it

Iterate:

1. Use the tracker normally.
2. Notice something wrong ("my frown barely registers").
3. Record another session or two covering it.
4. Retrain, re-export, reload.
5. Compare with the Strength slider.

Tips:

- **More neutral data** is the cheapest fix for twitchy resting faces.
- **Record across days.** A model trained on one session learns that session's exact camera position.
- **Keep old datasets.** Sessions accumulate; training uses everything in the folder.
- **Keep old models.** Rename them (`personalFaceModel_v3.onnx`) so you can go back.

---

## 8. Troubleshooting

**The preview is blank**
The face camera isn't running. Start it on the Home page.

**"Not loaded: No personal model at ..."**
The file isn't where the app expects. Check the path shown under the status line.

**"Not loaded: Model was trained against a different expression schema"**
The model predates a change to the expression list. Retrain against your existing recordings — your
data is still fine.

**"Not loaded: Model has no expression_schema_sha256 metadata"**
Exported by an older or hand-rolled script. Re-export with `babble_personal.export`.

**Tracking got worse after enabling it**
Slide Strength to 0% to confirm stock behaves normally (it should be identical). Then check the
training report for regressions. Common causes: too little data, no neutral sessions, or the camera
crop changed since recording. Recording a fresh session after moving the camera fixes the last one.

**Training says "no supervised cells"**
You only have speech sessions with pseudo-labels disabled. Record a Neutral session.

**Training says every session went to validation**
You have one session per type. Record at least two of each.

**Nothing changes when I enable the model**
Check the status line says `Active`. If Strength is 0%, that's exactly stock by design.

**Personalization is off and something still feels different**
It shouldn't be. With no model installed the pipeline is the stock path; if you can reproduce a
difference, that's a bug worth reporting.

---

## 9. Privacy and your data

Everything stays on your machine. There is no upload, no telemetry, no cloud service.

| What | Where |
|---|---|
| Recorded frames and labels | `%APPDATA%\ProjectBabble\PersonalDataset\` |
| Trained models | `%APPDATA%\ProjectBabble\Models\` |
| Training runs | wherever you pointed `--out` |

Each session is a self-contained folder — back it up, move it, or delete it freely. **Open dataset
folder** on the Personalization page takes you straight there.

`.gitignore` is set up so face data and trained models can't be committed by accident. Datasets live
outside the repo anyway.

To erase everything: delete the `PersonalDataset` folder and `personalFaceModel.onnx`. Tracking
returns to stock immediately.

---

## 10. What doesn't exist yet

Being explicit so nothing surprises you:

- **Guided expression capture.** The plan calls for your VRChat avatar to demonstrate target
  expressions while you imitate it, with the commanded values becoming training labels. The
  mechanism that drives the avatar is built and tested; the on-screen cue sequence is not written.
  Only Neutral and Speech recording is available today.
- **The review/correction tool** for hand-fixing individual frames.
- **Automatic hard-example capture** ("save the last 10 seconds, that looked wrong").
- **Localization** — this page is English-only.

Current measured inference cost, for reference: the output-only model adds about 0.03 ms per frame
and the image-conditioned model about 0.2 ms, against a 10 ms budget.

---

## Quick reference

```bat
:: Record            Personalization page → type → Start / Stop

:: Train
%LOCALAPPDATA%\babble-train-venv\Scripts\python -m babble_personal.train ^
    --data "%APPDATA%\ProjectBabble\PersonalDataset" --model a --out D:\babble-runs

:: Export
%LOCALAPPDATA%\babble-train-venv\Scripts\python -m babble_personal.export ^
    --checkpoint D:\babble-runs\<run>\model.pt

:: Install          copy personalFaceModel.onnx → %APPDATA%\ProjectBabble\Models\
:: Enable           Personalization → Use personal model
:: Compare          Strength slider 0% ↔ 100%
```

Deeper detail: `training/README.md` for the training pipeline, `WORK_PROGRESS.md` for implementation
state, and `you-are-working-inside-harmonic-taco.md` for the full design.
