# Personalized Face Tracking — How To Use

A practical guide to training Baballonia to recognise **your** face on **your** camera.

This is not slider calibration. You record real footage of your face, a small model learns where the
stock model gets your expressions wrong, and that correction runs live in front of VRChat.

> **Status:** capture, training and runtime all work, and the whole thing is driven from one page
> in the app — no terminal needed. Avatar-guided calibration is built at the plumbing level but the
> on-screen cue sequence is not written yet, so today you record Neutral and Speech sessions.
> See [What Doesn't Exist Yet](#10-what-doesnt-exist-yet).

---

## The short version

Everything below is detail. The actual workflow is:

1. Open **Personalization** in the sidebar.
2. If Setup says the training tools aren't ready, press **Set Up Training Tools** and wait.
3. Press **Record Neutral** (rest your face ~45 s), **Stop**. Do it twice.
4. Press **Record Speech** (talk ~1 min), **Stop**. Do it twice.
5. Press **Train My Face Model** and wait a few minutes.
6. Use the **Stock / Personal** switch to compare.

That's it. Nothing else in this document is required reading.

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
11. [Command line (optional)](#11-command-line-optional)

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

### Set up training

Open **Personalization** and look at the **Setup** section. If it says
`Training tools: Not set up yet`, press **Set Up Training Tools**.

That creates a Python environment in `%LOCALAPPDATA%\babble-train-venv` and downloads the training
software (~200 MB). It runs once, takes a few minutes, and stays entirely on your machine. Progress
appears on the page; **Show Details** has the raw log if anything goes wrong.

You need Python 3.11+ installed for this. If Setup reports Python is missing, install it from
python.org and press **Refresh**.

**Python is only needed for training.** Baballonia itself never runs Python — it loads the finished
model through the ONNX Runtime it already ships.

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

Press **Record Neutral** or **Record Speech**, do the thing, press **Stop Recording**. The status
line reports how many frames were saved.

### How much to record

The **Recordings** section keeps score and tells you what to do next:

```
Neutral   2 of 2 recorded
Speech    1 of 2 recorded

1 more Speech recording recommended.
```

You need **at least two of each type**. That is not arbitrary: testing holds back a whole recording
session, so with only one of a type there is nothing left to check the model against, and the app
will tell you the results cannot be trusted.

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

Press **Train My Face Model**.

That's the whole step. The app runs the training, checks the result, exports the model, installs it,
and switches it on. Progress appears as it goes:

```
Preparing your recordings...
Learning your face...
Checking the results...
Exporting the model...
Installing the model...
```

It usually takes a few minutes. **Show Details** reveals the full technical log at any point, and
**Cancel** stops it cleanly.

By default this trains the **output-only** model, which learns purely from the relationships between
the stock model's own numbers. It is the cheap, honest baseline: if it fixes your problem, there is
no reason to use anything bigger. The image-conditioned model — which also looks at the camera
picture and can fix mistakes that need visual information — is available from the command line
(see [section 11](#11-command-line-optional)) and will get a UI option once there is real-world
evidence it earns its keep.

---

## 5. Installing and testing it

Installation is automatic — training finishes with the model installed and active. The Setup section
will show:

```
✓ Personal model: Active
```

### The A/B test

Under **Compare**, switch between:

```
( ) Stock     (•) Personal
```

Do it while pulling faces, and then while actually talking to someone in VRChat. Stock is always one
click away, and nothing you do here can break normal tracking.

Under **Advanced** there is a **Strength** slider (0% is exactly stock, 100% fully personalized) and
a live table of every expression showing stock value, personal value, and the difference, sorted by
what the model is changing most. That table is the fastest way to see what your model actually
learned — and to catch it moving something it shouldn't.

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

**"There was nothing to learn from"**
You have no Neutral recordings. Those frames are what teach the model your resting face.

**"Record at least two sessions of each type"**
Testing holds back a whole session, so one of a type leaves nothing to check against.

**"The Python training environment is missing or incomplete"**
Press **Set Up Training Tools**. If it fails, **Show Details** has the exact error — usually no
internet connection during the PyTorch download.

**"Training tools: Not found"**
The app can't locate the training scripts. They ship in a `training` folder next to the executable
(or in the source tree if you're running from source). If that folder is missing, copy it from the
repository.

**Nothing changes when I switch to Personal**
Check Setup says `Personal model: Active`. Under Advanced, confirm Strength isn't 0% — that is
exactly stock by design.

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

## 11. Command line (optional)

The app drives the same tools you can run yourself. Nothing is hidden, and using the CLI does not
conflict with the buttons — the app picks up whatever model you install.

```bat
set PY=%LOCALAPPDATA%\babble-train-venv\Scripts\python
cd /d <repo>\training

:: Train the image-conditioned model (no UI option yet)
%PY% -m babble_personal.train --data "%APPDATA%\ProjectBabble\PersonalDataset" ^
     --model b --out "%APPDATA%\ProjectBabble\PersonalTraining"

:: Export and install by hand
%PY% -m babble_personal.export --checkpoint "<run folder>\model.pt"
copy "<run folder>\personalFaceModel.onnx" "%APPDATA%\ProjectBabble\Models\"
```

Then press **Reload Model** under Advanced.

Useful flags:

| Flag | Effect |
|---|---|
| `--model b` | Image-conditioned adapter (sees the camera picture) |
| `--val-sessions <id> ...` | Choose which sessions are held out |
| `--epochs 60` | Train longer |
| `--shrinkage 0.05` | Stay closer to stock (more conservative) |
| `--no-speech-pseudo-labels` | Ignore the weak speech labels entirely |

Each run folder contains `metrics.txt` (the full report), `summary.json` (what the app reads),
`history.json`, and the checkpoint.

---

## Quick reference

```
Setup      Personalization → Set Up Training Tools   (once)
Record     Record Neutral ×2, Record Speech ×2
Train      Train My Face Model
Compare    Stock / Personal
Undo       switch to Stock, or delete
           %APPDATA%\ProjectBabble\Models\personalFaceModel.onnx
```

Deeper detail: `training/README.md` for the training pipeline, `WORK_PROGRESS.md` for implementation
state, and `you-are-working-inside-harmonic-taco.md` for the full design.
