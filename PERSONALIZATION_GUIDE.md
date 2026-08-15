# Personalized Face Tracking — How To Use

A practical guide to training Baballonia to recognise **your** face on **your** camera.

This is not slider calibration. You record real footage of your face, a small model learns where the
stock model gets your expressions wrong, and that correction runs live in front of VRChat.

> **Status:** capture, A/B/C training, immutable model history, quick corrections, full guided face
> routines, and microphone-selectable audio assist are available from the app. Guided instructions
> and Eye V2 calibration use a true SteamVR headset overlay. See
> [What Doesn't Exist Yet](#13-what-doesnt-exist-yet) for the hardware-validation gaps.

---

## The short version

Everything below is detail. The actual workflow is:

1. Open **Personalization** in the sidebar.
2. If Setup says the training tools aren't ready, press **Set Up Training Tools** and wait.
3. Press **Record Neutral** (rest your face ~45 s), **Stop**. Do it twice.
4. Press **Record Speech** (talk ~1 min), **Stop**. Do it twice.
5. Choose Model B, press **Train Model B**, and wait a few minutes.
6. Use **Face model selection** to switch between Stock and any exact historical A/B/C artifact.

Then, once that works: put on the headset, choose a Guided routine, and copy the avatar while the
SteamVR overlay gives explicit instructions. Press **My mouth was closed** whenever you catch the
tracker getting it wrong. Both add evidence to the next explicit training run; neither mutates the
currently running model immediately.

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
10. [Fixing mistakes as they happen](#10-fixing-mistakes-as-they-happen)
11. [Guided calibration](#11-guided-calibration)
12. [Audio enhancement (optional)](#12-audio-enhancement-optional)
13. [What doesn't exist yet](#13-what-doesnt-exist-yet)
14. [Command line (optional)](#14-command-line-optional)

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

A good set, recorded across a few separate sittings:

| Session | Why |
|---|---|
| Neutral ×2–3 | Your resting face, which is what fixes expressions firing at rest |
| Neutral in different room lighting | Stops the model reading brightness as expression |
| Speech ×2–3 | Natural expression combinations |

**Wear the headset the way you normally do.** Do not deliberately reposition it, and do not try to
reproduce an exact position either. You cannot put a headset on the same way twice to the precision
the camera sees — a few millimetres of shift is a visible change in a 224×224 crop of your mouth —
so recording across separate sittings covers that drift by itself. Deliberately odd camera angles
are worse than useless: they spend model capacity on situations you will never actually be in.

The one thing that genuinely breaks a model is changing the **crop** (the ROI settings on the Home
page) after recording. That is a real change to what the model sees. If you adjust the crop, record
fresh sessions and retrain.

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

The **Model** dropdown picks what to train:

| Choice | What it sees | When to use it |
|---|---|---|
| **A — expressions only** | The stock model's 45 numbers | The cheap, honest baseline. Very good at removing a constant bias; it never sees your face. |
| **B — expressions + camera image** | Also the camera frame | Can fix mistakes that need visual information. Slower to train, more to overfit. |
| **C — shared visual embedding** | Stock values plus the stock network's visual features | Experimental. Prepare its feature cache first, and keep B unless C wins on your avatar. |

Train both and keep whichever wins on your own held-out recordings — the results screen tells you
which did better. In practice B has been the better one here, but that is a measurement, not a rule.

---

## 5. Installing and testing it

Training and activation are reported separately. A successful export does not claim to be active
unless the runtime loaded that exact artifact. The Face model selection card shows both
**Requested** and **Actually active**, including any fallback reason.

### Switching models

Choose Stock or any historical A/B/C artifact in **Face model selection**, then press **Use**. The
library is newest-first and shows exact file/run provenance. A rejected or incompatible manual ONNX
does not uninstall the working model. Do the comparison while pulling faces and while actually
talking in VRChat; Stock is always one click away.

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

Each session is a self-contained folder. **Show Training Data in Folder** on the Personalization
page takes you straight there so a genuinely bad session can be moved, backed up, or deleted. Press
**Refresh Counts** afterward: removing a session can change which newest session is held out.

`.gitignore` is set up so face data and trained models can't be committed by accident. Datasets live
outside the repo anyway.

To erase everything: delete the `PersonalDataset` folder and `personalFaceModel.onnx`. Tracking
returns to stock immediately.

---

## 10. Fixing mistakes as they happen

The tracker will occasionally get something wrong — most commonly, your avatar's jaw hanging
slightly open while your mouth is closed. That moment is the single most useful piece of training
data you can produce, because it is a real failure of your real model, and you are the only one who
knows it happened.

The problem is timing: by the time you have noticed, the moment has passed. So the app keeps the
last ten seconds of tracking in memory (nothing is written to disk unless you say so).

### Quick correction

When you see it happen, press **My mouth was closed** on the Personalization page. That saves the
last few seconds — 5 by default, or pick 2 or 10 — as an example of what went wrong.

It records only the claim you actually made. "My mouth was closed" says something about your jaw and
nothing about the rest of your face; if you happened to be smiling, that is left unlabelled rather
than wrongly recorded as neutral.

Collect a handful, then press **Train My Face Model** as usual. Corrections carry more weight than
any other label, because they are the only ones a human checked.

Nothing retrains automatically. A single correction is one noisy example; a dozen describe a
pattern, and training when you choose makes it possible to tell whether they helped.

---

## 11. Guided calibration

Recording yourself resting teaches the model what "nothing" looks like. Nothing in a Neutral or
Speech session teaches it what a *correct* open jaw looks like — which is the other half of telling
those two apart.

Guided calibration fixes that. Your avatar performs an expression vector chosen by the app, you copy
it, and because the app knows what it asked for, it knows what your face should have been doing.

1. Put your headset on with VRChat running and your avatar visible in a mirror.
2. Choose a focused expression or the Full Face pass, then press **Start Guided Calibration**.
3. Copy the avatar while the headset overlay names the expression, counts down, and shows
   **HOLD** / **RELAX**, intensity, repetition, and overall progress.

The controller overlay offers Retry, Skip, and Cancel. Retry replaces only that attempt; Skip or
Cancel writes an explicit invalid-attempt verdict so partial or missed holds cannot train at full
confidence. The trainer's `guided_quality.json` independently marks each repetition Good, Weak, or
Suppressed instead of throwing away the whole session.

Grimace is deliberately empirical: preview candidates A/B/C on the avatar in VR, confirm the one
that is natural and repeatable, and only then does a separate Grimace guided routine become
available. Previewing never records training data.

**If it refuses to start** and mentions an OSC prefix, clear that setting first. With a prefix set
your avatar would not move at all, and the recording would be labelled as expressions you never
made — which is worse than not recording.

Copying an avatar works far better than being told "open your jaw 50%", because nobody knows what
half a jaw feels like, but everybody can imitate a face.

---

## 12. Audio enhancement (optional)

Accurate tracking can still look *subtle* — your mouth is doing the right thing, just not very much
of it, which reads as flat from across a room.

**Personalization → Audio → Enhance expressions while speaking** uses your microphone's loudness to
make mouth movements the camera already sees a little larger while you talk.

Choose the exact input in **Audio Input** (or Automatic), then watch **Input Level**, **Voice**, and
**Selected**. The choice is persisted. A missing USB device falls back without stopping visual
tracking and is retried by the audio service even when the Personalization page is closed. Changing
the input restarts only microphone capture, never a face or eye pipeline.

It cannot invent an expression. If the tracker says you are not smiling, no amount of shouting puts
a smile on your avatar — audio only scales up movement that is genuinely there. The camera stays the
only thing deciding what your face is doing.

Off by default. With it off, or with no microphone, tracking is exactly what it was. Nothing is
recorded or uploaded; audio is analysed for loudness and pitch and immediately discarded.

If the enhancement seems to arrive slightly before or after your voice, adjust the sync offset in
settings (`AudioAssist_SyncOffsetMs`, default 50 ms).

---

## 13. What doesn't exist yet

Being explicit so nothing surprises you:

- **In-VR quick-correction button.** Guided Retry/Skip/Cancel are in the overlay, but the rolling
  **My mouth was closed** correction remains a desktop button.
- **A validated cross-runtime overlay.** The built-in presenter is a Windows SteamVR/OpenVR overlay.
  OpenXR/Monado/WiVRn do not yet have a proven overlay-over-VRChat path.
- **A universally correct Grimace vector.** Candidates exist so this avatar can determine it
  empirically; nothing is trainable until one is explicitly previewed and confirmed.
- **The review tool** for hand-fixing individual frames.
- **Audio phoneme assist** (matching mouth shapes to speech sounds). Deliberately later — simple
  loudness-based enhancement has to prove itself first.
- **Localization** — this page is English-only.

Measured costs, for reference, against a 10 ms per-frame budget: the output-only model adds about
0.03 ms, the image-conditioned model about 0.2 ms, the correction buffer under 1 ms, and audio
enhancement under 0.001 ms (its analysis runs on a separate thread).

---

## 14. Command line (optional)

The app drives the same tools you can run yourself. Nothing is hidden, and using the CLI does not
conflict with the buttons — the app picks up whatever model you install.

```bat
set PY=%LOCALAPPDATA%\babble-train-venv\Scripts\python
cd /d <repo>\training

:: Train the image-conditioned model from the CLI (the UI can also select B)
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
Guided     Choose routine → Start Guided Calibration (SteamVR + avatar mirror)
Train      Choose A/B/C → Train Model
Compare    Face model selection → Stock or exact historical artifact
Fix        My mouth was closed    (right after you see it go wrong)
Audio      Choose input → verify meter/voice → enable assist (optional, off by default)
Undo       switch to Stock, or delete
           %APPDATA%\ProjectBabble\Models\personalFaceModel.onnx
```

Deeper detail: `training/README.md` for the training pipeline, `WORK_PROGRESS.md` for implementation
state, and `you-are-working-inside-harmonic-taco.md` for the full design.
