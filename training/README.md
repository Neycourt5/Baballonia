# Personal face adapter — training

Trains a small user-specific model that corrects the stock Baballonia face model's 45-value output,
and exports it to ONNX for the app to load.

**Python is development-time only.** The Baballonia runtime loads the exported `.onnx` through the
ONNX Runtime it already ships and never invokes Python.

---

## Setup

Create the virtualenv **outside the repository**. This repo lives in a OneDrive-synced folder, and a
venv (or checkpoint directory) inside it causes constant sync churn and file locks.

```bat
py -3.13 -m venv %LOCALAPPDATA%\babble-train-venv
%LOCALAPPDATA%\babble-train-venv\Scripts\python -m pip install --upgrade pip

:: torch must come from the CPU index; plain PyPI pulls the CUDA build
%LOCALAPPDATA%\babble-train-venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
%LOCALAPPDATA%\babble-train-venv\Scripts\python -m pip install -r training\requirements.txt
```

Verified working: Python 3.13.5, torch 2.7.1+cpu, onnx 1.18.0, onnxruntime 1.22.0, numpy 2.5.2,
opencv 5.0.0. CPU-only is fine — these models are tiny.

Check the install:

```bat
%LOCALAPPDATA%\babble-train-venv\Scripts\python training\tests\test_pipeline_smoke.py
```

That fabricates sessions with known defects, trains, exports, and asserts the adapter actually
fixes them. All 7 checks should pass.

---

## Workflow

```
Collect  →  Train  →  Export  →  Install  →  Compare
```

### 1. Collect

Record sessions from the app's **Personalization** page. They land in
`%APPDATA%\ProjectBabble\PersonalDataset\<timestamp>_<type>\`.

Record **at least two sessions of each type**, so a whole session can be held out for validation:

* **Neutral** — ~45 s resting. The most valuable data you can record: it defines what your face
  looks like when it should be doing nothing.
* **Speech** — 60–90 s of natural talking.
* Vary headset placement across sessions so the model does not memorise one exact camera position.

### 2. Train

```bat
python -m babble_personal.train --data "%APPDATA%\ProjectBabble\PersonalDataset" --model a --out D:\babble-runs
```

* `--model a` — output-only baseline (~29k params). Start here: it answers how much is fixable from
  the stock outputs alone, and if it matches model B you should ship it.
* `--model b` — image-conditioned adapter (~44k params). Can fix errors that need visual
  information the stock outputs no longer contain.
* `--val-sessions <id> ...` — choose the held-out sessions explicitly.
* `--no-speech-pseudo-labels` — drop the weak stock-derived labels from speech sessions.
* `--shrinkage` — higher keeps the model closer to stock (default `1e-2`).

Point `--out` outside the repo (it is gitignored anyway, but OneDrive still syncs it).

Validation is **always split by session**. Adjacent frames are near-duplicates, so a per-frame split
would report an excellent score that means nothing.

### 3. Export

```bat
python -m babble_personal.export --checkpoint D:\babble-runs\<run>\model.pt ^
    --base-model src\Baballonia\faceModel.onnx
```

Produces `personalFaceModel.onnx` and refuses to finish unless ONNX Runtime reproduces PyTorch
within 1e-4. Both model types export the same signature:

```
image [1,1,224,224] float32   grayscale /255, exactly the stock model's input
stock [1,45]        float32   raw pre-filter stock output
    → personal [1,45] float32 = clip(stock + residual, 0, 1)
```

### 4. Install

Copy it to `%APPDATA%\ProjectBabble\Models\personalFaceModel.onnx`, then enable the personal model
in the app. Deleting the file restores stock behavior immediately.

### 5. Compare

Use the app's stock/personal toggle and blend slider. Numbers first, then judge it in VRChat — but
do not let "feels better" override a model that measurably regressed intentional expressions.

---

## What the metrics mean

`train.py` prints, and `evaluate.py` computes, three things for stock and personal side by side:

* **Per-expression MAE** on confidently-labelled frames — accuracy where we know the answer.
  Low-confidence cells (ramps, "stay relaxed" priors, speech pseudo-labels) are excluded, since
  scoring against a guess only measures how well the model copied the guess.
* **Neutral false-activation rate** — how often an expression exceeds 0.15 while you are resting.
  This is the "my avatar smiles when my face is still" complaint, quantified.
* **Cross-talk** — how much unrelated expressions light up during a cue.

A personal model is only better if it improves neutral stability and cross-talk **without** raising
MAE on intentional expressions. Suppressing everything scores well on two of the three and produces
a dead-feeling face; the report deliberately makes that visible by listing regressions.

---

## How labels are produced

Supervision quality is this project's biggest risk, so every target carries a weight and unlabelled
cells carry zero. Combined with the residual-shrinkage term in the loss, "no confident label" means
"leave the stock prediction alone" rather than "invent a target".

| Source | Weight | Notes |
|---|---|---|
| Neutral session, all expressions | 1.0 | Strongest signal available |
| Guided hold, cued expressions | 1.0 (0.4 tongue) | After a 0.5 s settle trim |
| Guided rest between cues | 1.0 | Back to neutral |
| Guided ramp, cued expressions | 0.5 | Commanded value is less trustworthy while moving |
| Non-cued expressions during a hold | 0.25 | Minus per-cue co-activation exclusions |
| Speech, jaw/mouth only | 0.3 | Stock pseudo-labels; the one circular source, kept weak |
| Manual correction | 2.0 | Overrides everything automatic |
| Transitions between levels | 0 | Masked: animator smoothing + human reaction delay |

`estimate_cue_lag_seconds()` recovers the roughly constant delay between a commanded cue and the
user's response by cross-correlation, and reports a correlation the caller can use to discard
repetitions where the cue clearly was not followed.

---

## Layout

```
babble_personal/
  schema.py    45 names, canonical order, SHA-256 shared with C#
  dataset.py   session discovery/loading, session-level splitting
  labels.py    targets + confidence weights, cue lag estimation
  models.py    OutputMlpAdapter (A), ImageResidualAdapter (B), masked loss
  train.py     CLI training with early stopping and checkpointing
  evaluate.py  per-expression MAE, neutral false activation, cross-talk
  export.py    ONNX export + metadata + torch/ORT parity check
tests/
  test_pipeline_smoke.py   end-to-end on synthetic data with injected defects
```

`schema.py` and `PersonalizationSchema.cs` must agree exactly. The hash is
`sha256("\n".join(names))` over UTF-8 with no trailing newline; it is embedded in every exported
model and checked at load time, so a reordered schema is rejected instead of silently mapping
corrections onto the wrong expressions. A C# test pins the literal value.
