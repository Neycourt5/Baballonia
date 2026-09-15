# Personal face tracking

Baballonia can train a local correction model from your face-camera recordings. Calibration changes input/output ranges; personal training learns a correction. Guided avatar demonstrations are approximate teaching cues rather than measured anatomy.

For this preview's **Model C → C2** workflow, start with [Train your own Model C and C2](docs/C2_GETTING_STARTED.md).

## Available models

| Choice under Train next | Inputs | Preparation |
|---|---|---|
| A — Expressions only (simple) | 45 stock expression values | Ordinary recordings and tools. |
| B — Expressions + camera image | Expressions and a separate small image network | Ordinary recordings and tools. |
| C — Shared visual features (existing recipe) | Expressions and stock network features | **Prepare Model C** before training. |
| C2 — Experimental | Working personal C plus shared features | Separate reviewed C2 candidate workflow. |

A/B/C remain available for comparison. No model is guaranteed to win on another face, camera, or avatar. C2 is a separate stage, not another entry in the A/B/C dropdown.

## Camera and training tools

1. Run **Baballonia.Desktop.exe**, select/start your face camera on **Home**, and adjust the crop.
2. Open **Personalization → Setup → Set Up Training Tools** when needed.
3. Use Python 3.11–3.13, preferably 3.13. Setup downloads CPU PyTorch and other packages into `%LOCALAPPDATA%\babble-train-venv`. The pinned PyTorch does not support Python 3.14.
4. Keep the distribution's `training` folder alongside the app. A source build finds it by walking up to the repository.

Python runs when training/preparing models; ordinary ONNX tracking does not need it. Setup needs internet access. Use **Show Details** to inspect errors. The [C2 guide](docs/C2_GETTING_STARTED.md#2-prepare-a-personal-model-c) explains explicitly selecting Python 3.13.

## Record

Use **Record Neutral**, rest naturally for about 45 seconds, then **Stop Recording**. Use **Record Speech**, talk for roughly a minute, then stop. Two neutral and two speech sessions across separate normal sittings are recommended. The code allows a smaller starting set, but fewer sessions weaken held-out coverage.

Changing the crop or camera changes your model's input. Review counts and the **Training data** summary. Recordings/corrections remain cumulative; held-out sessions are displayed separately and do not update the model. Old neutral labels are not anatomical ground truth. C2 uses its own per-attempt review and labels.

## Train and select

Under **Train → Train next**, select A, B, or C. For C, choose **Prepare Model C** and wait until both runner and recorded features are ready. Press **Train Model A/B/C**.

Legacy A/B/C training attempts to install the exported result. Check **Requested** and **Actually active** under **Face model selection**. Choose an exact historical model and **Use Model A/B/C**, use **Use This Model** after a run, or return to **Stock**. Read fallback reasons rather than assuming a selected filename is active.

C2 training saves a candidate without activation. Its **Try C2 for 60 seconds**, **Keep using this C2**, and **Return to Model C** controls are covered in the [C2 guide](docs/C2_GETTING_STARTED.md).

## Guided calibration and quick corrections

Under **Guided calibration**, choose a routine and press **Start**. A compatible avatar demonstrates movements; the SteamVR overlay can show instructions and hold/relax phases. Confirm the avatar displays the intended movement before using it as a cue. This A/B/C flow and the C2 expander have different recording/label recipes.

**Quick correction** can retain a short recent-frame buffer for explicit corrections such as **My mouth was closed**. They teach a later training run and do not immediately change the active model. Closed lips are not a direct measurement of a closed jaw.

The smile/grimace labs preview avatar poses. Confirm only a pose the avatar displays correctly and you can imitate naturally. These optional experiments do not silently enter every guided routine.

## Audio Assist

**Enhance expressions while speaking** uses microphone loudness to amplify movements already seen by the visual model. Select the input and check its level/status; **Restart audio** restarts microphone capture. Audio failure is intended to leave face/eye inference running.

C2 recording, training, and temporary trials require audio off because reports do not replay it. A kept C2 can use Audio Assist after visual correction. It does not save speech audio as part of face training.

## Compare and troubleshoot

Compare your actual avatar at rest, during small/full jaw opening, gentle/full smiles, speech, mixed expressions, and return to rest. Repeat across normal wearing sessions. Preserve a working model while assessing another. Correlated frames are not independent wearings, approximate cues are not anatomical measurements, and good metrics do not establish better avatar behavior.

- Use **Show Recording Folder**, **Show Training Data in Folder**, or **Open Run Folder** to locate personal data.
- Missing Python, missing `training`, or incomplete tools are reported under Setup.
- **Prepare Model C** checks the derived stock-feature graph, recording features, and runner.
- C2 rejects changed model/camera/output context and retains the base model. Read its reason and restore the recorded setup or train another candidate.
- Face and eye calibration are distinct. Check eye model/ranges, synchronization, and receiver when investigating gaze.

Keep recordings, calibration, logs, and trained weights private. Do not upload a profile as a bug report. See [privacy](docs/PRIVACY_AND_PROVENANCE.md), [eye diagnostics](docs/EYE_GAZE_DIAGNOSTICS.md), [validation](docs/C2_VALIDATION.md), and the lower-level [A/B/C trainer](training/README.md).
