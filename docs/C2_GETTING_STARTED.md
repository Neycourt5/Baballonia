# Train your own Model C and C2

The preview ships stock runtime models and the training implementation. It does not ship the maintainer's personal weights or recordings. You can use stock tracking without training anything. C2 trains your face; eye tracking is separate.

## 1. Get ordinary tracking working

Extract the complete Windows ZIP and run **Baballonia.Desktop.exe**. Select your face camera on **Home**, start it, and wait for a moving preview. Set a usable crop and wear the headset as you normally do. Keep that camera and crop consistent while preparing and comparing a candidate.

## 2. Prepare a personal Model C

1. Open **Personalization** (page title **Personalize My Face**).
2. Under **Setup**, choose **Set Up Training Tools** if needed. Use Python **3.11–3.13**, preferably **3.13**; the pinned PyTorch does not support Python 3.14. Setup creates a separate environment and downloads CPU training dependencies. Keep the package's `training` folder beside the executable.
3. Use **Record Neutral** for about 45 seconds, then **Stop Recording**. Record two neutral sessions.
4. Use **Record Speech** for roughly a minute of natural talking, then **Stop Recording**. Record two speech sessions across normal separate wearings rather than splitting one recording to manufacture independent checks.
5. Under **Train → Train next**, select **C — Shared visual features (existing recipe)**.
6. Choose **Prepare Model C**. It builds a derived stock graph exposing visual features, computes features for existing recordings, enables that runner, and checks readiness. It does not retrain the stock face network.
7. Choose **Train Model C**. Read the result and **Face model selection** status. Legacy A/B/C training can install its result; training and activation success are reported separately. Use **Use This Model** or select the model and **Use Model C** if activation is needed.
8. Confirm **Actually active** identifies Model C and its embedding runner is active. Compare rest, small/comfortable jaw opening, smiles, speech, and return to rest before using C as a C2 reference.

Two neutral and two speech sessions are recommended. The code permits a smaller starting set of one neutral plus another ordinary session, but that does not supply the same held-out coverage. Recording counts alone do not prove expression quality.

If automatic setup picks another Python version, explicitly create the environment first in PowerShell:

```powershell
py -3.13 -m venv "$env:LOCALAPPDATA\babble-train-venv"
```

Then use **Set Up Training Tools**. If an existing environment uses the wrong Python version, preserve it separately before recreating it; the setup button does not switch an existing environment's interpreter.

See [the personalization guide](../PERSONALIZATION_GUIDE.md) for the older A/B/C guided calibration and correction workflow. C2 uses the separate review process below.

## 3. Collect reviewed C2 practice

Keep your working **Model C** selected. Turn **Enhance expressions while speaking** (Audio Assist) off for recording, training, and comparison trials. C2 reports do not replay audio.

Expand **C2 — Experimental** at the top of Personalization. Follow **Next action** through the core tasks:

| Task | Purpose |
|---|---|
| Relaxed jaw and smile | Explicit jaw/smile absence at rest. |
| Small jaw opening | A gentle onset cue. |
| Comfortable jaw opening | A comfortable positive range cue. |
| Gentle smile | A subtle positive smile cue. |
| Comfortable natural smile | A stronger comfortable smile cue. |
| Talking and return to rest | Preservation/replay without invented exact speech targets. |

For each task:

1. Choose **Record this attempt** and follow the instruction through the countdown, settling interval, and short hold.
2. Review the saved attempt. Choose **Accept hold / reuse existing for replay** only if you actually followed the displayed instruction.
3. Otherwise choose **Missed / not shown correctly / skip**, then retry. Exclusion keeps the recording on disk.

**Pause / continue later** keeps accepted progress; an unfinished attempt is excluded. Optional toothy-smile, smile-while-talking, and asymmetry tasks are replay checks, not assertions about exact jaw or lip values.

You may select suitable older recordings and **reuse existing for replay**. Their old labels are not automatically treated as verified C2 supervision. Accepted relaxed and positive practice are needed for each expression C2 learns.

## 4. Make an independent check when possible

Remove and reseat the headset. Choose **I have reseated the headset — start a new wearing session**, enable **Check-only — keep separate from practice used to teach C2**, and record/review the core again.

Stop/Start and app restart preserve the wearing-session identity. Practice and checks cannot share that identity. Checks do not train the head; the workflow prevents reuse of a consumed check wearing session in later training runs. Without separate checks, reports remain exploratory.

The present workflow evaluates a check as part of the candidate training run. A dedicated evaluate-an-existing-candidate workflow is not implemented.

## 5. Train, compare, and explicitly keep

1. Choose **Train C2 experimental candidate**. It creates a separate candidate, local caches, and a report. It does not select the candidate or replace Model C.
2. Choose **Compare selected report** and review **Advanced — local report and operation details**. Metrics describe labelled/replayed behavior; there is no automatic quality or promotion decision.
3. Choose **Try C2 for 60 seconds**. Compare facial movements and speech on your actual avatar. The trial ends on timeout, leaving the page, or failure.
4. If you prefer it, choose **Keep using this C2**. Confirm **Using now** and **Actually active** identify C2. The selection persists across navigation and restart.
5. Choose **Return to Model C** to clear that persistent selection. Return to C before starting a different temporary trial while a kept C2 is active.

**Keep** records your preference, not an independently proven improvement. A kept C2 can use Audio Assist after visual C2 correction. Keep audio off for controlled visual comparisons.

## Optional jaw-open curve

**App Settings / Enable Jaw Open Curve** starts off, preserving your existing jaw motion and kept C2 even if an older saved curve value is above 1. When enabled, values above 1 reduce intermediate openings; 1 keeps the current response. The curve is applied before face calibration.

Choose your curve before collecting/training a new C2. Changing its effective value makes an incompatible active C2 fall back to Model C with an explanation. To reuse an older candidate, restore its curve/settings and explicitly keep it again; the app does not rewrite that candidate or your recordings.

## If C2 stops or refuses to load

Read the displayed reason. Activation checks candidate bytes, schema/contract metadata, reference C and blend, camera/crop, smoothing, and calibration. Changing these can make a candidate incompatible. Calibration for expressions outside corrected support can change during kept everyday use.

An app folder can move while identical stock/feature model bytes remain compatible. Moving personal C or changing its bytes/settings may invalidate the contract. Do not edit contract/report hashes to bypass checks. Restore the recorded setup or train a new candidate; the runtime retains the base model on failure.

## Assess on your own hardware

Check rest false motion, small/full jaw opening, jaw versus lip separation, gentle/full smiles, tooth visibility, speech, smile while speaking, and return to rest. Repeat across normal wearings. Check camera reconnects and eye throughput while face tracking/recording/training run. An avatar that cannot display an expression cannot validate it.

Your recordings, models, reports, and calibration are personal data. See [privacy and storage](PRIVACY_AND_PROVENANCE.md).
