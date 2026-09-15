# C2 implementation and design

C2 is a separate experimental correction stage over a working personal **Model C**. It preserves A/B/C training choices and stock inference. [User instructions](C2_GETTING_STARTED.md) cover recording and selection.

## Runtime path

```text
face camera → saved transform → stock expressions + shared visual features
            → personal Model C → optional C2 → optional Audio Assist
            → One Euro smoothing → calibration/remap → OSC output
```

Face and eyes use separate background workers and pipeline locks. C2 consumes Model C's 45 raw expression values and the stock network's 1280-element feature vector; it does not run another camera backbone. ONNX inputs are `reference` and `embedding`; the named output is `personal`.

The contract is `c2-raw-reference-v1`, with label recipe `c2-reviewed-holds-v1`. Output stays in Model C's pre-filter domain. Approximate teaching cues are inverted through the recorded output mapping before supervising the raw head.

A known sender comparison checks `JawOpen`, whereas the standard schema uses `/jawOpen`. The jaw curve is ineffective on that route. C2 records effective exponent 1 and preserves that behavior. Enabling the curve would change existing candidate meaning and requires a separate change and validation.

## Labels and preservation

Only accepted steady-hold cue dimensions receive explicit weights. Countdown, settling, transitions, and unknown expressions are not made into exact labels. Explicit absence and approximate positive cues have different weights. A weak preservation term retains reference behavior on unlabelled supported dimensions.

Core supervision covers JawOpen, MouthSmileLeft, and MouthSmileRight. Support requires accepted positive and absent practice; channels outside candidate support pass through from Model C at runtime. Toothy smiles and speech replay do not imply exact jaw/lip anatomy.

The default trainer uses a small C-style residual head, fixed seed, bounded epochs, and masked labels. A separate Python `--linear` option provides a linear residual experiment. There is no automatic architecture search, promotion, or universal improvement claim.

## Recording and comparison

Recordings, reviews, manifests, contracts, candidates, and wearing identity live under the active profile's `C2` directory. New images are paired with stock outputs and float32 features from the same inference. Bounded writer queues/finalization expose incomplete sessions rather than accepting them silently.

Review sidecars pin source hashes. Legacy recordings are read-only reuse inputs; replay regenerates compatible stock/features together. Old neutral/correction labels do not become verified C2 ground truth automatically.

Practice and checks use explicit wearing-session groups. Stop/Start and app restart retain identity; explicit headset reseating starts another. Shared practice/check origins and duplicate frames across origins are rejected. Checks do not train the head, and consumed checks cannot be reused for later candidates through this workflow. Lost legacy provenance limits independence claims.

Export verifies CPU PyTorch/ONNX Runtime agreement before a completed summary is written. Reports describe raw/emitted error, rest activation, range, clipping, and preservation. They remain `comparison_needed` / `recommended: false`; a user's **Keep** choice is separate. Software timestamps measure replay timing, not camera-to-photon latency.

## Selection, persistence, and fallback

`C2ModelManager` owns selection independently of the page. A trial lasts up to 60 seconds and ends when leaving the page. **Keep using this C2** writes the exact directory to `C2/selection.json`, retains it across navigation, and restores it after personal-model startup. **Return to Model C** clears the saved choice. Training does not activate a candidate.

Activation checks candidate hash, schema/contract metadata, personal C, blend, camera/backend, smoothing, and calibration. Identical stock and feature files may move with an app build. A kept candidate checks calibration only for corrected channels; collection and trials require the full context.

Audio Assist must be off for collection/training/trials because comparisons do not replay it. A kept C2 may use audio after visual correction. Invalid context, missing features, malformed tensors, or non-finite output retain Model C and display a reason. Swaps wait for in-flight inference before disposing the old candidate.

## Source map

| Code | Responsibility |
|---|---|
| `src/Baballonia/Services/Personalization/C2/C2Contracts.cs` | Tasks, output meaning, context/review records. |
| `C2Workspace.cs` in the same directory | Inventory, reviews, wearing identity, manifests. |
| `C2RuntimeContext.cs` | Candidate validation and live compatibility. |
| `C2CandidateCorrector.cs` | ONNX inference, preserved channels, fallback. |
| `C2ModelManager.cs` | Trial, Keep, restoration, Return to C. |
| `src/Baballonia/ViewModels/SplitViewPane/C2ViewModel.cs` | Recording and comparison flow. |
| `src/Baballonia/Views/C2View.axaml` | Controls and instructions. |
| `training/babble_personal/c2.py` | Prepare, train, export, compare. |

C2 also connects to the recorder, face pipeline, training service, startup, and Personalization page. It is distinct from the optional `src/BabblePersonalizer` experiment.

Headset usability, anatomical quality, independent comparison, real-provider performance, simultaneous throughput, storage/provider/camera stress, and downstream avatar behavior need separate evidence. See [validation](C2_VALIDATION.md).
