![Baballonia Promo](BaballoniaPromo.png)

# Baballonia — C2 experimental fork

An unofficial experimental fork of [Project-Babble/Baballonia](https://github.com/Project-Babble/Baballonia), maintained at [Neycourt5/Baballonia](https://github.com/Neycourt5/Baballonia). It adds personal face training, C2 candidate comparison, and custom eye and camera recovery behavior to the upstream XR eye/face tracker.

C2 has worked well for the maintainer's own setup. That is a personal observation, not evidence that it improves tracking for everyone. This preview shares the implementation so other users can train and assess their own models.

## Download and run on Windows

1. Open this fork's [Releases](https://github.com/Neycourt5/Baballonia/releases).
2. Choose **C2 Experimental Preview — 2026-09-15** (`c2-preview-2026.09.15`) and download **Baballonia-C2-Windows-x64.zip**.
3. Extract the entire ZIP into a folder.
4. Run **Baballonia.Desktop.exe**. Keep the DLLs, `Modules`, models, and other folders beside it.
5. Select and start your cameras on **Home**, then check their previews and crop settings.

The Windows x64 package includes its .NET runtime; installing Python is only necessary for training. A release asset is available once it appears on the release page. See the release notes and included `README.txt` for completed validation.

**A fresh profile starts with stock face and eye models.** The maintainer's personal Model C, C2 weights, tuned eye model, camera recordings, and calibration are not included. Existing installations use their own saved profile; extracting this ZIP does not create an isolated profile. See [profiles and privacy](docs/PRIVACY_AND_PROVENANCE.md#profiles-and-local-data) before running two copies at once.

## What this fork changes

| Area | Available behavior |
|---|---|
| Personal face models | Local recording and A/B/C training, model history and selection, guided teaching cues, and quick corrections. |
| C2 | Short reviewed jaw/smile tasks, separate candidate training, reports, a 60-second trial, explicit **Keep using this C2**, restart restoration, and **Return to Model C**. |
| Eyes | Eyelid and gaze synchronization, synchronized squint/wide-eye controls, adjustable squint strength, and configurable wink handling. |
| BlinkGuard | Optional post-blink gaze hold and stable-sample reacquisition intended to reduce reopening jumps. Off by default. |
| Expressions Alpha | Processing support for expression-capable eye models and eye calibration; the maintainer's tuned model is not bundled. |
| Speech and jaw | Optional microphone loudness assist after visual correction. The jaw-open curve control remains present; a known sender-key mismatch makes it ineffective on the standard `/jawOpen` path. |
| Camera recovery | Lifecycle/watchdog recovery for missing, stale, failed, and reconnected face/eye sources. Hardware-specific recovery still needs testing. |
| Diagnostics | Opt-in eye-stage traces and BlinkGuard counters/captures for investigating timing and mapping. |

These additions are experimental. Full synchronization can reduce independent eye movement; camera layout, model, calibration, receiver, and avatar rig all affect the result. [Eye behavior and checks](docs/EYE_GAZE_DIAGNOSTICS.md) describe the limits.

## What is C2?

**Model C** is a personal correction model that reuses visual features from the stock face network. **C2** is a separate correction stage trained against your working Model C, using short facial holds you explicitly review. It keeps unsupported expression channels with Model C and retains Model C for fallback.

C2 is specific to its recordings, base Model C, camera configuration, and output settings. Training creates a separate candidate and does not activate it. **Keep using this C2** saves your choice across pages and restarts; **Return to Model C** clears it.

Start with [Train your own Model C and C2](docs/C2_GETTING_STARTED.md). The [personalization guide](PERSONALIZATION_GUIDE.md) covers A/B/C, guided recordings, and audio. [C2 design](docs/C2_DESIGN_AND_DECISIONS.md) explains the implementation.

## Requirements

- Windows x64 for this package, with a supported face camera and/or eye-camera source and working drivers. Face and eye tracking can be used separately.
- A compatible graphics driver for DirectML acceleration; available execution providers depend on the machine.
- For source builds: Git, .NET SDK 10, and the dependencies below.
- For personal training only: Python **3.11–3.13**, preferably **3.13**, internet access for **Set Up Training Tools**, and disk space for recordings and CPU training packages. Python 3.14 is incompatible with the pinned PyTorch 2.7.1 training dependency.
- For headset instruction overlays: SteamVR and a compatible connected headset. Instructions are also available in the desktop UI.
- For VRChat facial output: VRCFaceTracking, a compatible Babble module, and an avatar with the required expressions. Native VRChat eye OSC provides eye look rather than full facial output.

### Hardware and game references

Upstream documents support for official/DIY Babble trackers and Vive Facial Tracker; eye sources include EyetrackVR and Bigscreen Beyond 2E. Other devices can need a separate camera bridge: [ReVision](https://github.com/Blue-Doggo/ReVision) for Vive Pro Eye, [Varjo Streamer](https://docs.babble.diy/docs/software/baballonia/varjo-streamer) for Varjo Aero, and [BrokenEye](https://github.com/ghostiam/BrokenEye) for HP Reverb G2 Omnicept/Pimax Crystal. Beyond 2E on Linux can use [go-bsb-cams](https://github.com/LilliaElaine/go-bsb-cams). These preserve upstream integration references; they are not a hardware test matrix for this preview.

See the upstream [quickstart](https://docs.babble.diy/docs/babbleofficaltracker) and [integration documentation](https://docs.babble.diy/docs/software/integrations), including Resonite and ChilloutVR. VRCFaceTracking is available on [Steam](https://store.steampowered.com/app/3329480/VRCFaceTracking/); its [module documentation](https://docs.vrcft.io/docs/vrcft-software/vrcft#module-registry) covers installation. [VRChat native eye OSC](https://docs.vrchat.com/docs/osc-eye-tracking) is a separate output route.

## Build from source

Build the Desktop project rather than the whole solution, which also contains mobile and separate VRCFaceTracking module projects.

```powershell
git clone --recurse-submodules https://github.com/Neycourt5/Baballonia.git
cd Baballonia
git checkout c2-preview-2026.09.15
git submodule update --init --recursive
powershell -ExecutionPolicy Bypass -File .\download_dependencies.ps1
. .\scripts\resolve-dotnet.ps1
& (Resolve-BaballoniaDotNet) build .\src\Baballonia.Desktop\Baballonia.Desktop.csproj -c Release
```

The tag becomes available when the preview is published. Dependency downloads require internet access. To test and package from a clean checkout:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-tests.ps1 -Configuration Release
powershell -ExecutionPolicy Bypass -File .\scripts\package-windows.ps1 -Version 0.0.0-c2-preview.20260915 -RequireCleanSource
```

The package script creates a self-contained Windows x64 folder and ZIP under a new `artifacts/release` directory. It retains native camera, OpenCV, Avalonia, and ONNX dependencies and includes `training` for in-app personal training. It does not use trimming or single-file publishing. [Validation and reproduction](docs/C2_VALIDATION.md) record tests, build results, and outstanding hardware checks.

## Repository map

- `src/Baballonia` and `src/Baballonia.Desktop`: application and desktop host.
- `src/Baballonia.Tests` and `training/tests`: regression and synthetic training tests.
- `training/babble_personal`: face training used by the app.
- `docs/`: current instructions, design, diagnostics, and validation.
- `src/BabblePersonalizer`, `docs/BabblePersonalizer`, and `experimental/`: separate optional work, outside the active in-app C2 pipeline. See their documentation before use.
- `artifacts/`, local profiles, recordings, and build outputs are excluded from Git.

## Privacy and validation

Personal recordings, training caches, models, calibration, and logs stay out of this repository and release package. Personalization runs locally. The inherited calibration upload integration is disabled unless explicitly configured; see [privacy and provenance](docs/PRIVACY_AND_PROVENANCE.md) for its configuration and scope.

Portable-PDB source hashes were compared with the preserved C2 Keep build. This establishes C# source correspondence, not a quality result or a fresh hardware test. [Validation status](docs/C2_VALIDATION.md) separates automated checks from headset/camera checks.

## Upstream credit and license

Baballonia is built by [Project Babble](https://github.com/Project-Babble/Baballonia) and its contributors. This fork preserves the [LICENSE](LICENSE), [CREDITS.md](CREDITS.md), third-party notices, and existing copyright attribution. It is not an official Project Babble release and is not relicensed.

For official upstream builds, use [Project Babble on Steam](https://store.steampowered.com/app/4091970/Project_Babble_Baballonia/) or [upstream Releases](https://github.com/Project-Babble/Baballonia/releases). Those downloads are distinct from this fork's C2 preview. See [CONTRIBUTING.md](CONTRIBUTING.md) for upstream contribution guidance.
