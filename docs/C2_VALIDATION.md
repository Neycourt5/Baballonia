# C2 preview validation

This document separates source evidence, automated mechanics checks, and hardware observations. The release notes and packaged `BUILD-INFO.json` identify the exact committed source and final clean-checkout build results.

## Preserved build and source

The preview source preserves C2 Keep's persistent selection, Audio Assist/navigation handling, calibration compatibility, eye range repair, diagnostics, and camera lifecycle/recovery work. The original working source/build and private profile were retained independently of the publication branch.

Portable-PDB checksums match all existing inspected main/Desktop/SDK/CaptureBin C# documents, with no mismatches. DLL-to-PDB identity and normalized checksums also match. See [provenance](PRIVACY_AND_PROVENANCE.md#c2-keep-correspondence). This establishes inspected source correspondence, not fresh package behavior.

## Automated results during preview preparation

| Check | Result | Scope |
|---|---|---|
| Python training suite | **137 passed**, 12 existing PyTorch ONNX deprecation warnings; 14.54 seconds. | Fresh Python 3.13 environment, public training requirements plus pytest, synthetic tests in `training/tests`; no personal profile/dataset. |
| Windows dependency bootstrap | Public pinned downloads verified; idempotent rerun verified. | Trainer, calibration overlay, firmware tool and required licenses. |
| C# Release suite | **739 passed, 9 skipped, 0 failed** out of 748; 44.1712 seconds. Build succeeded with warnings. | Isolated profile; hardware classes excluded by the script. The remaining skips require optional personal or derived model fixtures. |
| Synthetic calibration follow-up | **27 passed, 1 skipped, 0 failed**. | Targeted checks after replacing copied calibration vectors in three tests with synthetic values; skip requires an optional real model. Runtime code unchanged. |
| Initial Windows package smoke | Launcher, isolated stock-model startup, and UI navigation verified; ZIP layout/CRC checked. | DirectML initialized both public stock models. Onboarding, Home, Personalization/C2, Calibration, and App Settings were exercised with UI Automation. |

These are preparation-stage checks, not a substitute for the exact committed-package record. Counts overlap and must not be added together. Historical development-log counts are not reused as release verification. Synthetic tests exercise contracts, training/export mechanics, ownership, and fallback; they do not prove expression quality.

## Reproduce from clean source

Initialize declared submodules and use the committed dependency bootstrap. Build the Desktop project rather than the entire mobile/module solution. The package script requires the .NET 10 SDK and locates it with `scripts/resolve-dotnet.ps1`.

```powershell
git submodule update --init --recursive
powershell -ExecutionPolicy Bypass -File .\download_dependencies.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run-tests.ps1 -Configuration Release
powershell -ExecutionPolicy Bypass -File .\scripts\package-windows.ps1 -Version 0.0.0-c2-preview.20260915 -RequireCleanSource
```

The test script isolates synthetic model/profile writes and explicitly excludes tests requiring real camera/serial/firmware devices, network provisioning, external trainer operation, or SteamVR overlays. Record passed/failed/skipped counts, not only process success.

For a new isolated Python test environment:

```powershell
py -3.13 -m venv artifacts/python313-tests
$testPython = '.\artifacts\python313-tests\Scripts\python.exe'
& $testPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
& $testPython -m pip install -r training/requirements.txt pytest
$env:PYTHONPATH = (Join-Path (Get-Location) 'training')
& $testPython -m pytest training/tests -q
```

The package script creates a new folder under `artifacts/release`, publishes self-contained Release win-x64 without trimming/single-file mode, retains native/runtime dependencies and `training`, and removes only package development symbols/import libraries. It records source information and creates **Baballonia-C2-Windows-x64.zip**. It does not copy the user's personal profile or old build folder.

## Release record

The release notes record the exact commit/tag, final clean-checkout test counts/skips, build warnings, ZIP SHA256, and launch result. Packaged `BUILD-INFO.json` records source/build identity. Follow those records for the downloadable asset; the initial smoke above does not identify the final ZIP.

CI is appropriate only after a clean checkout demonstrably reproduces the dependency-complete package without private material. Manual release validation is retained when CI reproduction is not established.

## Application verification and remaining hardware checks

| Check | Evidence required |
|---|---|
| Launcher | Initial isolated smoke launched `Baballonia.Desktop.exe`; final asset verification is recorded in the release notes. |
| Default models | Verified: DirectML initialized both public stock models. Personalization reported **Actually active: Default Baballonia Stock**; C2 reported **Using now: Stock**, with no candidate. |
| Personalization UI | Verified through UI Automation: onboarding/Home, Personalization and C2 expander, Calibration, and App Settings navigation. This checks rendering/state, not a full training journey. |
| Personal C/C2 | Hardware/user check: train a compatible user-owned candidate; verify **Actually active**, Keep across navigation/restart, and Return to C. Personal candidate restoration was not exercised in the clean smoke profile. |
| Face camera | Not started during smoke: the preserved original C2 app was using the cameras. Verify fresh moving frames on the target machine. |
| Eye camera | Not started during smoke for the same device-ownership reason. Verify actual tracking/model/ranges on the target machine. |
| Recovery | Not hardware-tested during preparation. Disconnect/reconnect each real face/eye device and verify recovery without relaunch. |
| Package contents | Initial ZIP layout/CRC checked. Final native dependencies, Modules, training files, and byte/privacy guards are checked by the package workflow and recorded with the release. |

Fresh-profile startup does not establish personalized C2 activation. Test profiles must not silently import the maintainer's weights/calibration to manufacture that result.

## Hardware checks to perform

- Rest, small/full jaw opening, jaw/lip separation, gentle/full smiles, tooth visibility, speech, smile+speech, and return to rest on a compatible avatar.
- Separate normal headset wearings. Approximate cues and one user's preference do not establish general quality.
- Eyelid/gaze/squint/wide-eye sync, wink handling, convergence, and BlinkGuard reopening behavior; see [eye diagnostics](EYE_GAZE_DIAGNOSTICS.md).
- Headset instruction visibility, pause/retry, missed-task exclusion, and recording status while wearing it.
- Simultaneous face/eye throughput/freshness during recording/training and recovery for every real camera/backend.
- Installed VRCFaceTracking receiver and avatar mapping; a source module or queued OSC tuple does not prove receiver behavior.

Do not mark hardware checks passed based on compiling code or synthetic tests. No physical latency, universal hardware compatibility, or model-quality claim follows from the automated suite.
