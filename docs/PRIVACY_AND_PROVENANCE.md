# Privacy and source provenance

## Public model strategy

The preview distributes stock/default runtime models and training source. It excludes the maintainer's personal Model C, C2 candidate, tuned eye model, recordings, caches, calibration, and settings.

C2 is trained against a particular personal C and camera/output context. Other users train their own; personal weights are not required for stock startup. Release preparation does not replace models/settings in the preserved original development setup.

## Profiles and local data

Ordinary Windows runs use `%APPDATA%\ProjectBabble` for persistent data. Personal material includes `PersonalDataset`, `EyePersonalDataset`, `PersonalTraining`, `C2`, `Models`, settings, calibration, and diagnostics. Candidate directories can contain prepared session arrays as well as ONNX weights and reports; they are not automatically safe to share.

Normal copies can share the same profile. To try a build independently, set a name in the same PowerShell window before launching:

```powershell
$env:BABALLONIA_PROFILE = 'c2-preview-test'
.\Baballonia.Desktop.exe
```

This uses `ProjectBabble-c2-preview-test` without existing personal models/settings. `BABALLONIA_DATA_ROOT` is honored only with a named profile and is intended for isolated development/tests. Avoid running two copies against the same profile.

## Publication exclusions and uploads

Generated builds, `artifacts`, local profiles/datasets, recordings/frames/video, training checkpoints/caches, diagnostics/logs/crash dumps, temporary environments, and secret configuration are excluded from publication. Exclusion does not require deleting local work.

Required upstream models/resources, synthetic fixtures, useful source/scripts/docs, licenses, credits, and copyright notices remain. Copied legacy trainer samples and checkpoints (`footage_unlabeled.npz`, `gaze_model_best.pt`, `model_best.pt`) are excluded from new commits and the ZIP: the pinned standalone trainer does not read them. Original local copies remain available. Numeric calibration regressions use synthetic examples rather than copied personal settings. A filename alone does not establish whether an asset is private.

Personalization records and trains locally. Training-tool setup downloads dependencies. The separate inherited eye-calibration upload integration requires external `BABALLONIA_UPLOAD_ACCESS_KEY_ID` and `BABALLONIA_UPLOAD_SECRET_ACCESS_KEY` configuration; no upload credentials are bundled. Without both, upload controls are unavailable and no upload client/file read/upload is attempted. User consent remains required even when configured. This uploader is distinct from local C2 training.

Audit logs and machine-specific preservation inventories are kept privately outside the public source. A source/package review is not a guarantee about every historical commit. Do not publish profiles, raw logs, recordings, or personal model folders as bug reports.

## C2 Keep correspondence

The preserved build was rebuilt on September 10, 2026: main DLL at 22:33:55 and Desktop DLL/EXE at 22:34:08 (recorded local filesystem times). Its directory was reused for Keep, Audio Assist/navigation, and calibration follow-ups; the directory name alone is not a source revision.

Its matching source is based on `f359e704fb5484d89fd0455f1fd217965db9b3fd` **plus preserved uncommitted C2 and related changes**. That historical commit alone does not contain C2.

Read-only portable-PDB inspection found:

| Assembly | Existing source documents matching recorded hashes | Generated documents absent from disk | Existing-source mismatches |
|---|---:|---:|---:|
| Baballonia | 234 | 130 | 0 |
| Baballonia.Desktop | 15 | 1 | 0 |
| Baballonia.SDK | 5 | 0 | 0 |
| Baballonia.CaptureBin.IO | 6 | 0 | 0 |

CodeView identifiers/timestamps and normalized SHA256 PDB checksums match all nine Baballonia DLLs, including camera Modules. This binds the source checksums to the preserved binaries. Missing documents are generated sources under `obj`.

These checks establish correspondence for inspected C# inputs. They do not independently reconstruct XAML/resources, Python, build properties, or dependency downloads. Fresh build/tests and hardware review are separate evidence in [validation](C2_VALIDATION.md).

## Synthetic C2 test fixture

`src/Baballonia.Tests/Assets/PersonalModels/c2Adapter.onnx` was inspected as an ONNX graph. All 57,600 matrix weights are zero; all 45 biases are zero except one 0.1 adjustment. It contains Gemm/Add/Clip/Identity operations, bounds 0/1, and `fixture-context` metadata, with no external tensor data or training state. It is a deterministic mechanics fixture, not a face-trained model.

SHA256: `8490599550ffecec14c2c6bc3d1af227385b22ebd18fceb17a3aeee70c70adf7`. Tests use it for named output selection, blending, preservation, and incompatible-context fallback.
