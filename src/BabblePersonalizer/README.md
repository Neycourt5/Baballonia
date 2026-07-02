# Babble Personalizer

Standalone, local-first tooling for inspecting a user's own Baballonia `faceModel.onnx`, matching Baballonia camera preprocessing, previewing stock inference, and recording calibration streams.

The project never bundles a stock model. Select your locally installed model after launch.

## Quick start

1. Build `BabblePersonalizer.csproj` for Windows x64.
2. Launch the executable and select the existing stock `faceModel.onnx`.
3. Confirm the displayed tensor contract and expression list.
4. Select the mouth camera and match crop, rotation, gamma, and mirroring.
5. Compare the raw and exact inference-input previews.
6. Start guided calibration. Saving inference images is off unless explicitly enabled.
7. Follow three training ramps and one held-out ramp for each safely guided expression.
8. Review confidence, response curves, live comparison, and held-out results on Diagnostics.

Data is written beneath `%LOCALAPPDATA%\BabblePersonalizerData`. Nothing is uploaded. Phase 2 generates a versioned correction profile but does not generate or install a replacement ONNX yet; those actions remain disabled until strict graph and inference validation are implemented in Phase 3.

See [the Phase 0 investigation](../../docs/BabblePersonalizer/PHASE0-INVESTIGATION.md) for the model inventory and design contract.
See [the Phase 2 calibration design](../../docs/BabblePersonalizer/PHASE2-CALIBRATION.md) for fitting, validation, and profile details.
