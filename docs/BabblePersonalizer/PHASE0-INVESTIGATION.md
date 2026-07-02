# Babble Personalizer — Phase 0 investigation

## Architecture and reusable behavior

Baballonia is a .NET 10 Avalonia application. Face frames are acquired through a capture abstraction, converted to Gray8, transformed by `ImageTransformer`, normalized by `MatToFloatTensorConverter`, evaluated with ONNX Runtime, passed through a One Euro filter, remapped by manual calibration, then sent over OSC/VRCFT.

The standalone application uses the compatible stack (Avalonia, OpenCvSharp, ONNX Runtime) but does not reference the Baballonia application project. `BaballoniaCompatiblePreprocessor` independently reproduces the small preprocessing contract so a public Personalizer release does not drag in Baballonia's host, models, senders, firmware, or UI.

There is no guided face-calibration wizard in this checkout. The Calibration page contains manual min/max sliders. Desktop overlay calibration is an eye/gaze capture routine and is not applicable to the lower-face model.

## Selected repository model contract

The checked-in model was inspected without modification.

| Property | Value |
|---|---|
| File | `src/Baballonia/faceModel.onnx` |
| SHA-256 | `14A907B116C885B1A383F4297569D305E752CE7499B18A6040FD5126A7859719` |
| ONNX opset | 17 |
| Input | `x.1`, float32, `[1, 1, 224, 224]` |
| Layout | NCHW, one grayscale channel |
| Output | `1210`, float32, `[dynamic batch, 45]` |
| Metadata | none |
| Expression-name resolution | validated legacy runtime sender/calibration ordering |

The application accepts metadata-driven counts when `blendshape_names` exists and exactly matches the output. A metadata-free model is accepted only for the known 45-output legacy contract; any other unresolved count is rejected.

## Exact preprocessing contract

1. Acquire a camera frame and request Gray8 in the Baballonia pipeline.
2. Validate ROI. Invalid, non-positive, out-of-bounds, or full-width/full-height ROI falls back to the whole image.
3. Clone the ROI.
4. If multi-channel, convert BGR to gray, or extract channel index 0 when the existing `UseRedChannel` option is enabled (the implementation is preserved exactly even though OpenCV normally calls channel 0 blue for BGR input).
5. Apply `ConvertTo` with the configured gamma value as a linear multiplier.
6. Rotate/mirror with Baballonia's affine matrix, or resize directly to 224×224.
7. Convert to float32 and scale by `1/255`.
8. Copy HWC pixels into a `[1,C,H,W]` NCHW tensor.

Windows uses DirectShow for OpenCV capture. Baballonia requests configured width, height, and FPS from the backend but devices may negotiate different values. The Personalizer warns through camera startup/failure status and stores requested settings with every session; negotiated-format reporting is a Phase 1 follow-up if the SDK is available.

## Canonical direct-output inventory

Every row below is a direct unsigned continuous `[0,1]` model output, is One Euro filtered, has manual min/max calibration, and is sent to the shown OSC destination. The source symbols are `ParameterSenderService.FaceExpressionMap`, `CalibrationService._faceExpressionMap`, `CalibrationViewModel`, and `BabbleExpressions.BabbleExpressionMap`.

| # | Canonical/model name | Category | Counterpart | Calibration strategy | Sender |
|---:|---|---|---|---|---|
| 0 | CheekPuffLeft | Cheek | Right | Left guided ramp | `/cheekPuffLeft` |
| 1 | CheekPuffRight | Cheek | Left | Right guided ramp | `/cheekPuffRight` |
| 2 | CheekSuckLeft | Cheek | Right | Left guided ramp | `/cheekSuckLeft` |
| 3 | CheekSuckRight | Cheek | Left | Right guided ramp | `/cheekSuckRight` |
| 4 | JawOpen | Jaw | — | Symmetric guided ramp | `/jawOpen` |
| 5 | JawForward | Jaw | — | Symmetric guided ramp | `/jawForward` |
| 6 | JawLeft | Jaw | JawRight (opposing) | Left guided ramp | `/jawLeft` |
| 7 | JawRight | Jaw | JawLeft (opposing) | Right guided ramp | `/jawRight` |
| 8 | NoseSneerLeft | Nose | Right | Left guided ramp | `/noseSneerLeft` |
| 9 | NoseSneerRight | Nose | Left | Right guided ramp | `/noseSneerRight` |
| 10 | MouthFunnel | Mouth/lip | — | Symmetric guided ramp | `/mouthFunnel` |
| 11 | MouthPucker | Mouth/lip | — | Symmetric guided ramp | `/mouthPucker` |
| 12 | MouthLeft | Mouth/lip | MouthRight (opposing) | Left guided ramp | `/mouthLeft` |
| 13 | MouthRight | Mouth/lip | MouthLeft (opposing) | Right guided ramp | `/mouthRight` |
| 14 | MouthRollUpper | Mouth/lip | Lower (related) | Symmetric guided ramp | `/mouthRollUpper` |
| 15 | MouthRollLower | Mouth/lip | Upper (related) | Symmetric guided ramp | `/mouthRollLower` |
| 16 | MouthShrugUpper | Mouth/lip | Lower (related) | Symmetric guided ramp | `/mouthShrugUpper` |
| 17 | MouthShrugLower | Mouth/lip | Upper (related) | Symmetric guided ramp | `/mouthShrugLower` |
| 18 | MouthClose | Mouth/lip | JawOpen (contextual) | Maximum-only | `/mouthClose` |
| 19 | MouthSmileLeft | Mouth/lip | Right | Left guided ramp | `/mouthSmileLeft` |
| 20 | MouthSmileRight | Mouth/lip | Left | Right guided ramp | `/mouthSmileRight` |
| 21 | MouthFrownLeft | Mouth/lip | Right | Left guided ramp | `/mouthFrownLeft` |
| 22 | MouthFrownRight | Mouth/lip | Left | Right guided ramp | `/mouthFrownRight` |
| 23 | MouthDimpleLeft | Mouth/lip | Right | Left guided ramp | `/mouthDimpleLeft` |
| 24 | MouthDimpleRight | Mouth/lip | Left | Right guided ramp | `/mouthDimpleRight` |
| 25 | MouthUpperUpLeft | Mouth/lip | Right | Left guided ramp | `/mouthUpperUpLeft` |
| 26 | MouthUpperUpRight | Mouth/lip | Left | Right guided ramp | `/mouthUpperUpRight` |
| 27 | MouthLowerDownLeft | Mouth/lip | Right | Left guided ramp | `/mouthLowerDownLeft` |
| 28 | MouthLowerDownRight | Mouth/lip | Left | Right guided ramp | `/mouthLowerDownRight` |
| 29 | MouthPressLeft | Mouth/lip | Right | Left guided ramp | `/mouthPressLeft` |
| 30 | MouthPressRight | Mouth/lip | Left | Right guided ramp | `/mouthPressRight` |
| 31 | MouthStretchLeft | Mouth/lip | Right | Left guided ramp | `/mouthStretchLeft` |
| 32 | MouthStretchRight | Mouth/lip | Left | Right guided ramp | `/mouthStretchRight` |
| 33 | TongueOut | Tongue | — | Symmetric guided ramp | `/tongueOut` |
| 34 | TongueUp | Tongue | TongueDown | Symmetric guided ramp | `/tongueUp` |
| 35 | TongueDown | Tongue | TongueUp | Symmetric guided ramp | `/tongueDown` |
| 36 | TongueLeft | Tongue | TongueRight | Left guided ramp | `/tongueLeft` |
| 37 | TongueRight | Tongue | TongueLeft | Right guided ramp | `/tongueRight` |
| 38 | TongueRoll | Tongue | — | Symmetric guided ramp | `/tongueRoll` |
| 39 | TongueBendDown | Tongue | TongueCurlUp | Symmetric guided ramp | `/tongueBendDown` |
| 40 | TongueCurlUp | Tongue | TongueBendDown | Symmetric guided ramp | `/tongueCurlUp` |
| 41 | TongueSquish | Tongue | TongueFlat (related) | Manual review first | `/tongueSquish` |
| 42 | TongueFlat | Tongue | TongueSquish (related) | Manual review first | `/tongueFlat` |
| 43 | TongueTwistLeft | Tongue | Right | Manual review first | `/tongueTwistLeft` |
| 44 | TongueTwistRight | Tongue | Left | Manual review first | `/tongueTwistRight` |

No direct face output is signed or binary in this selected contract. The dynamic model/profile types still reserve those treatments for a future metadata-defined model.

## Derived and sender-only aliases

The face tensor has no derived outputs. VRCFaceTracking fans direct channels out to Unified Expressions: `MouthFunnel` and `MouthPucker` each feed four lip quadrants; `MouthLeft/Right` feed upper and lower directional shapes; upper/lower roll feed paired lip-suck shapes; smile maps to mouth-corner pull; and `MouthClose` maps to `MouthClosed`. These are sender aliases and must not be appended to ONNX output.

## Separate eye, upper-face, and legacy fields

The eye pipeline is separate. Its active post-conversion output is six values: `LeftEyeX`, `LeftEyeY`, `LeftEyeLid`, `RightEyeX`, `RightEyeY`, and `RightEyeLid`. Older/default inference mappings also name eye widen and brow channels; calibration and VRCFT contain disabled or compatibility entries for widen, squint, lower, and brow. Desktop overlay capture has brow raise/angry, widen, squint, and pupil-dilate fields. None is part of this lower-face ONNX tensor, and none belongs in the mouth-camera wizard or generated replacement output.

## Correction and future ONNX composition

For sufficiently confident unsigned channel `i`:

`z = max(0, raw[i] - neutral[i] - deadZone[i]) / reliableRange[i]`

`mapped = monotonicPiecewiseLinear_i(clamp(z * sideScale[i], 0, 1))`

Conservative cross-talk may subtract a sparse thresholded term only when repeated held-out data supports it. Unresolved, binary, unsupported, or low-confidence channels are identity passthrough. Phase 3 will append standard ONNX operators after the renamed stock terminal output and restore the exact original output name, type, dimensions, batch behavior, ordering, and metadata.

## Application surfaces

1. Stock model selection and validation.
2. Camera selection, requested format, crop/rotation/mirroring, and raw/exact-input previews.
3. Dynamic guided calibration (Phase 2).
4. Representative/suspicious sample review (Phase 4).
5. Held-out stock-versus-personalized validation (Phase 2).
6. Export/install/backup/restore (Phase 3).
7. Live diagnostics and reports (Phases 2–4).

## Persistence

`BabblePersonalizerData` contains `Sessions`, `Profiles`, `Exports`, `Reports`, and `Backups`. A session contains `metadata.json`, `samples.csv`, `diagnostics.json`, `manual-labels.json`, and an optional `frames` directory. Metadata records the complete model contract/hash, ordered expression hash, camera/preprocessing configuration, application version, session ID, and frame-recording state. The bounded writer exposes dropped-sample and dropped-image counters.

The profile schema planned for Phase 2 stores entries by canonical name plus original index/name—not an undocumented positional array—and validates model hash, expression hash/order, tensor contract, parameter type, and camera/preprocessing hashes before use.

## Licensing, privacy, and risk

The repository uses the Babble Software Distribution License 1.0. Reused behavior and any distributed derivative need its license and applicable third-party notices (Avalonia, OpenCvSharp, ONNX Runtime). The original model is never included. All processing and storage are local; frame recording is opt-in.

Risks requiring gates: missing/incorrect metadata, backend-negotiated camera format differences, provider-specific ONNX support, preserving a dynamic batch dimension during graph composition, cross-talk overcorrection, and lack of genuine image labels. The current work calibrates output correction only; it does not fine-tune the visual backbone.
