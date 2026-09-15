# Built-in SteamVR overlay repair

Preview `c2-preview-2026.09.15.2` contains a candidate repair for the reported calibration-panel flicker. Automated rendering and lifecycle tests support the implementation; **a headset test must still confirm whether it resolves the reported symptom**.

## Scope

This covers `OpenVrCalibrationPresenter`, used for guided face calibration, C2 headset instructions, eye-personalization capture, and expression previews on Windows SteamVR. Home's legacy eye-calibration flow launches a separate `BabbleCalibration` executable; this renderer change does not repair or validate that external application.

The existing small instruction panel and larger gaze-target layout remain. The jaw-open curve stays opt-in and off by default; no model weights, recordings, calibration, or C2 contracts are rewritten by this overlay update.

## What changed

- Completed Skia frames enter a private D3D11 upload texture, then `CopyResource` publishes each whole image into one persistent shared texture on the adapter selected by SteamVR. After the copy, a D3D11 event query confirms completion before the presenter submits its `DXGISharedHandle` through `SetOverlayTexture`. The completion wait is bounded at 500 ms and fails on timeout or device errors. The presenter no longer reloads the overlay from raw CPU pixels on every update.
- The shared surface uses premultiplied RGBA pixels and matching compositor blending. It is detached from OpenVR before its graphics resources are released.
- Unchanged frames skip uploads. Cosmetic progress/countdown updates retain the 100 ms minimum interval; instructions, phases, buttons, and gaze-target changes submit immediately. Only successful submissions reset the consecutive-failure count.
- C2 keeps saved-frame counts in the desktop status. Headset instructions remain the task's stable text, with prepare/settle/hold, countdown, and progress carried separately. Phase transitions remain unlabelled until the presenter accepts the update and input polling succeeds. Failure or Cancel clears the cue immediately and excludes the unfinished attempt through the existing recording workflow.
- Completion callbacks carry a generation identifier so a delayed callback from an earlier session cannot close a newer panel.

The earlier CPU double buffer and throttling did not remove the repeated raw-image submission path. Source inspection establishes that behavior and the C2 text churn; it does not establish which compositor/driver event caused a particular visible flash.

## API basis

Valve recommends texture submission when possible over [`SetOverlayRaw`](https://github.com/ValveSoftware/openvr/wiki/IVROverlay::SetOverlayRaw). Its current [OpenVR header](https://github.com/ValveSoftware/openvr/blob/master/headers/openvr.h) specifies that a `DXGISharedHandle` is used directly by the overlay renderer and must receive atomic copy/resolve operations; it also defines `GetDXGIOutputInfo` for selecting the graphics adapter.

Microsoft documents that [`UpdateSubresource`](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-updatesubresource) captures the source bytes before returning, allowing immediate reuse of the CPU bitmap. [`Flush`](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-flush) submits queued work asynchronously; the documented event-query/GetData pattern supplies the additional GPU-completion check used here. That confirms the shared copy has completed, not that SteamVR has displayed it in a headset. The implementation makes no measured headset-latency or display-completion guarantee.

## Automated checks

| Test area | What it exercises | What it cannot establish |
|---|---|---|
| Presenter and fake OpenVR | Initial upload before show, stable resource reuse, duplicate/cosmetic scheduling, immediate target/instruction updates, failure handling, teardown, and stale completion callbacks. | SteamVR compositor behavior or visible flicker. |
| D3D11 through Windows WARP | Real shared-resource creation and readback through a second software D3D11 device; stable handle, completion before readback, repeated complete pixel copies, padded rows, CPU-buffer reuse/disposal, invalid input, and disposal. | Physical GPU compatibility, driver timing, or headset blending. |
| C2 instruction handoff | Stable task text, phase/progress fields, presenter-before-cue ordering, cancellation/failure clearing, and desktop capture. | User pose quality or correct avatar output. |

These tests use synthetic inputs. See the release notes for the executed test counts, clean-package result, and exact commit; `BUILD-INFO.json` identifies the packaged source. [Validation instructions](C2_VALIDATION.md) reproduce the suite.

## Headset checks

Use one running Baballonia instance and your own profile. Start SteamVR and the cameras needed for the selected capture; confirm their desktop previews are fresh. Keep your current jaw-curve switch and model selection unchanged for this comparison.

1. **C2:** with your own working Model C active, open Personalization's C2 panel and enable headset instructions. Record a short task. Observe GET READY, SETTLE, and HOLD, then repeat another task. The instruction should remain readable as the desktop saved-frame count increases; the panel should not blank or flash during the hold. Cancel once from the headset and verify that the unfinished attempt is excluded.
2. **Guided face calibration:** choose a short guided routine. Watch the instruction panel during countdowns and prepare/settle/hold/relax transitions while looking at your avatar. Exercise Retry, Skip, and Cancel where offered; verify that recording stops and live tracking resumes. Start another session promptly after a completion message and confirm that an earlier completion timeout does not close it.
3. **Eye personalization:** start its guided capture with headset presentation available. Observe the move between the instruction panel and gaze-target layout, then center/outer fixation targets. The dot should stay visible and stable within each hold. Check the pointer's Cancel action. This is the built-in eye-personalization capture, distinct from Home's legacy eye calibration.
4. Repeat the relevant checks with VRChat in the foreground and with the SteamVR dashboard opened and closed. If flicker remains, note the release version, GPU/driver and headset/SteamVR versions, which calibration flow was used, and whether it occurred during a steady hold or only at transitions. Review any recording affected by missing instructions before accepting it for training.

Passing these manual checks on one setup does not establish universal hardware support or improved model accuracy.
