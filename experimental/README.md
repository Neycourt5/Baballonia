# Filed-away experiments

Code that was built, then deliberately taken out of the shipped application. It is kept here
because it worked and may be worth revisiting - not because it is scheduled.

Nothing in this directory is compiled. No project references it, and adding one is the only way it
can come back.

## eye-v2

Personal eye-tracking calibration, removed on 2026-08-15 so eye tracking behaves exactly as stock
Baballonia does.

- `Services/EyeV2/` - the V2-A personal mapper (per-eye affine gaze fit plus lid anchors), the V2-B
  geometry hybrid (`ClassicEyeGeometryExtractor`, pupil/eyelid estimation), the calibration service
  and its on-disk store, and the mode manager with its Default/V2-A/V2-B fallback chain.
- `Services/EyeOscDiagnostics.cs` - eye OSC send snapshots and the installed VRCFT module
  inspector, used by the Advanced eye debug panel.
- `Module/EyeExpressionRouter.cs` - the shared address table that let the app and the VRCFT module
  be held to the same ten-channel contract by a test.
- `Tests/` - the suites for all of the above.

### What stayed in the application

Two things from this work are *not* experimental and remain in the shipped build, because stock eye
tracking depends on them:

- **The twelve-to-six output projection** in `EyeProcessingPipeline`. The eye model in use emits
  twelve named outputs; the rest of Baballonia speaks the six-value legacy contract. The projection
  maps one onto the other *by output name*. Without it this model cannot be used at all - taking
  the first six values positionally would silently read the wrong ones.
- **The raw eye pipeline events**. Passive diagnostic taps with no subscribers in the shipped build.

### Bringing it back

Move `Services/EyeV2` and `Services/EyeOscDiagnostics.cs` back under `src/Baballonia/Services/`,
restore the three `AddSingleton` registrations in `App.axaml.cs`, re-add the mapper stage to
`EyeProcessingPipeline`, and widen the eye OSC table in `ParameterSenderService` past its six stock
channels. The git history before this removal has all of it working.
