# Eye behavior and diagnostics

This fork preserves expression-capable processing, eyelid/gaze/squint/wide-eye synchronization, squint strength, calibration, and optional BlinkGuard. The package starts with a stock eye model; the maintainer's tuned Expressions Alpha model and calibration are not included.

## Controls and defaults

Settings exposes **Eyelid Sync**, **Squint Sync**, **Wide-Eye Sync**, **Gaze Sync**, sync source, wink handling, and squint strength. Fresh defaults are eyelid sync 0.75, the other listed sync amounts 0, source Average, and wink detection on. Saved profile values take precedence.

Model-provided squint/wide-eye outputs are preferred to openness-derived outputs. Channels and calibration depend on the eye model. Stronger-eye synchronization is meaningful for expression magnitude; gaze treats Stronger as Average. A deliberate wink can release coupling when wink detection is enabled.

More synchronization reduces differences between eyes. It can also reduce independent motion/convergence cues; compare both social appearance and ordinary gaze before changing it.

## BlinkGuard

**Settings → BlinkGuard (post-blink gaze) → Enable BlinkGuard** is off by default. It holds gaze through closure, requires stable reopening samples, and eases toward the accepted direction. It aims to reduce reopening jumps while passing ordinary gaze unchanged in its Normal state.

BlinkGuard affects the social/VRCFT path. Dynamic foveated rendering reads an earlier unfiltered stream. Counters/capture controls expose state and rejected samples. The current camera-model path does not supply vendor confidence; validity, eyelid state, and temporal consistency drive the guard.

## Output and calibration checks

The native eye tuple is left pitch, left yaw, right pitch, right yaw. An asymmetric regression sentinel checks it. This does not establish anatomical camera identity, installed receiver behavior, or avatar mapping.

The calibration loader repairs exact legacy gaze ranges `(0,1,0,1)` to bipolar `(-1,1,-1,1)`, retaining explicit trims and other channels. This prevents that legacy default from clipping negative gaze; it is not a general recalibration.

## Opt-in stage trace

Enable **Settings → Advanced → Show Debug menu**. On **Debug**, choose **Start eye diagnostics**, reproduce the symptom, then **Mark eye issue** promptly and **Stop eye diagnostics**.

The bounded trace includes source sequence/receipt where available, raw/personal/filtered values, geometry, calibration, lid sync, BlinkGuard, gaze/expression sync, and queued sender output. **Mark** saves JSON under the active profile's `Diagnostics/Eyes`. It does not capture images or audio. Device/settings information can still be personal; review before sharing.

Software timestamps are not exposure times. Sender rows do not establish receiver timing; missing confidence/receiver identity is not guessed. Find the first stage of divergence and distinguish amplitude/saturation from timing. Trace collection adds work, so check throughput when enabling it.

## Hardware checklist

These require the actual headset and avatar:

- Small/large horizontal and vertical sweeps, each eye and both directions.
- Individual and simultaneous blink/reopen; gaze jumps and reacquisition.
- Near/far convergence and independent motion at the chosen sync amount.
- Gentle/full squint and wide-eye behavior; vary one control at a time.
- Saved gaze center/ranges after restart and behavior through the installed receiver.
- Disconnect/reconnect each camera; confirm recovery, including stale/failing sources.
- Simultaneous face/eye tracking, recording, and training for responsiveness/freshness.

Preserve working settings during comparisons. Report hardware/model/receiver context without sharing personal calibration or recordings.
