# Changes

Concise implementation history for the combined personalized Baballonia build.
Detailed agent handoff state lives in `WORK_PROGRESS.md`.

Branch: `combined`, based on `expr-opt` @ `d917e25` (Expressions Alpha + 2 post-Steam eye-brow
mapping fixes).

## Stage A — combined source and baseline

- Created the combined working tree as a local clone of the user's fork, branched from the
  Expressions Alpha (`expr-opt`) tip. No behaviour changes to the app.
- Restored the ability to build and run the test project (`d9bdb61`, cherry-picked from the fork):
  NuGet package-version alignment and a stale namespace import. Before this, `dotnet test` could
  not restore at all.
- Added `scripts/run-tests.ps1`, which runs the suite with the hardware-dependent tests (serial
  firmware boards, wifi provisioning, physical cameras, SteamVR overlay, external trainer)
  excluded. Those hang for minutes and crash the test host on a normal machine.

User-visible effect: none yet — this is the unmodified Expressions Alpha application.

## Stage B — personalized face tracking ported (`624992b`)

- The personalized face system from the v22 build now runs on the Expressions Alpha pipeline:
  personal model runtime and hot-swap, embedding-capable inference, dataset recorder, guided
  capture and cues, grimace/smile labs, VR calibration presenter, audio expression assist,
  training orchestration and Python scripts, and the Personalization page.
- The face pipeline converts between the Alpha keyed value map and the positional 45-expression
  vector personalization uses, so every previously trained model and recorded frame stays valid.
- The app now refuses to run personalization if the face model's expression order ever stops
  matching the personalization schema, instead of silently sending expressions to the wrong slots.

User-visible: the Personalization page is back, and the personal face model is applied again.
Eye tracking is unchanged.

## Stage C — face parity verified (`a84d0d5`)

- Added a parity test suite that checks the real installed models (stock, derived embedding,
  trained adapter) load, validate, and produce a personalized result that is finite, in range,
  and actually different from stock — with blend 0 still exactly stock.
- Confirmed by running the app: one process loads both the personalized face model and the tuned
  12-output eye model on DirectML, reporting "Personal model active: embedding_head_v1, blend 100%".
- `BABALLONIA_PROFILE` is carried over so development can run against a copied profile instead of
  your live settings. It is off by default and the shipped app behaves exactly as before.

User-visible: none directly — this is the checkpoint that the face half behaves like v22 before
eye work begins. **One manual check is still outstanding** (see WORK_PROGRESS.md).

## Stage D — combined baseline (`cb7d360`)

- Confirmed personalized face tracking and Expressions Alpha eye tracking running together in one
  process: 90.1 eye / 40.2 face inference fps, both models on DirectML. No execution-provider
  change is warranted; the two models coexist without contention.

## Stage E — capture freshness and hygiene (`56865e2`)

- Cameras now record when they last actually delivered a frame, and capture teardown is bounded so
  a wedged camera can no longer leak its device handle and make the next open fail with "camera
  opened somewhere else".

## Stage F — eye capture lifecycle fixes (`21bc00d`)

- Fixed the frozen-half bug: when one eye camera stopped, the other eye was silently stitched
  against its last frame forever. The stale half now expires after 500 ms and the working eye is
  mirrored instead.
- Fixed transposed camera Start/Stop, several per-frame image leaks, camera switching that ignored
  the address, stale temporal frames after a camera swap, and turning the smoothing filter off
  leaving it installed until restart.

## Stage G — automatic camera recovery (`d729935`, `3504ba9`, `87b4abf`)

- Face and both eye cameras now recover on their own after being unplugged, each independently: a
  reconnect on one never restarts the others. Backoff runs 1→2→4→8→10 s so a permanently absent
  camera cannot spin.
- Bigeye needed two extra detections beyond the usual timeout. Its DirectShow driver keeps
  reporting activity after unplug, so running cameras are also checked against physical device
  enumeration once a second; and it keeps returning the *same cached frame*, so image content is
  sampled four times a second and repeated frames count as a stall.

## Stage H — relax instead of freeze (`47e247b`)

- When all eye cameras are lost, the eyes now ease to neutral over 500 ms — gaze centered, lids
  open — instead of freezing on whatever expression you had. The face does the same, toward
  neutral. Each pipeline reacts only to its own cameras, and pressing Stop yourself does not
  trigger it.

## Stage I — no invalid values reach VRChat (`d71e4d9`)

- Fixed a divide-by-zero that could send NaN as vertical gaze during a full blink.
- Eye output is now guaranteed finite and in range at both the pipeline and the OSC boundary.
- Fixed calibration defaults being overwritten between eye and face, the native/DFR lid keys never
  matching, and swapped slider bounds.

## Stage J — guided per-eye calibration (`e1f5ee9`)

- New **Quick Eye Setup** on the Calibration page: relax, close gently, open wide, and the app
  learns what those poses mean for each of your eyes independently. Resting eyes then sit at 0.75
  openness, a gentle close reaches 0, and wide reaches 1.
- Gaze is recentered per eye at the same time. Robust percentiles are used rather than averages, so
  one stray blink cannot skew the result, and nonsensical captures are rejected with a retry rather
  than saved.
- Calibration persists across restarts and applies immediately without restarting the camera. Your
  existing manual Lower/Upper sliders still work as the final trim.

## Stage K — widen and squint (`93f8d76`)

- Your tuned eye model already predicts real per-eye Widen, Squint and Brow, so those are passed
  straight through, as before. They are only ever *derived* from lid openness for models that lack
  them, such as the stock eye model.
- The derived version is deliberately conservative: it will not change state on a single noisy
  frame, will not chatter on a threshold, and suppresses squint during blinks — otherwise every
  blink reads as two deliberate squints on the way past.

## Stage L — blink shaping (`6f9868f`)

- Blinks close instantly, with nothing added to the closing edge. Only the reopen is smoothed, over
  about 100 ms, which removes the flutter just after an eye opens without making blinks feel laggy.
- Per-eye, so winks are unaffected.
- OneEuro filtering is deliberately left alone pending real jitter/latency measurements.

## Stage M — diagnostics (`791f95a`)

- The Debug page now also shows: each camera's lifecycle state, reconnects this session, corrupt
  eye frames dropped, and the live eye output — calibrated openness with the raw value beside it,
  widen/squint per eye labelled with where they came from, and gaze per eye.

---

# Phase 2 — Personalized eye tracking

## Stage N0 — Quick Eye Setup would not save (`8472327`)

- **Fixed:** Quick Eye Setup rejected valid captures with "Left eye samples overlap." It required
  your eyelid to visibly open *further* than relaxed when you open wide — but on your tuned eye
  model, opening wide shows up in the Widen channel, and the lid is often already near its limit
  when relaxed. Calibration now accepts this and tells you it happened.
- Failure messages now quote the actual measured values, so a genuine problem says which of the
  three steps went wrong instead of just "overlap".
- Every calibration attempt logs all six measured anchors, so a bad setup can be diagnosed later.

## Stage S — eyelid sync and adjustable conjugate gaze (`a3d0aff`)

- **New: eyelid sync.** Both eyelids are now pulled toward each other, which removes the
  one-eye-lower half-squint and the lagging-eyelid look caused by the two cameras disagreeing.
  **Winking still works** — a deliberate wink releases the coupling instantly, and coupling
  re-engages once your eyes agree again. On by default; `AppSettings_EyeLidSyncAmount` (0 to
  disable, 1 for maximum).
- **New: conjugate gaze dial.** `AppSettings_EyeGazeConjugateAmount` controls how strongly both
  eyes are forced to point the same direction. Default 0 — no change from before, because vertical
  gaze was already shared between the eyes and horizontal already averaged with outward divergence
  blocked. Raise it only if the eyes still look like they cross or wander apart.

## Stage T — saved calibrations, and making saving visible (`e5d5a3f`)

- **Your calibrations are now kept.** Previously each Save overwrote the last one. There is now a
  dropdown on the Calibration page listing every calibration you have saved, plus
  **"Default (no calibration)"** so you can A/B against no calibration at all.
- **Delete** removes a bad capture. Deleting the one in use drops you to uncalibrated rather than
  silently switching to another, so what is running never changes by surprise.
- **Saving now shows you something.** It prints the six values it measured. Note that Quick Eye
  Setup deliberately does *not* move the Lower/Upper sliders — those are a separate manual trim,
  and the panel now says so.
- Any calibration you had before this update is kept and appears in the list as
  "Previous calibration".
- **Eyelid Sync** and **Conjugate Gaze** now have entries in Settings instead of being editable
  only by hand in the settings file.

## Stage N — corrector seam (`ef10de4`)

- Internal groundwork for personalized eye tracking: the pipeline can now accept a small trained
  correction layer between the eye model and everything downstream. Nothing is user-visible yet,
  and with nothing installed the output is unchanged.
- The stock (non-tuned) eye model is detected and left completely alone — it has no widen/squint
  channels for a correction layer to work with.

## Stage O — guided eye capture (`8954a6f`)

- Added the guided recording routine that personalized eye tracking will learn from: eighteen poses
  over about two minutes — look at a dot in nine directions, open wide, squint, close, wink each
  eye, and blink normally.
- The dot appears in your headset with everything else on the panel hidden, so you are looking at
  the target rather than reading instructions next to it.
- Recordings are tiny (no images are stored) and are tagged with the eye model they came from, so
  they can never be used to train against a different model.
- Not reachable from the UI yet — the button comes with the rest of the eye-personalization page.

## Stage P — gaze correction fitted from your recording (`067e87e`)

- A recorded guided session can now be turned into a per-eye gaze correction: six numbers per eye
  fitted from the nine dots you looked at, applied live to the model's output.
- It corrects offset, scale *and* rotation, so a headset that sits slightly differently than last
  time is handled.
- It refuses rather than guessing. If you did not follow the dots closely, if the correction needed
  would be implausibly large, or if the result would not actually beat no correction at all,
  nothing is saved and tracking is unchanged.
- A saved fit is tied to the eye model it came from. Retrain your eye model in VR and the fit
  switches itself off rather than silently applying numbers that no longer mean anything.
- Gaze only for now — eyelids, widen and squint are untouched.
- Still no UI; that comes next.

## Stage R — the eye personalization screen, and the VR overlay flashing fix (`2fcc7bb`)

- **New: Calibration → Personalized Eye Correction.** Record a ~110-second guided capture (follow a
  dot in nine directions, then wide/squint/closed/winks/blinks), fit a correction from it, and
  compare it against the unmodified model with one click.
- A strength slider and **Base (0%) / Personalized (100%)** buttons make the comparison instant,
  with a live twelve-row table showing every eye channel before and after.
- **Fixed: the VR calibration overlay flashing.** The overlay was redrawing into the same texture
  SteamVR was still reading, and re-uploading it about sixteen times a second just to nudge a
  progress bar. It now draws into an alternate buffer and holds progress-only updates to ten times
  a second. Instructions, phase changes and the dot still update immediately. This affects both the
  face and eye guided calibration; the older separate eye-calibration overlay is unchanged.
- The Debug page now shows how long the eye correction takes and how much it is changing.

## Stage U — squinting and wide-eye that actually register

- **The guided capture now fits your squint and wide-eye, not just gaze.** If your hardest squint
  only ever reads as a faint one, the fit rescales the channel so your own strongest expression
  reaches full strength.
- **Your resting face still reads as resting.** This is a curve anchored on the poses you actually
  performed, not a volume knob. Turning a hard squint up with a multiplier would turn a neutral
  face into a permanently slightly-squinting one; anchoring rest and the extreme separately is the
  entire point of doing it this way.
- **Gaze and expressions are now judged separately.** Previously nothing saved unless the gaze fit
  beat the base model — which would have thrown away a good squint fit from someone whose gaze was
  already fine. A gaze fit that does not help is now simply dropped, and the expression curves save
  anyway.
- **It refuses when it should.** If you performed the light and strong versions of a pose at
  similar intensity, the model cannot tell them apart, and a curve fitted through them would
  amplify noise into expression. It says so and leaves that channel alone. For someone whose model
  already behaves, leaving it alone is the right answer.
- **Beyond your strongest pose the curve holds flat** rather than extrapolating, so it can never
  overshoot into an expression you never demonstrated.
- Eyelids and eyebrows are deliberately untouched. Quick Eye Setup already owns eyelid openness,
  and two systems correcting the same channel is how a correction ends up fighting a calibration.
- The strength slider and the Base/Personalized buttons cover the new curves too, so the
  before-and-after comparison still shows everything at once.
- The summary tells you what changed in terms you can feel — *"Squint now reads 2.6× stronger at
  your strongest pose"* — instead of printing coefficients.
- Existing saved fits keep working. They are treated as gaze-only until you record again.

## Stage V — eyelid sync that actually engages, and a "better eye"

- **Fixed: eyelid sync did nothing, even turned all the way up.** The wink escape hatch was keyed on
  how much the two lids disagreed — but a big disagreement between the two cameras is the exact
  problem the coupling exists to fix, so it switched itself off for precisely the people who needed
  it. A resting asymmetry, or the first blink after one, could latch it off permanently.
- A wink is now recognised by its shape — one eye genuinely closed while the other stays
  genuinely open — instead of by disagreement. Two eyes that differ while both are open are two
  cameras arguing, not a wink. Winks still release instantly; blinks still stay coupled.
- **New: Better Eye.** If one of your cameras reads you better than the other, name it and the other
  eye follows it instead of the two being averaged. Averaging drags the good eye halfway toward the
  bad one. Naming an eye also means only that eye closing counts as a wink, since a wink is
  deliberate and will show up on the camera that tracks you properly.
- **New: Squint Sync and Wide-Eye Sync.** Same idea for expressions: if your squint reads well on one
  eye and poorly on the other, the other can copy it. Off by default — which eye reads better
  depends on how your cameras sit — and separate controls, because one channel can read well
  while the other does not. Both release during a wink.
- **The Debug page now shows whether the eyes are coupled or released.** A coupling that has switched
  itself off looked identical to one that was never turned on, which is why this bug was invisible.

## Stage W — the eye calibration that was silently doing nothing

- **Fixed: saved Quick Eye Setup calibrations were being ignored.** A rounding error in a range check
  rejected the very profile the setup had just built — in floating point, `1.0 + 0.05` minus
  `1.0` is `0.04999995`, which failed a "must be at least 0.05" test by a hair.
- Nothing announced this. A rejected profile falls back to passing values through untouched, so the
  setup reported success and tracking looked like it was working — it was simply uncalibrated.
- This is why one eye could sit lower than the other: the two cameras were sending raw values on
  different scales (one reading 0.55 with the eye shut, the other 0.71) with nothing normalising
  them. It also meant eyelid sync was averaging two numbers that did not mean the same thing.
- **You do not need to re-record.** The saved anchors were always correct; only the check rejecting
  them was wrong. Your existing calibration starts working on the next launch.

## Stage X — wide-eye reaches full again, and lids stay together under a squint

- **Fixed: wide-eye was capped.** Once the calibration started applying, opening your eyes as wide as
  possible only reached three-quarters of the range, so the avatar sat permanently half-lidded and
  the surprise expression never landed at full strength.
- The mapping reserves the top of the range for "wider than relaxed". That is correct when the
  eyelid channel can actually report it — but on a channel where relaxed already reads at the
  maximum, that reserved quarter can never be reached. Relaxed now maps to fully open on those
  channels, and wide-eye expression comes from the Widen channel as intended. Cameras that do have
  room above relaxed keep the old behaviour exactly.
- **Fixed: the eyelids drifted apart while holding a squint.** Blink shaping runs after eyelid sync
  and keeps separate state for each eye, so two lids that were coupled could come apart again —
  most visibly mid-squint. They are pulled back together afterwards.
- Winks are unaffected: still instant, still one-sided, and the re-coupling delay is unchanged.

## Stage Y — whichever eye sees the expression, both eyes do it

- **New: Better Eye → Stronger.** Averaging a squint that one camera reads strongly and the
  other barely sees produces a half-hearted squint on both eyes — syncing that makes the
  expression *worse*. Stronger takes whichever eye saw it more, so an expression only one camera
  notices still reaches both eyes at full strength, from either side.
- Stronger applies to squint and wide-eye only. On eyelids it would mean "more closed", which would
  let one mis-reading camera shut both eyes, so eyelids keep averaging.
- **New: Detect Winks can be turned off.** If you never wink, the escape hatch only costs you —
  every false positive is a moment of the lopsidedness you turned syncing on to remove.
- **Note on squint strength.** Restoring the full wide-eye range in the previous change also means
  everything below relaxed now reads about a third more open than it did while wide-eye was broken,
  so a squint moves the eyelids less than before. That is the cost of getting wide-eye back —
  and it is why Stronger squint sync is worth turning on.

## Stage Z — the two eyes now arrive together

- **Fixed: one eye lagged the other when looking sideways**, most obviously the eye looking outward.
  With typical settings the two eyes finished the same movement about **100 ms** apart.
- The cause was in the smoothing filter. It relaxes its smoothing based on how fast each channel is
  moving — sensible on its own, but the outward-looking eye sits near the corner of the camera's
  view where the model reports a smaller movement, so it was smoothed *harder* for the very reason it
  needed it least. Penalised twice for the same weakness.
- The two eyes now share one speed reading, taking whichever is moving faster, so a movement one eye
  sees clearly relaxes the smoothing for both. They start and finish together.
- This can only reduce smoothing, never add it, so nothing became laggier — there is a test that
  runs it both ways and checks exactly that. Face tracking is untouched.
- This was a long-standing issue, not something introduced recently; it simply became the most
  obvious remaining flaw once the eyelid problems were fixed.

## Stage AA — the eyes were not lagging, they were disagreeing

- Measured from a real recorded session rather than guessed at: the two eyes' horizontal movement
  lines up with **zero** delay between them, but agrees with each other only about **29%** of the
  time. They are not arriving late — they are estimating separately and disagreeing, which looks
  to you like one eye moving before the other.
- **New: Gaze Sync.** Pulls both eyes to look at the same place. At 1 they are perfectly conjugate;
  lower values keep some of the inward turn you make when focusing on something close.
- It deliberately does not reuse the older Conjugate Gaze setting, which acts too early in the
  pipeline — before each eye's resting position is corrected — so it averages two values
  that do not yet mean the same thing and can push the eyes further apart.
- Vertical gaze already looked fine because the two eyes were always being collapsed to a single
  shared value; horizontal had no such coupling until now.
- Off by default, since the evidence is from one pair of cameras.

## Stage AB — one direction driven harder than the other

- Measured from a real recording: both eyes travel about **1.8x further** one way than the other. The
  cause is that Quick Eye Setup places your gaze centre using the relaxed sample alone — and for
  this user that sample sits 4.5–5.7° away from the middle of their actual range, which
  makes one side of the range longer than the other.
- **No new correction was needed.** The personalized gaze fit already solves this, because it works
  out both the centre and the gain from all nine dots instead of trusting one sample. Fitted against
  the existing recording it takes the right eye from **16.3° to 3.4°** of error.
- **New: a warning after fitting.** Your saved gaze centre is measured without the correction and
  then applied on top of it, so after fitting the two stack and gaze sits off to one side. The panel
  now says so clearly and asks you to re-run Quick Eye Setup, which clears the notice. Switching
  between saved calibrations, or choosing none, clears it too.
- The warning only appears when a fit actually changes gaze — a squint or wide-eye fit leaves
  your centre alone and says nothing.

## Stage AC — opening your mouth a little no longer drops the whole jaw

- **New: Jaw Open Curve.** If opening your mouth slightly makes the avatar's entire jaw swing down,
  raise this. A small opening stays small, while opening wide still reaches full.
- It is an exponent rather than a volume control, and that distinction is the whole point: simply
  turning jaw open down far enough to fix the small opening would also mean the avatar could never
  open its mouth properly again. This bends how fast the jaw travels without moving where it ends.
- **1.00 is the default and changes nothing whatsoever** — not approximately, exactly. Try 1.50
  or 2.00. Below 1 makes the jaw more eager, for the opposite complaint.
- Applied before your own Lower/Upper calibration, so anything you have set there still works on top.
  Only jaw open is affected; no other expression changes.

## Stage BG — BlinkGuard: no more eye snap after a blink

- **New: BlinkGuard**, in Settings, **off by default**. It stops your avatar's eyes flicking
  somewhere random for a few frames after you blink.
- While an eyelid is down, gaze is held where it was rather than following whatever the tracker
  thinks it can see behind the lid. When the eye reopens, the first readings have to agree with each
  other before they are believed — so a single wild frame never reaches your avatar.
- **It is not a smoother.** Ordinary eye movement and fast saccades are passed through completely
  untouched; the filter only acts between the lid closing and gaze becoming trustworthy again.
- **It will not drag your eyes back.** If you genuinely look somewhere else while blinking, several
  agreeing readings at the new position win, and it eases across in about 80 ms.
- Gaze is never forced to centre, and your blink animation itself is untouched.
- Winks work: each eye has its own state, so a closed eye holding does not disturb the open one.
- Presets Subtle / Balanced / Strong, with every threshold exposed under Advanced.
- **Diagnostics built in**, because this needs tuning on real hardware: live state, counters for
  blinks, glitches prevented, timeouts and reacquisition times, plus a bounded capture you can start,
  stop and export as CSV.
- Foveated rendering is unaffected — it reads an earlier, unfiltered copy of the gaze stream.
