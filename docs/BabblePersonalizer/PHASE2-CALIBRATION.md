# Babble Personalizer — Phase 2 calibration and profile format

## Guided routine

The routine is generated from the selected model's validated direct-output inventory. It records:

1. 15 seconds of relaxed neutral face.
2. Three slow training repetitions for each safely guided parameter: neutral → 25% → 50% → 75% → maximum → 75% → 50% → 25% → neutral.
3. Rest and transition samples between poses.
4. 30 seconds normal speech.
5. 20 seconds exaggerated speech.
6. Eight seconds of held-out neutral.
7. A fourth, held-out ramp for every guided parameter.

Requested intensity is stored as a soft pacing cue. It is not treated as authoritative ground truth. `ManualReviewOnly` and unresolved parameters receive no invented instruction and remain passthrough.

The UI supports cancellation, retrying the current parameter, skipping it, progress reporting, live stability/noise, dropped-item counters, and optional local inference-image recording.

## Robust profile generation

Training and held-out samples are separated before fitting. For each direct parameter, the profile contains:

- canonical and original model names/index;
- category, type, range, side relationship, and strategy;
- neutral median, P05, P95, scaled MAD, and dead zone;
- robust P05/P95 active bounds and active range;
- repetition consistency, ramp monotonicity, hysteresis, and stability;
- accepted/rejected counts and rejection reasons;
- monotonic piecewise-linear response curve;
- conservative left/right scale;
- confidence and explicit passthrough reason.

Single-frame spikes and robust outliers are rejected. Maximums use percentiles, never a single peak. Corrections require confidence ≥ 0.35; otherwise the original stock value passes through unchanged.

Cross-activations are recorded only across different expression categories and require three supported repetitions. Phase 2 does not apply these terms: strong candidates are retained for Phase 4 manual review, preventing premature suppression of natural combinations.

## Runtime preview correction

For an enabled unsigned continuous parameter:

1. subtract neutral median;
2. apply the noise-aware dead zone;
3. divide by the robust usable range;
4. apply bounded side scaling;
5. interpolate the monotonic response curve;
6. clip to the parameter's declared range.

Binary, unresolved, unsupported, disabled, or low-confidence parameters are identity passthrough. The direct-output count and ordering never change.

## Held-out validation

The fourth repetition is never used to fit the profile. Validation compares stock and personalized outputs for target error, monotonicity, and isolated one-frame spike rate. The application reports improvement only when:

- at least three parameters have held-out samples;
- the aggregate personalized score clears the stock score by more than 0.01;
- more than half of evaluated parameters improve without worse monotonicity or spike rate.

Guided intensity remains a soft reference, so this validates repeatability and response behavior—not anatomical ground truth.

## Profile JSON

Profiles are stored under `%LOCALAPPDATA%\BabblePersonalizerData\Profiles` with schema version 2. The root records model input/output contracts, stock-model hash, ordered expression-list hash, camera configuration/hash, session ID, correction-method version, per-parameter entries, cross-activation diagnostics, and the held-out report.

Loading rejects wrong schema, corrupt JSON, model-hash mismatch, expression-list mismatch, dimension mismatch, name/order mismatch, or parameter-count mismatch. Camera/preprocessing mismatch is a visible warning because it does not change expression identity but can invalidate calibration quality.

## Diagnostics

The diagnostics screen shows live stock/personalized values, recent selected-expression history, neutral/noise/dead-zone values, robust range, side scale, response-curve points, confidence, stability, consistency, hysteresis, accepted/rejected counts, cross-activation observations, and held-out validation results.

Phase 2 still does not create a replacement ONNX. Combined graph generation and installation remain gated on Phase 3 contract validation.
