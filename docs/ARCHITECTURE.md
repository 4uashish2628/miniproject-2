# Architecture and code walkthrough

## Separation of responsibilities

The browser owns camera permission, video decoding, MediaPipe inference and the overlay. The Python process owns geometric features, rules, training, classification and local landmark storage. Both communicate over loopback HTTP. Model and WebAssembly assets load from the local server after the setup download. Browser inference is sampled near 10 Hz and runs synchronously on the main thread, a deliberate simplicity tradeoff. Moving it to a Web Worker is future work if interface responsiveness becomes a problem.

## Landmark geometry

MediaPipe returns normalized image coordinates with x measured relative to width and y relative to height. Using those directly in Euclidean angles distorts geometry when the aspect ratio is not one. The feature code first multiplies x by width/height. It centers at the midpoint of the hips and divides by the shoulder-midpoint-to-hip-midpoint distance. Translation and uniform scale mostly cancel, but rotation and projection do not.

For joints A, B and C, the angle at B is arccos of the normalized dot product of A−B and C−B. Numerical clipping keeps the cosine within [−1, 1]. Degenerate vectors trigger rejection. Angles are divided by pi for classifier inputs and converted back to degrees in the interface.

## Eighteen features per frame

1–8: left/right knee, hip, elbow and shoulder angles. 9: ankle horizontal separation. 10–11: wrist height relative to its shoulder. 12: hip height relative to the mean ankle height. 13: trunk tilt relative to image vertical. 14: signed knee-height difference. 15–16: left/right wrist horizontal offset from the hip center. 17–18: left/right ankle vertical offset from the hip center. Positions use torso-length units. The stored z coordinate is excluded from this version.

The `static` comparison is the mean of those 18 features across the window. The `temporal` feature vector concatenates 18 means, 18 standard deviations, 18 peak-to-peak ranges and 18 mean absolute derivatives per second. The derivative uses timestamps after uniform resampling, so slightly irregular browser sampling does not silently change its units. This is an order-sensitive speed summary, but it discards detailed trajectory ordering; an LSTM could be a future extension.

## Temporal behavior

Predictions are requested about every 0.5 seconds over the recent two seconds. Three consecutive accepted labels confirm an activity. During a label change the UI displays Confirming, and an unknown prediction clears confirmation immediately. This trades approximately one extra second of confirmation delay for less flicker. It does not inherit an old confident label indefinitely.

Repetition counters use frame-level knee angles or arm/leg geometry once the activity is confirmed. Squats must reach an upright state, bend below the lower threshold, then return upright. Jumping jacks must reach the closed state, open arms and legs, then close. Separate upper/lower thresholds provide hysteresis. A minimum 0.6-second interval suppresses very rapid double counts. Lost tracking or an activity change resets the partial-cycle state while preserving completed totals. The first unarmed partial cycle does not count.

## Stored artifacts

Each clip JSON includes a coded subject, label, source, consent, image aspect ratio and frames. Every frame stores a timestamp plus 33 × 4 landmark values. Filenames are generated UUIDs. The API validates shape, finite values, ranges and strictly increasing timestamps. A synthetic source is rejected by the collection endpoint and real training loader. These source tags rely on truthful labeling and are not authenticity proofs.

Training artifacts contain the forest, feature version, selected threshold and full evaluation record. Only load models generated locally. Joblib uses pickle-based serialization. Run directories preserve prior metrics, per-window predictions and manifest hashes; clips themselves are not duplicated. Keep a separate immutable backup of the frozen landmark dataset if you need exact replay later.

## API endpoints

| Method and route | Purpose |
|---|---|
| GET /api/status | Model and pose-asset readiness |
| POST /api/predict | Quality-gated rule or forest inference |
| POST /api/clips | Save a permitted real landmark clip |
| GET /api/dataset | Per-participant, per-activity counts |
| POST /api/train | Fit both models, choose validation thresholds, evaluate test participants |
| GET /api/results | Latest saved results or pending state |
| GET /api/results/download | Download latest metric JSON |
| GET /api/demo/{label} | Synthetic visualization input only |

The server is a local single-user academic application, not an internet deployment. The default host is 127.0.0.1, allowed Host values are restricted and cross-origin mutation requests are rejected. It has no user accounts or production authorization system. Do not expose it publicly.
