# Experiment protocol

## Scope and hypothesis

The system recognizes six deliberately limited activities from full-body, single-person RGB video: standing, sitting, marching in place, squats, jumping jacks and waving. Test whether temporal features improve macro F1 compared with mean pose features on participants never used for training. A negative result is valid and should be explained.

## Collection plan

Recommended pilot: six consenting adults, five eight-second clips per activity, two sessions with the same participant ID, and a fixed camera. This produces 180 clips and 24 minutes of recorded activity, excluding setup and breaks. It does not guarantee sufficient data or accuracy. A larger test group would strengthen the study.

Assign P01–P04 to training, P05 to validation and P06 to testing before final collection. These are example IDs, not real enrolled participants. Record all six activities in every split. Keep every session and viewing angle from the same participant in that participant's split. Do not rename a person to make extra subjects.

Ask permission to save body landmarks for an academic prototype. Describe where files are stored and who will see them. Avoid names, faces in saved evidence, and identifying surroundings. The app stores x, y, z, visibility and timestamps, not source video. If you choose to retain separate videos for rep-count evaluation, obtain permission for those too and keep them local. Participants can decline any movement or stop at any time.

## Label definitions

| Label | Recording instruction | Common ambiguity |
|---|---|---|
| standing | Stand comfortably with arms down and little motion | Waiting between exercises also looks like standing |
| sitting | Sit still on a stable chair with knees visible, slight diagonal view | Frontal projection can hide bent knees |
| marching | Alternate lifting feet/knees without traveling | Very small steps may resemble standing |
| squats | Repeat comfortable bend-and-return cycles | Sitting down and squat transitions can resemble each other |
| jumping_jacks | Repeat closed/open/closed arm-and-leg cycles | Occluded wrists or cropped feet invalidate tracking |
| waving | Wave one raised hand side to side while standing | Arms-up stillness is outside the defined label |

Begin the activity before the countdown finishes. Do one label per clip and use varied but comfortable speeds. Keep source frame rate at least 15 FPS where possible; the app samples at approximately 10 Hz. Record variations in clothing, distance and moderate lighting across clips. Avoid a camera change that occurs only in one class, as the model may learn that cue.

## Quality and windowing

Core joints are shoulders, elbows, wrists, hips, knees and ankles. Visibility must be at least 0.5 at all core joints in a valid frame, and the joints must remain approximately within the image bounds. The torso must have adequate projected length.

Windows are two seconds long with a 0.5-second training stride. Each accepted window needs at least 12 valid frames, at least 85% valid frames, at least 1.8 seconds of valid span, a recent valid frame, and no valid-frame gap above 0.35 seconds. Features resample to 21 evenly spaced time points. Short tracking gaps can be interpolated; long gaps are rejected.

The collector rejects a whole clip with fewer than two valid windows. The dataset therefore reflects a full-body recording protocol. Record rejected collection attempts in your log if reporting robustness. Within accepted clips, rejected evaluation windows count as unknown errors, so the reported accuracy denominator includes them.

## Reproducible comparison

Model A uses 18 window-mean pose features. Model B uses 72 features: mean, standard deviation, range and mean absolute derivative for each pose feature. Both use 200 trees, max depth 12, minimum leaf size 2, balanced class weighting and seed 42. Both share the same clips, quality checks and splits. This comparison tests adding temporal variation to window-mean geometry, not a single-frame classifier against a video classifier.

The validation set selects a threshold from 0, 0.45, 0.55, 0.65 and 0.75 for each model, maximizing macro F1 among candidates with coverage at least 70%, or among all candidates if none meet that constraint. Ties prefer greater coverage and then lower threshold. A chosen threshold of zero is legitimate: rejection did not improve that validation objective. Do not silently force a favorable threshold based on test performance.

Run the Experiment tab once the dataset is frozen. The script computes four test variants: A raw, B raw, A with its validation threshold, and B with its validation threshold. All variants retain quality rejection. Inference uses the temporal model and its selected threshold.

Report accuracy, six-class macro F1, per-class precision/recall/F1, confusion matrix, coverage and accuracy among accepted predictions. For accuracy and macro F1, unknown predictions count as missed true activities. Accepted-only accuracy must be paired with coverage. The unknown column is not a seventh trained class. Low model votes do not guarantee reliable out-of-distribution detection.

Preserve run ID, Python/package versions, clip hashes, participant IDs by split, class counts, and camera conditions. Window metrics are descriptive because adjacent windows overlap. Do not report confidence intervals treating every window as an independent person. If time allows, predefine multiple participant folds and report variability, keeping model selection separate from each test fold.

## Additional measurements

The automated classification experiment does **not** evaluate smoothing or repetition counters. For that, use separately consented continuous clips with hand-counted full cycles and labeled activity transitions.

1. Record three 20–30 second clips each for squats and jumping jacks, after recognition is stable. Manually count completed cycles from the same defined start time as the app.
2. Record manual count, app count and absolute error per clip. Report mean absolute error over clips and explain missed first cycles or occlusion resets.
3. For continuous mixed activity clips, annotate transition times. Record label changes and transition delay before and after confirmation. Exported session CSV contains raw and displayed activity labels for this comparison.
4. Record browser, CPU, resolution and observed pose samples/second. The prediction request time includes local request overhead and Python features/classification but excludes pose detection and the two-second window delay. Do not call it full end-to-end latency.
5. Include a poor-lighting/partial-occlusion demonstration and unfamiliar movements. Describe failures instead of claiming universal unknown detection.

## Results worksheet

| Item | Value to enter after real testing |
|---|---|
| Dataset participants / clips / usable windows | Pending collection |
| Training / validation / test participant IDs | Pending collection |
| Mean-pose accuracy / macro F1 | Pending experiment |
| Temporal accuracy / macro F1 | Pending experiment |
| Selected thresholds and coverage | Pending experiment |
| Rep count mean absolute error | Pending annotated videos |
| Label switch rate and transition delay | Pending annotated videos |
| Pose sampling rate and laptop specification | Pending target laptop |

Do not paste simulated fixture scores into this worksheet. Do not claim improvement until the controlled comparison supports it.
