# Verification record

Checked on 6 October 2026 in the development environment: macOS arm64, Python 3.12.14, Node 24 and the Codex in-app browser.

## Passed checks

- **18 Python tests** passed, including six procedural activity patterns, normalization and aspect correction, irregular/lower-rate sampling, tracking rejection, malformed input, local API behavior, split isolation and end-to-end forest training/evaluation.
- **4 JavaScript tests** passed for consecutive-label confirmation, full-cycle squat counting, jumping-jack state separation, resets and invalid geometry.
- JavaScript syntax checks and Python compilation succeeded.
- The actual local MediaPipe Lite model loaded in the browser and detected one person with **33 landmarks** in Google's public test image: https://storage.googleapis.com/mediapipe-assets/pose.jpg. This was a pose-inference smoke test using a still image, not a video recognition benchmark.
- The dashboard displayed synthetic squat recognition and growing complete-cycle counts. Jumping-jack simulation also produced counts. After adjusting the minimum accepted window to 12 frames, the corrected simulator showed the intended 10 samples/second and accepted continuous windows. This is not a measured webcam frame-rate claim.
- Dataset and Experiment correctly displayed an empty real dataset and pending empirical results. Synthetic recording is disabled in the interface and rejected by the collection/training backend.
- The 11-page report was rendered and visually inspected. The 13-slide deck was rendered and visually inspected, with package/layout checks and successful re-import. Native Microsoft Word/PowerPoint rendering was not tested.

The test suite emits a dependency deprecation warning from Starlette's test client and AnyIO. Tests pass; this warning concerns a future compatibility change in test tooling.

## Still requiring participant testing

- Camera permissions and continuous webcam capture on the presentation laptop.
- Real-video decoding for the particular input formats the team intends to use.
- Six-class classifier accuracy on actual training/validation/test participants.
- Repetition-count error, transition delay, robustness to camera viewpoint and poor illumination.
- Performance and setup on Windows/Linux. Instructions are provided, but this development session ran on macOS.

No empirical performance numbers were obtained from real participants. Procedural tests use temporary directories and do not leave a trained demonstration model or fabricated metric file in the project's real data, models or results folders.
