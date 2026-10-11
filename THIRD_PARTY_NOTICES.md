# Research references and third party notices

## Academic sources

**Primary research project (October 2026 revision):** Lukas Hedegaard, Negar Heidari, Alexandros Iosifidis. *Continual spatio-temporal graph convolutional networks*. Pattern Recognition 140 (2023), 109528. [DOI](https://doi.org/10.1016/j.patcog.2023.109528). The published paper is open access under CC BY. The implementation is retrieved from [continual-skeletons](https://github.com/LukasHedegaard/continual-skeletons), commit `a8fe2937a33f24cce65c1f8c2fc41081bceda721`. The pretrained activity classifier and neural architecture are the authors' work. Our work is the recovery/quality policy, reproducible stress-test runner, inference compatibility adapter and results interface.

The official repository displays an Apache-2.0 license badge but contains no standalone LICENSE file at the pinned commit. It is cloned locally by setup rather than vendored or relicensed here. Model weights are downloaded separately from the authors' linked Google Drive folder and checked against the recorded hash. Preserve and review the original authors' terms when redistributing their materials.

Kinetics-skeleton consists of OpenPose extractions distributed by the ST-GCN authors and linked from continual-skeletons. The experiment downloads a deterministic selection from the original archive, validates ZIP CRCs and SHA256 hashes, and retains original labels. Its manifest records exact public-source URLs and selected IDs. Public raw clips and weights are not committed; small pose excerpts in the dashboard illustrate the measured examples. Synthetic joint dropout/noise is an experimental intervention on real pose recordings, not a replacement activity dataset.

PyTorch and Continual Inference are third-party runtime dependencies. Their software licenses and notices remain applicable. `research/upstream.py` reads pinned, hash-checked model definitions from the external checkout and omits only training-framework scaffolding. Numerical equivalence is recorded with every completed evaluation.

**Earlier MediaPipe prototype references:**

1. Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang and Matthias Grundmann. **BlazePose: On-device Real-time Body Pose Tracking**. 2020. [arXiv:2006.10204](https://arxiv.org/abs/2006.10204). Foundation for the pretrained pose pipeline. The paper's device benchmarks are not measurements of this project.
2. Colin Arrowsmith, David Burns, Thomas Mak, Michael Hardisty and Cari Whyne. **Physiotherapy Exercise Classification with Single-Camera Pose Detection and Machine Learning**. Sensors 23(1), 363, 2023 issue, published 29 December 2022. [DOI:10.3390/s23010363](https://doi.org/10.3390/s23010363). Motivation for classifying landmark time series. PoseLab does not reproduce the paper's clinical exercise classes, dataset or CNN results.
3. Google AI Edge. [Pose Landmarker web guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js). Implementation reference for browser pose inference.
4. Scikit-learn. [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html). Classifier and grouped evaluation references.

## Upstream implementations and provenance

- MediaPipe: [official repository](https://github.com/google-ai-edge/mediapipe), Apache-2.0 for its code. The project uses `@mediapipe/tasks-vision` version 0.10.21, with downloads from jsDelivr's npm mirror. Refer to the upstream package for its notices and the official model page for model terms.
- Pose model: Google's official `pose_landmarker_lite/float16/1/pose_landmarker_lite.task` bundle. [Model overview](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). The setup script records exact download URLs and SHA-256 hashes in `static/asset_manifest.json`; a hash records local provenance, not an independent authenticity audit.
- NumPy and scikit-learn: BSD-3-Clause. FastAPI and Pydantic: MIT. Uvicorn: BSD-3-Clause. Transitive dependencies retain their respective licenses.
- The earlier collection interface, feature pipeline and repetition logic were authored for this project. The revised research project clones and reuses continual-skeletons as described above. Do not describe either the pretrained pose detector or the pretrained CoST-GCN weights as newly trained by the students.

Before redistributing packaged vendor files or model weights, retain the applicable upstream license notices. The source-only project archive excludes downloaded vendor/model assets and provides a setup script to retrieve them.
