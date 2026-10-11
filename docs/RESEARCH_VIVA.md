# Research viva guide

**Team roll numbers:** 23bcs019, 23bcs007, 23bcs021

**Mentor:** SUNIL CK

## A short explanation of the project

We use an existing pretrained graph neural network to recognize human actions from body-joint sequences. We investigate what happens when a pose detector temporarily loses joints. Our extension repairs short gaps using earlier observations and can reject predictions when evidence is insufficient. We compare the original model and three ablations on identical real pose sequences with controlled corruption.

## What does the base model do?

The base paper is [Continual Spatio-Temporal Graph Convolutional Networks](https://doi.org/10.1016/j.patcog.2023.109528), published in *Pattern Recognition* in 2023. Its code is at the [authors' repository](https://github.com/LukasHedegaard/continual-skeletons). A skeleton is represented as a graph: joints are nodes and body connections are edges. Spatial graph convolutions combine information across joints; temporal convolutions learn how that information changes across frames. CoST-GCN reuses intermediate computation as new frames arrive, reducing redundant work in streaming inference.

This project uses the authors' pretrained **single joint-stream CoST-GCN\*** checkpoint. The classifier accepts 18-joint 2D OpenPose data with x, y and confidence channels and produces 400 activity scores. We did not train this classifier ourselves. Our benchmark uses the weight-compatible clip path and records a numerical equivalence check against the continual definition. Its timing is batch processing per 300-frame clip, not live camera frames per second.

## What did we add?

The extension is a causal, bounded pose-recovery module and a calibrated rejection policy. Causal means it uses current and earlier observations, never future observations. For each joint/person slot, an alpha-beta tracker maintains position and velocity. Alpha adjusts the estimated position when a new observation arrives; beta adjusts velocity. Missing joints can be filled only for a bounded number of frames, with bounded displacement and a configurable confidence multiplier. Long gaps remain invalid. Reliable observed coordinates pass through unchanged.

Calibration selected **confident_hold_5**, with alpha **1**, beta **0**, and maximum gap **5 frames**. The selected setting has beta = 0, so the deployed repair holds the last observed position rather than extrapolating motion. Repair multiplies the last observed confidence by 0.9 for each missing frame. Repair can start only when the last real observation has confidence at least 0.5. The torso-jump guard resets a person slot above the configured threshold 0.15; it is not an identity tracker. This is a practical extension and a measured experiment relative to this base paper, not a claim that interpolation, tracking, or rejection has never been studied before.

## Why compare four variants?

1. **Original model** measures the unchanged checkpoint.
2. **Joint recovery only** isolates whether repaired input improves recognition.
3. **Rejection only** measures the effect of withholding uncertain predictions without repairing input.
4. **Recovery + rejection** measures their combination.

Without these ablations, an apparent improvement could come only from declining to answer harder cases. Recovery-only accuracy is therefore the main paired contrast. Rejection thresholds were selected separately for the original and recovered inputs on calibration data.

## How is the comparison fair?

We use 200 calibration clips and 400 different test clips. Selection is deterministic from the official validation partition, without selecting clips because predictions are correct. The classifier still predicts all 400 classes; 251 true classes occur in this sampled test set. All variants receive the same clips and identical corruption masks. We freeze settings before evaluating test labels. The 2 empty test clips remain in the analysis.

The conditions are original poses, short joint gaps, short whole-limb gaps, longer joint gaps and coordinate noise. The pose data are real, but the added masks/noise are simulated. This evaluates controlled pose corruption, not demonstrated robustness to physical occlusion in a webcam scene.

## What do the metrics mean?

- **All-input accuracy:** correct predictions divided by every test clip; abstentions count as errors.
- **Macro-F1:** the average class F1 over true classes represented in this subset, giving rare and frequent represented classes equal weight.
- **Coverage:** the fraction of all clips receiving a prediction.
- **Accepted accuracy:** correctness among accepted clips only; always report it with coverage.
- **Percentage-point difference:** recovery accuracy minus original-model accuracy, rather than relative percentage improvement.
- **Paired 95% bootstrap interval:** resample the same clip pairs to estimate uncertainty in the recovery-minus-baseline difference. An interval crossing zero does not establish a consistent improvement in this pilot.

The rejection thresholds are **0.080** for original inputs and **0.080** for repaired inputs. They maximize calibration accepted accuracy subject to at least 80% calibration coverage. That constraint does not guarantee 80% coverage on every unseen corruption condition.

## What should we say about the actual results?

Use the complete tables in [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md), including losses and unchanged results. Never quote only the condition with the largest gain. A short-gap repair method may help missing joints while leaving noisy observed coordinates unchanged or making some cases worse. A negative result defines where this implementation is unsuitable; it must remain in the presentation.

## What are the practical limits?

The pilot is smaller than the paper's full test set. Upstream validation clips may have influenced the authors' checkpoint selection. The repair uses confidence-ranked person slots rather than stable identities, so a slot can change people. Long gaps cannot be reconstructed reliably, pose-only classification misses object/context information, and measured batch timings exclude pose estimation and browser overhead. We do not claim full benchmark reproduction, state-of-the-art accuracy, medical suitability, or validated real-time webcam performance.

The earlier MediaPipe six-activity dashboard is a separate demo. The new 400-class research checkpoint expects OpenPose's exact topology and preprocessing; merely renaming joints or supplying MediaPipe landmarks would not establish correctness. A future webcam adapter needs separate data and evaluation.

## What is the research question?

Can bounded causal repair improve a fixed pretrained skeleton-action classifier under short pose-detection gaps, and what accuracy-versus-coverage tradeoff results from calibrated rejection? We answer that question with an unchanged model, controlled perturbations, ablations, held-out clips and uncertainty intervals.
