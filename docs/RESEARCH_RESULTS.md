# Measured research results

Generated from `results/research/comparison.json`, measured at **2026-10-11T23:17:40.737167+05:30**. Source SHA256: `346808497b94c5ee389f38629a4e46429b4895611430b73fae048e29643986b9`.

The pilot uses **200 calibration clips** and **400 separate test clips** from real Kinetics-skeleton (ST-GCN OpenPose extraction) data. The classifier retains **400 output classes**; **251 true classes** occur in this test subset. **2 empty clips** are retained.

Base paper: [Continual Spatio-Temporal Graph Convolutional Networks](https://doi.org/10.1016/j.patcog.2023.109528), *Pattern Recognition* 140 (2023), 109528. [Authors' code](https://github.com/LukasHedegaard/continual-skeletons).

## Complete measured table

Accuracy and macro-F1 count abstentions as incorrect. Accepted accuracy is conditional on receiving a prediction and must always be read together with coverage. Macro-F1 averages the true classes present in this test subset; it is not a 400-class macro average.

| Condition | Variant | All-input accuracy | Macro-F1 | Accepted accuracy | Coverage | Processing ms/clip |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original poses | Original model | 28.75% | 27.33% | 28.75% | 100.00% | 31.91 |
| Original poses | With joint recovery | 29.00% | 27.98% | 29.00% | 100.00% | 47.15 |
| Original poses | With rejection only | 28.50% | 27.98% | 33.93% | 84.00% | 31.91 |
| Original poses | Recovery + rejection | 28.75% | 28.42% | 33.82% | 85.00% | 47.15 |
| 25% joint dropout · 5-frame blocks | Original model | 11.50% | 8.78% | 11.50% | 100.00% | 30.87 |
| 25% joint dropout · 5-frame blocks | With joint recovery | 15.25% | 12.66% | 15.25% | 100.00% | 43.58 |
| 25% joint dropout · 5-frame blocks | With rejection only | 11.50% | 9.15% | 15.03% | 76.50% | 30.87 |
| 25% joint dropout · 5-frame blocks | Recovery + rejection | 15.00% | 12.64% | 19.11% | 78.50% | 43.58 |
| 25% limb dropout · 5-frame blocks | Original model | 15.25% | 13.43% | 15.25% | 100.00% | 31.04 |
| 25% limb dropout · 5-frame blocks | With joint recovery | 18.50% | 16.34% | 18.50% | 100.00% | 43.68 |
| 25% limb dropout · 5-frame blocks | With rejection only | 14.50% | 13.31% | 18.18% | 79.75% | 31.04 |
| 25% limb dropout · 5-frame blocks | Recovery + rejection | 17.75% | 16.44% | 21.98% | 80.75% | 43.68 |
| 25% joint dropout · 15-frame blocks | Original model | 14.75% | 13.21% | 14.75% | 100.00% | 31.06 |
| 25% joint dropout · 15-frame blocks | With joint recovery | 15.00% | 12.95% | 15.00% | 100.00% | 43.49 |
| 25% joint dropout · 15-frame blocks | With rejection only | 14.50% | 13.09% | 18.53% | 78.25% | 31.06 |
| 25% joint dropout · 15-frame blocks | Recovery + rejection | 14.75% | 13.12% | 18.44% | 80.00% | 43.49 |
| Coordinate noise · σ = 0.02 | Original model | 10.25% | 8.92% | 10.25% | 100.00% | 30.94 |
| Coordinate noise · σ = 0.02 | With joint recovery | 11.75% | 10.49% | 11.75% | 100.00% | 43.79 |
| Coordinate noise · σ = 0.02 | With rejection only | 10.00% | 8.88% | 11.53% | 86.75% | 30.94 |
| Coordinate noise · σ = 0.02 | Recovery + rejection | 10.75% | 9.47% | 12.57% | 85.50% | 43.79 |

## Recovery compared with the original model

These are paired accuracy differences in **percentage points**, using identical test clips and corruption masks. Rejection is excluded from this particular contrast.

| Condition | Accuracy difference | Paired 95% interval | Corrected clips | Regressed clips |
| --- | ---: | --- | ---: | ---: |
| Original poses | +0.25 pp | -1.50 pp to +2.00 pp | 7 | 6 |
| 25% joint dropout · 5-frame blocks | +3.75 pp | +1.50 pp to +6.25 pp | 21 | 6 |
| 25% limb dropout · 5-frame blocks | +3.25 pp | +0.75 pp to +5.75 pp | 20 | 7 |
| 25% joint dropout · 15-frame blocks | +0.25 pp | -1.50 pp to +2.00 pp | 7 | 6 |
| Coordinate noise · σ = 0.02 | +1.50 pp | +0.00 pp to +3.00 pp | 8 | 2 |

- Original poses: recovery increased all-input accuracy (difference +0.25 pp) (paired 95% interval -1.50 pp to +2.00 pp). The interval includes zero, so this pilot does not establish a consistent gain.
- 25% joint dropout · 5-frame blocks: recovery increased all-input accuracy (difference +3.75 pp) (paired 95% interval +1.50 pp to +6.25 pp). This interval concerns the sampled clips under this fixed corruption protocol; it is not evidence of universal superiority.
- 25% limb dropout · 5-frame blocks: recovery increased all-input accuracy (difference +3.25 pp) (paired 95% interval +0.75 pp to +5.75 pp). This interval concerns the sampled clips under this fixed corruption protocol; it is not evidence of universal superiority.
- 25% joint dropout · 15-frame blocks: recovery increased all-input accuracy (difference +0.25 pp) (paired 95% interval -1.50 pp to +2.00 pp). The interval includes zero, so this pilot does not establish a consistent gain.
- Coordinate noise · σ = 0.02: recovery increased all-input accuracy (difference +1.50 pp) (paired 95% interval +0.00 pp to +3.00 pp). The interval includes zero, so this pilot does not establish a consistent gain.

## What was selected

Calibration selected **confident_hold_5**. The selected setting has beta = 0, so the deployed repair holds the last observed position rather than extrapolating motion. Repair multiplies the last observed confidence by 0.9 for each missing frame. Repair can start only when the last real observation has confidence at least 0.5. The torso-jump guard resets a person slot above the configured threshold 0.15; it is not an identity tracker.

```json
{
  "enabled": true,
  "min_confidence": 0,
  "max_gap_frames": 5,
  "alpha": 1,
  "beta": 0,
  "confidence_decay": 0.9,
  "max_speed": 0.08,
  "max_displacement": 0.2,
  "identity_jump_threshold": 0.15,
  "min_repair_confidence": 0.5
}
```

## Scope and limitations

- The pose coordinates and action labels are real public data. Missing-joint blocks and coordinate jitter are synthetic perturbations of those real poses. Physical occlusion in new camera recordings has not been validated.
- This is a sampled robustness pilot, not reproduction of the paper's full benchmark, and it does not establish state-of-the-art performance or worldwide methodological novelty.
- Calibration and test clips are disjoint within the upstream validation partition. That partition may have informed the authors' pretrained checkpoint selection. This is not a newly collected subject-independent test set.
- All methods use identical pretrained weights, preprocessing, clips and corruption masks. Repair and rejection settings were frozen on calibration data before test evaluation; the classifier was not retrained.
- Mean synchronized model processing time per 300-frame clip in batches; recovery runs on CPU. Excludes transfers, camera pose estimation, I/O and browser. Not per-frame streaming latency.
- The research classifier is not integrated into a validated webcam pipeline. The existing MediaPipe six-activity demo is a separate system; its outputs are not results from this 400-class checkpoint.
- Kinetics uses OpenPose 18-joint 2D coordinates and confidence. MediaPipe 33-joint landmarks and NTU 25-joint 3D skeletons are not interchangeable inputs without an adapter and fresh validation.
- Person slots are ranked by detector confidence and can switch identities. Bounded extrapolation cannot reliably repair long gaps, abrupt motion, or identity swaps. Reliable visible observations are not denoised by this recovery method.
- A confidence threshold improves the reliability of accepted predictions only by a tradeoff with coverage; it does not prove the classifier itself became more accurate.

The HTML report and both charts are in `results/research`. Per-clip predictions and frozen calibration settings are separate benchmark outputs.
