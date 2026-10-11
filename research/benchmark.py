"""Calibrate on 200 clips, freeze settings, then evaluate 400 separate clips.

All inputs and labels are real public Kinetics data. Joint dropout and jitter
are explicitly synthetic stress conditions, not real occlusion recordings.
"""

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from sklearn.metrics import f1_score

from research.corruptions import CorruptionConfig, apply_corruption
from research.data import load_partition
from research.recovery import RecoveryConfig, recover_pose_sequence
from research.upstream import (
    ROOT,
    COMMIT,
    CHECKPOINT_SHA256,
    SOURCE_HASHES,
    load_model,
    verify_equivalence,
)

CONDITIONS = [
    ("clean", "Original poses", CorruptionConfig()),
    (
        "burst_missing",
        "25% joint dropout · 5-frame blocks",
        CorruptionConfig(dropout_fraction=0.25, block_frames=5),
    ),
    (
        "limb_missing",
        "25% limb dropout · 5-frame blocks",
        CorruptionConfig(dropout_fraction=0.25, block_frames=5, mode="limb"),
    ),
    (
        "long_missing",
        "25% joint dropout · 15-frame blocks",
        CorruptionConfig(dropout_fraction=0.25, block_frames=15),
    ),
    ("jitter", "Coordinate noise · σ = 0.02", CorruptionConfig(jitter_std=0.02)),
]
CALIBRATION_CONDITIONS = {"clean", "burst_missing", "limb_missing"}
CANDIDATES = {
    "confident_hold_5": RecoveryConfig(
        min_confidence=0,
        min_repair_confidence=0.5,
        alpha=1,
        beta=0,
        max_gap_frames=5,
        confidence_decay=0.9,
        identity_jump_threshold=0.15,
    ),
    "hold_3_last_score": RecoveryConfig(
        min_confidence=0,
        alpha=1,
        beta=0,
        max_gap_frames=3,
        confidence_decay=1,
        identity_jump_threshold=0.15,
    ),
    "hold_5_last_score": RecoveryConfig(
        min_confidence=0,
        alpha=1,
        beta=0,
        max_gap_frames=5,
        confidence_decay=1,
        identity_jump_threshold=0.15,
    ),
    "hold_5": RecoveryConfig(
        min_confidence=0,
        alpha=1,
        beta=0,
        max_gap_frames=5,
        confidence_decay=0.9,
        identity_jump_threshold=0.15,
    ),
    "motion_5": RecoveryConfig(
        min_confidence=0,
        max_gap_frames=5,
        confidence_decay=0.9,
        identity_jump_threshold=0.15,
    ),
    "motion_10": RecoveryConfig(
        min_confidence=0,
        max_gap_frames=10,
        confidence_decay=0.9,
        identity_jump_threshold=0.15,
    ),
    "conservative_5": RecoveryConfig(
        min_confidence=0.1,
        max_gap_frames=5,
        confidence_decay=0.75,
        identity_jump_threshold=0.15,
    ),
}
VARIANTS = [
    {
        "id": "baseline",
        "label": "Original model",
        "description": "Authors' pretrained CoST-GCN* equivalent clip path; unchanged weights.",
    },
    {
        "id": "recovery",
        "label": "With joint recovery",
        "description": "Same classifier with bounded causal pose repair.",
    },
    {
        "id": "rejection",
        "label": "With rejection only",
        "description": "Original model, with calibration-selected rejection threshold.",
    },
    {
        "id": "combined",
        "label": "Recovery + rejection",
        "description": "Recovery and a separately calibrated prediction threshold.",
    },
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def predictions(model, samples, batch_size):
    logits, elapsed = [], 0.0
    device = next(model.parameters()).device

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(samples[start : start + batch_size])
            ).to(device)
            sync()
            begin = time.perf_counter()
            value = model(batch)
            sync()
            elapsed += time.perf_counter() - begin
            logits.append(value.cpu().numpy())
    values = np.concatenate(logits)
    probabilities = torch.softmax(torch.from_numpy(values), dim=1).numpy()
    return probabilities, elapsed * 1000 / len(samples)


def corrupt(samples, rows, condition_id, config):
    outputs, severities = [], []
    for sample, row in zip(samples, rows):
        seed = int.from_bytes(
            hashlib.sha256(f"2026:{row['id']}:{condition_id}".encode()).digest()[:8],
            "little",
        )
        result = apply_corruption(sample, replace(config, seed=seed))
        outputs.append(result.corrupted)
        severities.append(result.metadata["actual_dropout_fraction"])
    return np.stack(outputs), severities


def recover(samples, lengths, config):
    outputs, elapsed = [], 0.0
    for sample, length in zip(samples, lengths):
        value = sample.copy()
        begin = time.perf_counter()
        if length:
            value[:, :length] = recover_pose_sequence(
                sample[:, :length], config
            ).repaired
        elapsed += time.perf_counter() - begin
        outputs.append(value)
    return np.stack(outputs), elapsed * 1000 / len(samples)


def quality_scores(samples, lengths):
    # Avoid counting zero padding or imputed joints as genuinely observed pose.
    return np.array(
        [float((x[2, :t] > 0).mean()) if t else 0.0 for x, t in zip(samples, lengths)]
    )


def choose_threshold(probabilities, labels, quality):
    """Maximize accepted accuracy on calibration, subject to >=80% coverage."""
    confidence = probabilities.max(1)
    pred = probabilities.argmax(1)
    candidates = []
    for threshold in np.linspace(0, 0.5, 101):
        accepted = (confidence >= threshold) & (quality > 0)
        if accepted.mean() >= 0.8:
            candidates.append(
                (
                    float((pred[accepted] == labels[accepted]).mean()),
                    float(accepted.mean()),
                    float(threshold),
                )
            )
    if not candidates:
        return 0.0
    return max(candidates, key=lambda v: (v[0], v[1], -v[2]))[2]


def metric(probabilities, labels, threshold=0.0, quality=None):
    raw = probabilities.argmax(1)
    accepted = probabilities.max(1) >= threshold
    if quality is not None:
        accepted &= quality > 0
    pred = np.where(accepted, raw, -1)
    # Fixed, documented label set = ground-truth classes present in this split.
    label_set = np.unique(labels)
    return (
        {
            "accuracy": float((pred == labels).mean()),
            "macro_f1": float(
                f1_score(
                    labels, pred, labels=label_set, average="macro", zero_division=0
                )
            ),
            "coverage": float(accepted.mean()),
            "accepted_accuracy": (
                float((pred[accepted] == labels[accepted]).mean())
                if accepted.any()
                else None
            ),
            "n_samples": len(labels),
            "n_accepted": int(accepted.sum()),
            "correct": int((pred == labels).sum()),
            "threshold": threshold,
            "raw_top5_accuracy": float(
                np.mean(
                    [y in np.argsort(p)[-5:] for y, p in zip(labels, probabilities)]
                )
            ),
        },
        pred,
        accepted,
    )


def paired_ci(before, after, seed=2026):
    # Clip-paired bootstrap; not evidence of population-wide superiority.
    rng = np.random.default_rng(seed)
    difference = after.astype(float) - before.astype(float)
    means = difference[
        rng.integers(0, len(difference), size=(3000, len(difference)))
    ].mean(1)
    return {
        "accuracy_gain": float(difference.mean()),
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "improved_clips": int(((~before) & after).sum()),
        "regressed_clips": int((before & (~after)).sum()),
        "method": "paired clip bootstrap, 3000 resamples",
    }


def calibrate(args, model):
    x, y, rows, lengths, manifest = load_partition(args.data, "calibration")
    # Persist design before evaluation of test labels/predictions.
    design = {
        "selection": manifest["sampling"],
        "seed": manifest["seed"],
        "conditions": [
            {"id": i, "label": l, "config": asdict(c)} for i, l, c in CONDITIONS
        ],
        "candidates": {k: asdict(v) for k, v in CANDIDATES.items()},
        "selection_objective": "Mean all-input top1 accuracy across clean, burst_missing and limb_missing calibration clips; ties by candidate order.",
        "rejection_objective": "Highest calibration accepted accuracy with >=80% coverage; ties prefer coverage.",
        "calibration_ids": [r["id"] for r in rows],
        "calibration_count": len(rows),
        "upstream_commit": COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "device": str(next(model.parameters()).device),
        "batch_size": args.batch_size,
        "upstream_source_hashes": SOURCE_HASHES,
        "implementation_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "research").glob("*.py"))
        },
    }
    write_json(args.out / "design.json", design)
    scores = {name: [] for name in CANDIDATES}
    baseline_probs, all_labels, all_quality = [], [], []
    baseline_scores = []
    candidate_probs = {name: [] for name in CANDIDATES}
    for condition, label, config in CONDITIONS:
        if condition not in CALIBRATION_CONDITIONS:
            continue
        damaged, _ = corrupt(x, rows, condition, config)
        p, _ = predictions(model, damaged, args.batch_size)
        baseline_probs.append(p)
        baseline_scores.append(float((p.argmax(1) == y).mean()))
        all_labels.append(y)
        all_quality.append(quality_scores(damaged, lengths))
        print(
            f"Calibration {condition}: original {(p.argmax(1)==y).mean():.3f}",
            flush=True,
        )
        for name, recovery_config in CANDIDATES.items():
            repaired, _ = recover(damaged, lengths, recovery_config)
            p, _ = predictions(model, repaired, args.batch_size)
            score = float((p.argmax(1) == y).mean())
            scores[name].append(score)
            candidate_probs[name].append(p)
            print(f"  {name}: {score:.3f}", flush=True)
    winner = max(CANDIDATES, key=lambda name: np.mean(scores[name]))
    labels = np.concatenate(all_labels)
    qualities = np.concatenate(all_quality)
    frozen = {
        "selected_candidate": winner,
        "baseline_calibration_scores": baseline_scores,
        "development_note": "Development used three calibration-only rounds: four initial recipes, two additional last-score hold recipes, and finally one high-confidence hold recipe. The earlier rounds are retained under calibration-round1 and calibration-round2. All seven candidates were compared in the final calibration. No held-out test predictions were used to choose the method.",
        "recovery": asdict(CANDIDATES[winner]),
        "calibration_scores": scores,
        "calibration_means": {k: float(np.mean(v)) for k, v in scores.items()},
        "baseline_threshold": choose_threshold(
            np.concatenate(baseline_probs), labels, qualities
        ),
        "recovery_threshold": choose_threshold(
            np.concatenate(candidate_probs[winner]), labels, qualities
        ),
        "design_sha256": hashlib.sha256(
            (args.out / "design.json").read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_commit": COMMIT,
        "frozen_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
    }
    write_json(args.out / "calibration.json", frozen)
    print("Frozen calibration", json.dumps(frozen), flush=True)


def evaluate(args, model):
    frozen = json.loads((args.out / "calibration.json").read_text())
    x, y, rows, lengths, manifest = load_partition(args.data, "test")
    design = json.loads((args.out / "design.json").read_text())
    if set(design["calibration_ids"]) & {r["id"] for r in rows}:
        raise ValueError("Calibration/test overlap")
    if (
        hashlib.sha256((args.out / "design.json").read_bytes()).hexdigest()
        != frozen["design_sha256"]
    ):
        raise ValueError("Experiment design changed after calibration")
    if design["conditions"] != [
        {"id": i, "label": l, "config": asdict(c)} for i, l, c in CONDITIONS
    ]:
        raise ValueError("Corruption conditions changed after calibration")
    if (
        design["device"] != str(next(model.parameters()).device)
        or design["batch_size"] != args.batch_size
    ):
        raise ValueError("Use the same device and batch size as calibration")
    for path, digest in design["implementation_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Implementation changed after calibration: {path}")
    config = RecoveryConfig(**frozen["recovery"])
    # Use a calibration clip for the bridge check, keeping it independent of test outcomes.
    calibration_sample = load_partition(args.data, "calibration")[0][:1]
    equivalence = verify_equivalence(torch.from_numpy(calibration_sample))
    with torch.inference_mode():
        cpu_logits = load_model()(torch.from_numpy(calibration_sample))
        device_logits = model(
            torch.from_numpy(calibration_sample).to(next(model.parameters()).device)
        ).cpu()
    torch.testing.assert_close(cpu_logits, device_logits, atol=5e-4, rtol=1e-4)
    equivalence["cpu_vs_experiment_device_max_abs"] = float(
        (cpu_logits - device_logits).abs().max()
    )
    write_json(args.out / "equivalence.json", equivalence)
    results = []
    csv_rows = []
    example_candidates = []
    classmap = json.loads((args.data / "classes.json").read_text())
    for condition, label, corruption_config in CONDITIONS:
        damaged, severities = corrupt(x, rows, condition, corruption_config)
        repaired, repair_ms = recover(damaged, lengths, config)
        original_p, base_ms = predictions(model, damaged, args.batch_size)
        recovery_p, recovered_ms = predictions(model, repaired, args.batch_size)
        quality = quality_scores(damaged, lengths)
        entries = []
        variant_preds = {}
        for variant, probs, threshold, qual, inference_ms, prep_ms in [
            ("baseline", original_p, 0, None, base_ms, 0),
            ("recovery", recovery_p, 0, None, recovered_ms, repair_ms),
            (
                "rejection",
                original_p,
                frozen["baseline_threshold"],
                quality,
                base_ms,
                0,
            ),
            (
                "combined",
                recovery_p,
                frozen["recovery_threshold"],
                quality,
                recovered_ms,
                repair_ms,
            ),
        ]:
            value, pred, accepted = metric(probs, y, threshold, qual)
            value.update(
                variant=variant,
                classifier_ms=inference_ms,
                recovery_ms=prep_ms,
                latency_ms=inference_ms + prep_ms,
            )
            entries.append(value)
            variant_preds[variant] = pred
            for i, row in enumerate(rows):
                csv_rows.append(
                    dict(
                        id=row["id"],
                        condition=condition,
                        variant=variant,
                        truth=int(y[i]),
                        truth_label=row["label"],
                        prediction=int(pred[i]),
                        prediction_label=classmap.get(
                            str(pred[i]), "insufficient information"
                        ),
                        confidence=float(probs[i].max()),
                        accepted=bool(accepted[i]),
                        correct=bool(pred[i] == y[i]),
                        actual_dropout=severities[i],
                    )
                )
        change = paired_ci(
            variant_preds["baseline"] == y, variant_preds["recovery"] == y
        )
        results.append(
            {
                "id": condition,
                "label": label,
                "description": f"Controlled skeleton stress test; config: {asdict(corruption_config)}",
                "mean_actual_dropout": float(np.mean(severities)),
                "results": entries,
                "paired_recovery": change,
            }
        )
        print(
            f"TEST {condition}: original {entries[0]['accuracy']:.3f}, recovery {entries[1]['accuracy']:.3f}, combined {entries[3]['accuracy']:.3f} coverage {entries[3]['coverage']:.3f}",
            flush=True,
        )
        # Deterministic first successes AND failures, explicitly qualitative examples.
        if condition == "burst_missing":
            categories = [
                (
                    "Improved",
                    (variant_preds["baseline"] != y) & (variant_preds["recovery"] == y),
                ),
                (
                    "Regressed",
                    (variant_preds["baseline"] == y) & (variant_preds["recovery"] != y),
                ),
                (
                    "Both incorrect",
                    (variant_preds["baseline"] != y) & (variant_preds["recovery"] != y),
                ),
            ]
            edges = [
                [1, 2],
                [2, 3],
                [3, 4],
                [1, 5],
                [5, 6],
                [6, 7],
                [2, 8],
                [8, 9],
                [9, 10],
                [5, 11],
                [11, 12],
                [12, 13],
                [1, 0],
                [0, 14],
                [0, 15],
                [14, 16],
                [15, 17],
            ]
            for category, mask in categories:
                for i in np.flatnonzero(mask)[:1]:

                    def frames(value):
                        v = value[:, : lengths[i] : 2, :, 0].transpose(1, 2, 0).copy()
                        v[:, :, 0] += 0.5
                        v[:, :, 1] = 0.5 - v[:, :, 1]
                        return np.round(v, 4).tolist()

                    example_candidates.append(
                        {
                            "id": rows[i]["id"],
                            "label": rows[i]["label"],
                            "category": category,
                            "condition": condition,
                            "fps": 15,
                            "edges": edges,
                            "prediction_baseline": classmap[
                                str(variant_preds["baseline"][i])
                            ],
                            "prediction_combined": classmap.get(
                                str(variant_preds["combined"][i]),
                                "insufficient information",
                            ),
                            "prediction_recovery": classmap[
                                str(variant_preds["recovery"][i])
                            ],
                            "before": frames(damaged[i]),
                            "after": frames(repaired[i]),
                        }
                    )
    payload = {
        "status": "complete",
        "title": "PoseLab · Measured research comparison",
        "generated_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "protocol": {
            "dataset": manifest["dataset"],
            "split": "400 held-out clips sampled from official validation; separate 200-clip calibration",
            "sample_count": len(y),
            "test_count": len(y),
            "calibration_count": manifest["calibration_count"],
            "class_count": 400,
            "classes_present": len(np.unique(y)),
            "seed": manifest["seed"],
            "checkpoint": "stgcnmod_kinetics_joint.ckpt (authors' pretrained single joint stream)",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "upstream_commit": COMMIT,
            "source_url": manifest["source_folder"],
            "selection": manifest["sampling"],
            "subset_limitation": "A 400-clip pilot, not reproduction of full-paper benchmark scores; controlled joint masking is not physical camera occlusion. Upstream validation data may have informed the authors' checkpoint selection.",
            "description": "Same pretrained weights, same original preprocessing, same clips and corruption masks. Recovery and rejection settings frozen on a disjoint 200-clip calibration set.",
            "corruptions": "5-frame joint and limb dropout, 15-frame dropout, coordinate jitter; clean condition retained.",
            "hardware": platform.platform(),
            "device": str(next(model.parameters()).device),
            "torch_threads": torch.get_num_threads(),
            "batch_size": args.batch_size,
            "timing_scope": "Mean synchronized model processing time per 300-frame clip in batches; recovery runs on CPU. Excludes transfers, camera pose estimation, I/O and browser. Not per-frame streaming latency.",
            "macro_f1_scope": f"Macro average over the {len(np.unique(y))} true classes represented in the held-out test subset; classifier retains all 400 outputs.",
            "empty_clips_retained": sum(not r["has_skeleton"] for r in rows),
        },
        "variants": VARIANTS,
        "conditions": results,
        "examples": example_candidates,
        "calibration": frozen,
        "equivalence": equivalence,
        "notes": [
            "All accuracy/F1 values count abstentions as incorrect. Accepted accuracy must be interpreted together with coverage.",
            "Settings were selected on calibration clips only; no training or fine-tuning on test clips.",
            "Recovery reuses prior positions within confidence-ranked person slots. These slots are not guaranteed persistent identities; identity switches remain a limitation.",
            "Model quality is limited by the authors' 400-class 2D pose-only checkpoint. NTU 3D benchmark scores do not apply to this experiment.",
            "Qualitative examples are the first gain, regression and shared failure in deterministic test order, where available. Their captions are clip-level predictions, not per-frame predictions.",
        ],
    }
    write_json(args.out / "comparison.json", payload)
    write_json(ROOT / "static/research-results.json", payload)
    write_json(args.out / "data-manifest.json", manifest)
    with (args.out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved real results to {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".build/research-data")
    parser.add_argument("--out", type=Path, default=ROOT / "results/research")
    parser.add_argument(
        "--phase", choices=["all", "calibrate", "evaluate"], default="all"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    model = load_model().to(device)
    with torch.inference_mode():
        model(torch.zeros(1, 3, 300, 18, 2, device=device))
    if args.phase in {"all", "calibrate"}:
        calibrate(args, model)
    if args.phase in {"all", "evaluate"}:
        evaluate(args, model)


if __name__ == "__main__":
    main()
