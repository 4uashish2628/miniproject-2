"""Subject-separated evaluation. Never randomly split overlapping windows."""

import csv
import hashlib
import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from .config import LABELS, FEATURE_VERSION, STRIDE_SECONDS
from .features import extract_window, PoseQualityError, TEMPORAL_NAMES
from .schema import Clip

TRAIN_LOCK = threading.Lock()


def clip_windows(clip):
    ts = np.asarray([f.t for f in clip.frames])
    for end in np.arange(ts[0] + 2, ts[-1] + 0.001, STRIDE_SECONDS):
        window = [f for f in clip.frames if end - 2 - 1e-6 <= f.t <= end + 1e-6]
        try:
            yield extract_window(window, clip.aspect)
        except PoseQualityError:
            yield None


def load_clips(directory):
    clips = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            clip = Clip.model_validate_json(path.read_text())
        except Exception as e:
            raise ValueError(f"Invalid clip {path.name}: {e}") from e
        if clip.source == "synthetic":
            raise ValueError(
                "Synthetic clips are forbidden in real training. Remove them from data/clips."
            )
        if not clip.consent:
            raise ValueError(f"Clip {path.name} has no recorded consent.")
        clips.append((path, clip))
    return clips


def inventory(directory):
    clips = load_clips(directory)
    subjects = sorted({c.subject for _, c in clips})
    return {
        "total_clips": len(clips),
        "subjects": subjects,
        "labels": list(LABELS),
        "counts": {
            s: {
                a: sum(c.subject == s and c.label == a for _, c in clips)
                for a in LABELS
            }
            for s in subjects
        },
    }


def split_clips(clips, validation_subjects, test_subjects):
    val, test = set(validation_subjects), set(test_subjects)
    subjects = {c.subject for _, c in clips}
    if not val or not test or val & test or not (val | test) <= subjects:
        raise ValueError(
            "Choose existing, different participants for validation and testing."
        )
    train = subjects - val - test
    if not train:
        raise ValueError(
            "At least three participants are needed: training, validation and test."
        )
    groups = {"train": train, "validation": val, "test": test}
    for name, group in groups.items():
        missing = set(LABELS) - {c.label for _, c in clips if c.subject in group}
        if missing:
            raise ValueError(
                f"{name} is missing activities: {', '.join(sorted(missing))}."
            )
    return groups


def evaluate(y, pred, valid):
    y, pred = np.asarray(y), np.asarray(pred)
    accepted = pred != "unknown"
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": (
            float(accuracy_score(y[accepted], pred[accepted]))
            if accepted.any()
            else None
        ),
        "windows": len(y),
        "quality_rejected": int(len(y) - sum(valid)),
        "labels": [*LABELS, "unknown"],
        "confusion_matrix": confusion_matrix(
            y, pred, labels=[*LABELS, "unknown"]
        ).tolist(),
        "per_class": classification_report(
            y, pred, labels=list(LABELS), output_dict=True, zero_division=0
        ),
    }


def predictions(model, rows, feature, threshold):
    out = []
    for row in rows:
        if row is None:
            out.append("unknown")
            continue
        probs = model.predict_proba([row[feature]])[0]
        out.append(
            str(model.classes_[np.argmax(probs)])
            if probs.max() >= threshold
            else "unknown"
        )
    return out


def train_project(clip_dir, model_dir, result_dir, validation_subjects, test_subjects):
    clips = load_clips(clip_dir)
    groups = split_clips(clips, validation_subjects, test_subjects)
    batches = {}
    manifest = []
    for split, subjects in groups.items():
        rows, y, provenance = [], [], []
        for path, clip in clips:
            if clip.subject not in subjects:
                continue
            manifest.append(
                {
                    "clip": path.name,
                    "subject": clip.subject,
                    "label": clip.label,
                    "split": split,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            for index, row in enumerate(clip_windows(clip)):
                rows.append(row)
                y.append(clip.label)
                provenance.append(
                    {"clip": path.name, "subject": clip.subject, "window": index}
                )
        if not rows:
            raise ValueError(f"No full two-second windows in {split}.")
        if set(y[i] for i, r in enumerate(rows) if r is not None) != set(LABELS):
            raise ValueError(
                f"{split} needs usable full-body windows for every activity."
            )
        batches[split] = (rows, y, provenance)

    fitted, thresholds, validation, results, test_predictions = {}, {}, {}, {}, {}
    for key in ("static", "temporal"):
        rows, y, _ = batches["train"]
        good = [i for i, r in enumerate(rows) if r is not None]
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        model.fit(np.asarray([rows[i][key] for i in good]), [y[i] for i in good])
        fitted[key] = model
        vr, vy, _ = batches["validation"]
        candidates = []
        for threshold in (0, 0.45, 0.55, 0.65, 0.75):
            metrics = evaluate(
                vy, predictions(model, vr, key, threshold), [r is not None for r in vr]
            )
            candidates.append({"threshold": threshold, **metrics})
        eligible = [m for m in candidates if m["coverage"] >= 0.7] or candidates
        best = max(
            eligible, key=lambda m: (m["macro_f1"], m["coverage"], -m["threshold"])
        )
        thresholds[key] = best["threshold"]
        validation[key] = [
            {k: c[k] for k in ("threshold", "macro_f1", "coverage")} for c in candidates
        ]
        tr, ty, _ = batches["test"]
        for variant, threshold in (
            (f"{key}_raw", 0),
            (f"{key}_selective", best["threshold"]),
        ):
            pred = predictions(model, tr, key, threshold)
            test_predictions[variant] = pred
            results[variant] = evaluate(ty, pred, [r is not None for r in tr])

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    )
    report = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": "real_landmark_recordings",
        "feature_version": FEATURE_VERSION,
        "sklearn_version": sklearn.__version__,
        "random_seed": 42,
        "split_subjects": {k: sorted(v) for k, v in groups.items()},
        "clip_manifest": manifest,
        "thresholds": thresholds,
        "validation": validation,
        "test": results,
        "window_counts": {k: len(v[0]) for k, v in batches.items()},
        "feature_importance": sorted(
            zip(TEMPORAL_NAMES, fitted["temporal"].feature_importances_.tolist()),
            key=lambda p: -p[1],
        ),
        "notes": [
            "Overlapping windows stay within participants.",
            "Report held-out participant and clip counts alongside window metrics.",
            "Test windows are correlated; they are not independent trials.",
            "Confidence is an uncalibrated forest vote score.",
            "Live smoothing and repetition counting require a separate video evaluation.",
            "Repeated inspection of the same test set weakens its held-out status.",
        ],
    }
    run_dir = Path(result_dir) / run_id
    run_dir.mkdir(parents=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    rows, truth, provenance = batches["test"]
    with (run_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip", "subject", "window", "truth", *test_predictions]
        )
        writer.writeheader()
        for i, meta in enumerate(provenance):
            writer.writerow(
                {
                    **meta,
                    "truth": truth[i],
                    **{k: v[i] for k, v in test_predictions.items()},
                }
            )
    payload = {
        "model": fitted["temporal"],
        "threshold": thresholds["temporal"],
        "feature_version": FEATURE_VERSION,
        "report": report,
    }
    # Keep the previous model intact if serializing the new model fails.
    temp = Path(model_dir) / "activity.tmp"
    joblib.dump(payload, temp)
    temp.replace(Path(model_dir) / "activity.joblib")
    (Path(result_dir) / "latest.json").write_text(json.dumps(report, indent=2))
    return report
