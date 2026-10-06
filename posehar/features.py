"""All training and inference use this one feature implementation.

Image x is corrected by width/height before Euclidean geometry. Features
use the hip midpoint as origin and shoulder-to-hip distance as scale.
2D joint angles are projected angles, not clinical joint measurements.
"""

import numpy as np
from .config import WINDOW_SECONDS, SAMPLE_HZ

CORE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
NAMES = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "ankle_spread",
    "left_wrist_height",
    "right_wrist_height",
    "hip_height",
    "trunk_tilt",
    "knee_height_diff",
    "left_wrist_x",
    "right_wrist_x",
    "left_ankle_height",
    "right_ankle_height",
]
TEMPORAL_NAMES = [
    f"{stat}:{name}" for stat in ("mean", "std", "range", "speed") for name in NAMES
]


class PoseQualityError(ValueError):
    pass


def angle(a, b, c):
    u, v = a - b, c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-7:
        raise PoseQualityError("Overlapping joints: move fully into view.")
    return float(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1)) / np.pi)


def frame_features(landmarks, aspect=1):
    p = np.asarray(landmarks, dtype=float)
    if p.shape != (33, 4) or not np.isfinite(p).all():
        raise PoseQualityError("No valid body landmarks.")
    if p[CORE, 3].min() < 0.5:
        raise PoseQualityError(
            "Keep shoulders, wrists, hips, knees and ankles visible."
        )
    if (p[CORE, :2] < -0.05).any() or (p[CORE, :2] > 1.05).any():
        raise PoseQualityError("Step back so your whole body fits in the frame.")
    xy = p[:, :2] * [aspect, 1]
    hip = (xy[23] + xy[24]) / 2
    shoulder = (xy[11] + xy[12]) / 2
    scale = np.linalg.norm(shoulder - hip)
    if scale < 0.06:
        raise PoseQualityError("Move closer or face the camera at a slight angle.")
    q = (xy - hip) / scale
    triples = [
        (23, 25, 27),
        (24, 26, 28),
        (11, 23, 25),
        (12, 24, 26),
        (11, 13, 15),
        (12, 14, 16),
        (23, 11, 13),
        (24, 12, 14),
    ]
    values = [angle(q[a], q[b], q[c]) for a, b, c in triples]
    trunk = shoulder - hip
    values += [
        abs(q[27, 0] - q[28, 0]),
        q[11, 1] - q[15, 1],
        q[12, 1] - q[16, 1],
        (q[27, 1] + q[28, 1]) / 2,
        float(np.arctan2(abs(trunk[0]), -trunk[1]) / np.pi),
        q[25, 1] - q[26, 1],
        q[15, 0],
        q[16, 0],
        q[27, 1],
        q[28, 1],
    ]
    return np.asarray(values), float(p[CORE, 3].mean())


def extract_window(frames, aspect=1):
    ts = np.asarray([f.t if hasattr(f, "t") else f["t"] for f in frames])
    if len(ts) < 12 or ts[-1] - ts[0] < WINDOW_SECONDS - 0.2:
        raise PoseQualityError("Collecting a two-second movement window.")
    start = ts[-1] - WINDOW_SECONDS
    selected = [(t, f) for t, f in zip(ts, frames) if t >= start - 1e-6]
    good_t, good_x, quality = [], [], []
    for t, f in selected:
        try:
            row, score = frame_features(
                f.landmarks if hasattr(f, "landmarks") else f["landmarks"], aspect
            )
            good_t.append(t)
            good_x.append(row)
            quality.append(score)
        except PoseQualityError:
            pass
    if len(good_t) < 12 or len(good_t) / len(selected) < 0.85:
        raise PoseQualityError("Tracking is incomplete. Keep your full body visible.")
    if (
        good_t[-1] < ts[-1] - 0.12
        or good_t[-1] - good_t[0] < 1.8
        or max(np.diff(good_t)) > 0.35
    ):
        raise PoseQualityError(
            "Tracking was interrupted. Hold position for two seconds."
        )
    grid = np.linspace(good_t[0], good_t[-1], SAMPLE_HZ * 2 + 1)
    raw = np.asarray(good_x)
    x = np.column_stack([np.interp(grid, good_t, raw[:, i]) for i in range(len(NAMES))])
    mean = x.mean(axis=0)
    temporal = np.concatenate(
        [
            mean,
            x.std(axis=0),
            np.ptp(x, axis=0),
            np.abs(np.diff(x, axis=0)).mean(axis=0) / (grid[1] - grid[0]),
        ]
    )
    return {
        "static": mean,
        "temporal": temporal,
        "series": x,
        "quality": float(np.mean(quality)),
        "current": dict(zip(NAMES, x[-1].tolist())),
    }


def rule_prediction(feat):
    """A transparent heuristic baseline. Scores are never probabilities."""
    x = feat["series"]
    m, r, s = x.mean(0), np.ptp(x, axis=0), x.std(0)
    if r[8] > 0.55 and min(r[9], r[10]) > 0.65:
        return "jumping_jacks", "Both arms and leg spacing change together."
    if min(r[0], r[1]) > 0.15 and r[11] > 0.22:
        return "squats", "Both knees bend and the hips move vertically."
    if r[13] > 0.28 and max(r[16], r[17]) > 0.25:
        return "marching", "The left and right legs rise alternately."
    if (m[9] > -0.1 and s[14] > 0.12) or (m[10] > -0.1 and s[15] > 0.12):
        return "waving", "A raised wrist moves from side to side."
    if (m[0] + m[1]) / 2 < 0.78:
        return "sitting", "Bent knees with little movement."
    if (m[0] + m[1]) / 2 > 0.85 and max(s[16], s[17]) < 0.1:
        return "standing", "Straight legs with little movement."
    return "unknown", "Movement does not match a baseline rule."
