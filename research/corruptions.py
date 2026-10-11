"""Reproducible pose corruptions shared by both comparison pipelines.

These are controlled skeleton-coordinate stress tests, not claims that digital
masking reproduces all effects of physical occlusion. Generate each corrupted
clip once and feed that exact array to both baseline and recovery. Neither
repair nor model inference should receive the corruption mask or clean input.
"""

from dataclasses import dataclass

import numpy as np

from research.recovery import validate_sequence


# OpenPose COCO-18 indices: right/left arms, right/left legs. Head and neck are
# deliberately excluded from the limb condition. Custom layouts must provide
# their own groups instead of silently receiving this mapping.
OPENPOSE_LIMBS = ((2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13))


@dataclass(frozen=True)
class CorruptionConfig:
    dropout_fraction: float = 0.0
    block_frames: int = 5
    mode: str = "joint"
    jitter_std: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if (
            not np.isfinite(self.dropout_fraction)
            or not 0 <= self.dropout_fraction <= 1
        ):
            raise ValueError("dropout_fraction must be finite and in [0, 1]")
        if not np.isfinite(self.jitter_std) or self.jitter_std < 0:
            raise ValueError("jitter_std must be finite and nonnegative")
        if (
            isinstance(self.block_frames, (bool, np.bool_))
            or not isinstance(self.block_frames, (int, np.integer))
            or self.block_frames < 1
        ):
            raise ValueError("block_frames must be a positive integer")
        if self.mode not in ("joint", "limb"):
            raise ValueError("mode must be 'joint' or 'limb'")
        if (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")


@dataclass
class CorruptionResult:
    corrupted: np.ndarray
    dropout_mask: np.ndarray
    jitter_mask: np.ndarray
    metadata: dict


def apply_corruption(
    sequence: np.ndarray,
    config: CorruptionConfig | None = None,
    *,
    joint_groups: tuple[tuple[int, ...], ...] | None = None,
) -> CorruptionResult:
    """Mask temporal blocks, then jitter remaining observed x/y coordinates.

    dropout_fraction targets a fraction of all originally observed joint-frames.
    Whole blocks are retained, so the actual fraction may overshoot the target;
    a limb condition may saturate before reaching it because only limbs are
    eligible. Always report metadata['actual_dropout_fraction'] alongside the
    requested value. Already invalid joints (confidence 0) are never changed.
    """
    array = validate_sequence(sequence)
    config = config or CorruptionConfig()
    _, frames, joints, people = array.shape
    if joint_groups is not None and config.mode != "limb":
        raise ValueError("Custom joint_groups are only supported for limb mode")
    if config.mode == "joint":
        groups = tuple((joint,) for joint in range(joints))
    else:
        if joint_groups is None:
            if joints != 18:
                raise ValueError("Default limb groups require OpenPose's 18 joints")
            groups = OPENPOSE_LIMBS
        else:
            groups = tuple(tuple(group) for group in joint_groups)
            if not groups or any(not group for group in groups):
                raise ValueError("Custom limb groups must be nonempty")
            for group in groups:
                if len(set(group)) != len(group) or any(
                    isinstance(joint, (bool, np.bool_))
                    or not isinstance(joint, (int, np.integer))
                    or not 0 <= joint < joints
                    for joint in group
                ):
                    raise ValueError(
                        "Custom limb groups contain duplicate/invalid joints"
                    )

    rng = np.random.default_rng(config.seed)
    observed = array[2] > 0
    corrupted = array.copy()
    dropout = np.zeros_like(observed)
    eligible = np.zeros_like(observed)
    eligible[:, sorted({joint for group in groups for joint in group}), :] = True
    eligible &= observed
    observed_count = int(observed.sum())
    eligible_count = int(eligible.sum())
    requested_count = int(np.ceil(config.dropout_fraction * observed_count))
    target = min(requested_count, eligible_count)
    events = []

    if target and frames:
        duration = min(config.block_frames, frames)
        starts = frames - duration + 1
        candidates = starts * len(groups) * people
        removed = 0
        # A random permutation gives deterministic sampling without replacement
        # and a finite bound even for sparse clips or saturated limb conditions.
        for flat_index in rng.permutation(candidates):
            start, remainder = divmod(int(flat_index), len(groups) * people)
            group_index, person = divmod(remainder, people)
            group = groups[group_index]
            block = np.zeros_like(observed)
            block[start : start + duration, group, person] = True
            newly_removed = block & observed & ~dropout
            added = int(newly_removed.sum())
            if not added:
                continue
            dropout |= newly_removed
            removed += added
            events.append(
                {
                    "start": start,
                    "stop": start + duration,
                    "joints": list(group),
                    "person": person,
                }
            )
            if removed >= target:
                break
        corrupted[:, dropout] = 0

    jitter_mask = (
        observed & ~dropout if config.jitter_std > 0 else np.zeros_like(observed)
    )
    if config.jitter_std > 0:
        noise = rng.normal(0, config.jitter_std, size=(2, frames, joints, people))
        corrupted[:2, jitter_mask] += noise[:, jitter_mask].astype(array.dtype)
    metadata = {
        "seed": int(config.seed),
        "mode": config.mode,
        "block_frames": int(config.block_frames),
        "requested_dropout_fraction": float(config.dropout_fraction),
        "actual_dropout_fraction": (
            float(dropout.sum() / observed_count) if observed_count else 0.0
        ),
        "observed_joint_frames": observed_count,
        "dropped_joint_frames": int(dropout.sum()),
        "eligible_joint_frames": eligible_count,
        "saturated": requested_count > eligible_count,
        "jitter_std": float(config.jitter_std),
        "events": events,
    }
    return CorruptionResult(corrupted, dropout, jitter_mask, metadata)
