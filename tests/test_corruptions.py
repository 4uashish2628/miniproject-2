import numpy as np
import pytest

from research.corruptions import CorruptionConfig, apply_corruption


def full_pose(frames=20, people=1):
    result = np.ones((3, frames, 18, people), dtype=np.float32)
    result[0] = 0.15
    result[1] = -0.2
    result[2] = 0.9
    return result


def test_same_seed_produces_identical_corruption_without_mutating_input():
    sequence = full_pose()
    before = sequence.copy()
    config = CorruptionConfig(
        dropout_fraction=0.2, block_frames=4, jitter_std=0.01, seed=12
    )
    first, second = apply_corruption(sequence, config), apply_corruption(
        sequence, config
    )
    np.testing.assert_array_equal(first.corrupted, second.corrupted)
    np.testing.assert_array_equal(first.dropout_mask, second.dropout_mask)
    np.testing.assert_array_equal(sequence, before)
    assert first.metadata == second.metadata
    assert first.metadata["actual_dropout_fraction"] >= 0.2


def test_limb_dropout_masks_whole_contiguous_limb_blocks():
    sequence = full_pose(12)
    result = apply_corruption(
        sequence,
        CorruptionConfig(dropout_fraction=0.01, block_frames=4, mode="limb", seed=1),
    )
    assert len(result.metadata["events"]) == 1
    event = result.metadata["events"][0]
    assert event["stop"] - event["start"] == 4
    assert len(event["joints"]) == 3
    assert result.dropout_mask.sum() == 12
    assert np.all(result.corrupted[:, result.dropout_mask] == 0)
    assert np.all(result.corrupted[2, ~result.dropout_mask] == 0.9)


def test_corruptions_preserve_already_invalid_joints_and_confidence_for_jitter():
    sequence = full_pose(8, people=2)
    sequence[2, 3:5, :, 0] = 0
    sequence[2, :, :, 1] = 0
    invalid = sequence[2] == 0
    result = apply_corruption(
        sequence, CorruptionConfig(dropout_fraction=0.5, jitter_std=0.02, seed=5)
    )
    np.testing.assert_array_equal(result.corrupted[:, invalid], sequence[:, invalid])
    assert not result.dropout_mask[invalid].any()
    assert not result.jitter_mask[invalid].any()
    np.testing.assert_array_equal(
        result.corrupted[2, result.jitter_mask], sequence[2, result.jitter_mask]
    )
    assert not np.any(result.jitter_mask & result.dropout_mask)


def test_joint_saturation_and_limb_saturation_report_actual_fraction():
    sequence = full_pose(7)
    joint = apply_corruption(
        sequence, CorruptionConfig(dropout_fraction=1, block_frames=50)
    )
    assert joint.dropout_mask.all()
    assert joint.metadata["actual_dropout_fraction"] == 1
    limb = apply_corruption(sequence, CorruptionConfig(dropout_fraction=1, mode="limb"))
    assert limb.metadata["saturated"]
    assert limb.metadata["actual_dropout_fraction"] == pytest.approx(12 / 18)


def test_zero_corruption_is_identity_and_empty_input_is_supported():
    sequence = full_pose()
    result = apply_corruption(sequence)
    np.testing.assert_array_equal(result.corrupted, sequence)
    assert not result.dropout_mask.any()
    assert not result.jitter_mask.any()
    empty = apply_corruption(sequence[:, :0], CorruptionConfig(dropout_fraction=0.5))
    assert empty.corrupted.shape == (3, 0, 18, 1)
    assert empty.metadata["actual_dropout_fraction"] == 0


def test_non_openpose_layout_requires_explicit_limb_mapping():
    sequence = full_pose()[:, :, :6]
    with pytest.raises(ValueError, match="18"):
        apply_corruption(sequence, CorruptionConfig(mode="limb"))
    result = apply_corruption(
        sequence,
        CorruptionConfig(mode="limb", dropout_fraction=0.1),
        joint_groups=((0, 1), (2, 3)),
    )
    assert result.dropout_mask.any()
    with pytest.raises(ValueError, match="duplicate/invalid"):
        apply_corruption(
            sequence, CorruptionConfig(mode="limb"), joint_groups=((0, 99),)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dropout_fraction": 1.1},
        {"dropout_fraction": np.nan},
        {"block_frames": 0},
        {"jitter_std": -1},
        {"mode": "random"},
        {"seed": -1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        CorruptionConfig(**kwargs)
