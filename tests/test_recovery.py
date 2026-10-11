import numpy as np
import pytest

from research.recovery import (
    CausalPoseRecovery,
    RecoveryConfig,
    recover_pose_sequence,
)


def moving_joint(frames=16, joints=1, people=1):
    result = np.zeros((3, frames, joints, people), dtype=np.float32)
    result[0] = (np.arange(frames) * 0.01 - 0.2)[:, None, None]
    result[1] = 0.1
    result[2] = 0.9
    return result


def test_short_gap_improves_on_last_observation_without_changing_observations():
    clean = moving_joint()
    damaged = clean.copy()
    damaged[:, 10:13] = 0
    original = damaged.copy()
    result = recover_pose_sequence(damaged)
    repaired_error = np.abs(result.repaired[:2, 10:13] - clean[:2, 10:13]).mean()
    hold_error = np.abs(clean[:2, 9:10] - clean[:2, 10:13]).mean()
    assert repaired_error < hold_error
    assert result.imputed_mask[10:13].all()
    np.testing.assert_array_equal(
        result.repaired[:, result.observed_mask], clean[:, result.observed_mask]
    )
    np.testing.assert_array_equal(damaged, original)
    np.testing.assert_array_equal(result.original_confidence, damaged[2])
    assert not result.observed_mask[10:13].any()
    assert np.all(np.diff(result.repaired[2, 10:13, 0, 0]) < 0)


def test_repair_is_causal_under_arbitrary_suffix_changes():
    sequence = moving_joint()
    sequence[:, 8:11] = 0
    changed = sequence.copy()
    changed[:, 11:] = moving_joint(5) * np.array([8, -5, 0])[:, None, None, None]
    baseline = recover_pose_sequence(sequence)
    modified = recover_pose_sequence(changed)
    prefix = recover_pose_sequence(sequence[:, :11])
    np.testing.assert_array_equal(baseline.repaired[:, :11], modified.repaired[:, :11])
    np.testing.assert_array_equal(baseline.repaired[:, :11], prefix.repaired)
    np.testing.assert_array_equal(baseline.imputed_mask[:11], prefix.imputed_mask)


def test_long_and_initial_gaps_remain_invalid_and_reacquisition_resets_velocity():
    sequence = moving_joint(20)
    sequence[:, :2] = 0
    sequence[:, 8:16] = 0
    sequence[:, 17] = 0
    result = recover_pose_sequence(sequence, RecoveryConfig(max_gap_frames=3))
    assert not result.imputed_mask[:2].any()
    assert result.imputed_mask[8:11].all()
    assert not result.imputed_mask[11:16].any()
    assert np.all(result.repaired[:, 11:16] == 0)
    assert result.unresolved_mask[11:16].all()
    # The first observation after an expired track starts with zero velocity.
    np.testing.assert_array_equal(result.repaired[:2, 17], sequence[:2, 16])


def test_center_coordinates_are_valid_and_low_confidence_is_not_a_new_observation():
    sequence = np.zeros((3, 5, 1, 1), dtype=np.float32)
    sequence[2, 0] = 0.8
    sequence[:2, 1:4] = 20
    sequence[2, 1:4] = 0.1
    result = recover_pose_sequence(sequence, RecoveryConfig(max_gap_frames=2))
    assert result.observed_mask[0].all()
    assert result.imputed_mask[1:3].all()
    assert not result.imputed_mask[3:].any()
    assert np.all(result.repaired[:2] == 0)
    np.testing.assert_array_equal(result.original_confidence, sequence[2])


def test_prediction_displacement_and_speed_are_bounded():
    sequence = moving_joint(15)
    sequence[0, :5, 0, 0] = np.arange(5) * 20
    sequence[:, 5:] = 0
    config = RecoveryConfig(
        max_gap_frames=10, alpha=0.3, beta=1, max_speed=0.04, max_displacement=0.1
    )
    result = recover_pose_sequence(sequence, config)
    points = result.repaired[:2, 4:, 0, 0]
    assert np.all(np.linalg.norm(np.diff(points, axis=1), axis=0) <= 0.04001)
    assert np.all(np.linalg.norm(points - points[:, :1], axis=0) <= 0.10001)


def test_people_and_joints_do_not_share_tracking_state():
    sequence = moving_joint(5, joints=2, people=2)
    sequence[:, :, 1, 1] = 0
    sequence[:, 2:, 0, 0] = 0
    sequence[0, :, 0, 1] = 0.4
    result = recover_pose_sequence(sequence)
    assert not result.imputed_mask[:, 1, 1].any()
    assert np.all(result.repaired[:, :, 1, 1] == 0)
    np.testing.assert_array_equal(result.repaired[:, :, :, 1], sequence[:, :, :, 1])


def test_disabled_is_exact_identity_and_zero_gap_disables_imputation():
    sequence = moving_joint()
    sequence[2, 3] = 0.1
    sequence[:, 5] = 0
    disabled = recover_pose_sequence(sequence, RecoveryConfig(enabled=False))
    np.testing.assert_array_equal(disabled.repaired, sequence)
    assert disabled.repaired.dtype == sequence.dtype
    assert not np.shares_memory(disabled.repaired, sequence)
    assert not disabled.imputed_mask.any()
    assert not recover_pose_sequence(
        sequence, RecoveryConfig(max_gap_frames=0)
    ).imputed_mask.any()


def test_streaming_matches_batch_and_reset_clears_prior_clip():
    sequence = moving_joint()
    sequence[:, 5:7] = 0
    tracker = CausalPoseRecovery()
    streamed = np.stack(
        [tracker.step(frame).repaired for frame in sequence.transpose(1, 0, 2, 3)],
        axis=1,
    )
    np.testing.assert_array_equal(streamed, recover_pose_sequence(sequence).repaired)
    tracker.reset()
    assert not tracker.step(np.zeros((3, 1, 1), np.float32)).imputed_mask.any()


@pytest.mark.parametrize(
    "config",
    [
        {"max_gap_frames": -1},
        {"max_gap_frames": 1.5},
        {"alpha": 0},
        {"beta": 2},
        {"max_speed": np.inf},
        {"confidence_decay": -0.2},
    ],
)
def test_invalid_recovery_config_rejected(config):
    with pytest.raises(ValueError):
        RecoveryConfig(**config)


def test_invalid_input_and_changed_stream_shape_rejected():
    sequence = moving_joint()
    sequence[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        recover_pose_sequence(sequence)
    with pytest.raises(ValueError, match="layout"):
        recover_pose_sequence(np.zeros((2, 10, 18, 1), np.float32))
    tracker = CausalPoseRecovery()
    tracker.step(moving_joint(1)[:, 0])
    with pytest.raises(ValueError, match="dimensions"):
        tracker.step(moving_joint(1, joints=2)[:, 0])


def test_empty_sequence_is_supported():
    sequence = np.zeros((3, 0, 18, 2), np.float32)
    result = recover_pose_sequence(sequence)
    assert result.repaired.shape == sequence.shape
    assert result.imputed_mask.shape == (0, 18, 2)


def test_torso_jump_resets_history_without_reordering_people_or_changing_observations():
    sequence = moving_joint(12, joints=18, people=2)
    sequence[0, :, :, 1] += 0.6
    # At frame 7 the confidence-ranked person slots swap. The newly assigned
    # person in slot 0 has no visible wrist: old slot-0 history must not fill it.
    sequence[:, 7:] = sequence[:, 7:, :, ::-1].copy()
    sequence[:, 7, 4, 0] = 0
    config = RecoveryConfig(identity_jump_threshold=0.15)
    result = recover_pose_sequence(sequence, config)
    assert result.identity_resets == 2
    assert not result.imputed_mask[7, 4, 0]
    assert np.all(result.repaired[:, 7, 4, 0] == 0)
    np.testing.assert_array_equal(
        result.repaired[:, result.observed_mask], sequence[:, result.observed_mask]
    )
    # Original observation confidence cannot be rewritten by the reset guard.
    np.testing.assert_array_equal(result.original_confidence, sequence[2])


def test_torso_reset_guard_is_causal_and_counts_only_current_result():
    sequence = moving_joint(12, joints=18)
    sequence[0, 8:] += 0.5
    sequence[:, 8, 4] = 0
    config = RecoveryConfig(identity_jump_threshold=0.15)
    whole = recover_pose_sequence(sequence, config)
    prefix = recover_pose_sequence(sequence[:, :8], config)
    assert whole.identity_resets == 1
    assert prefix.identity_resets == 0
    np.testing.assert_array_equal(whole.repaired[:, :8], prefix.repaired)
    tracker = CausalPoseRecovery(config)
    counts = [tracker.step(sequence[:, t]).identity_resets for t in range(12)]
    assert counts == [0] * 8 + [1] + [0] * 3
    assert tracker.identity_resets == 1
    tracker.reset()
    assert tracker.identity_resets == 0


def test_torso_guard_requires_two_common_anchors_and_retains_bounded_gap_policy():
    sequence = moving_joint(12, joints=18)
    sequence[0, 8:] += 0.5
    sequence[:, 8, [2, 5, 8, 11]] = 0  # Only neck remains a common anchor.
    sequence[:, 8, 4] = 0
    result = recover_pose_sequence(
        sequence, RecoveryConfig(identity_jump_threshold=0.15)
    )
    assert result.identity_resets == 0
    assert result.imputed_mask[8, 4, 0]


def test_optional_guard_rejects_wrong_joint_layout_and_invalid_threshold():
    with pytest.raises(ValueError, match="18"):
        recover_pose_sequence(
            moving_joint(), RecoveryConfig(identity_jump_threshold=0.15)
        )
    with pytest.raises(ValueError, match="identity_jump_threshold"):
        RecoveryConfig(identity_jump_threshold=0)


def test_openpose_confidence_above_one_is_preserved_and_negative_is_rejected():
    sequence = moving_joint(6, joints=18)
    sequence[2, :3] = 1.05
    sequence[:, 3] = 0
    result = recover_pose_sequence(sequence)
    np.testing.assert_array_equal(result.original_confidence, sequence[2])
    np.testing.assert_array_equal(result.repaired[:, :3], sequence[:, :3])
    assert np.all(result.repaired[2, 3] == pytest.approx(1.05 * 0.75))
    sequence[2, 0, 0, 0] = -0.01
    with pytest.raises(ValueError, match="nonnegative"):
        recover_pose_sequence(sequence)


def test_minimum_repair_confidence_withholds_fill_but_preserves_weak_observations():
    sequence = moving_joint(10, joints=2)
    sequence[2, :, 0] = 0.3
    sequence[2, :, 1] = 1.05
    sequence[:, 6:8] = 0
    config = RecoveryConfig(min_confidence=0, min_repair_confidence=0.5)
    result = recover_pose_sequence(sequence, config)
    assert not result.imputed_mask[6:8, 0].any()
    assert np.all(result.repaired[:, 6:8, 0] == 0)
    assert result.imputed_mask[6:8, 1].all()
    # Confidence below the repair gate is still a genuine observed input.
    np.testing.assert_array_equal(result.repaired[:, :6], sequence[:, :6])
    np.testing.assert_array_equal(result.original_confidence, sequence[2])
    assert result.observed_mask[:6].all()
    assert result.identity_resets == 0


def test_repair_gate_uses_latest_real_score_not_an_older_high_score_or_decayed_score():
    sequence = moving_joint(12, joints=2)
    sequence[2, 5, 0] = 0.2  # Overrides the earlier high-confidence history.
    sequence[2, 5, 1] = 0.5  # Exactly at the repair threshold.
    sequence[:, 6:9] = 0
    result = recover_pose_sequence(
        sequence,
        RecoveryConfig(
            min_confidence=0, min_repair_confidence=0.5, confidence_decay=0.5
        ),
    )
    assert not result.imputed_mask[6:9, 0].any()
    # All three repairs use last REAL score .5, even though their assigned
    # output confidence decays below the gate on the first missing frame.
    assert result.imputed_mask[6:9, 1].all()
    np.testing.assert_allclose(result.repaired[2, 6:9, 1, 0], [0.25, 0.125, 0.0625])
    np.testing.assert_array_equal(result.repaired[:, 5], sequence[:, 5])


def test_repair_gate_default_is_backward_compatible_and_invalid_gate_is_rejected():
    sequence = moving_joint()
    sequence[:, 8:10] = 0
    implicit = recover_pose_sequence(sequence)
    explicit = recover_pose_sequence(sequence, RecoveryConfig(min_repair_confidence=0))
    np.testing.assert_array_equal(implicit.repaired, explicit.repaired)
    np.testing.assert_array_equal(implicit.imputed_mask, explicit.imputed_mask)
    for threshold in (-0.01, np.nan, np.inf):
        with pytest.raises(ValueError, match="min_repair_confidence"):
            RecoveryConfig(min_repair_confidence=threshold)
