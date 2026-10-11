import ast
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("continual")

from research.benchmark import (
    choose_threshold,
    metric,
    paired_ci,
    quality_scores,
    recover,
)
from research.recovery import RecoveryConfig
from research.upstream import _definitions, verify_equivalence


def test_rejected_predictions_do_not_inflate_all_input_accuracy():
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.55, 0.45]])
    truth = np.array([0, 1, 1])
    values, prediction, accepted = metric(probabilities, truth, threshold=0.8)
    assert values["accuracy"] == pytest.approx(1 / 3)
    assert values["coverage"] == pytest.approx(1 / 3)
    assert values["accepted_accuracy"] == 1
    assert prediction.tolist() == [0, -1, -1]
    assert accepted.tolist() == [True, False, False]


def test_no_pose_is_rejected_even_with_high_classifier_score():
    values, _, _ = metric(
        np.array([[0.99, 0.01]]), np.array([0]), quality=np.array([0.0])
    )
    assert values["coverage"] == 0
    assert values["accuracy"] == 0
    assert values["accepted_accuracy"] is None


def test_calibration_threshold_preserves_minimum_coverage():
    p = np.array([[0.9, 0.1]] * 8 + [[0.51, 0.49]] * 2)
    y = np.array([0] * 8 + [1] * 2)
    q = np.ones(10)
    threshold = choose_threshold(p, y, q)
    values, _, _ = metric(p, y, threshold, q)
    assert values["coverage"] >= 0.8


def test_paired_bootstrap_counts_regressions_and_gains():
    result = paired_ci(
        np.array([True, False, True, False]), np.array([True, True, False, False])
    )
    assert result["accuracy_gain"] == 0
    assert result["improved_clips"] == result["regressed_clips"] == 1
    assert result["ci95"][0] <= 0 <= result["ci95"][1]


def test_recovery_never_fills_padding_after_video_end():
    sample = np.zeros((1, 3, 300, 18, 2), dtype=np.float32)
    sample[0, :, 0, :, 0] = np.array([0.1, 0.2, 0.8])[:, None]
    output, _ = recover(sample, [1], RecoveryConfig(min_confidence=0))
    np.testing.assert_array_equal(output[:, :, 1:], 0)
    assert quality_scores(sample, [1])[0] == 0.5


def test_only_hash_pinned_model_source_can_execute(tmp_path):
    path = tmp_path / "models/utils.py"
    path.parent.mkdir()
    path.write_text('raise RuntimeError("must not run")')
    with pytest.raises(ValueError, match="hash mismatch"):
        _definitions(path, {})


def test_equivalence_rejects_wrong_window_before_loading_models():
    import torch

    with pytest.raises(ValueError, match="300"):
        verify_equivalence(torch.zeros(1, 3, 100, 18, 2))


def test_colab_notebook_is_valid_python_and_contains_no_claimed_outputs():
    from pathlib import Path

    notebook = json.loads(Path("notebooks/Research_Comparison.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell.get("outputs", []) == []
