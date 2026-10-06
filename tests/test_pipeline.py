import copy
import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from posehar.features import (
    extract_window,
    frame_features,
    rule_prediction,
    PoseQualityError,
)
from posehar.synthetic import demo_clip
from posehar.schema import Clip, Sequence
from posehar.config import LABELS
from posehar.training import split_clips, train_project, load_clips
import posehar.app as app_module


@pytest.mark.parametrize("label", LABELS)
def test_rule_patterns(label):
    f = extract_window(demo_clip(label, 2))
    assert f["temporal"].shape == (72,)
    assert np.isfinite(f["temporal"]).all()
    assert rule_prediction(f)[0] == label


def test_geometry_invariant_to_translation_and_scale():
    raw = np.asarray(demo_clip("standing", 2)[0]["landmarks"])
    changed = raw.copy()
    changed[:, :2] = changed[:, :2] * 0.7 + [0.1, 0.05]
    np.testing.assert_allclose(
        frame_features(raw)[0], frame_features(changed)[0], atol=1e-7
    )


def test_aspect_correction():
    raw = np.asarray(demo_clip("waving", 2)[0]["landmarks"])
    wide = raw.copy()
    wide[:, 0] /= 1.7
    np.testing.assert_allclose(
        frame_features(raw)[0], frame_features(wide, 1.7)[0], atol=1e-7
    )


def test_occlusion_and_gap_rejected():
    frames = demo_clip("standing", 2)
    for f in frames:
        f["landmarks"][27][3] = 0.1
    with pytest.raises(PoseQualityError):
        extract_window(frames)
    frames = demo_clip("standing", 2)
    with pytest.raises(PoseQualityError):
        extract_window(frames[:7] + frames[12:])


def test_irregular_sampling_uses_time():
    frames = demo_clip("waving", 2)
    f = extract_window(frames)
    irregular = copy.deepcopy(frames)
    for i in range(1, len(irregular) - 1):
        irregular[i]["t"] += 0.01 * (-1) ** i
    g = extract_window(irregular)
    np.testing.assert_allclose(f["temporal"][:18], g["temporal"][:18], atol=0.03)


def test_lower_sampling_rate_keeps_a_full_time_window():
    from posehar.synthetic import skeleton

    frames = [{"t": i / 6, "landmarks": skeleton("standing", i / 6)} for i in range(13)]
    assert rule_prediction(extract_window(frames))[0] == "standing"


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_rejected(value):
    frames = demo_clip("standing", 2)
    frames[0]["landmarks"][0][0] = value
    with pytest.raises(ValueError):
        Sequence(frames=frames)


def test_timestamps_must_increase():
    frames = demo_clip("standing", 2)
    frames[1]["t"] = 0
    with pytest.raises(ValueError):
        Sequence(frames=frames)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "CLIP_DIR", tmp_path / "data")
    monkeypatch.setattr(app_module, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(app_module, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(app_module, "_cached_model", None)
    monkeypatch.setattr(app_module, "_model_stamp", None)
    return TestClient(app_module.app)


def test_api_modes_and_safe_rejection(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/status").json()["trained"] is False
    body = {"frames": demo_clip("standing", 2), "aspect": 1, "engine": "rules"}
    assert client.post("/api/predict", json=body).json()["label"] == "standing"
    body["engine"] = "trained"
    assert client.post("/api/predict", json=body).status_code == 409
    body["frames"] = body["frames"][:5]
    assert client.post("/api/predict", json=body).json()["label"] == "unknown"


def test_dataset_rejects_synthetic_and_unconsented(client):
    body = {
        "frames": demo_clip("standing", 4),
        "label": "standing",
        "subject": "P01",
        "source": "synthetic",
        "consent": True,
    }
    assert client.post("/api/clips", json=body).status_code == 422
    body["source"] = "camera"
    body["consent"] = False
    assert client.post("/api/clips", json=body).status_code == 422
    body["consent"] = True
    # Procedural input is deliberately used as a fixture in this temporary directory only.
    assert client.post("/api/clips", json=body).status_code == 201
    assert client.get("/api/dataset").json()["counts"]["P01"]["standing"] == 1
    assert (
        client.post(
            "/api/clips", json=body, headers={"origin": "https://other.example"}
        ).status_code
        == 403
    )


def fixture_clips(root):
    root.mkdir()
    for subject in ("A", "B", "C"):
        for label in LABELS:
            body = {
                "subject": subject,
                "label": label,
                "source": "camera",
                "consent": True,
                "frames": demo_clip(label, 4, phase=0.1 * ord(subject)),
            }
            (root / f"{subject}-{label}.json").write_text(json.dumps(body))


def test_splits_and_end_to_end_training(tmp_path):
    # End-to-end SOFTWARE TEST. Its procedural metrics are never project findings.
    clips = tmp_path / "clips"
    fixture_clips(clips)
    loaded = load_clips(clips)
    groups = split_clips(loaded, ["B"], ["C"])
    assert groups == {"train": {"A"}, "validation": {"B"}, "test": {"C"}}
    with pytest.raises(ValueError):
        split_clips(loaded, ["B"], ["B"])
    result = train_project(
        clips, tmp_path / "models", tmp_path / "results", ["B"], ["C"]
    )
    assert (tmp_path / "models" / "activity.joblib").exists()
    assert result["feature_version"] == "pose18-temporal-v1"
    assert result["split_subjects"]["test"] == ["C"]
    assert all(
        row["subject"] == "C"
        for row in result["clip_manifest"]
        if row["split"] == "test"
    )
    assert len(result["test"]["temporal_raw"]["confusion_matrix"]) == 7
    assert result["test"]["temporal_raw"]["windows"] > 0
    assert (tmp_path / "results" / result["run_id"] / "test_predictions.csv").exists()


def test_synthetic_files_blocked(tmp_path):
    data = {
        "subject": "A",
        "label": "standing",
        "source": "synthetic",
        "consent": True,
        "frames": demo_clip("standing", 4),
    }
    (tmp_path / "fake.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Synthetic"):
        load_clips(tmp_path)
