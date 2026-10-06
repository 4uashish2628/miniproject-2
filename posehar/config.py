from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("POSELAB_DATA", str(ROOT))).resolve()
CLIP_DIR = DATA_ROOT / "data" / "clips"
MODEL_DIR = DATA_ROOT / "models"
RESULT_DIR = DATA_ROOT / "results" / "runs"
LABELS = ("standing", "sitting", "marching", "squats", "jumping_jacks", "waving")
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
SAMPLE_HZ = 10
FEATURE_VERSION = "pose18-temporal-v1"
