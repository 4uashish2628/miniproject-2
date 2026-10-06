import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posehar.config import CLIP_DIR, MODEL_DIR, RESULT_DIR
from posehar.training import train_project

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train with separate validation and test participants"
    )
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True)
    args = p.parse_args()
    result = train_project(CLIP_DIR, MODEL_DIR, RESULT_DIR, args.validation, args.test)
    print("Saved experiment:", result["run_id"])
    print("Test results:", result["test"])
