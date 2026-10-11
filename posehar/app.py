import json
import time
from pathlib import Path
from uuid import uuid4
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .config import ROOT, CLIP_DIR, MODEL_DIR, RESULT_DIR, LABELS, FEATURE_VERSION
from .features import extract_window, PoseQualityError, rule_prediction
from .schema import PredictRequest, Clip, TrainRequest
from .training import train_project, inventory, clip_windows, TRAIN_LOCK
from .synthetic import demo_clip

app = FastAPI(title="PoseLab", version="1.0.0", docs_url="/api/docs")
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
_cached_model = None
_model_stamp = None


@app.middleware("http")
async def local_requests_only(request: Request, call_next):
    # Local app: prevent cross-site writes to participant recordings or model files.
    origin = request.headers.get("origin")
    if (
        request.method in {"POST", "PUT", "DELETE"}
        and origin
        and origin != str(request.base_url).rstrip("/")
    ):
        return JSONResponse(
            {"detail": "Use the local PoseLab tab for this action."}, status_code=403
        )
    return await call_next(request)


def active_model():
    global _cached_model, _model_stamp
    path = MODEL_DIR / "activity.joblib"
    if not path.exists():
        raise HTTPException(
            409, "Train a model with real recordings in the Dataset tab first."
        )
    stamp = path.stat().st_mtime_ns
    if stamp != _model_stamp:
        # Only locally generated artifacts are supported. Never load downloaded joblib files.
        model = joblib.load(path)
        if model["feature_version"] != FEATURE_VERSION:
            raise HTTPException(409, "Feature version changed. Retrain the classifier.")
        _cached_model, _model_stamp = model, stamp
    return _cached_model


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "research.html")


@app.get("/lab")
def activity_lab():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/research")
def research_comparison():
    return FileResponse(ROOT / "static" / "research.html")


@app.get("/research/report")
def research_report():
    path = ROOT / "results" / "research" / "report.html"
    if not path.is_file():
        raise HTTPException(404, "The measured report has not been generated yet.")
    return FileResponse(path)


@app.get("/research/download/{kind}")
def research_download(kind: str):
    artifacts = {
        "report": ROOT / "output" / "pdf" / "PoseLab_Research_Results.pdf",
        "slides": ROOT / "deliverables" / "PoseLab_Research_Presentation.pptx",
    }
    path = artifacts.get(kind)
    if path is None or not path.is_file():
        raise HTTPException(404, "This research artifact is not available.")
    return FileResponse(path, filename=path.name)


@app.get("/api/status")
def status():
    return {
        "trained": (MODEL_DIR / "activity.joblib").exists(),
        "activities": LABELS,
        "pose_assets_ready": all(
            (ROOT / "static" / p).exists()
            for p in [
                "vendor/vision_bundle.mjs",
                "vendor/wasm/vision_wasm_internal.wasm",
                "models/pose_landmarker_lite.task",
            ]
        ),
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    start = time.perf_counter()
    try:
        f = extract_window(req.frames, req.aspect)
    except PoseQualityError as e:
        return {
            "label": "unknown",
            "quality": 0,
            "reason": str(e),
            "score": None,
            "engine": req.engine,
            "current": {},
        }
    if req.engine == "rules":
        label, reason = rule_prediction(f)
        score, probabilities = None, {}
    else:
        payload = active_model()
        probs = payload["model"].predict_proba([f["temporal"]])[0]
        score = float(probs.max())
        label = (
            str(payload["model"].classes_[np.argmax(probs)])
            if score >= payload["threshold"]
            else "unknown"
        )
        reason = (
            "Learned from recorded landmark sequences."
            if label != "unknown"
            else "Model vote score is below the validation threshold."
        )
        probabilities = dict(zip(payload["model"].classes_.tolist(), probs.tolist()))
    return {
        "label": label,
        "quality": f["quality"],
        "score": score,
        "reason": reason,
        "engine": req.engine,
        "probabilities": probabilities,
        "current": f["current"],
        "classifier_ms": (time.perf_counter() - start) * 1000,
    }


@app.get("/api/demo/{label}")
def demo(label: str):
    if label not in LABELS:
        raise HTTPException(404, "Unknown activity")
    return {"source": "synthetic", "frames": demo_clip(label), "aspect": 1}


@app.get("/api/dataset")
def dataset():
    try:
        return inventory(CLIP_DIR)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/api/clips", status_code=201)
def save_clip(clip: Clip):
    if clip.source == "synthetic":
        raise HTTPException(
            422, "Synthetic examples cannot enter the research dataset."
        )
    if not clip.consent:
        raise HTTPException(
            422, "Confirm permission before saving participant landmarks."
        )
    if clip.frames[-1].t - clip.frames[0].t < 4:
        raise HTTPException(422, "Record at least four seconds.")
    if sum(w is not None for w in clip_windows(clip)) < 2:
        raise HTTPException(
            422,
            "Insufficient usable windows. Keep the full body visible and record again.",
        )
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clip_id = uuid4().hex
    (CLIP_DIR / f"{clip_id}.json").write_text(clip.model_dump_json())
    return {
        "id": clip_id,
        "frames": len(clip.frames),
        "subject": clip.subject,
        "label": clip.label,
    }


@app.post("/api/train")
def train(req: TrainRequest):
    if not TRAIN_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Training is already running.")
    try:
        return train_project(
            CLIP_DIR, MODEL_DIR, RESULT_DIR, req.validation_subjects, req.test_subjects
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        TRAIN_LOCK.release()


@app.get("/api/results")
def results():
    path = RESULT_DIR / "latest.json"
    return json.loads(path.read_text()) if path.exists() else {"pending": True}


@app.get("/api/results/download")
def download_results():
    path = RESULT_DIR / "latest.json"
    if not path.exists():
        raise HTTPException(404, "Run the experiment first.")
    return FileResponse(path, filename="poselab_metrics.json")
