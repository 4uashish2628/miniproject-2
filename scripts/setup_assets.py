"""Download pinned MediaPipe runtime and official Lite model for offline use."""

from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "static"
VERSION = "0.10.21"
BASE = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{VERSION}"
FILES = {
    "vendor/vision_bundle.mjs": BASE + "/vision_bundle.mjs",
    "vendor/wasm/vision_wasm_internal.js": BASE + "/wasm/vision_wasm_internal.js",
    "vendor/wasm/vision_wasm_internal.wasm": BASE + "/wasm/vision_wasm_internal.wasm",
    "vendor/wasm/vision_wasm_nosimd_internal.js": BASE
    + "/wasm/vision_wasm_nosimd_internal.js",
    "vendor/wasm/vision_wasm_nosimd_internal.wasm": BASE
    + "/wasm/vision_wasm_nosimd_internal.wasm",
    "models/pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def main():
    records = []
    for name, url in FILES.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            with urllib.request.urlopen(url, timeout=90) as response:
                body = response.read()
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(body)
            temporary.replace(path)
        records.append(
            {
                "file": name,
                "source": url,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (ROOT / "asset_manifest.json").write_text(
        json.dumps({"mediapipe_version": VERSION, "files": records}, indent=2)
    )
    print("Pose assets ready. The app can now run without internet.")


if __name__ == "__main__":
    main()
