import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start PoseLab on your laptop")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port} in Chrome or Edge. Stop with Ctrl+C.")
    uvicorn.run("posehar.app:app", host="127.0.0.1", port=args.port)
