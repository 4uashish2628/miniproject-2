#!/usr/bin/env python3
"""Fetch the pinned authors' source and verified research checkpoint.

This script installs no packages and never resets an existing checkout or
replaces a mismatched checkpoint. Run from any directory with Python 3.10+.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/LukasHedegaard/continual-skeletons.git"
SOURCE_COMMIT = "a8fe2937a33f24cce65c1f8c2fc41081bceda721"
CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1RcHgORWHlwUcM9B5GpvR_BYsh6glkRc0&export=download&confirm=t"
)
CHECKPOINT_SHA256 = "bec5c9563cc017a6e4739c2255ad76b24a046e445210b8fc965816435229452f"
CHECKPOINT_BYTES = 25_938_363


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_source(path: Path) -> None:
    if not path.is_dir() or not (path / ".git").exists():
        raise ValueError(f"Existing source path is not a Git checkout: {path}")
    actual = git_output(path, "rev-parse", "HEAD")
    if actual != SOURCE_COMMIT:
        raise ValueError(
            f"Existing checkout is at {actual}, expected {SOURCE_COMMIT}. "
            "Move it aside or choose --source-dir; it will not be reset."
        )
    if git_output(path, "status", "--porcelain"):
        raise ValueError(
            f"Existing upstream checkout has local changes: {path}. "
            "Preserve those changes and use a clean checkout; nothing was overwritten."
        )


def verify_checkpoint(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError(f"Existing checkpoint has unexpected size: {path}")
    if digest_file(path) != CHECKPOINT_SHA256:
        raise ValueError(
            f"Existing checkpoint checksum does not match: {path}. "
            "It will not be overwritten; move it aside before retrying."
        )


def fetch_source(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="posehar-source-", dir=destination.parent
    ) as temporary:
        checkout = Path(temporary) / "checkout"
        subprocess.run(
            ["git", "clone", "--no-checkout", SOURCE_URL, str(checkout)], check=True
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", SOURCE_COMMIT],
            check=True,
        )
        verify_source(checkout)
        if destination.exists():
            raise FileExistsError(
                f"Destination appeared during download: {destination}"
            )
        checkout.rename(destination)


def fetch_checkpoint(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="posehar-checkpoint-", dir=destination.parent
    ) as temporary:
        downloaded = Path(temporary) / "checkpoint.ckpt"
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"Unexpected download status: {response.status}")
            if "text/html" in response.headers.get("Content-Type", ""):
                raise RuntimeError(
                    "Download returned a web page instead of a checkpoint"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != CHECKPOINT_BYTES:
                raise RuntimeError("Checkpoint download has unexpected declared size")
            written = 0
            with downloaded.open("wb") as output:
                while written <= CHECKPOINT_BYTES:
                    chunk = response.read(
                        min(1024 * 1024, CHECKPOINT_BYTES + 1 - written)
                    )
                    if not chunk:
                        break
                    output.write(chunk)
                    written += len(chunk)
            if written != CHECKPOINT_BYTES:
                raise RuntimeError(
                    "Checkpoint download has unexpected or incomplete size"
                )
        verify_checkpoint(downloaded)
        if destination.exists():
            raise FileExistsError(
                f"Destination appeared during download: {destination}"
            )
        downloaded.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / ".build/continual-skeletons"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / ".build/stgcnmod_kinetics_joint.ckpt"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing assets without any writes or network requests",
    )
    args = parser.parse_args()
    if shutil.which("git") is None:
        parser.error("Git is required. Install Git before running this setup.")
    source = args.source_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    try:
        # Preflight all existing assets before starting any downloads.
        if source.exists() or args.verify_only:
            verify_source(source)
        if checkpoint.exists() or args.verify_only:
            verify_checkpoint(checkpoint)
        if not args.verify_only:
            if not source.exists():
                print("Downloading the authors' pinned source...", flush=True)
                fetch_source(source)
            if not checkpoint.exists():
                print("Downloading the authors' 25.9 MB checkpoint...", flush=True)
                fetch_checkpoint(checkpoint)
    except (ValueError, OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        parser.exit(1, f"Research setup stopped: {exc}\n")
    print(f"Source verified: {source}\nCommit: {SOURCE_COMMIT}")
    print(f"Checkpoint verified: {checkpoint}\nSHA256: {CHECKPOINT_SHA256}")
    print(
        "Assets are ready. Install requirements-research.txt separately to run the comparison."
    )


if __name__ == "__main__":
    main()
