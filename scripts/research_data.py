#!/usr/bin/env python3
"""Download a deterministic, prediction-independent Kinetics-skeleton pilot.

Uses bounded HTTP byte ranges against the official archive linked by the
CoST-GCN authors. It never downloads the full 7.47 GB archive. Raw JSON poses
are retained unchanged; model preprocessing belongs to the benchmark runner.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import threading
import urllib.request
import zipfile
import zlib


SOURCE_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1PvIM_FjDRVSKh_kRDiWBpKkNNp3fUH3o&export=download&confirm=t"
)
SOURCE_FOLDER = (
    "https://drive.google.com/drive/folders/1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ"
)
PROVENANCE_URL = (
    "https://github.com/LukasHedegaard/continual-skeletons#dataset-preparation"
)
ARCHIVE_SIZE = 7_468_059_313
MAX_REQUEST_BYTES = 36 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RangeArchive(io.RawIOBase):
    """Seekable remote ZIP with exact response validation and transfer limits."""

    def __init__(self, cache_dir: Path, max_transfer_bytes: int):
        self.position = 0
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_transfer_bytes = max_transfer_bytes
        self.transferred = 0
        self.lock = threading.Lock()

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = ARCHIVE_SIZE + offset
        if not 0 <= self.position <= ARCHIVE_SIZE:
            raise ValueError("Seek outside known archive")
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = ARCHIVE_SIZE - self.position
        size = min(size, ARCHIVE_SIZE - self.position)
        data = self.fetch(self.position, size)
        self.position += len(data)
        return data

    def fetch(self, offset: int, size: int) -> bytes:
        if not 0 <= size <= MAX_REQUEST_BYTES:
            raise ValueError(f"Refusing oversized byte-range request: {size}")
        if size == 0:
            return b""
        if offset < 0 or offset + size > ARCHIVE_SIZE:
            raise ValueError("Range outside known archive")
        cache = self.cache_dir / f"{offset}-{size}.bin"
        if cache.exists():
            cached = cache.read_bytes()
            if len(cached) == size:
                return cached
        with self.lock:
            if self.transferred + size > self.max_transfer_bytes:
                raise RuntimeError("Configured transfer budget would be exceeded")
            self.transferred += size
        expected = f"bytes {offset}-{offset + size - 1}/{ARCHIVE_SIZE}"
        request = urllib.request.Request(SOURCE_URL)
        request.add_header("Range", f"bytes={offset}-{offset + size - 1}")
        with urllib.request.urlopen(request, timeout=60) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range") != expected
            ):
                raise RuntimeError("Server did not honor the exact bounded byte range")
            data = response.read(size)
        if len(data) != size:
            raise RuntimeError("Truncated range response")
        cache.write_bytes(data)
        return data

    def member(self, entry: zipfile.ZipInfo) -> bytes:
        # Fetch local header + payload together. ZIP local metadata is bounded
        # separately; filenames and extra fields fit inside this generous bound.
        size = min(entry.compress_size + 4096, ARCHIVE_SIZE - entry.header_offset)
        block = self.fetch(entry.header_offset, size)
        if block[:4] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        name_length, extra_length = struct.unpack_from("<HH", block, 26)
        start = 30 + name_length + extra_length
        if start > 4096:
            raise ValueError("Unexpectedly large ZIP local metadata")
        compressed = block[start : start + entry.compress_size]
        if entry.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(compressed, -15)
        elif entry.compress_type == zipfile.ZIP_STORED:
            data = compressed
        else:
            raise ValueError("Unsupported compression")
        if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
            raise ValueError("ZIP member integrity check failed")
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".build/research-data"))
    parser.add_argument("--calibration", type=int, default=200)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--seed", default="2026")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-transfer-mib", type=int, default=128)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = RangeArchive(args.out / "range-cache", args.max_transfer_mib * 1024**2)
    with zipfile.ZipFile(archive) as remote_zip:
        entries = remote_zip.infolist()
    print(f"Read archive index: {len(entries)} entries", flush=True)
    label_entries = [
        e for e in entries if "label" in e.filename and e.filename.endswith(".json")
    ]
    print("Label files:", [e.filename for e in label_entries], flush=True)
    label_entry = next(
        e for e in label_entries if e.filename.endswith("kinetics_val_label.json")
    )
    labels_raw = archive.member(label_entry)
    (args.out / "kinetics_val_label.json").write_bytes(labels_raw)
    labels = json.loads(labels_raw)
    candidates = [
        e
        for e in entries
        if "/kinetics_val/" in e.filename and e.filename.endswith(".json")
    ]
    # Include empty-pose clips in sampling. Their status is recorded, never
    # filtered by label confidence or recognition success.
    candidates.sort(
        key=lambda e: (
            sha256(f"{args.seed}:{Path(e.filename).stem}".encode()),
            e.filename,
        )
    )
    total = args.calibration + args.test
    if len(candidates) < total:
        raise RuntimeError(
            f"Only {len(candidates)} candidate clips for {total} requested"
        )
    selected = candidates[:total]
    print(f"Validation candidates: {len(candidates)}; selected: {total}", flush=True)
    if args.inspect_only:
        print("First entries:", [e.filename for e in selected[:5]])
        print("First label:", next(iter(labels.items())))
        return
    class_map = {}
    for value in labels.values():
        class_map[str(value["label_index"])] = value["label"]
    (args.out / "classes.json").write_text(json.dumps(class_map, indent=2) + "\n")
    rows = []

    def download(index_entry):
        index, entry = index_entry
        clip_id = Path(entry.filename).stem
        if not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
            raise ValueError("Unexpected clip identifier")
        partition = "calibration" if index < args.calibration else "test"
        destination = args.out / partition / f"{clip_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raw = destination.read_bytes()
            if len(raw) != entry.file_size or zlib.crc32(raw) != entry.CRC:
                raise ValueError("Cached JSON member does not match archive CRC")
        else:
            raw = archive.member(entry)
            destination.write_bytes(raw)
        clip = json.loads(raw)
        reference = labels[clip_id]
        if clip["label_index"] != reference["label_index"]:
            raise ValueError("Official label and clip disagree")
        return {
            "id": clip_id,
            "partition": partition,
            "path": str(destination.relative_to(args.out)),
            "archive_member": entry.filename,
            "sha256": sha256(raw),
            "bytes": len(raw),
            "label_index": reference["label_index"],
            "label": reference["label"],
            "has_skeleton": reference["has_skeleton"],
            "selection_hash": sha256(f"{args.seed}:{clip_id}".encode()),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item) for item in enumerate(selected)]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 25 == 0 or len(rows) == 1:
                print(f"Downloaded {len(rows)}/{total} clips", flush=True)
    rows.sort(key=lambda row: row["selection_hash"])
    manifest = {
        "dataset": "Kinetics-skeleton (ST-GCN OpenPose extraction)",
        "source_folder": SOURCE_FOLDER,
        "source_archive_url": SOURCE_URL,
        "source_archive_bytes": ARCHIVE_SIZE,
        "paper_repository_source": PROVENANCE_URL,
        "official_partition": "validation",
        "sampling": "Ascending SHA256(seed + ':' + clip_id); first calibration_count for calibration, subsequent test_count for test; no empty-pose or class filtering",
        "seed": args.seed,
        "calibration_count": args.calibration,
        "test_count": args.test,
        "candidate_count": len(candidates),
        "class_count_original": len(class_map),
        "calibration_classes": len(
            {r["label_index"] for r in rows if r["partition"] == "calibration"}
        ),
        "test_classes": len(
            {r["label_index"] for r in rows if r["partition"] == "test"}
        ),
        "empty_skeleton_count": sum(not r["has_skeleton"] for r in rows),
        "labels_sha256": sha256(labels_raw),
        "download_bytes_this_run": archive.transferred,
        "clips": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "clips"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
