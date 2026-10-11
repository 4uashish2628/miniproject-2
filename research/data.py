"""Validated pilot data using the authors' exact Kinetics preprocessing."""

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from research.upstream import UPSTREAM, _definitions


def load_partition(root, partition):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = [r for r in manifest["clips"] if r["partition"] == partition]
    ns = dict(np=np, json=json, os=os, pickle=pickle, Dataset=Dataset)
    _definitions(
        UPSTREAM / "datasets/data_preparation/kinetics400_prep.py", ns, assignments=True
    )
    feeder = ns["Feeder_kinetics"](
        str(root / partition),
        str(root / "kinetics_val_label.json"),
        ignore_empty_sample=False,
    )
    index = {Path(name).stem: i for i, name in enumerate(feeder.sample_name)}
    samples, lengths = [], []
    for row in rows:
        raw = (root / row["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {row['id']}")
        clip = json.loads(raw)
        length = max((f["frame_index"] for f in clip["data"]), default=-1) + 1
        if not 0 <= length <= 300:
            raise ValueError("Unexpected frame index beyond the authors' input window")
        sample, label = feeder[index[row["id"]]]
        if label != row["label_index"]:
            raise ValueError("Manifest and original feeder labels disagree")
        samples.append(sample.astype(np.float32))
        lengths.append(length)
    return (
        np.stack(samples),
        np.array([r["label_index"] for r in rows]),
        rows,
        lengths,
        manifest,
    )
