"""Run the authors' neural-network definitions without their legacy trainer.

Model method bodies come directly from the pinned external checkout. Only the
Ride training mixins/imports are omitted. Both clip and continual definitions
are available for a numerical equivalence check. No upstream code is vendored.
"""

import ast
import hashlib
import io
import logging
import math
import pickle
import subprocess
from collections import OrderedDict, defaultdict
from operator import attrgetter
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import continual as co
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".build/continual-skeletons"
COMMIT = "a8fe2937a33f24cce65c1f8c2fc41081bceda721"
CHECKPOINT = ROOT / ".build/stgcnmod_kinetics_joint.ckpt"
CHECKPOINT_SHA256 = "bec5c9563cc017a6e4739c2255ad76b24a046e445210b8fc965816435229452f"
SOURCE_HASHES = {
    "models/utils.py": "b809080b603d12e31009d92245364285837290fd254fc8e15ab18a3980323d2c",
    "models/base.py": "621994a185086b1b6e0f99c0af60207fb0469139e1603cf9f4695cdb1025a06c",
    "datasets/graph.py": "84bb4a533ef4b90d5c379a6ceb753182569d41d073a8c9f54d02dd4f868c5786",
    "datasets/kinetics.py": "f994b7121a4bd972e528f0f2e4abb8e94afafd34190ac53b98cbb51fda43364d",
    "models/st_gcn_mod/st_gcn_mod.py": "60a83bf4f8b735e0f15c48aa0d7c11d053c7a94607036678f7e972350b0e1373",
    "models/cost_gcn_mod/cost_gcn_mod.py": "bc472b1f0deed58cbcb440c27934ee3dec5861864b8cf67bacd3e73d1d597162",
    "datasets/data_preparation/kinetics400_prep.py": "95fcef6f68c6be5ae6979b6083de2d6ca98d5ffbad9dcb6f362bbad1559782cd",
}


class _TrainingMetadata:
    """Inert placeholder for obsolete training-only checkpoint objects."""


def _metadata_getattr(obj, name):
    if isinstance(obj, _TrainingMetadata) and name in {
        "_annealing_cos",
        "_annealing_linear",
    }:
        return None
    raise pickle.UnpicklingError(f"Unexpected checkpoint attribute: {name}")


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("collections", "defaultdict"): defaultdict,
            ("__builtin__", "dict"): dict,
            ("__builtin__", "getattr"): _metadata_getattr,
            ("torch", "FloatStorage"): torch.FloatStorage,
            ("torch", "LongStorage"): torch.LongStorage,
            ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch.optim.lr_scheduler", "OneCycleLR"): _TrainingMetadata,
            ("torch.optim.sgd", "SGD"): _TrainingMetadata,
            (
                "pytorch_lightning.callbacks.model_checkpoint",
                "ModelCheckpoint",
            ): _TrainingMetadata,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(
                f"Unexpected checkpoint class: {module}.{name}"
            )
        return allowed[module, name]


def checkpoint_state(path=CHECKPOINT):
    path = Path(path)
    verified_bytes = path.read_bytes()
    digest = hashlib.sha256(verified_bytes).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError(
            "Checkpoint hash differs from the inspected authors' checkpoint."
        )
    # Legacy Lightning stores optimizer objects. Load only an explicit whitelist;
    # the three training classes become inert metadata containers.
    module = SimpleNamespace(
        __name__="restricted_checkpoint",
        Unpickler=_CheckpointUnpickler,
        load=pickle.load,
    )
    payload = torch.load(
        io.BytesIO(verified_bytes),
        map_location="cpu",
        pickle_module=module,
        weights_only=False,
    )
    state = payload["state_dict"]
    if state["fc.weight"].shape != (400, 256):
        raise ValueError("Expected the 400-class Kinetics checkpoint.")
    return state


def _definitions(path, namespace, bases=None, assignments=False):
    """Import original definitions; substitute framework bases, never methods."""
    source = path.read_bytes()
    relative = next((k for k in SOURCE_HASHES if str(path).endswith("/" + k)), None)
    if (
        relative is None
        or hashlib.sha256(source).hexdigest() != SOURCE_HASHES[relative]
    ):
        raise ValueError(f"Upstream source hash mismatch: {path}")
    tree = ast.parse(source.decode().lstrip(), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef) and bases and node.name in bases:
                node.bases = [
                    ast.parse(base, mode="eval").body for base in bases[node.name]
                ]
            body.append(node)
        elif assignments and isinstance(node, ast.Assign):
            body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def load_model(continual=False, upstream=UPSTREAM, checkpoint=CHECKPOINT):
    upstream = Path(upstream)
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != COMMIT:
        raise ValueError(f"Upstream checkout must be pinned to {COMMIT}.")
    ns = dict(
        np=np,
        nn=nn,
        torch=torch,
        co=co,
        math=math,
        prod=np.prod,
        _BatchNorm=_BatchNorm,
        _ConvNd=_ConvNd,
        OrderedDict=OrderedDict,
        Sequence=Sequence,
        List=List,
        Tuple=Tuple,
        Tensor=Tensor,
        attrgetter=attrgetter,
        Configs=object,
        logger=logging.getLogger("research.upstream"),
    )
    _definitions(upstream / "models/utils.py", ns)
    _definitions(upstream / "models/base.py", ns, {"CoModelBase": ["co.Sequential"]})
    _definitions(upstream / "datasets/graph.py", ns)
    _definitions(upstream / "datasets/kinetics.py", ns, assignments=True)
    name = "CoStGcnMod" if continual else "StGcnMod"
    folder = "cost_gcn_mod" if continual else "st_gcn_mod"
    _definitions(
        upstream / f"models/{folder}/{folder}.py",
        ns,
        {name: ["CoModelBase"] if continual else ["nn.Module"]},
    )
    cls = ns[name]
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.input_shape = (3, 300, 18, 2)
    model.num_classes = 400
    model.graph = ns["graph"]
    model.hparams = SimpleNamespace(
        pool_size=-1, pool_padding=-1, forward_mode="clip", profile_model=False
    )
    cls.__init__(model, model.hparams)
    if continual:
        ns["CoModelBase"].on_init_end(model, model.hparams)
    state = checkpoint_state(checkpoint)
    if continual:
        state = model.map_state_dict(state)
    model.load_state_dict(state, strict=True)
    return model.eval()


def verify_equivalence(sample, atol=5e-4):
    """Compare clip execution to the authors' continual batch and step paths."""
    if (
        sample.ndim != 5
        or tuple(sample.shape[1:]) != (3, 300, 18, 2)
        or sample.dtype != torch.float32
        or not torch.isfinite(sample).all()
    ):
        raise ValueError(
            "Equivalence check requires finite float32 [N,3,300,18,2] inputs."
        )
    clip = load_model()
    stream = load_model(continual=True)
    with torch.inference_mode():
        a = clip(sample)
        b = stream(sample)
        stream.clean_state()
        c = stream.forward_steps(sample)
        if c.ndim == 3:
            c = c[:, :, -1]
    torch.testing.assert_close(a, b, atol=atol, rtol=1e-4)
    torch.testing.assert_close(a, c, atol=atol, rtol=1e-4)
    return {
        "passed": True,
        "clip_vs_continual_max_abs": float((a - b).abs().max()),
        "clip_vs_step_max_abs": float((a - c).abs().max()),
        "atol": atol,
        "classes": a.shape[1],
        "input_shape": list(sample.shape),
    }
