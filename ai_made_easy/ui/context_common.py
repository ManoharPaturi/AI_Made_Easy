"""Helpers shared by AppContext (context.py) and its action mixins (context_*.py)."""
from __future__ import annotations

import json
import os
from pathlib import Path

_TRAINER_BLOCK = "train.trainer"
_SETTINGS = "aime/workbench"


def api_module():  # noqa: ANN201
    from ai_made_easy.core import api

    return api


def tracking_label(target: str) -> str:
    from ai_made_easy.core.tracking import LABELS

    return LABELS.get(target, target)


def _json_or(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _load_trained(workdir: Path, class_name: str):
    """Import the run's training script as a module and load its checkpoint."""
    import importlib.util

    import torch

    script = next(iter(sorted(workdir.glob("*_train_pytorch.py"))), None)
    if script is None:
        raise RuntimeError("no PyTorch training script in the run folder")
    spec = importlib.util.spec_from_file_location("aime_trained", script)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    model = getattr(module, class_name)()
    model.load_state_dict(torch.load(workdir / module.CHECKPOINT, map_location="cpu"))
    model.eval()
    norm = ((tuple(module.NORM_MEAN), tuple(module.NORM_STD))
            if hasattr(module, "NORM_MEAN") else None)
    return model, tuple(module.INPUT_SHAPE), norm
