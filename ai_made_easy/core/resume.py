"""Resume an interrupted run (3.9; Keras and remote runs in 3.10).

Training scripts save what training needs after each epoch: PyTorch's ``resume.pt`` (the
model, optimizer, scheduler, AMP scaler, averaged weights, the best weights so far and the
random state), Keras' ``resume.weights.h5`` (weights and optimizer) with
``resume.extra.pkl`` (the random state and the early-stopping and plateau counters);
``resume.json`` says which epoch. A run that stopped, failed or was interrupted continues
as the same run: its recorded epochs are cut back to the saved one and the script runs
again in its folder, starting from the next epoch. The files are removed when a run
finishes, so finished runs carry no extra weight.

A run trained elsewhere (SSH, Modal, a container, Kaggle, Colab) resumes there: its
backend brings the saved files back with the outputs when the run ends, and the resumed
run's bundle carries them out again.
"""
from __future__ import annotations

import json
from pathlib import Path

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runs.history import RESUME_FILES

RESUMABLE = ("stopped", "failed", "interrupted")
INFO, SCRIPT = "resume.json", "resume_run.py"
WEIGHTS = ("resume.pt", "resume.weights.h5")   # PyTorch, Keras

_WRAPPER = '''"""Resume this run from its last saved epoch (written by AI Made Easy)."""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("resumed_model", Path({script!r}))
module = importlib.util.module_from_spec(spec)
sys.modules["resumed_model"] = module
spec.loader.exec_module(module)
module.RESUME = True
module.main()
'''


class ResumeError(ValueError):
    """A run that can't be resumed."""


def saved_epoch(mgr, run_id: str) -> int | None:  # noqa: ANN001
    """The epoch the run can resume after, or None."""
    folder = mgr.history.path(run_id)
    try:
        info = json.loads((folder / INFO).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    saved = any((folder / name).is_file() for name in WEIGHTS)
    return int(info["epoch"]) if saved and "epoch" in info else None


def _training_script(folder: Path) -> Path:
    found = sorted(folder.glob("*_train_pytorch.py")) + sorted(folder.glob("*_train_keras.py"))
    if not found:
        raise ResumeError("resume continues networks trained with PyTorch or Keras")
    return found[0]


def _backend(record):  # noqa: ANN001, ANN202 — a RunRecord; a Backend or None
    """Where the run trained (None: this computer)."""
    if not record.backend:
        return None
    from ai_made_easy.core.runner import backends

    try:
        return backends.resolve(record.backend)
    except backends.BackendError as exc:
        raise ResumeError(f"run {record.run_id} trained on {record.backend}, which is not "
                          f"set up any more: {exc}") from exc


def check(mgr, run_id: str) -> int:  # noqa: ANN001
    """The epoch to resume after; raises ResumeError saying why a run can't resume."""
    from ai_made_easy.core.runner.manager import FINAL_STATES

    record = mgr.history.get(run_id)
    if mgr.is_live(run_id) and mgr.get(run_id).state not in FINAL_STATES:
        raise ResumeError(f"run {run_id} is still running")
    if record.status not in RESUMABLE:
        raise ResumeError(f"run {run_id} is {record.status}: only a stopped, failed or "
                          "interrupted run resumes")
    _training_script(mgr.history.path(run_id))
    _backend(record)
    epoch = saved_epoch(mgr, run_id)
    if epoch is None:
        where = (" (or its outputs did not come back from " + record.backend + ")"
                 if record.backend else "")
        raise ResumeError(f"run {run_id} saved no epoch to resume from: it stopped during "
                          f"its first epoch{where}, or was trained before resume existed")
    return epoch


def _bundle(run_id: str, folder: Path, training: Path, graph: Graph):  # noqa: ANN202
    """The run's bundle again, with the saved state and the resuming script."""
    from ai_made_easy.core.runner.backends.bundle import (
        Bundle,
        relocate_data,
        training_requirements,
    )

    _, files = relocate_data(graph)
    files[training.name] = training
    files.update({name: folder / name for name in RESUME_FILES if (folder / name).is_file()})
    return Bundle(run_id=run_id, run_dir=folder, script=SCRIPT, files=files,
                  requirements=training_requirements(training.read_text(encoding="utf-8")))


def start(mgr, run_id: str) -> int:  # noqa: ANN001
    """Continue the run after its saved epoch, where it trained; returns that epoch."""
    epoch = check(mgr, run_id)
    record = mgr.history.get(run_id)
    folder = mgr.history.path(run_id)
    training = _training_script(folder)
    script = folder / SCRIPT
    script.write_text(_WRAPPER.format(script=training.name), encoding="utf-8")
    graph = Graph.from_dict(record.graph)
    backend = _backend(record)
    bundle = None
    if backend is not None:
        from ai_made_easy.core.runner.backends.bundle import MAX_EMBEDDED_BYTES

        bundle = _bundle(run_id, folder, training, graph)
        if backend.kind in ("kaggle", "colab") and bundle.size() > MAX_EMBEDDED_BYTES:
            raise ResumeError(f"the saved state is {bundle.size() / 1e6:.0f} MB: a {backend.kind} "
                              f"run carries at most {MAX_EMBEDDED_BYTES / 1e6:.0f} MB inside "
                              "its one file")
    mgr.history.keep_epochs(run_id, epoch)
    mgr.relaunch(run_id, script, graph, backend=backend, bundle=bundle)
    return epoch
