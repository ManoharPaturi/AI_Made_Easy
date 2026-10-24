"""Resume an interrupted run (3.9).

PyTorch training scripts save everything training needs after each epoch (``resume.pt``:
the model, optimizer, scheduler, AMP scaler, averaged weights, the best weights so far and
the random state; ``resume.json`` says which epoch). A run that stopped, failed or was
interrupted continues as the same run: its recorded epochs are cut back to the saved one
and the script runs again in its folder, starting from the next epoch. The file is removed
when a run finishes, so finished runs carry no extra weight.
"""
from __future__ import annotations

import json

from ai_made_easy.core.graph import Graph

RESUMABLE = ("stopped", "failed", "interrupted")
STATE, INFO, SCRIPT = "resume.pt", "resume.json", "resume_run.py"

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
    return int(info["epoch"]) if (folder / STATE).is_file() and "epoch" in info else None


def check(mgr, run_id: str) -> int:  # noqa: ANN001
    """The epoch to resume after; raises ResumeError saying why a run can't resume."""
    from ai_made_easy.core.runner.manager import FINAL_STATES

    record = mgr.history.get(run_id)
    if mgr.is_live(run_id) and mgr.get(run_id).state not in FINAL_STATES:
        raise ResumeError(f"run {run_id} is still running")
    if record.status not in RESUMABLE:
        raise ResumeError(f"run {run_id} is {record.status}: only a stopped, failed or "
                          "interrupted run resumes")
    if record.backend:
        raise ResumeError(f"run {run_id} trained on {record.backend}: resume runs on this "
                          "computer")
    from ai_made_easy.core.pipelines import stages as S

    try:
        S.script_of(mgr.history.path(run_id))
    except FileNotFoundError as exc:
        raise ResumeError("resume continues PyTorch networks") from exc
    epoch = saved_epoch(mgr, run_id)
    if epoch is None:
        raise ResumeError(f"run {run_id} saved no epoch to resume from (it stopped during "
                          "its first epoch, or was trained before 3.9)")
    return epoch


def start(mgr, run_id: str) -> int:  # noqa: ANN001
    """Continue the run after its saved epoch; returns that epoch."""
    from ai_made_easy.core.pipelines import stages as S

    epoch = check(mgr, run_id)
    folder = mgr.history.path(run_id)
    script = folder / SCRIPT
    script.write_text(_WRAPPER.format(script=S.script_of(folder).name), encoding="utf-8")
    mgr.history.keep_epochs(run_id, epoch)
    mgr.relaunch(run_id, script, Graph.from_dict(mgr.history.get(run_id).graph))
    return epoch
