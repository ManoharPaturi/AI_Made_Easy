"""Persistent run history: every training run gets a folder and a record.

Layout (``$AIME_HOME/runs/<run_id>/``)::

    run.json        RunRecord — graph snapshot, parameters, status, metrics
    epochs.jsonl    one epoch event per line, appended live
    <script>.py     the generated training script (also the run's cwd, so
                    checkpoints, metrics.json, predictions.json … land here)

Run ids sort chronologically (``20261007-142501-a1b2``). Pure Python.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ai_made_easy.core.paths import subdir

FINAL_STATES = ("finished", "failed", "stopped")


@dataclass
class RunRecord:
    run_id: str
    name: str
    project: str = ""
    framework: str = "pytorch"
    kind: str = "train"
    status: str = "created"  # created | running | finished | failed | stopped
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    returncode: int | None = None
    graph: dict = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    best_metrics: dict[str, float] = field(default_factory=dict)
    epochs_done: int = 0
    env: dict[str, Any] = field(default_factory=dict)
    # measured on the first epoch: peak_memory_mb, step_ms, batch_size, device
    resources: dict[str, Any] = field(default_factory=dict)
    data_fingerprint: str = ""
    tags: list[str] = field(default_factory=list)
    note: str = ""
    parent: str = ""  # sweep id for trials
    trial: dict[str, Any] = field(default_factory=dict)  # sampled sweep values
    error: str = ""
    pid: int | None = None

    @property
    def duration(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.created_at

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration"] = self.duration
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RunRecord:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def graph_params(graph_dict: dict) -> dict[str, Any]:
    """Flat ``node_id.param -> value`` view of every block parameter.

    This is what run comparison diffs and sweeps address.
    """
    out: dict[str, Any] = {}
    for node in graph_dict.get("nodes", []):
        for key, value in (node.get("params") or {}).items():
            out[f"{node['id']}.{key}"] = value
    return out


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


# Keys where smaller is better when picking an epoch's "best" value.
def lower_is_better(metric: str) -> bool:
    m = metric.lower()
    return any(t in m for t in ("loss", "error", "mae", "mse", "rmse", "mape", "mase", "wape",
                                "crps", "pinball", "nll", "perplexity", "wer", "cer"))


class RunHistory:
    """File-backed run records; safe to use from several threads."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else subdir("runs")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------ create
    def create(self, graph_dict: dict, *, framework: str = "pytorch",
               project: str = "", kind: str = "train", parent: str = "",
               trial: dict | None = None, tags: list[str] | None = None,
               data_fingerprint: str | None = None) -> RunRecord:
        if data_fingerprint is None:
            from ai_made_easy.core.data.fingerprint import graph_fingerprint

            try:
                data_fingerprint = graph_fingerprint(graph_dict)
            except Exception:  # noqa: BLE001 — a fingerprint must never block a run
                data_fingerprint = ""
        record = RunRecord(
            run_id=new_run_id(), name=str(graph_dict.get("name") or "model"),
            project=project, framework=framework, kind=kind, graph=graph_dict,
            params=graph_params(graph_dict), parent=parent, trial=dict(trial or {}),
            tags=list(tags or []), data_fingerprint=data_fingerprint)
        self.path(record.run_id).mkdir(parents=True, exist_ok=False)
        self.save(record)
        return record

    # ------------------------------------------------------------- paths
    def path(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
            raise KeyError(f"invalid run id {run_id!r}")
        return self.root / run_id

    def exists(self, run_id: str) -> bool:
        try:
            return (self.path(run_id) / "run.json").exists()
        except KeyError:
            return False

    # -------------------------------------------------------------- read
    def get(self, run_id: str) -> RunRecord:
        file = self.path(run_id) / "run.json"
        if not file.exists():
            raise KeyError(f"unknown run {run_id!r}")
        with self._lock:
            return RunRecord.from_dict(json.loads(file.read_text()))

    def list(self, *, project: str | None = None, parent: str | None = None,
             kind: str | None = None) -> list[RunRecord]:
        records = []
        for file in sorted(self.root.glob("*/run.json"), reverse=True):
            try:
                rec = RunRecord.from_dict(json.loads(file.read_text()))
            except (OSError, ValueError, TypeError):
                continue
            if project is not None and rec.project != project:
                continue
            if parent is not None and rec.parent != parent:
                continue
            if kind is not None and rec.kind != kind:
                continue
            records.append(rec)
        return records

    def epochs(self, run_id: str) -> list[dict]:
        file = self.path(run_id) / "epochs.jsonl"
        if not file.exists():
            return []
        out = []
        for line in file.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    # ------------------------------------------------------------- write
    def save(self, record: RunRecord) -> None:
        with self._lock:
            file = self.path(record.run_id) / "run.json"
            tmp = file.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(record), indent=1, default=str))
            tmp.replace(file)

    def update(self, run_id: str, **fields: Any) -> RunRecord:
        with self._lock:
            record = self.get(run_id)
            for key, value in fields.items():
                if key not in RunRecord.__dataclass_fields__:
                    raise AttributeError(f"RunRecord has no field {key!r}")
                setattr(record, key, value)
            self.save(record)
            return record

    def append_epoch(self, run_id: str, event: dict) -> None:
        with self._lock:
            with open(self.path(run_id) / "epochs.jsonl", "a") as fh:
                fh.write(json.dumps(event) + "\n")
            record = self.get(run_id)
            record.epochs_done += 1
            for key, value in (event.get("metrics") or {}).items():
                if not isinstance(value, (int, float)) or key == "lr":
                    continue
                best = record.best_metrics.get(key)
                if best is None or (value < best if lower_is_better(key) else value > best):
                    record.best_metrics[key] = float(value)
            self.save(record)

    def finalize(self, run_id: str, status: str, returncode: int | None = None,
                 error: str = "") -> RunRecord:
        """Mark a run finished and pull ``metrics.json`` into the record."""
        with self._lock:
            record = self.get(run_id)
            record.status = status
            record.returncode = returncode
            record.finished_at = time.time()
            if error:
                record.error = error[-4000:]
            metrics_file = self.path(run_id) / "metrics.json"
            if metrics_file.exists():
                try:
                    data = json.loads(metrics_file.read_text())
                    record.final_metrics = {k: v for k, v in data.items()
                                            if isinstance(v, (int, float))}
                except ValueError:
                    pass
            self.save(record)
            return record

    def delete(self, run_id: str) -> None:
        with self._lock:
            path = self.path(run_id)
            if not (path / "run.json").exists():
                raise KeyError(f"unknown run {run_id!r}")
            shutil.rmtree(path)

    def mark_interrupted(self) -> int:
        """Runs left 'running' by a process that no longer exists become 'failed'."""
        count = 0
        for rec in self.list():
            if rec.status in ("created", "running") and not _alive(rec.pid):
                self.finalize(rec.run_id, "failed", error="interrupted (the app exited)")
                count += 1
        return count


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def compare(records: list[RunRecord]) -> dict[str, Any]:
    """Side-by-side view: metrics of every run + only the params that differ."""
    keys = sorted({k for r in records for k in r.params})
    differing = [k for k in keys
                 if len({json.dumps(r.params.get(k), sort_keys=True, default=str)
                         for r in records}) > 1]
    metric_keys = sorted({k for r in records for k in (*r.final_metrics, *r.best_metrics)})
    return {
        "runs": [r.run_id for r in records],
        "params": {k: [r.params.get(k) for r in records] for k in differing},
        "final_metrics": {k: [r.final_metrics.get(k) for r in records] for k in metric_keys},
        "best_metrics": {k: [r.best_metrics.get(k) for r in records] for k in metric_keys},
        "status": [r.status for r in records],
        "duration": [r.duration for r in records],
        "data": [r.data_fingerprint for r in records],
        "same_data": len({r.data_fingerprint for r in records}) <= 1,
    }
