"""Headless run manager: training runs in pure Python (threads, no Qt).

The engine behind ``aime run``, the MCP training tools, sweeps and the web
server. Every run is recorded in the persistent :class:`RunHistory`; the run
folder is also the training script's working directory, so checkpoints and
``metrics.json`` land next to ``run.json``.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner import backends
from ai_made_easy.core.runner.protocol import worker_script_path  # noqa: F401 — re-exported
from ai_made_easy.core.runs.history import FINAL_STATES, RunHistory

Listener = Callable[[str, dict], None]
FRAMEWORKS = ("auto", "pytorch", "keras", "sklearn")  # built-ins; see families.FRAMEWORK_PACKAGES


def python_executable() -> str:
    """Interpreter for training runs: ``$AIME_PYTHON``, else this interpreter
    (or python3 on PATH when running from a frozen app bundle)."""
    import os
    import shutil
    import sys

    override = os.environ.get("AIME_PYTHON")
    if override:
        return override
    if not getattr(sys, "frozen", False):
        return sys.executable
    return shutil.which("python3") or shutil.which("python") or sys.executable


def resolve_framework(graph: Graph, framework: str = "auto") -> str:
    from ai_made_easy.core.families import resolve_framework as _resolve

    from ai_made_easy.core.families import FRAMEWORK_PACKAGES

    if framework != "auto" and framework not in FRAMEWORK_PACKAGES:
        raise ValueError(f"unknown framework {framework!r}; one of "
                         f"{('auto', *FRAMEWORK_PACKAGES)}")
    return _resolve(graph, framework)


class TrainingRun:
    def __init__(self, run_id: str, script_path: Path, workdir: Path,
                 history: RunHistory | None = None):
        self.run_id = run_id
        self.script_path = script_path
        self.workdir = workdir
        self.history = history
        self.state = "starting"  # starting | running | finished | failed | stopped
        self.returncode: int | None = None
        self.started_at = time.time()
        self.events: list[dict] = []
        self.logs: list[str] = []
        self.epochs: list[dict] = []
        self.latest_samples: dict | None = None   # last "samples" event (live sample grid)
        self.error: str | None = None
        self.process: subprocess.Popen | None = None
        self.listeners: list[Listener] = []
        self._lock = threading.Lock()
        self.name = workdir.name
        # where it trains (see core.runner.backends) and what remote backends keep
        self.backend: backends.Backend = backends.LocalBackend()
        self.bundle = None             # the run bundle, for remote backends
        self.remote_offset = 0         # events.jsonl lines received so far
        self.held_done: dict | None = None
        self.accelerator: dict = {}
        self.remote_dir = ""
        self.remote_url = ""
        self.slug = ""
        self.graph: Graph | None = None  # the design, for budget checks on the backend's GPU
        self.mirror = None             # MLflow / W&B trackers (core.tracking.RunMirror)
        self.closing_mirror = None     # the mirror finishing after the run ended

    def record(self, event: dict) -> None:
        finished = False
        with self._lock:
            self.events.append(event)
            kind = event.get("type")
            if kind == "epoch":
                self.epochs.append(event)
            elif kind == "log":
                self.logs.append(str(event.get("line", "")))
            elif kind == "samples":
                self.latest_samples = event
            elif kind == "error":
                self.error = str(event.get("traceback", ""))
            elif kind == "done" and self.state not in FINAL_STATES:
                code = int(event.get("returncode", 1))
                self.returncode = code
                self.state = "finished" if code == 0 else "failed"
                finished = True
        if self.mirror is not None and kind == "epoch":
            self.mirror.epoch(event)
        if self.history is not None:
            if kind == "epoch":
                self.history.append_epoch(self.run_id, event)
            elif kind == "env":
                self.history.update(self.run_id, env={k: v for k, v in event.items()
                                                      if k != "type"})
            elif kind == "resources":
                self.history.update(self.run_id, resources={k: v for k, v in event.items()
                                                            if k != "type"})
            if finished:
                self.history.finalize(self.run_id, self.state, self.returncode,
                                      self.error or "")
        if finished:
            self._finish_tracking()
        for listener in list(self.listeners):
            try:
                listener(self.run_id, event)
            except Exception:  # noqa: BLE001 — a bad subscriber must not kill the pump
                pass

    def process_exited(self, code: int) -> None:
        """Worker died without a ``done`` event (killed, segfault, …)."""
        if self.state in FINAL_STATES:
            if self.state == "stopped" and self.history is not None:
                self.history.finalize(self.run_id, "stopped", code)
            self._finish_tracking()
            return
        self.record({"type": "done", "returncode": code if code else 1})

    def _finish_tracking(self) -> None:
        mirror, self.mirror = self.mirror, None
        if mirror is not None:
            mirror.finish(self.state, Path(self.workdir))
            self.closing_mirror = mirror

    def start_tracking(self, names: list[str]) -> None:
        """Mirror this run to MLflow / W&B (``core.tracking``) from its events."""
        from ai_made_easy.core import tracking

        if not names or self.history is None:
            return
        record = self.history.get(self.run_id).to_dict()

        def log(line: str) -> None:
            self.record({"type": "log", "line": f"[tracking] {line}"})

        def started(info: list[dict]) -> None:
            if info and self.history is not None:
                self.history.update(self.run_id, tracking=info)

        self.mirror = tracking.RunMirror(record, names, log=log, on_start=started)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "state": self.state,
                "returncode": self.returncode,
                "epochs_done": len(self.epochs),
                "total_epochs": self.epochs[0].get("total") if self.epochs else None,
                "latest_metrics": self.epochs[-1].get("metrics") if self.epochs else None,
                "error": self.error,
                "workspace": str(self.workdir),
                "backend": self.backend.spec,
                "live": self.backend.live,
                "accelerator": self.accelerator,
                "elapsed_seconds": round(time.time() - self.started_at, 1),
                "log_tail": self.logs[-10:],
            }


class RunManager:
    """Owns headless training runs; safe to call from any thread."""

    def __init__(self, history: RunHistory | None = None, python: str | None = None) -> None:
        self.history = history if history is not None else RunHistory()
        self.python = python
        self._runs: dict[str, TrainingRun] = {}
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        """Receive ``(run_id, event)`` for every event of every run."""
        self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def start(self, graph: Graph, framework: str = "auto", *, project: str = "",
              parent: str = "", trial: dict | None = None,
              tags: list[str] | None = None, on: str | None = None) -> str:
        """Train the design; ``on`` picks where (see :func:`backends.resolve`)."""
        backend = backends.resolve(on)
        framework = resolve_framework(graph, framework)
        record = self.history.create(graph.to_dict(), framework=framework,
                                     project=project, parent=parent, trial=trial,
                                     tags=tags)
        workdir = self.history.path(record.run_id)
        bundle = None
        try:
            if backend.remote:
                from ai_made_easy.core.runner.backends.bundle import prepare

                bundle = prepare(graph, framework, record.run_id, workdir)
                script = workdir / bundle.script
            else:
                script = export_training(graph, framework, workdir)
        except Exception as exc:
            self.history.finalize(record.run_id, "failed", None,
                                  f"could not generate the training script: {exc}")
            raise
        return self._launch(record.run_id, script, workdir, backend, bundle, graph)

    def start_stage(self, graph: Graph, write, *, framework: str = "pytorch",
                    kind: str = "stage", project: str = "", parent: str = "",
                    trial: dict | None = None, tags: list[str] | None = None) -> str:
        """Run a script that ``write(workdir) -> script path`` puts in a new run folder
        (pipeline stages); ``graph`` is the design the run's model comes from."""
        record = self.history.create(graph.to_dict(), framework=framework, project=project,
                                     kind=kind, parent=parent, trial=trial, tags=tags)
        workdir = self.history.path(record.run_id)
        try:
            script = write(workdir)
        except Exception as exc:
            self.history.finalize(record.run_id, "failed", None,
                                  f"could not prepare the stage: {exc}")
            raise
        return self._launch(record.run_id, script, workdir, graph=graph)

    def _launch(self, run_id: str, script, workdir,  # noqa: ANN001
                backend: backends.Backend | None = None, bundle=None,  # noqa: ANN001
                graph: Graph | None = None) -> str:
        record = self.history.get(run_id)
        run = TrainingRun(record.run_id, script, workdir, self.history)
        run.name = record.name
        run.listeners = self._listeners
        run.backend = backend or backends.LocalBackend()
        run.bundle = bundle
        run.graph = graph
        self._runs[record.run_id] = run
        run.state = "running"
        self.history.update(record.run_id, status="running",
                            backend="" if run.backend.kind == "local" else run.backend.spec)
        self._track(run, graph)
        try:
            run.backend.launch(run, self.python or python_executable())
        except Exception as exc:
            run.record({"type": "error", "traceback": str(exc)})
            run.process_exited(1)
        return record.run_id

    def _track(self, run: TrainingRun, graph: Graph | None) -> None:
        from ai_made_easy.core import tracking

        try:
            names = tracking.selected(graph.meta if graph is not None else None)
        except tracking.TrackingError as exc:
            run.record({"type": "log", "line": f"[tracking] {exc}"})
            return
        run.start_tracking(names)

    def attach(self, run_id: str) -> dict:
        """Follow a remote run again (after the app restarted or the link dropped)."""
        if run_id in self._runs and self._runs[run_id].state not in FINAL_STATES:
            return self._runs[run_id].status()
        record = self.history.get(run_id)
        if record.backend in ("", "local"):
            raise backends.BackendError("only remote runs can be followed again")
        if record.status in FINAL_STATES:
            raise backends.BackendError(f"run {run_id} already {record.status}")
        backend = backends.resolve(record.backend)
        workdir = self.history.path(run_id)
        run = TrainingRun(run_id, workdir / "script.py", workdir, self.history)
        run.name = record.name
        run.listeners = self._listeners
        run.backend = backend
        run.state = "running"
        remote = record.remote or {}
        run.accelerator = remote.get("accelerator") or {}
        run.remote_dir = remote.get("dir", "")
        run.remote_url = remote.get("url") or remote.get("notebook", "")
        run.slug = remote.get("slug", "")
        self.history.reset_epochs(run_id)
        self._runs[run_id] = run
        backend.reattach(run)
        return run.status()

    def import_results(self, path: str | Path) -> dict:
        """Complete a hand-off run (Colab without Drive) from its results zip."""
        from ai_made_easy.core.runner.backends.bundle import merge_folder

        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as archive:
            root = Path(tmp).resolve()
            for member in archive.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise ValueError(f"unsafe path in the zip: {member}")
            archive.extractall(root)
            meta_files = sorted(root.rglob("bundle.json"), key=lambda p: len(p.parts))
            if not meta_files:
                raise ValueError("not an AI Made Easy results zip (no bundle.json)")
            folder = meta_files[0].parent
            run_id = json.loads(meta_files[0].read_text())["run_id"]
            record = self.history.get(run_id)
            if record.status in FINAL_STATES:
                raise ValueError(f"run {run_id} is already {record.status}")
            workdir = self.history.path(run_id)
            merge_folder(folder, workdir)
        run = self._runs.get(run_id)
        if run is None or run.state in FINAL_STATES:
            run = TrainingRun(run_id, workdir / "script.py", workdir, self.history)
            run.listeners = self._listeners
            run.state = "running"
            self._runs[run_id] = run
        self.history.reset_epochs(run_id)
        replay = backends.RemoteBackend()
        run.remote_offset = 0
        events = workdir / "events.jsonl"
        for line in events.read_text().splitlines() if events.exists() else []:
            replay.feed(run, line)
        exit_file = workdir / "exit_code"
        code = exit_file.read_text().strip() if exit_file.exists() else "1"
        replay.finish(run, int(code) if code.lstrip("-").isdigit() else 1)
        return run.status()

    def get(self, run_id: str) -> TrainingRun:
        try:
            return self._runs[run_id]
        except KeyError:
            raise KeyError(f"unknown run_id {run_id!r}") from None

    def is_live(self, run_id: str) -> bool:
        return run_id in self._runs

    def latest_samples(self, run_id: str) -> dict | None:
        """The newest sample grid / text a live run produced (None when there is none)."""
        run = self._runs.get(run_id)
        return run.latest_samples if run is not None else None

    def status(self, run_id: str) -> dict:
        if run_id in self._runs:
            return self._runs[run_id].status()
        rec = self.history.get(run_id)  # a run from an earlier session
        return {"run_id": run_id, "state": rec.status, "returncode": rec.returncode,
                "epochs_done": rec.epochs_done, "total_epochs": None,
                "latest_metrics": rec.final_metrics or rec.best_metrics or None,
                "error": rec.error or None, "workspace": str(self.history.path(run_id)),
                "elapsed_seconds": rec.duration, "log_tail": []}

    def metrics(self, run_id: str) -> list[dict]:
        if run_id in self._runs:
            return list(self._runs[run_id].epochs)
        return self.history.epochs(run_id)

    def stop(self, run_id: str) -> dict:
        run = self.get(run_id)
        if run.state not in FINAL_STATES:
            run.state = "stopped"
            run.backend.stop(run)
        return run.status()

    def stop_all(self) -> None:
        for run_id in list(self._runs):
            self.stop(run_id)

    def wait(self, run_id: str, timeout: float = 3600.0) -> dict:
        run = self.get(run_id)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if run.state in FINAL_STATES:
                # the pump finalises history right after the done event
                rec_deadline = time.time() + 5
                while (time.time() < rec_deadline
                       and self.history.get(run_id).status not in FINAL_STATES):
                    time.sleep(0.05)
                if run.closing_mirror is not None:  # MLflow / W&B finish mirroring
                    run.closing_mirror.join(max(1.0, deadline - time.time()))
                return run.status()
            time.sleep(0.1)
        raise TimeoutError(f"run {run_id} did not finish within {timeout}s")
