"""Headless run manager: training runs in pure Python (threads, no Qt).

The engine behind ``aime run``, the MCP training tools, sweeps and the web
server. Every run is recorded in the persistent :class:`RunHistory`; the run
folder is also the training script's working directory, so checkpoints and
``metrics.json`` land next to ``run.json``.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.protocol import parse_event, worker_script_path
from ai_made_easy.core.runs.history import FINAL_STATES, RunHistory

Listener = Callable[[str, dict], None]
FRAMEWORKS = ("auto", "pytorch", "keras", "sklearn")


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

    if framework not in FRAMEWORKS:
        raise ValueError(f"unknown framework {framework!r}; one of {FRAMEWORKS}")
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
            return
        self.record({"type": "done", "returncode": code if code else 1})

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
              tags: list[str] | None = None) -> str:
        framework = resolve_framework(graph, framework)
        record = self.history.create(graph.to_dict(), framework=framework,
                                     project=project, parent=parent, trial=trial,
                                     tags=tags)
        workdir = self.history.path(record.run_id)
        try:
            script = export_training(graph, framework, workdir)
        except Exception as exc:
            self.history.finalize(record.run_id, "failed", None,
                                  f"could not generate the training script: {exc}")
            raise
        run = TrainingRun(record.run_id, script, workdir, self.history)
        run.listeners = self._listeners
        self._runs[record.run_id] = run

        proc = subprocess.Popen(
            [self.python or python_executable(), str(worker_script_path()), str(script)],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        run.process = proc
        run.state = "running"
        self.history.update(record.run_id, status="running", pid=proc.pid)

        def pump_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                run.record({"type": "log", "line": line.rstrip(), "stream": "stderr"})

        err_thread = threading.Thread(target=pump_stderr, daemon=True)

        def pump_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                event = parse_event(line)
                if event is not None:
                    run.record(event)
            code = proc.wait()
            err_thread.join(timeout=5)
            run.process_exited(code)

        err_thread.start()
        threading.Thread(target=pump_stdout, daemon=True).start()
        return record.run_id

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
        if run.process and run.process.poll() is None:
            run.state = "stopped"
            run.process.terminate()
            try:
                run.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                run.process.kill()
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
                return run.status()
            time.sleep(0.1)
        raise TimeoutError(f"run {run_id} did not finish within {timeout}s")
