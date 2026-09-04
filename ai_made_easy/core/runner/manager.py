"""Headless run manager: training runs in pure Python (threads, no Qt).

This is the engine behind `aime run` and the MCP ``start_training`` tool —
the UI keeps its QProcess controller, agents get this one.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.protocol import parse_event, worker_script_path


class TrainingRun:
    def __init__(self, run_id: str, script_path: Path, workdir: Path):
        self.run_id = run_id
        self.script_path = script_path
        self.workdir = workdir
        self.state = "starting"  # starting | running | finished | failed | stopped
        self.returncode: int | None = None
        self.started_at = time.time()
        self.events: list[dict] = []
        self.logs: list[str] = []
        self.epochs: list[dict] = []
        self.error: str | None = None
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def record(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)
            kind = event.get("type")
            if kind == "epoch":
                self.epochs.append(event)
            elif kind == "log":
                self.logs.append(str(event.get("line", "")))
            elif kind == "error":
                self.error = str(event.get("traceback", ""))
            elif kind == "done":
                code = int(event.get("returncode", 1))
                self.returncode = code
                if self.state != "stopped":
                    self.state = "finished" if code == 0 else "failed"

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

    def __init__(self) -> None:
        self._runs: dict[str, TrainingRun] = {}

    def start(self, graph: Graph, framework: str = "pytorch") -> str:
        import tempfile

        if framework != "pytorch":
            raise ValueError("headless runs support the pytorch framework so far")
        workdir = Path(tempfile.mkdtemp(prefix="aime_run_"))
        script = export_training(graph, framework, workdir)
        run_id = uuid.uuid4().hex[:12]
        run = TrainingRun(run_id, script, workdir)
        self._runs[run_id] = run

        proc = subprocess.Popen(
            [sys.executable, str(worker_script_path()), str(script)],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        run.process = proc
        run.state = "running"

        def pump_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                event = parse_event(line)
                if event is not None:
                    run.record(event)
            proc.wait()

        def pump_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                run.record({"type": "log", "line": line.rstrip()})

        threading.Thread(target=pump_stdout, daemon=True).start()
        threading.Thread(target=pump_stderr, daemon=True).start()
        return run_id

    def get(self, run_id: str) -> TrainingRun:
        try:
            return self._runs[run_id]
        except KeyError:
            raise KeyError(f"unknown run_id {run_id!r}") from None

    def status(self, run_id: str) -> dict:
        return self.get(run_id).status()

    def metrics(self, run_id: str) -> list[dict]:
        return list(self.get(run_id).epochs)

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

    def wait(self, run_id: str, timeout: float = 3600.0) -> dict:
        run = self.get(run_id)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if run.state in ("finished", "failed", "stopped"):
                return run.status()
            time.sleep(0.1)
        raise TimeoutError(f"run {run_id} did not finish within {timeout}s")
