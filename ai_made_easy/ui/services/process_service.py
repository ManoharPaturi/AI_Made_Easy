"""ProcessService: THE one subprocess pattern for everything runnable —
training runs, forward-pass test runs, and ONNX/TorchScript export
scripts. One QProcess + worker-protocol pipeline, one set of signals.
"""
from __future__ import annotations

import queue
import tempfile
from pathlib import Path

from PySide6 import QtCore

from ai_made_easy.core.codegen import export as export_model
from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.protocol import parse_event, worker_script_path
from ai_made_easy.core.runs.history import RunHistory


def python_executable() -> str:
    """Interpreter for training / export runs.

    Order: $AIME_PYTHON, the "python" setting, the running interpreter (when not
    frozen), then python3 on PATH. Frozen app bundles cannot run scripts with
    their own executable, so they need a real Python environment.
    """
    import os

    from ai_made_easy.core.runner import manager

    override = os.environ.get("AIME_PYTHON") or QtCore.QSettings("aime/workbench").value(
        "python", "")
    return str(override) if override else manager.python_executable()


class ProcessService(QtCore.QObject):
    log_received = QtCore.Signal(str)
    epoch_received = QtCore.Signal(dict)
    samples_received = QtCore.Signal(dict)   # live sample grid / text of generative runs
    error_received = QtCore.Signal(str)
    finished = QtCore.Signal(int, str)  # returncode, kind
    history_changed = QtCore.Signal()

    def __init__(self, log, run_store, parent=None, history: RunHistory | None = None):
        super().__init__(parent)
        self.log = log
        self.run_store = run_store
        self._proc: QtCore.QProcess | None = None
        self._buf = ""
        self._kind = ""
        self.last_workdir: Path | None = None
        self.history = history if history is not None else RunHistory()
        self.current_run_id: str | None = None
        self._stopping = False
        # runs on another machine (SSH, Kaggle, Colab, Modal, Docker) go through the
        # core RunManager; its events are drained into the same signals on a timer
        self._remote_manager = None
        self._remote_run: str | None = None
        self._remote_events: queue.Queue = queue.Queue()
        self._remote_timer = QtCore.QTimer(self)
        self._remote_timer.setInterval(200)
        self._remote_timer.timeout.connect(self._drain_remote)
        try:
            self.history.mark_interrupted()
        except OSError:
            pass

    # ------------------------------------------------------------ state

    def is_running(self) -> bool:
        if self._remote_run is not None:
            return True
        return (self._proc is not None
                and self._proc.state() != QtCore.QProcess.ProcessState.NotRunning)

    def stop(self) -> None:
        if self._remote_run is not None:
            self._stopping = True
            self._remote_manager.stop(self._remote_run)
            return
        if not self.is_running():
            return
        self._stopping = True
        self._proc.terminate()
        QtCore.QTimer.singleShot(3000, self._force_kill)

    def _force_kill(self) -> None:
        if self.is_running():
            self._proc.kill()

    # ------------------------------------------------------------ start

    def _start(self, script: Path, workdir: Path, kind: str) -> None:
        if self.is_running():
            self.log.error(f"a {self._kind} run is already active")
            return
        self._buf = ""
        self._kind = kind
        self._stopping = False
        proc = QtCore.QProcess(self)
        proc.setProgram(python_executable())
        proc.setArguments([str(worker_script_path()), str(script)])
        proc.setWorkingDirectory(str(workdir))
        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.readyReadStandardError.connect(self._on_stderr)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(
            lambda err: self.error_received.emit(f"process error: {err}"))
        self._proc = proc
        self.run_store.set(self.run_store.RUNNING, kind)
        proc.start()

    def run_training(self, graph: Graph, project: str = "", on: str | None = None) -> None:
        """Train here, or ``on`` a backend (``ssh:<host>``, ``kaggle``, a saved name …)."""
        import importlib.util

        from ai_made_easy.core.families import resolve_framework

        if on and on != "local":
            self._run_remote(graph, project, on)
            return

        try:
            framework = resolve_framework(graph)
        except ValueError as exc:
            self.log.error(str(exc))
            return
        from ai_made_easy.core.families import FRAMEWORK_PACKAGES

        needed, package = FRAMEWORK_PACKAGES.get(framework, ("torch", "torch"))
        if importlib.util.find_spec(needed) is None:
            self.log.error(f"{package} is not installed in this environment "
                           f"(pip install {package})")
            return
        record = self.history.create(graph.to_dict(), framework=framework, project=project)
        workdir = self.history.path(record.run_id)
        try:
            script = export_training(graph, framework, workdir)
        except Exception as exc:  # noqa: BLE001 — reported to the user
            self.history.finalize(record.run_id, "failed", None, str(exc))
            self.log.error(f"could not generate the training script: {exc}")
            return
        self.last_workdir = workdir
        self.current_run_id = record.run_id
        self.log.info(f"training started — run {record.run_id}, folder: {workdir}")
        self._start(script, workdir, "train")
        if self._proc is not None:
            self.history.update(record.run_id, status="running",
                                pid=int(self._proc.processId()) or None)
        self.history_changed.emit()

    def _run_remote(self, graph: Graph, project: str, on: str) -> None:
        from ai_made_easy.core.runner.manager import RunManager

        if self.is_running():
            self.log.error(f"a {self._kind} run is already active")
            return
        if self._remote_manager is None:
            self._remote_manager = RunManager(self.history, python=python_executable())
            self._remote_manager.subscribe(
                lambda run_id, event: self._remote_events.put((run_id, event)))
        self._stopping = False
        self._kind = "train"
        try:
            run_id = self._remote_manager.start(graph, project=project, on=on)
        except Exception as exc:  # noqa: BLE001 — a missing tool, no sign-in, a bad host
            self.log.error(f"could not start the run on {on}: {exc}")
            return
        self._remote_run = self.current_run_id = run_id
        self.last_workdir = self.history.path(run_id)
        self.log.info(f"training started on {on} — run {run_id}, folder: {self.last_workdir}")
        if not self._remote_manager.get(run_id).backend.live:
            self.log.info("curves arrive when the run reports back (Kaggle and Colab are "
                          "polled, not streamed)")
        self.run_store.set(self.run_store.RUNNING, "train")
        self._remote_timer.start()
        self.history_changed.emit()

    def _drain_remote(self) -> None:
        run_id = self._remote_run
        while True:
            try:
                rid, event = self._remote_events.get_nowait()
            except queue.Empty:
                break
            if rid == run_id:
                self._dispatch(event, record=False)  # the RunManager records history
        if run_id is None:
            self._remote_timer.stop()
            return
        status = self._remote_manager.status(run_id)
        if status["state"] not in ("finished", "failed", "stopped"):
            return
        if self.history.get(run_id).status not in ("finished", "failed", "stopped"):
            return  # the record finalises right after the done event
        self._remote_timer.stop()
        self._remote_run = None
        code = status["returncode"] if status["returncode"] is not None else (
            0 if status["state"] == "finished" else 1)
        self.history_changed.emit()
        self.run_store.set(self.run_store.FINISHED if status["state"] == "finished"
                           else self.run_store.FAILED, "train")
        self.finished.emit(int(code), "train")

    def run_test(self, graph: Graph) -> None:
        workdir = Path(tempfile.mkdtemp(prefix="aime_test_"))
        try:
            script = export_model(graph, "pytorch", workdir)
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"could not generate the model: {exc}")
            return
        self.log.info(f"testing a forward pass in {workdir}")
        self._start(script, workdir, "test")

    def run_script(self, script: Path, workdir: Path, kind: str) -> None:
        self.log.info(f"running {kind} → {script}")
        # runtime export scripts print plain output (no worker protocol)
        proc = QtCore.QProcess(self)
        proc.setProgram(python_executable())
        proc.setArguments([str(script)])
        proc.setWorkingDirectory(str(workdir))
        proc.readyReadStandardOutput.connect(lambda: self._emit_lines(
            bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")))
        proc.readyReadStandardError.connect(lambda: self._emit_lines(
            bytes(proc.readAllStandardError()).decode("utf-8", "replace")))
        def _done(code, *_):
            self.run_store.set(
                self.run_store.FINISHED if int(code) == 0
                else self.run_store.FAILED, kind)
            self.finished.emit(int(code), kind)
            self.log.info(f"{kind} finished (exit code {int(code)})")

        proc.finished.connect(_done)
        self.run_store.set(self.run_store.RUNNING, kind)
        proc.start()

    # ------------------------------------------------------------- io

    def _emit_lines(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self.log_received.emit(line.strip())

    def _on_stdout(self) -> None:
        self._buf += bytes(self._proc.readAllStandardOutput()).decode(
            "utf-8", "replace")
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._dispatch(parse_event(line))

    def _on_stderr(self) -> None:
        text = bytes(self._proc.readAllStandardError()).decode("utf-8", "replace")
        for line in text.splitlines():
            if line.strip():
                self.log_received.emit(line.strip())

    def _dispatch(self, event: dict | None, record: bool = True) -> None:
        if event is None:
            return
        kind = event.get("type")
        run_id = self.current_run_id if self._kind == "train" and record else None
        if run_id and kind == "epoch":
            self._safely(self.history.append_epoch, run_id, event)
        elif run_id and kind == "env":
            self._safely(self.history.update, run_id,
                         env={k: v for k, v in event.items() if k != "type"})
        elif run_id and kind == "resources":
            self._safely(self.history.update, run_id,
                         resources={k: v for k, v in event.items() if k != "type"})
        if kind == "epoch":
            self.epoch_received.emit(event)
        elif kind == "samples":
            self.samples_received.emit(event)
        elif kind == "log":
            self.log_received.emit(str(event.get("line", "")))
        elif kind == "error":
            self._error = str(event.get("traceback", ""))
            self.error_received.emit(self._error)

    def _safely(self, fn, *args, **kwargs) -> None:  # noqa: ANN001
        try:
            fn(*args, **kwargs)
        except (OSError, KeyError, ValueError) as exc:
            self.log.warning(f"run history not updated: {exc}")

    def _on_finished(self, code, _status) -> None:
        code = int(code)
        if self._buf.strip():
            self._dispatch(parse_event(self._buf))
            self._buf = ""
        if self._kind == "train" and self.current_run_id:
            status = "stopped" if self._stopping else ("finished" if code == 0 else "failed")
            self._safely(self.history.finalize, self.current_run_id, status, code,
                         getattr(self, "_error", ""))
            self._error = ""
            self.history_changed.emit()
        state = (self.run_store.FINISHED if code == 0
                 else self.run_store.FAILED)
        self.run_store.set(state, self._kind)
        self.finished.emit(code, self._kind)
