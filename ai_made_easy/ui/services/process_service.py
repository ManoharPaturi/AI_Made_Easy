"""ProcessService: THE one subprocess pattern for everything runnable —
training runs, forward-pass test runs, and ONNX/TorchScript export
scripts. One QProcess + worker-protocol pipeline, one set of signals.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PySide6 import QtCore

from ai_made_easy.core.codegen import export as export_model
from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.protocol import parse_event, worker_script_path


class ProcessService(QtCore.QObject):
    log_received = QtCore.Signal(str)
    epoch_received = QtCore.Signal(dict)
    error_received = QtCore.Signal(str)
    finished = QtCore.Signal(int, str)  # returncode, kind

    def __init__(self, log, run_store, parent=None):
        super().__init__(parent)
        self.log = log
        self.run_store = run_store
        self._proc: QtCore.QProcess | None = None
        self._buf = ""
        self._kind = ""

    # ------------------------------------------------------------ state

    def is_running(self) -> bool:
        return (self._proc is not None
                and self._proc.state() != QtCore.QProcess.ProcessState.NotRunning)

    def stop(self) -> None:
        if not self.is_running():
            return
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
        proc = QtCore.QProcess(self)
        proc.setProgram(sys.executable)
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

    def run_training(self, graph: Graph) -> None:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            self.log.error("PyTorch is not installed in this environment "
                           "(pip install torch)")
            return
        workdir = Path(tempfile.mkdtemp(prefix="aime_train_"))
        script = export_training(graph, "pytorch", workdir)
        self.log.info(f"training started — workspace: {workdir}")
        self._start(script, workdir, "train")

    def run_test(self, graph: Graph) -> None:
        workdir = Path(tempfile.mkdtemp(prefix="aime_test_"))
        script = export_model(graph, "pytorch", workdir)
        self.log.info(f"testing forward pass in {workdir}")
        self._start(script, workdir, "test")

    def run_script(self, script: Path, workdir: Path, kind: str) -> None:
        self.log.info(f"running {kind} export → {script}")
        # runtime export scripts print plain output (no worker protocol)
        proc = QtCore.QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments([str(script)])
        proc.setWorkingDirectory(str(workdir))
        proc.readyReadStandardOutput.connect(lambda: self._emit_lines(
            bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")))
        proc.readyReadStandardError.connect(lambda: self._emit_lines(
            bytes(proc.readAllStandardError()).decode("utf-8", "replace")))
        proc.finished.connect(
            lambda code, *_: (self.finished.emit(int(code), kind),
                              self.log.info(f"{kind} export finished (exit {int(code)}) — see exports/")))
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

    def _dispatch(self, event: dict | None) -> None:
        if event is None:
            return
        kind = event.get("type")
        if kind == "epoch":
            self.epoch_received.emit(event)
        elif kind == "log":
            self.log_received.emit(str(event.get("line", "")))
        elif kind == "error":
            self.error_received.emit(str(event.get("traceback", "")))

    def _on_finished(self, code, _status) -> None:
        code = int(code)
        state = (self.run_store.FINISHED if code == 0
                 else self.run_store.FAILED)
        self.run_store.set(state, self._kind)
        self.finished.emit(code, self._kind)
