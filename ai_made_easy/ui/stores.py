"""Single-concern stores (Langflow convention): granular signals, dumb
components subscribe to slices. No store knows about any other store.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore


class ProjectStore(QtCore.QObject):
    """Project identity + dirty state. THE only write path for these."""

    name_changed = QtCore.Signal(str)
    path_changed = QtCore.Signal(object)  # Path | None
    dirty_changed = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = "untitled"
        self._path: Path | None = None
        self._dirty = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self):
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    def set_name(self, name: str) -> None:
        name = (name or "").strip() or "untitled"
        if name != self._name:
            self._name = name
            self.name_changed.emit(name)
            self.mark_dirty()

    def set_path(self, path) -> None:
        if path != self._path:
            self._path = path
            self.path_changed.emit(path)

    def mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            self.dirty_changed.emit(True)

    def mark_clean(self) -> None:
        if self._dirty:
            self._dirty = False
            self.dirty_changed.emit(False)

    def reset(self, name: str = "untitled") -> None:
        self.set_path(None)
        self.set_name(name)
        self.mark_clean()


class RunStore(QtCore.QObject):
    """Run lifecycle state. One writer (ProcessService), many readers."""

    state_changed = QtCore.Signal(str, str)  # state, kind
    IDLE, RUNNING, FINISHED, FAILED, STOPPED = (
        "idle", "running", "finished", "failed", "stopped")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = self.IDLE
        self._kind = ""

    @property
    def state(self) -> str:
        return self._state

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def is_running(self) -> bool:
        return self._state == self.RUNNING

    def set(self, state: str, kind: str = "") -> None:
        changed = (state, kind) != (self._state, self._kind)
        self._state, self._kind = state, kind
        if changed:
            self.state_changed.emit(state, kind)


class ValidationStore(QtCore.QObject):
    """Latest validation issues + valid flag."""

    issues_changed = QtCore.Signal(list)  # list[ValidationIssue]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._issues: list = []

    @property
    def issues(self) -> list:
        return self._issues

    @property
    def errors(self) -> list:
        return [i for i in self._issues if i.severity == "error"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def update(self, issues: list) -> None:
        self._issues = list(issues)
        self.issues_changed.emit(self._issues)


class LogBus(QtCore.QObject):
    """The one logging channel; ConsolePanel is its only renderer."""

    logged = QtCore.Signal(str, str)  # level, message

    def info(self, message: str) -> None:
        self.logged.emit("info", message)

    def warning(self, message: str) -> None:
        self.logged.emit("warning", message)

    def error(self, message: str) -> None:
        self.logged.emit("error", message)
