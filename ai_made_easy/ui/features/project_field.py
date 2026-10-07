"""Project name field (toolbar), two-way bound to the ProjectStore."""
from __future__ import annotations

from PySide6 import QtWidgets

from ai_made_easy.ui.stores import ProjectStore


class ProjectNameField(QtWidgets.QLineEdit):
    def __init__(self, store: ProjectStore, parent=None):
        super().__init__(store.name, parent)
        self.setObjectName("projectName")
        self.setToolTip("Project name (used for exported files)")
        self.setMaximumWidth(260)
        self._store = store
        self._guard = False
        self.editingFinished.connect(self._push)
        store.name_changed.connect(self._pull)

    def _push(self) -> None:
        if not self._guard:
            self._guard = True
            self._store.set_name(self.text())
            self._guard = False

    def _pull(self, name: str) -> None:
        if not self._guard and self.text() != name:
            self._guard = True
            self.setText(name)
            self._guard = False
