"""Dialogs: lazy, short-lived, value-returning (Orange convention).

BaseDialog gives every dialog the same chrome; the shortcuts sheet is
auto-rendered from the actions catalog (actions-as-data paying off).
"""
from __future__ import annotations

from PySide6 import QtWidgets


class BaseDialog(QtWidgets.QDialog):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

    def add_buttons(self) -> QtWidgets.QDialogButtonBox:
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)
        return buttons


class SampleGalleryDialog(BaseDialog):
    """One-click editable starter graphs (Orange Examples / Langflow templates)."""

    def __init__(self, parent, entries: list[tuple]):
        # entries: [(path, name, description), ...]
        super().__init__(parent, "Sample Projects")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Open a starter graph (one click, fully editable):"))
        self.listing = QtWidgets.QListWidget()
        for _path, name, desc in entries:
            self.listing.addItem(name if not desc else f"{name} — {desc}")
        self.listing.setCurrentRow(0)
        layout.addWidget(self.listing)
        self.add_buttons()

    def chosen_index(self) -> int:
        return self.listing.currentRow()


class SaveTemplateDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Save Selection as Template")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Template name (shows under the Custom palette category):"))
        self.field = QtWidgets.QLineEdit()
        layout.addWidget(self.field)
        self.add_buttons()

    def template_name(self) -> str:
        return self.field.text().strip()


class ShortcutsDialog(BaseDialog):
    """Auto-rendered from the actions catalog — every shortcut documented."""

    def __init__(self, parent, specs: list):
        super().__init__(parent, "Keyboard Shortcuts")
        layout = QtWidgets.QVBoxLayout(self)
        table = QtWidgets.QTableWidget(len(specs), 2)
        table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, spec in enumerate(specs):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(spec.text))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(spec.shortcut or "—"))
        layout.addWidget(table)
        self.add_buttons()
