"""Where runs train, on the desktop: the "Run on" selector and the Backends dialog.

Sign-in stays with each tool (``~/.ssh``, ``~/.kaggle/kaggle.json``, ``modal token``,
Google Drive for desktop); no field here takes a password, token or key.
"""
from __future__ import annotations

import json

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.runner import backends

SETTING = "run_on"
MANAGE = "__manage"
KIND_LABELS = {"local": "this computer", "ssh": "SSH", "kaggle": "Kaggle", "colab": "Colab",
               "modal": "Modal", "container": "Docker"}


def _label(row: dict) -> str:
    if row["kind"] == "local":
        return "This computer"
    kind = KIND_LABELS.get(row["kind"], row["kind"])
    return kind if row["name"] == row["kind"] else f"{row['name']} ({kind})"


class RunOnCombo(QtWidgets.QComboBox):
    """Backends to train on; the choice is remembered in the settings."""

    manage_requested = QtCore.Signal()

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setToolTip("Where the run trains")
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.activated.connect(self._on_activated)
        self.refresh()

    def refresh(self) -> None:
        wanted = str(QtCore.QSettings("aime/workbench").value(SETTING, "local") or "local")
        self.blockSignals(True)
        self.clear()
        try:
            rows = backends.listing()
        except Exception:  # noqa: BLE001 — a broken backends.json must not hide the panel
            rows = [backends.LocalBackend().describe()]
        for row in rows:
            text = _label(row) + ("" if row["live"] else "  · per poll")
            self.addItem(text, row["name"])
        self.addItem("Manage backends…", MANAGE)
        index = self.findData(wanted)
        self.setCurrentIndex(index if index >= 0 else 0)
        self.blockSignals(False)

    def _on_activated(self, index: int) -> None:
        if self.itemData(index) == MANAGE:
            self.refresh()  # put the previous choice back
            self.manage_requested.emit()
            return
        QtCore.QSettings("aime/workbench").setValue(SETTING, self.itemData(index))

    def selected(self) -> str:
        value = self.currentData()
        return "local" if value in (None, MANAGE) else str(value)


class BackendsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("Backends: where runs train")
        self.resize(760, 520)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Sign-in stays with each tool: ~/.ssh/config and keys, ~/.kaggle/kaggle.json, "
            "`modal token new`, Google Drive for desktop. AI Made Easy never asks for or "
            "stores a password, token or key.")
        note.setWordWrap(True)
        note.setObjectName("blockMeta")
        layout.addWidget(note)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Kind", "Options", "Check"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        self.test_btn = QtWidgets.QPushButton("Test")
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.test_btn.clicked.connect(self._test)
        self.remove_btn.clicked.connect(self._remove)
        row.addWidget(self.test_btn)
        row.addWidget(self.remove_btn)
        row.addStretch(1)
        layout.addLayout(row)

        box = QtWidgets.QGroupBox("Add a backend")
        self.form = QtWidgets.QFormLayout(box)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("gpu-box")
        self.kind_combo = QtWidgets.QComboBox()
        for kind in backends.KINDS:
            if kind != "local":
                self.kind_combo.addItem(KIND_LABELS[kind], kind)
        self.kind_combo.currentIndexChanged.connect(self._build_options)
        self.form.addRow("Name", self.name_edit)
        self.form.addRow("Kind", self.kind_combo)
        self.option_edits: dict[str, QtWidgets.QLineEdit] = {}
        save = QtWidgets.QPushButton("Save backend")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save)
        self.form.addRow("", save)
        self._save_button = save
        layout.addWidget(box)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._build_options()
        self.refresh()

    # ------------------------------------------------------------------ table
    def refresh(self) -> None:
        rows = backends.listing()
        self.table.setRowCount(len(rows))
        for i, b in enumerate(rows):
            options = " ".join(f"{k}={v if isinstance(v, str) else json.dumps(v)}"
                               for k, v in b["options"].items())
            kind = KIND_LABELS.get(b["kind"], b["kind"]) + ("" if b["live"] else " · per poll")
            for col, text in enumerate((b["name"], kind, options, "")):
                self.table.setItem(i, col, QtWidgets.QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _selected_name(self) -> str:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.text() if item else ""

    def _test(self) -> None:
        name = self._selected_name()
        if not name:
            return
        row = self.table.currentRow()
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem("checking…"))
        QtWidgets.QApplication.processEvents()
        try:
            result = backends.resolve(name).test()
        except backends.BackendError as exc:
            result = {"ok": False, "message": str(exc), "accelerator": {}}
        accel = result.get("accelerator") or {}
        text = ("✓ " if result["ok"] else "✗ ") + result["message"]
        if accel.get("gpu"):
            text += f" · {accel['gpu']}"
        if accel.get("device"):
            text += f" · budget: {accel['device']}"
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(text))
        self.last_test = result

    def _remove(self) -> None:
        name = self._selected_name()
        if not name or name in backends.KINDS:
            return
        try:
            backends.remove(name)
        except backends.BackendError as exc:
            QtWidgets.QMessageBox.warning(self, "Backends", str(exc))
        self.refresh()

    # ------------------------------------------------------------------- form
    def _build_options(self) -> None:
        for edit in self.option_edits.values():
            self.form.removeRow(edit)
        self.option_edits = {}
        kind = self.kind_combo.currentData()
        cls = backends._backend_classes()[kind]
        for i, (option, help_text) in enumerate(cls.options_help.items()):
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(help_text)
            edit.setToolTip(help_text)
            self.form.insertRow(2 + i, option, edit)
            self.option_edits[option] = edit

    def _save(self) -> None:
        options = {}
        for option, edit in self.option_edits.items():
            text = edit.text().strip()
            if not text:
                continue
            try:
                options[option] = json.loads(text)
            except ValueError:
                options[option] = text
        try:
            backends.save(self.name_edit.text().strip(), self.kind_combo.currentData(), **options)
        except backends.BackendError as exc:
            QtWidgets.QMessageBox.warning(self, "Backends", str(exc))
            return
        self.name_edit.clear()
        for edit in self.option_edits.values():
            edit.clear()
        self.refresh()
