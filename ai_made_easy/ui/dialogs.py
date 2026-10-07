"""Small modal dialogs: examples browser, save-as-block, shortcuts, about."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class BaseDialog(QtWidgets.QDialog):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

    def add_buttons(self, ok_text: str = "OK") -> QtWidgets.QDialogButtonBox:
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(ok_text)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)
        return buttons


class ExamplesDialog(BaseDialog):
    """Browse example projects: name, kind and description."""

    def __init__(self, parent, entries: list[tuple]):
        # entries: [(path, name, description, kind), ...]
        super().__init__(parent, "Open Example")
        self.resize(720, 420)
        layout = QtWidgets.QVBoxLayout(self)
        split = QtWidgets.QHBoxLayout()
        self.listing = QtWidgets.QListWidget()
        self.listing.setMinimumWidth(260)
        for entry in entries:
            self.listing.addItem(entry[1])
        split.addWidget(self.listing)
        self.detail = QtWidgets.QLabel()
        self.detail.setWordWrap(True)
        self.detail.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.detail.setObjectName("blockDesc")
        split.addWidget(self.detail, 1)
        layout.addLayout(split, 1)
        self._entries = entries
        self.listing.currentRowChanged.connect(self._show)
        self.listing.itemDoubleClicked.connect(lambda _: self.accept())
        self.listing.setCurrentRow(0)
        self.add_buttons("Open")

    def _show(self, row: int) -> None:
        if 0 <= row < len(self._entries):
            _path, name, desc, kind = self._entries[row]
            self.detail.setText(f"<p style='font-size:15px'><b>{name}</b></p>"
                                f"<p>{kind}</p><p>{desc or ''}</p>")

    def chosen_index(self) -> int:
        return self.listing.currentRow()


SampleGalleryDialog = ExamplesDialog


class SaveTemplateDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Save Selection as Block")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Name of the reusable block (listed under Custom):"))
        self.field = QtWidgets.QLineEdit()
        layout.addWidget(self.field)
        self.add_buttons("Save")

    def template_name(self) -> str:
        return self.field.text().strip()


class ShortcutsDialog(BaseDialog):
    """Generated from the actions catalog — every shortcut documented."""

    def __init__(self, parent, specs: list):
        super().__init__(parent, "Keyboard Shortcuts")
        self.resize(460, 520)
        layout = QtWidgets.QVBoxLayout(self)
        rows = [s for s in specs if s.shortcut]
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        from PySide6 import QtGui

        for row, spec in enumerate(rows):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(spec.text.replace("&", "")))
            native = QtGui.QKeySequence(spec.shortcut).toString(
                QtGui.QKeySequence.SequenceFormat.NativeText)
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(native))
        layout.addWidget(table)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class AboutDialog(BaseDialog):
    def __init__(self, parent, version: str, environment: list[tuple[str, str]]):
        super().__init__(parent, "About AI Made Easy")
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(f"<span style='font-size:18px'><b>AI Made Easy</b></span> "
                                 f"&nbsp;{version}")
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel(
            "Visual model builder: design neural networks and classic ML pipelines,\n"
            "validate them as you build, train in-app and export clean code."))
        grid = QtWidgets.QFormLayout()
        for name, value in environment:
            grid.addRow(f"{name}:", QtWidgets.QLabel(value))
        layout.addLayout(grid)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
