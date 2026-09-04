"""Report Card dialog: preview the auto-generated kid-worded Model Report
Card, let the learner write the two human fields, save as .md."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.model_card import build_card


class ReportCardDialog(QtWidgets.QDialog):
    def __init__(self, parent, name: str, dataset_comment: str,
                 trainer_params: dict, workdir):
        super().__init__(parent)
        self.setWindowTitle("🪪 Model Report Card")
        self.setModal(True)
        self.resize(640, 640)
        self._meta = (name, dataset_comment, trainer_params, workdir)

        layout = QtWidgets.QVBoxLayout(self)

        fields = QtWidgets.QFormLayout()
        self.superpower = QtWidgets.QLineEdit()
        self.superpower.setPlaceholderText(
            "e.g. It can tell cats from dogs in photos")
        self.careful = QtWidgets.QLineEdit()
        self.careful.setPlaceholderText(
            "e.g. Don't use it on blurry night photos")
        fields.addRow("🦸 Superpower", self.superpower)
        fields.addRow("⚠️ Be careful", self.careful)
        layout.addLayout(fields)

        self.view = QtWidgets.QTextBrowser()
        layout.addWidget(self.view, 1)

        buttons = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("💾 Save as Markdown…")
        save.clicked.connect(self._save)
        buttons.addStretch(1)
        buttons.addWidget(save)
        close = QtWidgets.QPushButton("Done")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        for field in (self.superpower, self.careful):
            field.textChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        name, ds, params, wd = self._meta
        self.view.setMarkdown(build_card(
            name, ds, params, wd,
            superpower=self.superpower.text().strip(),
            careful=self.careful.text().strip()))

    def _save(self) -> None:
        default = f"{self._meta[0].replace(' ', '_')}_report_card.md"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Report Card", default, "Markdown (*.md)")
        if path:
            from pathlib import Path
            Path(path).write_text(self.view.toMarkdown())
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Report card saved to\n{path}")
