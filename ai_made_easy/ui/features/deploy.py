"""Deploy dialog: build a model-server package from a run or registered model."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy.core.deploy import FORMAT_LABELS, FORMATS, DeployError, build_package


class DeployDialog(QtWidgets.QDialog):
    """Pick formats and a destination; the build runs on a worker thread."""

    _done = QtCore.Signal(object, str)  # PackageResult | None, error

    def __init__(self, parent, source: Path, framework: str, name: str, *,
                 version: str = "1", default_dir: Path | None = None,
                 python: str | None = None, subtitle: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Deploy Model")
        self.resize(720, 560)
        self._source = Path(source)
        self._framework = framework
        self._version = version
        self._python = python
        self.result_path: Path | None = None
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Builds a self-contained model server: a FastAPI app with /predict, the trained "
            "model and its fitted preprocessing, pinned requirements and a Dockerfile."
            + (f"\n{subtitle}" if subtitle else ""))
        intro.setObjectName("blockMeta")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(name)
        form.addRow("Service name", self.name)
        folder_row = QtWidgets.QHBoxLayout()
        base = default_dir or Path.home()
        self.folder = QtWidgets.QLineEdit(str(base / f"{_slug(name)}-server"))
        browse = QtWidgets.QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)
        form.addRow("Output folder", folder_row)
        layout.addLayout(form)

        box = QtWidgets.QGroupBox("Additional formats (optional)")
        grid = QtWidgets.QVBoxLayout(box)
        self.format_checks: dict[str, QtWidgets.QCheckBox] = {}
        for fmt in FORMATS.get(framework, ()):
            check = QtWidgets.QCheckBox(FORMAT_LABELS[fmt])
            check.setChecked(fmt in ("onnx", "torchscript"))
            grid.addWidget(check)
            self.format_checks[fmt] = check
        if not self.format_checks:
            grid.addWidget(QtWidgets.QLabel("No extra formats for this framework."))
        hint = QtWidgets.QLabel("Formats whose converter is not installed in the training "
                                "environment are skipped and reported.")
        hint.setObjectName("blockMeta")
        hint.setWordWrap(True)
        grid.addWidget(hint)
        layout.addWidget(box)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("logView")
        self.log.setPlaceholderText("Build output appears here.")
        layout.addWidget(self.log, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        buttons = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open Folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_folder)
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        self.build_btn = QtWidgets.QPushButton("Build Package")
        self.build_btn.setObjectName("primaryButton")
        self.build_btn.clicked.connect(self._build)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        buttons.addWidget(self.build_btn)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)
        self._done.connect(self._finished)

    def selected_formats(self) -> tuple[str, ...]:
        return tuple(f for f, c in self.format_checks.items() if c.isChecked())

    def _browse(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose an empty folder", str(Path(self.folder.text()).parent))
        if folder:
            self.folder.setText(folder)

    def _build(self) -> None:
        out = Path(self.folder.text()).expanduser()
        name = self.name.text().strip() or "model"
        if out.exists() and any(out.iterdir()):
            self.log.appendPlainText(f"{out} is not empty — choose another folder.")
            return
        self.build_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.progress.show()
        self.log.appendPlainText(f"Building {name} → {out}")
        formats = self.selected_formats()

        def work() -> None:
            try:
                result = build_package(self._source, out, formats=formats, name=name,
                                       version=self._version, python=self._python)
                self._done.emit(result, "")
            except (DeployError, OSError) as exc:
                self._done.emit(None, str(exc))
            except Exception as exc:  # noqa: BLE001 — surfaced in the dialog
                self._done.emit(None, f"{type(exc).__name__}: {exc}")

        threading.Thread(target=work, daemon=True).start()

    def _finished(self, result, error: str) -> None:  # noqa: ANN001
        self.progress.hide()
        self.close_btn.setEnabled(True)
        if error:
            self.build_btn.setEnabled(True)
            self.log.appendPlainText("Failed: " + error)
            return
        self.result_path = result.path
        self.log.appendPlainText(result.log.replace("EXPORT-DONE", "").strip())
        self.log.appendPlainText(f"\nPackage ready: {result.path}\n"
                                 f"Run it:  cd {result.path} && uvicorn app:app --port 8000")
        self.open_btn.setEnabled(True)

    def _open_folder(self) -> None:
        if self.result_path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.result_path)))


def _slug(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-") or "model"
