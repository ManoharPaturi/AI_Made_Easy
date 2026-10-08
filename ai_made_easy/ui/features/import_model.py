"""Import Model dialog: PyTorch / ONNX / Keras → an editable design."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.importers import ModelImportError, import_model

_KINDS = [("PyTorch module (.py file or installed module)", "pytorch"),
          ("ONNX model (.onnx)", "onnx"),
          ("Keras model (.keras / .h5)", "keras")]
_FILTERS = {"pytorch": "Python files (*.py)", "onnx": "ONNX models (*.onnx)",
            "keras": "Keras models (*.keras *.h5)"}


class ImportModelDialog(QtWidgets.QDialog):
    """Runs the importer on a worker thread and previews the fidelity report."""

    _done = QtCore.Signal(object, str)  # ImportResult | None, error

    def __init__(self, parent, python: str | None = None, start_dir: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Import Model")
        self.resize(760, 600)
        self._python = python
        self._start_dir = start_dir or Path.home()
        self.result = None
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Turns an existing model into blocks you can edit, validate, train and export. "
            "The rebuilt model is checked against the original: parameter count, output "
            "shape and, when weights can be mapped, the outputs themselves.")
        intro.setObjectName("blockMeta")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        self.kind = QtWidgets.QComboBox()
        for label, kind in _KINDS:
            self.kind.addItem(label, kind)
        form.addRow("Format", self.kind)
        row = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QLineEdit()
        self.source.setPlaceholderText("path/to/model.py  or  torchvision.models")
        browse = QtWidgets.QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.source, 1)
        row.addWidget(browse)
        form.addRow("Source", row)
        self.attr = QtWidgets.QLineEdit()
        self.attr.setPlaceholderText("class or factory, e.g. Net or resnet18")
        form.addRow("Model class", self.attr)
        self.kwargs = QtWidgets.QLineEdit("{}")
        self.kwargs.setPlaceholderText('constructor arguments as JSON, e.g. {"num_classes": 10}')
        form.addRow("Arguments", self.kwargs)
        self.shape = QtWidgets.QLineEdit()
        self.shape.setPlaceholderText("per sample, without the batch: 3, 224, 224")
        form.addRow("Input shape", self.shape)
        self.dtype = QtWidgets.QComboBox()
        self.dtype.addItems(["float32", "int64"])
        self.dtype.setToolTip("int64 for token ids (models that start with an Embedding)")
        form.addRow("Input type", self.dtype)
        layout.addLayout(form)
        self._torch_rows = [self.attr, self.kwargs, self.dtype]
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self._kind_changed()

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setObjectName("logView")
        self.report.setPlaceholderText("The import report appears here.")
        layout.addWidget(self.report, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.import_btn = QtWidgets.QPushButton("Import")
        self.import_btn.clicked.connect(self._run)
        self.open_btn = QtWidgets.QPushButton("Open as Project")
        self.open_btn.setObjectName("primaryButton")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.accept)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        for b in (self.import_btn, self.open_btn, cancel):
            buttons.addWidget(b)
        layout.addLayout(buttons)
        self._done.connect(self._finished)

    def _kind_changed(self) -> None:
        torch = self.kind.currentData() == "pytorch"
        for widget in self._torch_rows:
            widget.setEnabled(torch)
        self.shape.setPlaceholderText(
            "per sample, without the batch: 3, 224, 224" if torch
            else "optional — only for inputs with dynamic dimensions")

    def _browse(self) -> None:
        kind = self.kind.currentData()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a model", str(self._start_dir), _FILTERS[kind])
        if path:
            self.source.setText(path)

    def request(self) -> dict:
        """The import request from the form (raises ValueError on bad input)."""
        kind = self.kind.currentData()
        source = self.source.text().strip()
        if not source:
            raise ValueError("choose a model file")
        shape_text = self.shape.text().replace("x", ",").replace("(", "").replace(")", "")
        try:
            shape = [int(d) for d in shape_text.split(",") if d.strip()] or None
        except ValueError as exc:
            raise ValueError("the input shape must be integers, e.g. 3, 224, 224") from exc
        request = {"kind": kind, "source": source, "input_shape": shape}
        if kind == "pytorch":
            if not self.attr.text().strip():
                raise ValueError("enter the model class or factory name")
            if not shape:
                raise ValueError("PyTorch imports need the input shape")
            try:
                kwargs = json.loads(self.kwargs.text() or "{}")
            except ValueError as exc:
                raise ValueError("arguments must be a JSON object") from exc
            if not isinstance(kwargs, dict):
                raise ValueError("arguments must be a JSON object")
            request.update(attr=self.attr.text().strip(), kwargs=kwargs,
                           dtype=self.dtype.currentText())
        return request

    def _run(self) -> None:
        try:
            request = self.request()
        except ValueError as exc:
            self.report.setPlainText(str(exc))
            return
        self.import_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.progress.show()
        self.report.setPlainText("Importing…")

        def work() -> None:
            req = dict(request)
            kind = req.pop("kind")
            try:
                self._done.emit(import_model(kind, python=self._python, **req), "")
            except ModelImportError as exc:
                self._done.emit(None, str(exc))
            except Exception as exc:  # noqa: BLE001 — shown in the dialog
                self._done.emit(None, f"{type(exc).__name__}: {exc}")

        threading.Thread(target=work, daemon=True).start()

    def _finished(self, result, error: str) -> None:  # noqa: ANN001
        self.progress.hide()
        self.import_btn.setEnabled(True)
        if error:
            self.report.setPlainText("Import failed:\n" + error)
            return
        self.result = result
        text = result.summary()
        if result.warnings:
            text += "\n\nNotes:\n  - " + "\n  - ".join(result.warnings)
        if result.unsupported:
            text += ("\n\nReplace or remove these parts of the model and import again, or "
                     "rebuild them from blocks after importing the rest.")
        self.report.setPlainText(text)
        self.open_btn.setEnabled(not result.unsupported
                                 and bool(result.verification.get("ok")))
