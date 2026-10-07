"""Data preview: inspect what a dataset block will feed the pipeline.

Opened by double-clicking a dataset block. Reads are small and local: the
first rows of tables, array shapes, class folders with health findings, or a
description of benchmark datasets.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.dataset_health import (
    AUDIO_SUFFIXES,
    IMAGE_SUFFIXES,
    TEXT_SUFFIXES,
    scan_class_folder,
)

_BENCHMARKS = {
    "mnist": "Handwritten digits 0–9 · 60,000 train / 10,000 test · [1, 28, 28]",
    "fashion_mnist": "Zalando clothing (10 classes) · 60,000 / 10,000 · [1, 28, 28]",
    "kmnist": "Kuzushiji characters (10 classes) · 60,000 / 10,000 · [1, 28, 28]",
    "cifar10": "Natural images (10 classes) · 50,000 / 10,000 · [3, 32, 32]",
    "cifar100": "Natural images (100 classes) · 50,000 / 10,000 · [3, 32, 32]",
    "svhn": "Street-view house numbers · 73,257 / 26,032 · [3, 32, 32]",
    "stl10": "Natural images (10 classes) · 5,000 / 8,000 · [3, 96, 96]",
    "usps": "Handwritten digits · 7,291 / 2,007 · [1, 16, 16]",
    "iris": "Iris flowers · 150 samples · 4 features · 3 classes",
    "wine": "Wine cultivars · 178 samples · 13 features · 3 classes",
    "breast_cancer": "Breast cancer diagnosis · 569 samples · 30 features · 2 classes",
    "digits": "8×8 digits · 1,797 samples · 64 features · 10 classes",
    "diabetes": "Diabetes progression · 442 samples · 10 features · regression",
    "california_housing": "California housing · 20,640 samples · 8 features · regression",
}


def _table(path: str, fmt: str = "csv", rows: int = 8) -> tuple[list[str], list[list[str]], str]:
    p = Path(path).expanduser()
    if not p.exists():
        return [], [], f"File not found: {p}"
    try:
        import pandas as pd

        reader = {"csv": pd.read_csv, "tsv": lambda f: pd.read_csv(f, sep="\t"),
                  "parquet": pd.read_parquet, "excel": pd.read_excel,
                  "json": lambda f: pd.read_json(f, lines=str(f).endswith(".jsonl"))}[fmt]
        frame = reader(p)
        head = frame.head(rows)
        info = (f"{len(frame):,} rows × {len(frame.columns)} columns · "
                f"{int(frame.isna().sum().sum()):,} missing values")
        return [str(c) for c in head.columns], head.astype(str).values.tolist(), info
    except Exception as exc:  # noqa: BLE001 — shown to the user
        return [], [], f"Could not read {p.name}: {exc}"


def _folder_summary(root: str, suffixes: set[str]) -> str:
    report = scan_class_folder(Path(root).expanduser(), suffixes)
    lines = [f"{report.total:,} samples in {len(report.classes)} class(es)"]
    lines += [f"  {c.name}: {c.count:,}" for c in report.classes]
    findings = [f for f in report.findings if f.severity == "warning"]
    if findings:
        lines.append("")
        lines += [f"Warning: {f.message}" + (f" — {f.hint}" if f.hint else "") for f in findings]
    return "\n".join(lines)


def describe(type_id: str, params: dict) -> str:
    if type_id in ("data.torchvision", "data.sklearn"):
        name = params.get("dataset", "")
        return _BENCHMARKS.get(name, name) + "\n\nDownloaded automatically on first use."
    if type_id == "data.image_folder":
        return _folder_summary(params.get("root", ""), IMAGE_SUFFIXES)
    if type_id == "data.text_folder":
        return _folder_summary(params.get("root", ""), TEXT_SUFFIXES)
    if type_id == "data.audio_folder":
        return _folder_summary(params.get("root", ""), AUDIO_SUFFIXES)
    if type_id == "data.numpy":
        p = Path(params.get("path", "")).expanduser()
        if not p.exists():
            return f"File not found: {p}"
        import numpy as np

        with np.load(p) as archive:
            return "\n".join(f"{k}: shape {list(archive[k].shape)}, dtype {archive[k].dtype}"
                             for k in archive.files)
    if type_id == "data.json":
        p = Path(params.get("path", "")).expanduser()
        if not p.exists():
            return f"File not found: {p}"
        text = p.read_text().strip()
        records = (json.loads(text) if text.startswith("[")
                   else [json.loads(line) for line in text.splitlines()[:5] if line.strip()])
        return "\n".join(json.dumps(r)[:300] for r in records[:5])
    if type_id == "data.huggingface":
        return (f"Hugging Face dataset '{params.get('repo_id')}' (split "
                f"'{params.get('split')}'), downloaded on first use.")
    if type_id == "data.synthetic":
        return (f"{params.get('kind')} data · {params.get('n_samples')} samples · "
                f"{params.get('n_features')} features"
                + (f" · {params.get('n_classes')} classes"
                   if params.get("kind") != "regression" else ""))
    return "\n".join(f"{k}: {v}" for k, v in params.items())


class DataPreviewDialog(QtWidgets.QDialog):
    def __init__(self, parent, definition, params: dict):  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle(f"{definition.display_name} — Preview")
        self.resize(760, 480)
        layout = QtWidgets.QVBoxLayout(self)
        type_id = definition.type_id
        tables = {"data.csv": "path", "data.text_csv": "path", "data.timeseries_csv": "path"}
        if type_id in tables:
            columns, rows, info = _table(params.get("path", ""), params.get("format", "csv"))
            meta = QtWidgets.QLabel(info)
            meta.setObjectName("blockMeta")
            layout.addWidget(meta)
            table = QtWidgets.QTableWidget(len(rows), len(columns))
            table.setHorizontalHeaderLabels(columns)
            table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    table.setItem(r, c, QtWidgets.QTableWidgetItem(value))
            layout.addWidget(table, 1)
        else:
            view = QtWidgets.QPlainTextEdit(describe(type_id, params))
            view.setObjectName("codeView")
            view.setReadOnly(True)
            layout.addWidget(view, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 0, QtCore.Qt.AlignmentFlag.AlignRight)
