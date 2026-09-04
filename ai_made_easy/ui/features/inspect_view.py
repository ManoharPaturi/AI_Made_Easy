"""Inspect view: "What is it looking at?" — Grad-CAM over the input plus a
first-layer feature grid. The heavy lifting happens in a generated inspect
script (run in the training workdir); this dialog just renders the .npy /
.json artifacts and re-runs with new inputs on request.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

_HEAT_LOW = (255, 171, 45)     # amber
_HEAT_HIGH = (229, 72, 77)     # red


def _array_to_image(arr: np.ndarray) -> QtGui.QImage:
    """Float array (C,H,W / H,W / grid H,W) → display QImage."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] in (1, 3):  # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 3 and a.shape[0] not in (1, 3) and a.shape[-1] not in (1, 3):
        a = a[0]  # unexpected channel stack — show the first slice
    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] not in (1, 3):
        a = a[:, :, :1]
    lo, hi = float(a.min()), float(a.max())
    a = ((a - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    h, w, _ = a.shape
    return QtGui.QImage(a.copy(), w, h, 3 * w,
                        QtGui.QImage.Format.Format_RGB888)


def _overlay(base: np.ndarray, cam: np.ndarray) -> QtGui.QImage:
    """Red-amber heatmap at ~45% alpha over the (C,H,W / H,W / H,W,1) input."""
    if base.ndim == 3 and base.shape[0] in (1, 3):
        base = np.transpose(base, (1, 2, 0))
    if base.ndim == 3 and base.shape[2] == 1:
        base = base[:, :, 0]
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    lo, hi = float(base.min()), float(base.max())
    img = ((base - lo) / (hi - lo + 1e-9))
    cam = np.asarray(cam, dtype=np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    h, w = cam.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (c_low, c_high) in enumerate(zip(_HEAT_LOW, _HEAT_HIGH)):
        heat = c_low + (c_high - c_low) * cam
        out[:, :, i] = np.clip(img[:, :, i] * 255 * 0.55 + heat * 0.45, 0, 255)
    return QtGui.QImage(out.copy(), w, h, 3 * w,
                        QtGui.QImage.Format.Format_RGB888)


class InspectDialog(QtWidgets.QDialog):
    """Renders inspect artifacts; `rerun_requested(str)` re-runs the script
    with a sample index or '--image <path>'."""

    rerun_requested = QtCore.Signal(str)

    def __init__(self, parent, workdir: Path):
        super().__init__(parent)
        self.setWindowTitle("👀 What is it looking at?")
        self.setModal(True)
        self.resize(820, 560)
        self.workdir = Path(workdir)

        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        self.index_spin = QtWidgets.QSpinBox()
        self.index_spin.setRange(0, 299)
        controls.addWidget(QtWidgets.QLabel("Test example #"))
        controls.addWidget(self.index_spin)
        run_btn = QtWidgets.QPushButton("🔎 Explain this one")
        run_btn.clicked.connect(
            lambda: self.rerun_requested.emit(str(self.index_spin.value())))
        controls.addWidget(run_btn)
        img_btn = QtWidgets.QPushButton("📁 My own image…")
        img_btn.clicked.connect(self._pick_image)
        controls.addWidget(img_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.sentence = QtWidgets.QLabel("Run an example to see the heatmap.")
        self.sentence.setStyleSheet("font-weight: 700; padding: 4px;")
        self.sentence.setWordWrap(True)
        layout.addWidget(self.sentence)

        views = QtWidgets.QHBoxLayout()
        self.cam_label = QtWidgets.QLabel("heatmap appears here")
        self.cam_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setMinimumSize(280, 280)
        self.cam_label.setStyleSheet("background: #FFFDF8;"
                                     "border: 1px solid #E1DDCF;"
                                     "border-radius: 10px;")
        views.addWidget(self.cam_label, 1)
        self.feats_label = QtWidgets.QLabel("what the first layer saw")
        self.feats_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.feats_label.setMinimumSize(280, 280)
        self.feats_label.setStyleSheet("background: #FFFDF8;"
                                       "border: 1px solid #E1DDCF;"
                                       "border-radius: 10px;")
        views.addWidget(self.feats_label, 1)
        layout.addLayout(views, 1)

        note = QtWidgets.QLabel(
            "Warm colours = where the model looked when it decided.\n"
            "If the warmth is on the background, the model learned the "
            "wrong thing — add more varied photos!")
        note.setWordWrap(True)
        note.setStyleSheet("color: #7A7565;")
        layout.addWidget(note)

        close = QtWidgets.QPushButton("Got it 👍")
        close.clicked.connect(self.accept)
        layout.addWidget(close)

        self.load_results()

    def _pick_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Pick an image", str(self.workdir), "Images (*.png *.jpg *.jpeg)")
        if path:
            self.rerun_requested.emit(f"--image {path}")

    def load_results(self) -> None:
        try:
            result = json.loads((self.workdir / "inspect.json").read_text())
        except (OSError, ValueError):
            self.sentence.setText("No inspection results yet — run one!")
            return
        top = result.get("top", [])
        if result.get("single"):
            self.sentence.setText(
                f"The model predicted {top[0]['prob']:.2f} "
                "(a number, not a class)")
        else:
            guesses = ", ".join(f"class {t['class']} ({t['prob']:.0%})"
                                for t in top[:3])
            self.sentence.setText(
                f"The model guessed {guesses} — mostly by looking at the "
                "warm spots below")
        cam_path = self.workdir / "cam.npy"
        inp_path = self.workdir / "input.npy"
        if cam_path.exists() and inp_path.exists():
            image = _overlay(np.load(inp_path), np.load(cam_path))
            self.cam_label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
                self.cam_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
        feats = self.workdir / "feats.npy"
        if feats.exists():
            image = _array_to_image(np.load(feats))
            self.feats_label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
                self.feats_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
