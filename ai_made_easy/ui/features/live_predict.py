"""LivePredictDialog: point the camera at things and watch the trained
model guess, in real time (~8 FPS).

The worker imports the run's generated training script as a module (its
model class, INPUT_SHAPE and optional NORM_* live at module level; its
main() is guarded) and loads the checkpoint in a QThread so the GUI never
blocks. Frames arrive from a QVideoSink, are mirrored + resized exactly
like the generated Image Folder loader (PIL resize → /255 → optional
normalize), and predicted with a plain softmax.
"""
from __future__ import annotations

import importlib.util
import queue
import sys
import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtMultimedia, QtWidgets


def load_predictor(workdir: Path):
    """Import the run's train script + checkpoint. Returns
    (model, input_shape, norm) or raises with a kid-friendly message."""
    import torch

    scripts = sorted(Path(workdir).glob("*_train_pytorch.py"))
    if not scripts:  # tolerate hand-renamed exports
        scripts = [p for p in sorted(Path(workdir).glob("*.py"))
                   if not p.name.startswith(("aime_inspect", "aime_"))]
    if not scripts:
        raise RuntimeError("no training script in this workspace")
    script = scripts[0]
    mod_name = f"aime_live_{script.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    model_cls = next(
        (obj for obj in vars(module).values()
         if isinstance(obj, type)
         and issubclass(obj, __import__("torch").nn.Module)
         and obj.__module__ == mod_name), None)
    if model_cls is None:
        raise RuntimeError("couldn't find the model in the training script")
    model = model_cls()
    ckpt = Path(workdir) / module.CHECKPOINT
    if not ckpt.exists():
        raise RuntimeError(f"checkpoint missing: {ckpt.name} — train first")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state)
    model.eval()
    shape = tuple(int(v) for v in module.INPUT_SHAPE)
    if len(shape) != 3 or shape[0] not in (1, 3):
        raise RuntimeError("live camera mode needs an image model "
                           "(input like 3,64,64)")
    norm = (tuple(module.NORM_MEAN), tuple(module.NORM_STD)) \
        if hasattr(module, "NORM_MEAN") else None
    return model, shape, norm


def frame_to_tensor(image: QtGui.QImage, shape, norm):
    """QImage → (1,C,H,W) float tensor using the training pipeline's math."""
    import numpy as np
    import torch
    from PIL import Image

    c, h, w = shape
    pil = Image.fromqimage(image)
    pil = pil.convert("L" if c == 1 else "RGB").resize((w, h))
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr)
    if c == 3:
        x = x.permute(2, 0, 1)
    else:
        x = x.unsqueeze(0)
    if norm is not None:
        mean = torch.tensor(norm[0], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(norm[1], dtype=torch.float32).view(-1, 1, 1)
        x = (x - mean) / std
    return x.unsqueeze(0)


class _Worker(QtCore.QThread):
    prediction = QtCore.Signal(list)

    def __init__(self, workdir: Path, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self.workdir = Path(workdir)
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._err: str | None = None

    def submit(self, image: QtGui.QImage) -> None:
        try:
            self._queue.put_nowait(image)
        except queue.Full:
            pass  # drop frames — we're behind, that's fine

    def stop(self) -> None:
        self._stop.set()
        self.wait(3000)

    # for tests: synchronous one-shot prediction
    def predict_now(self, image: QtGui.QImage) -> list:
        import torch

        model, shape, norm = load_predictor(self.workdir)
        with torch.no_grad():
            out = model(frame_to_tensor(image, shape, norm))
            probs = torch.softmax(out, dim=1)[0]
        return [round(float(v), 4) for v in probs]

    def run(self) -> None:
        import torch

        try:
            model, shape, norm = load_predictor(self.workdir)
        except Exception as exc:
            self._err = str(exc)
            self.prediction.emit([])
            return
        while not self._stop.is_set():
            try:
                image = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                with torch.no_grad():
                    out = model(frame_to_tensor(image, shape, norm))
                    probs = torch.softmax(out, dim=1)[0]
                self.prediction.emit([round(float(v), 4) for v in probs])
            except Exception:
                continue  # a bad frame shouldn't kill the loop


class LivePredictDialog(QtWidgets.QDialog):
    """Viewfinder + live top guess + confidence bars."""

    def __init__(self, parent, workdir: Path):  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("🔴 Live — try your model")
        self.setModal(True)
        self.resize(560, 620)
        self.workdir = Path(workdir)

        layout = QtWidgets.QVBoxLayout(self)
        self.video = QtWidgets.QLabel("starting camera…")
        self.video.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(420, 300)
        self.video.setStyleSheet("background: #1d1f24; color: #EDE9DC;")
        layout.addWidget(self.video, 1)

        self.result = QtWidgets.QLabel("…")
        self.result.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result.setStyleSheet("font-size: 22px; font-weight: 800;")
        layout.addWidget(self.result)

        self.bars_host = QtWidgets.QWidget()
        self.bars = QtWidgets.QVBoxLayout(self.bars_host)
        self.bars.setContentsMargins(12, 0, 12, 0)
        layout.addWidget(self.bars_host)

        row = QtWidgets.QHBoxLayout()
        pick = QtWidgets.QPushButton("📁 try a photo instead…")
        pick.clicked.connect(self._try_photo)
        row.addWidget(pick)
        close = QtWidgets.QPushButton("Stop 🔴")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)

        self._camera = None
        self._session: QtMultimedia.QMediaCaptureSession | None = None
        self._frame = None
        self._worker = _Worker(self.workdir, self)
        self._worker.prediction.connect(self._on_prediction)
        self._worker.start()
        self.finished.connect(self._cleanup)

        devices = QtMultimedia.QMediaDevices.videoInputs()
        if devices:
            self._camera = QtMultimedia.QCamera(devices[0])
            self._session = QtMultimedia.QMediaCaptureSession()
            self._session.setCamera(self._camera)
            tap = QtMultimedia.QVideoSink(self)
            self._session.setVideoSink(tap)
            tap.videoFrameChanged.connect(
                lambda frame: self._on_frame(frame.toImage()))
            self._camera.start()
            QtCore.QTimer.singleShot(2500, self._check_camera)
        else:
            self.video.setText("no camera found — use 📁 try a photo")

    def _check_camera(self) -> None:
        """Permission-denied cameras stay inactive with no frames — say so."""
        if self._camera is not None and not self._camera.isActive():
            self.video.setText(
                "camera not allowed yet 📷\n\nif macOS asked, click Allow — "
                "or turn it on in System Settings › Privacy › Camera — "
                "then reopen this window.\n\nMeanwhile: 📁 try a photo below")

    # ------------------------------------------------------------- frames
    def _on_frame(self, image: QtGui.QImage) -> None:
        if image.isNull():
            return
        self._frame = image
        self.video.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
            self.video.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        if self._worker.isRunning():
            self._worker.submit(image.mirrored(True, False))

    def _on_prediction(self, probs: list) -> None:
        if not probs:
            if self._worker._err:
                self.result.setText("can't run live 😕")
                self.video.setText(self._worker._err)
            return
        top = max(range(len(probs)), key=lambda i: probs[i])
        self.result.setText(f"class {top} · {probs[top]:.0%}")
        self._render_bars(probs)

    def _render_bars(self, probs: list) -> None:
        while self.bars.count():
            item = self.bars.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, p in enumerate(probs):
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(p * 100))
            bar.setTextVisible(True)
            bar.setFormat(f"class {i} — {p:.0%}")
            if i == max(range(len(probs)), key=lambda j: probs[j]):
                bar.setStyleSheet(
                    "QProgressBar::chunk { background: #63E6BE; }")
            self.bars.addWidget(bar)

    def _try_photo(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "pick a photo", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            image = QtGui.QImage(path)
            if not image.isNull():
                self._on_frame(image)

    def _cleanup(self) -> None:
        """Runs on accept/reject/close — accept() skips closeEvent."""
        self._worker.stop()
        if self._camera is not None:
            self._camera.stop()

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        self._cleanup()
        super().closeEvent(event)
