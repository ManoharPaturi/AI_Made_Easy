"""CaptureDialog: build an Image Folder dataset with your camera 📷 or
microphone 🎤 — hold to record.

Camera: QCamera + QMediaCaptureSession + QVideoWidget + QImageCapture (Qt
Multimedia only — no OpenCV). Frames are saved mirrored (what the kid sees
is what gets stored). Sound: optional `sounddevice` — 1 s clips become
log-spectrogram PNGs the Image Folder pipeline trains on unchanged.
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtMultimedia, QtWidgets

_EXPLAINED_KEY = "aime/capture/tcc_explained"
_BURST_MS = 350          # one frame while held every ~350 ms
_MIN_SOUND_S = 0.4       # ignore accidental taps shorter than this
_SR = 22050              # sample rate for mic recordings


def _new_class_dir(root: Path, name: str) -> Path:
    safe = ("".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
            .strip().replace(" ", "_").strip("_-") or "class")
    d = root / safe
    i = 2
    while d.exists() and any(d.iterdir()):
        d = root / f"{safe}_{i}"
        i += 1
    d.mkdir(parents=True, exist_ok=True)
    return d


def spectrogram_png(audio, path: Path) -> bool:
    """1-D float audio → log-spectrogram PNG (amber→red map)."""
    try:
        import numpy as np

        x = np.asarray(audio, dtype=np.float32)
        if x.size < _SR // 4:
            return False
        spec = _stft_logmag(x)
        h, w = spec.shape
        lo_c, hi_c = (255, 171, 45), (229, 72, 77)  # amber → red
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(3):
            img[:, :, i] = (lo_c[i]
                            + (hi_c[i] - lo_c[i]) * spec).astype(np.uint8)
        from PIL import Image

        Image.fromarray(img).save(path)
        return True
    except Exception:
        return False


def _stft_logmag(x) -> "np.ndarray":
    """Normalised log-magnitude spectrogram of a 1-D signal."""
    import numpy as np
    import torch

    t = torch.from_numpy(x)
    n_fft = 512
    hop = 160
    spec = torch.stft(t, n_fft=n_fft, hop_length=hop,
                      win_length=n_fft, return_complex=True,
                      window=torch.hann_window(n_fft))
    mag = torch.log1p(spec.abs()).numpy().T  # (time, freq)
    lo, hi = float(mag.min()), float(mag.max())
    return np.clip((mag - lo) / (hi - lo + 1e-9), 0.0, 1.0)


class CaptureDialog(QtWidgets.QDialog):
    """Hold-to-record examples into <root>/<class>/ for an Image Folder."""

    def __init__(self, parent, root: Path):  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("📷 Teach with your camera")
        self.setModal(True)
        self.resize(640, 560)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self._camera: QtMultimedia.QCamera | None = None
        self._session: QtMultimedia.QMediaCaptureSession | None = None
        self._still: QtMultimedia.QImageCapture | None = None
        self._sink = None
        self._sound_stream = None
        self._recording_sound = False
        self._sound_clip: list = []
        self._saved = 0

        layout = QtWidgets.QVBoxLayout(self)

        self.mode_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.mode_tabs, 1)
        self.mode_tabs.addTab(self._build_camera_tab(), "📷 Camera")
        self.mode_tabs.addTab(self._build_sound_tab(), "🎤 Sounds")

        self.class_row = QtWidgets.QHBoxLayout()
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.setMinimumWidth(180)
        self.class_row.addWidget(QtWidgets.QLabel("Class:"))
        self.class_row.addWidget(self.class_combo)
        new_btn = QtWidgets.QPushButton("➕ new class…")
        new_btn.clicked.connect(self._prompt_new_class)
        self.class_row.addWidget(new_btn, 1)
        self.count_label = QtWidgets.QLabel("")
        self.class_row.addWidget(self.count_label)
        layout.addLayout(self.class_row)

        self.hold_btn = QtWidgets.QPushButton("⏺  HOLD to record")
        self.hold_btn.setMinimumHeight(52)
        self.hold_btn.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.hold_btn.pressed.connect(self._start_recording)
        self.hold_btn.released.connect(self._stop_recording)
        layout.addWidget(self.hold_btn)

        self.status = QtWidgets.QLabel("pick a class, then hold the big button")
        self.status.setStyleSheet("color: #7A7565;")
        layout.addWidget(self.status)

        close = QtWidgets.QPushButton("Done ✅")
        close.clicked.connect(self.accept)
        layout.addWidget(close)

        self._burst = QtCore.QTimer(self, interval=_BURST_MS,
                                    timeout=self._capture_one)
        self.finished.connect(self._cleanup)
        self._refresh_classes()
        self._maybe_explain()

    def _cleanup(self) -> None:
        """Runs on accept/reject/close — accept() skips closeEvent."""
        self._burst.stop()
        if self._recording_sound:
            self._stop_sound()
        if self._camera is not None:
            self._camera.stop()

    # ------------------------------------------------------------- camera
    def _build_camera_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        devices = QtMultimedia.QMediaDevices.videoInputs()
        if not devices:
            self.hold_enabled_camera = False
            note = QtWidgets.QLabel(
                "🔍 No camera found.\n\nPlug one in (or grant camera "
                "permission to the app) and reopen this window.")
            note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            note.setWordWrap(True)
            lay.addWidget(note, 1)
            return page

        self.hold_enabled_camera = True
        from PySide6 import QtMultimediaWidgets

        self.view = QtMultimediaWidgets.QVideoWidget()
        lay.addWidget(self.view, 1)

        self._camera = QtMultimedia.QCamera(devices[0])
        self._session = QtMultimedia.QMediaCaptureSession()
        self._session.setCamera(self._camera)
        self._session.setVideoOutput(self.view)
        self._still = QtMultimedia.QImageCapture()
        self._session.setImageCapture(self._still)
        self._still.imageCaptured.connect(self._on_image_captured)
        self._camera.start()
        QtCore.QTimer.singleShot(
            2500, self._check_camera_permission)
        return page

    def _check_camera_permission(self) -> None:
        """A camera blocked by macOS privacy settings stays inactive."""
        if self._camera is not None and not self._camera.isActive():
            self.status.setText(
                "📷 camera not allowed yet — click Allow when macOS asks "
                "(or System Settings › Privacy › Camera), then reopen")

    # -------------------------------------------------------------- sound
    def _build_sound_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        self.mic_available = self._import_sounddevice() is not None
        if not self.mic_available:
            note = QtWidgets.QLabel(
                "🎤 Sound recording needs the optional `sounddevice` "
                "package.\n\nInstall it, reopen the app, and this tab "
                "records 1-second sounds as spectrogram pictures.")
            note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            note.setWordWrap(True)
            lay.addWidget(note, 1)
        else:
            how = QtWidgets.QLabel(
                "Hold the big button and make a sound — claps, your voice, "
                "a pet. Each hold becomes one spectrogram picture in the "
                "class folder.")
            how.setWordWrap(True)
            lay.addWidget(how, 1)
        return page

    @staticmethod
    def _import_sounddevice():
        try:
            import sounddevice as sd

            return sd
        except Exception:
            return None

    # ------------------------------------------------------------- classes
    def _classes(self) -> list[str]:
        return [d.name for d in sorted(self.root.iterdir())
                if d.is_dir()] if self.root.exists() else []

    def _refresh_classes(self) -> None:
        current = self.class_combo.currentText()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(self._classes())
        if current in self._classes():
            self.class_combo.setCurrentText(current)
        self.class_combo.blockSignals(False)
        self._update_count()

    def _current_class_dir(self) -> Path | None:
        name = self.class_combo.currentText().strip()
        if not name:
            return None
        return self.root / name

    def count_for(self, class_name: str) -> int:
        from ai_made_easy.core.dataset_health import _images_in

        d = self.root / class_name
        return len(_images_in(d)) if d.exists() else 0

    def _update_count(self) -> None:
        name = self.class_combo.currentText()
        if not name:
            self.count_label.setText("no classes yet — ➕ make one")
            return
        self.count_label.setText(f"{name}: {self.count_for(name)} photos")

    def _prompt_new_class(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New class", "What should the model learn to recognise?")
        if ok and name.strip():
            created = _new_class_dir(self.root, name.strip())
            self._refresh_classes()
            self.class_combo.setCurrentText(created.name)
            self._update_count()

    # ---------------------------------------------------------- recording
    def _start_recording(self) -> None:
        if self._current_class_dir() is None:
            self.status.setText("make a class first ➕")
            return
        if self.mode_tabs.currentIndex() == 0:
            if not self.hold_enabled_camera:
                self.status.setText("no camera available")
                return
            self._capture_one()
            self._burst.start()
            self.status.setText("recording… keep holding! 🎬")
        else:
            self._start_sound()

    def _stop_recording(self) -> None:
        self._burst.stop()
        if self._recording_sound:
            self._stop_sound()

    def _capture_one(self) -> None:
        if self._still is not None and self._still.isReadyForCapture():
            self._still.capture()

    def _on_image_captured(self, _id: int, image: QtGui.QImage) -> None:
        self._save_image(image)

    def _save_image(self, image: QtGui.QImage) -> Path | None:
        d = self._current_class_dir()
        if d is None:
            return None
        d.mkdir(parents=True, exist_ok=True)
        path = d / time.strftime(f"shot_%Y%m%d_%H%M%S_{self._saved:03d}.png")
        self._saved += 1
        mirrored = image.mirrored(True, False)  # match the viewfinder
        if not mirrored.save(str(path), "PNG"):
            return None
        self._update_count()
        return path

    # sound record path ------------------------------------------------
    def _start_sound(self) -> None:
        if not self.mic_available:
            self.status.setText("install `sounddevice` to record sounds")
            return
        sd = self._import_sounddevice()
        self._sound_clip = []

        def _cb(indata, frames, t, status):  # noqa: ANN001
            self._sound_clip.append(indata[:, 0].copy())

        try:
            self._sound_stream = sd.InputStream(
                samplerate=_SR, channels=1, dtype="float32", callback=_cb)
            self._sound_stream.start()
            self._recording_sound = True
            self.status.setText("listening… make the sound! 🎤")
        except Exception as exc:
            self.status.setText(f"mic problem: {exc}")

    def _stop_sound(self) -> None:
        self._recording_sound = False
        stream, self._sound_stream = self._sound_stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        import numpy as np

        audio = (np.concatenate(self._sound_clip)
                 if self._sound_clip else np.zeros(0))
        if audio.size < int(_SR * _MIN_SOUND_S):
            self.status.setText("too short — hold a bit longer")
            return
        path = self._save_spectrogram(audio)
        self.status.setText("saved spectrogram 🎼" if path
                            else "couldn't save that one")

    def _save_spectrogram(self, audio) -> Path | None:
        d = self._current_class_dir()
        if d is None:
            return None
        d.mkdir(parents=True, exist_ok=True)
        path = d / time.strftime("sound_%Y%m%d_%H%M%S.png")
        return path if spectrogram_png(audio, path) else None

    # ------------------------------------------------------------- close
    def _maybe_explain(self) -> None:
        settings = QtCore.QSettings()
        if settings.value(_EXPLAINED_KEY, False, type=bool):
            return
        settings.setValue(_EXPLAINED_KEY, True)
        QtWidgets.QMessageBox.information(
            self, "About camera permission",
            "macOS will ask once for permission to use the camera 📷\n\n"
            "Click Allow when the system question appears — photos never "
            "leave this computer.")

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        self._cleanup()
        super().closeEvent(event)
