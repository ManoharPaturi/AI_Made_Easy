"""Wave 2 — Teach with your world: capture, live prediction, spectrograms,
dataset health meter."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_made_easy.core.graph import Edge, Graph, NodeInstance  # noqa: E402


def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    apps = QtWidgets.QApplication.instance()
    return apps or QtWidgets.QApplication([])


# ------------------------------------------------------------ health meter

def test_health_findings(tmp_path):
    from ai_made_easy.core.dataset_health import scan_image_folder

    # missing root -> info
    r = scan_image_folder(tmp_path / "nope")
    assert r.findings and r.findings[0].severity == "info"

    (tmp_path / "cats").mkdir()
    (tmp_path / "dogs").mkdir()
    for i in range(3):  # cats: too few
        (tmp_path / "cats" / f"c{i}.png").write_bytes(b"x")
    for i in range(30):
        (tmp_path / "dogs" / f"d{i}.png").write_bytes(b"y")
    dup = (tmp_path / "dogs" / "copy.png")
    dup.write_bytes(b"y")  # exact duplicate of d0

    r = scan_image_folder(tmp_path)
    msgs = [f.message for f in r.findings]
    assert any("too few" in m and "cats" in m for m in msgs)
    assert any("unbalanced" in m for m in msgs)
    assert any("duplicate" in m for m in msgs)
    counts = {c.name: c.count for c in r.classes}
    assert counts == {"cats": 3, "dogs": 31}


def test_health_empty_folder_warning(tmp_path):
    from ai_made_easy.core.dataset_health import scan_image_folder

    (tmp_path / "empty_class").mkdir()
    r = scan_image_folder(tmp_path)
    assert any("is empty" in f.message for f in r.warnings)


def test_health_meter_widget_builds(tmp_path):
    _qapp()
    (tmp_path / "cats").mkdir()
    (tmp_path / "cats" / "a.png").write_bytes(b"x")
    from PySide6 import QtWidgets

    from ai_made_easy.ui.features.data_preview import _HealthMeter

    meter = _HealthMeter(tmp_path)
    labels = meter.findChildren(QtWidgets.QLabel)
    assert any("Dataset health" in l.text() for l in labels)
    assert meter.findChildren(QtWidgets.QProgressBar)  # balance bars


# ---------------------------------------------------------------- capture

def test_capture_dialog_state_machine(tmp_path, monkeypatch):
    _qapp()
    from PySide6 import QtGui, QtWidgets

    from ai_made_easy.ui.features.capture import CaptureDialog, _new_class_dir

    # no TCC explainer popup during the test
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    # offscreen: force the no-camera branch deterministically
    import ai_made_easy.ui.features.capture as cap

    monkeypatch.setattr(cap.QtMultimedia.QMediaDevices, "videoInputs",
                        staticmethod(lambda: []))

    dlg = CaptureDialog(None, tmp_path)
    assert not dlg.hold_enabled_camera  # graceful no-camera state

    d = _new_class_dir(tmp_path, "cats and dogs!")
    assert d.name == "cats_and_dogs"
    dlg._refresh_classes()
    assert dlg.class_combo.currentText() == "cats_and_dogs"

    img = QtGui.QImage(32, 32, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor("red"))
    saved = dlg._save_image(img)
    assert saved is not None and saved.exists()
    assert dlg.count_for("cats_and_dogs") == 1

    # burst never fires without a camera
    dlg._start_recording()
    assert not dlg._burst.isActive()


def test_spectrogram_png(tmp_path):
    import numpy as np

    from ai_made_easy.ui.features.capture import spectrogram_png

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    chirp = np.sin(2 * np.pi * (200 + 800 * t) * t).astype("float32")
    out = tmp_path / "spec.png"
    assert spectrogram_png(chirp, out)
    assert out.stat().st_size > 500  # a real picture, not an empty file

    assert not spectrogram_png(np.zeros(10, dtype="float32"),
                               tmp_path / "tiny.png")  # too short


# ------------------------------------------------------------ live predict

def _tiny_conv_graph():
    g = Graph(name="wave2live")
    g.add_node(NodeInstance("i", "core.input", {"shape": "1,8,8"}))
    g.add_node(NodeInstance("c", "core.conv2d", {"out_channels": 4}))
    g.add_node(NodeInstance("r", "core.relu"))
    g.add_node(NodeInstance("f", "core.flatten"))
    g.add_node(NodeInstance("d", "core.dense", {"units": 3}))
    g.add_node(NodeInstance("o", "core.output"))
    g.add_node(NodeInstance("tr", "train.trainer",
                            {"epochs": 1, "batch_size": 16}))
    for a, b in (("i", "c"), ("c", "r"), ("r", "f"), ("f", "d"), ("d", "o")):
        g.add_edge(Edge(a, "out", b, "in"))
    return g


@pytest.mark.skipif(os.environ.get("AIME_SKIP_TORCH"), reason="no torch")
def test_live_predictor_end_to_end(tmp_path):
    from ai_made_easy.core.codegen.training_gen import generate_training

    (tmp_path / "train_run.py").write_text(
        generate_training(_tiny_conv_graph(), "pytorch"))
    r = subprocess.run([sys.executable, "train_run.py"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]

    _qapp()
    from PySide6 import QtCore, QtGui

    from ai_made_easy.ui.features.live_predict import _Worker

    worker = _Worker(tmp_path)
    img = QtGui.QImage(16, 16, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor("steelblue"))
    probs = worker.predict_now(img)
    assert len(probs) == 3
    assert 0.999 <= sum(probs) <= 1.001

    # missing checkpoint -> kid-friendly error
    bad = tmp_path / "empty_wd"
    bad.mkdir()
    from ai_made_easy.ui.features.live_predict import load_predictor

    with pytest.raises(RuntimeError):
        load_predictor(bad)


def test_training_page_live_button():
    _qapp()
    from ai_made_easy.ui.features.runconsole import TrainingPage
    from ai_made_easy.ui.stores import RunStore

    page = TrainingPage(RunStore())
    assert not page.live_btn.isEnabled()
    page.set_results_available("/nonexistent_dir")
    assert not page.live_btn.isEnabled()


def test_context_wiring_covers_live():
    code = """
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from ai_made_easy.ui.app import _ensure_qt_plugin_path
_ensure_qt_plugin_path()
from PySide6 import QtWidgets
app = QtWidgets.QApplication([])
from ai_made_easy.ui.context import AppContext
ctx = AppContext()
calls = []
ctx._open_live_predict = lambda: calls.append("live")
ctx.training_page.live_clicked.emit()
assert calls == ["live"], f"live button unwired: {calls}"
from ai_made_easy.ui.context import ValidationIssue
issues = ctx._dataset_health_issues(ctx.graph_service.snapshot())
assert issues == []  # demo dataset is not an image folder
print("W2 WIRING OK")
"""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=Path(__file__).parent.parent, env=env,
                         timeout=120)
    assert "W2 WIRING OK" in out.stdout, (
        f"w2 wiring failed:\n{out.stdout}\n{out.stderr}")


def test_dataset_health_issues_flow(tmp_path):
    _qapp()
    from ai_made_easy.ui.context import AppContext

    ctx = AppContext()
    g = ctx.graph_service.snapshot()
    g.add_node(NodeInstance("d", "data.image_folder",
                            {"root": str(tmp_path)}))
    (tmp_path / "only_cats").mkdir()
    for i in range(4):
        (tmp_path / "only_cats" / f"c{i}.png").write_bytes(b"x")
    issues = ctx._dataset_health_issues(g)
    assert any("too few" in i.message for i in issues)
    assert all(i.severity == "warning" for i in issues)
