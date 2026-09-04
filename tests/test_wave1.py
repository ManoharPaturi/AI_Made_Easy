"""Wave 1 — See Inside: Mistake Museum, Grad-CAM inspect, confidence
predictions, Model Report Card."""
from __future__ import annotations

import ast
import json
import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_made_easy.core.graph import Edge, Graph, NodeInstance  # noqa: E402


def _conv_net(extra=()):
    g = Graph(name="wave1")
    g.add_node(NodeInstance("i", "core.input", {"shape": "1,8,8"}))
    g.add_node(NodeInstance("c", "core.conv2d", {"out_channels": 4}))
    g.add_node(NodeInstance("r", "core.relu"))
    g.add_node(NodeInstance("f", "core.flatten"))
    g.add_node(NodeInstance("d", "core.dense", {"units": 3}))
    g.add_node(NodeInstance("o", "core.output"))
    for a, b in (("i", "c"), ("c", "r"), ("r", "f"), ("f", "d"), ("d", "o")):
        g.add_edge(Edge(a, "out", b, "in"))
    for nid, tid, params in extra:
        g.add_node(NodeInstance(nid, tid, params))
    return g


# ------------------------------------------------------------ codegen

def test_training_scripts_dump_predictions_and_mistakes():
    from ai_made_easy.core.codegen.training_gen import generate_training

    for fw in ("pytorch", "keras"):
        code = generate_training(_conv_net(), fw)
        ast.parse(code)
        assert "predictions.json" in code, fw
        assert "mistakes.json" in code, fw


def test_inspect_script_shape():
    from ai_made_easy.core.codegen.training_gen import (
        generate_inspect, generate_training)

    code = generate_inspect(_conv_net())
    ast.parse(code)
    assert "_grad_cam" in code and "retain_grad" in code
    assert "inspect.json" in code and "feats.npy" in code
    # training mode must NOT contain the inspect main
    train = generate_training(_conv_net(), "pytorch")
    assert "_grad_cam" not in train
    # inspect ignores the kfold machinery
    kfold = generate_inspect(_conv_net([("kf", "train.kfold", {"k": 2})]))
    ast.parse(kfold)
    assert "kfold_cross_validate" not in kfold


def test_end_to_end_train_then_inspect(tmp_path):
    from ai_made_easy.core.codegen.training_gen import (
        generate_inspect, generate_training)

    g = _conv_net([("tr", "train.trainer", {"epochs": 1, "batch_size": 16})])
    (tmp_path / "train_run.py").write_text(generate_training(g, "pytorch"))
    r = subprocess.run([sys.executable, "train_run.py"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    preds = json.loads((tmp_path / "predictions.json").read_text())
    assert preds and "probs" in preds[0] and "true" in preds[0]
    assert (tmp_path / "mistakes.json").exists()
    # non-file image datasets get sample PNG dumps for the Mistake Museum
    assert (tmp_path / "samples").is_dir()
    assert any((tmp_path / "samples").glob("*.png"))
    assert preds[0]["file"] and (tmp_path / preds[0]["file"]).exists()

    (tmp_path / "aime_inspect_run.py").write_text(generate_inspect(g))
    r2 = subprocess.run([sys.executable, "aime_inspect_run.py", "0"],
                        cwd=tmp_path, capture_output=True, text=True,
                        timeout=120)
    assert r2.returncode == 0, r2.stderr[-500:]
    result = json.loads((tmp_path / "inspect.json").read_text())
    assert result["top"] and result["top"][0]["prob"] >= 0
    import numpy as np
    cam = np.load(tmp_path / "cam.npy")
    assert cam.shape == (8, 8)
    assert 0.0 <= float(cam.min()) <= float(cam.max()) <= 1.0
    feats = np.load(tmp_path / "feats.npy")
    assert feats.ndim == 2, f"feature grid must be 2-D, got {feats.shape}"
    assert feats.shape[0] == feats.shape[1]
    assert 0.0 <= float(feats.min()) and float(feats.max()) <= 1.0


# --------------------------------------------------------- model card

def test_model_card_fills_from_artifacts(tmp_path):
    from ai_made_easy.core.model_card import build_card

    (tmp_path / "predictions.json").write_text(json.dumps([
        {"index": 0, "true": 0, "probs": [0.9, 0.1], "file": None},
        {"index": 1, "true": 1, "probs": [0.8, 0.2], "file": None},
        {"index": 2, "true": 1, "probs": [0.3, 0.7], "file": None},
    ]))
    (tmp_path / "mistakes.json").write_text(json.dumps([
        {"index": 1, "true": 1, "probs": [0.8, 0.2], "file": None},
        {"index": 4, "true": 1, "probs": [0.6, 0.4], "file": None},
    ]))
    card = build_card("My Model", "Synthetic classification data",
                      {"epochs": 3}, tmp_path,
                      superpower="spots cats", careful="not night photos")
    assert "model_name: My Model" in card
    assert "accuracy: 0.6667" in card
    assert "67%" in card  # accuracy sentence
    assert "**class 0** when the right answer was **class 1** (2×)" in card
    assert "spots cats" in card and "not night photos" in card


def test_model_card_without_artifacts():
    from ai_made_easy.core.model_card import build_card

    card = build_card("Empty", "Synthetic data", {}, None)
    assert "accuracy: null" in card
    assert "unknown — train first" in card  # no museum claim without data


# ------------------------------------------------------------ dialogs

def _qapp():
    pytest.importorskip("PySide6")
    from ai_made_easy.ui.app import _ensure_qt_plugin_path
    _ensure_qt_plugin_path()
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_mistake_museum_dialog_builds(tmp_path):
    _qapp()
    (tmp_path / "mistakes.json").write_text(json.dumps([
        {"index": 3, "true": 2, "probs": [0.1, 0.2, 0.7], "file": None}]))
    (tmp_path / "predictions.json").write_text(json.dumps([
        {"index": i, "true": 0, "probs": [0.6, 0.4], "file": None}
        for i in range(4)]))
    from ai_made_easy.ui.features.mistake_museum import MistakeMuseumDialog

    dlg = MistakeMuseumDialog(None, tmp_path)
    assert dlg.windowTitle().startswith("🔍")


def test_inspect_dialog_renders_artifacts(tmp_path):
    _qapp()
    import numpy as np
    np.save(tmp_path / "input.npy", np.random.rand(1, 8, 8).astype("float32"))
    np.save(tmp_path / "cam.npy", np.random.rand(8, 8).astype("float32"))
    np.save(tmp_path / "feats.npy", np.random.rand(2, 32, 40).astype("float32"))
    (tmp_path / "inspect.json").write_text(json.dumps(
        {"image": None, "single": False,
         "top": [{"class": 1, "prob": 0.9},
                 {"class": 0, "prob": 0.1}]}))
    from ai_made_easy.ui.features.inspect_view import InspectDialog

    dlg = InspectDialog(None, tmp_path)
    assert "class 1 (90%)" in dlg.sentence.text()


def test_training_page_has_insight_buttons():
    _qapp()
    from ai_made_easy.ui.features.runconsole import TrainingPage
    from ai_made_easy.ui.stores import RunStore

    page = TrainingPage(RunStore())
    for btn in (page.museum_btn, page.inspect_btn, page.card_btn):
        assert not btn.isEnabled()
    page.set_results_available("/nonexistent_dir")
    assert not page.museum_btn.isEnabled()


def test_context_wiring_smoke():
    """Regression: _wire() once lost half its body to a misplaced method —
    the GUI buttons went dead while direct calls still worked. Click the
    real buttons on a booted AppContext and assert the services fire."""
    import os
    import subprocess
    import sys as _sys

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
ctx.process_service.run_training = lambda g: calls.append("train")
ctx.training_page.start_btn.click()
assert calls == ["train"], f"start_btn click did not reach run_training: {calls}"

# emit every intent signal and assert the wired service runs
ctx.process_service.stop = lambda: calls.append("stop")
ctx.training_page.stop_clicked.emit()
ctx._open_mistake_museum = lambda: calls.append("museum")
ctx.training_page.museum_clicked.emit()
ctx._start_inspect = lambda *a: calls.append("inspect")
ctx.training_page.inspect_clicked.emit()
ctx._open_report_card = lambda: calls.append("card")
ctx.training_page.card_clicked.emit()
ctx.graph_service.place_block = lambda *a: calls.append("place")
ctx.palette.place_requested.emit("core.dense")
ctx._open_mission = lambda s: calls.append("mission")
ctx.palette.missions.mission_selected.emit("x.json")
assert calls == ["train", "stop", "museum", "inspect", "card",
                 "place", "mission"], f"unwired intents: {calls}"
print("WIRING OK")
"""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    out = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True,
        cwd=Path(__file__).parent.parent, env=env, timeout=120)
    assert "WIRING OK" in out.stdout, (
        f"wiring smoke failed:\n{out.stdout}\n{out.stderr}")
