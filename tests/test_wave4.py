"""Wave 4 — Share & shine: web demo export, .aime bundle + swap-test,
animated wires, blocks↔python pane."""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ------------------------------------------------------------- web export

def _tiny_torch_model():
    import torch
    import torch.nn as nn

    torch.manual_seed(3)
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(4 * 4 * 4, 3))


def test_layers_from_torch_shapes():
    import numpy as np

    from ai_made_easy.core.web_export import layers_from_torch

    layers = layers_from_torch(_tiny_torch_model())
    kinds = [layer["type"] for layer in layers]
    assert kinds == ["conv2d", "activation", "maxpool2d", "flatten", "dense"]
    conv = layers[0]
    assert conv["config"]["filters"] == 4
    assert conv["weights"][0]["shape"] == [3, 3, 1, 4]  # HWIO (from OIHW)
    # base64 round-trips to the transposed weight
    w = np.frombuffer(base64.b64decode(conv["weights"][0]["b64"]),
                      dtype="float32").reshape(3, 3, 1, 4)

    expect = _tiny_torch_model()[0].weight.detach().permute(2, 3, 1, 0).numpy()
    assert np.allclose(w, expect)


def test_layers_reject_unsupported():
    import torch.nn as nn

    from ai_made_easy.core.web_export import layers_from_torch

    class Weird(nn.Sequential):
        def __init__(self):
            super().__init__(nn.LSTM(4, 4))

    with pytest.raises(RuntimeError, match="isn't supported"):
        layers_from_torch(Weird())


def test_build_web_demo_structure():
    from ai_made_easy.core.web_export import build_web_demo, layers_from_torch

    html = build_web_demo(layers_from_torch(_tiny_torch_model()),
                          ["a", "b", "c"], [1, 8, 8], title="tiny",
                          tfjs_path=Path(__file__).parent.parent
                          / "ai_made_easy" / "assets" / "tf.min.js")
    assert "<title>tiny — AI Made Easy web demo</title>" in html
    assert "tf.sequential" in html and "SELFTEST" in html
    assert 'const CLASSES = ["a", "b", "c"]' in html
    assert "atob(" in html  # base64 weights decode in-browser
    assert html.count("tf.tensor(") >= 1


def test_web_export_from_trained_workdir(tmp_path):
    """End-to-end: train tiny conv, export demo html from the workdir."""
    from ai_made_easy.core.codegen.training_gen import generate_training
    from ai_made_easy.core.graph import Edge, Graph, NodeInstance

    g = Graph(name="w4web")
    g.add_node(NodeInstance("i", "core.input", {"shape": "1,8,8"}))
    g.add_node(NodeInstance("c", "core.conv2d", {"out_channels": 4}))
    g.add_node(NodeInstance("r", "core.relu"))
    g.add_node(NodeInstance("p", "core.maxpool2d", {"kernel_size": 2}))
    g.add_node(NodeInstance("f", "core.flatten"))
    g.add_node(NodeInstance("d", "core.dense", {"units": 3}))
    g.add_node(NodeInstance("o", "core.output"))
    g.add_node(NodeInstance("tr", "train.trainer",
                            {"epochs": 1, "batch_size": 16}))
    for a, b in (("i", "c"), ("c", "r"), ("r", "p"), ("p", "f"),
                 ("f", "d"), ("d", "o")):
        g.add_edge(Edge(a, "out", b, "in"))
    (tmp_path / "train_run.py").write_text(generate_training(g, "pytorch"))
    r = subprocess.run([sys.executable, "train_run.py"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-400:]

    _qapp()
    from ai_made_easy.core.web_export import build_web_demo, layers_from_torch
    from ai_made_easy.ui.context import _load_trained

    (tmp_path / "w4web_train_pytorch.py").write_text(
        (tmp_path / "train_run.py").read_text())
    model, shape, norm = _load_trained(tmp_path, "W4web")
    layers = layers_from_torch(model)
    html = build_web_demo(layers, ["x", "y", "z"], list(shape),
                          tfjs_path=Path(__file__).parent.parent
                          / "ai_made_easy" / "assets" / "tf.min.js")
    assert "SELFTEST" in html and len(html) > 1_000_000


# ----------------------------------------------------------------- bundle

def test_bundle_roundtrip(tmp_path):
    from ai_made_easy.core.bundle import read_bundle, write_bundle

    # dataset + run artifacts
    ds = tmp_path / "data"
    (ds / "cats").mkdir(parents=True)
    (ds / "cats" / "a.png").write_bytes(b"cat")
    run = tmp_path / "run"
    run.mkdir()
    (run / "net_best.pt").write_bytes(b"weights")
    (run / "net_train_pytorch.py").write_text("# train")
    (run / "predictions.json").write_text("[]")

    out = write_bundle(tmp_path / "test.aime",
                       {"name": "mine", "nodes": [], "edges": []},
                       name="mine", card_md="# card", thumbnail_png=b"png",
                       dataset_dir=ds, workdir=run)
    data = read_bundle(out)
    assert data["manifest"]["format"] == "aime"
    assert data["manifest"]["name"] == "mine"
    assert data["graph"]["name"] == "mine"
    assert data["card"].startswith("# card")
    assert data["has_dataset"] and data["has_run"]
    assert (data["run_dir"] / "checkpoint.pt").exists()
    assert (data["dataset_dir"] / "cats" / "a.png").exists()
    assert data["manifest"]["entries"]["dataset_files"] == 1


def test_bundle_rejects_garbage(tmp_path):
    from ai_made_easy.core.bundle import read_bundle

    bad = tmp_path / "bad.aime"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("random.txt", "nope")
    with pytest.raises(ValueError):
        read_bundle(bad)


# -------------------------------------------------- wires + side code pane

def test_wire_flow_animator_toggles():
    _qapp()
    from ai_made_easy.ui.canvas import painter

    painter.install_flat_node_style()
    anim = painter.WireFlowAnimator()
    assert not painter._flow["on"]
    anim.set_running(True)
    assert painter._flow["on"]
    anim.set_running(False)
    assert not painter._flow["on"]
