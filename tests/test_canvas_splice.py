"""Canvas splice tests: composite expansion + template save/load through the
REAL canvas controller (Qt application required — skipped headless)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from ai_made_easy.core.codegen import generate  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_canvas(qapp):
    from ai_made_easy.ui.canvas import CanvasController

    return CanvasController()


def test_expand_selected_splices_wires(qapp):
    canvas = _make_canvas(qapp)
    graph = canvas.node_graph
    inp = graph.create_node("aim.data.InputNode", name="Input", pos=(-400.0, 0.0))
    macro = graph.create_node("aim.architectures.MLPNode", name="MLP",
                              pos=(-100.0, 0.0))
    macro.set_property("hidden", "8")
    macro.set_property("num_classes", 3)
    out = graph.create_node("aim.data.OutputNode", name="Output", pos=(900.0, 0.0))
    inp.output_ports()[0].connect_to(macro.input_ports()[0])
    macro.output_ports()[0].connect_to(out.input_ports()[0])

    graph.clear_selection()
    macro.set_selected(True)
    count = canvas.expand_selected()

    assert count == 1
    ir = canvas.to_ir("spliced")
    assert macro.id not in ir.nodes
    assert ir.validate() == []
    code = generate(ir, "pytorch")
    assert "self.dense_" in code and "return v_dense_" in code


def test_save_selection_roundtrip(qapp, tmp_path: Path, monkeypatch):
    from ai_made_easy.ui.canvas import templates as canvas_mod

    monkeypatch.setattr(canvas_mod, "TEMPLATES_DIR", tmp_path / "templates")

    canvas = _make_canvas(qapp)
    canvas.seed_demo()
    ir = canvas.to_ir("demo")
    # pick the conv..gap run as selection
    graph = canvas.node_graph
    first_conv = next(n for n in graph.all_nodes() if n.type_ == "aim.layers.Conv2DNode")
    relu = next(n for n in graph.all_nodes() if n.type_ == "aim.activations.ReLUNode")
    pool = next(n for n in graph.all_nodes() if n.type_ == "aim.layers.MaxPool2DNode")
    for n in graph.all_nodes():
        n.set_selected(False)
    for n in (first_conv, relu, pool):
        n.set_selected(True)
    path = canvas.save_selection_as_template("my block")
    data = json.loads(path.read_text())
    assert data["name"] == "my block"
    assert len(data["nodes"]) == 3
    assert data["entry"] == first_conv.id
    assert data["exit"] == pool.id

    # a fresh canvas picks it up as a Custom composite
    _canvas2 = _make_canvas(qapp)
    from ai_made_easy.core.composites import wrap_fragment
    from ai_made_easy.core.registry import get_registry

    assert get_registry().has("custom.my_block")
    frag = get_registry().get("custom.my_block").builder({})
    wrapped = wrap_fragment(frag, [1, 8, 8])
    assert wrapped.validate() == []
    code = generate(wrapped, "pytorch")
    assert "self.conv2d_" in code


def test_node_view_color_is_flat_rgba(qapp):
    """Regression: set_color(r, g, b) — a nested color tuple breaks paint."""
    canvas = _make_canvas(qapp)
    n = canvas.node_graph.create_node("aim.layers.Conv2DNode", name="Conv",
                                      pos=(0.0, 0.0))
    color = n.view.color
    assert isinstance(color, tuple) and len(color) == 4
    assert all(isinstance(c, int) for c in color), f"nested color: {color!r}"
    assert all(0 <= c <= 255 for c in color)
