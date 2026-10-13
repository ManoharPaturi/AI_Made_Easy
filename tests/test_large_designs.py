"""Large designs stay responsive: validation, shape inference and the desktop canvas scale
linearly. Budgets are ~10x the measured time, but far below the old quadratic / cubic
costs (validating 2000 blocks took over a second; opening 500 on the canvas, 20 s)."""
from __future__ import annotations

import copy
import time

import pytest

from ai_made_easy.core.graph import Edge, Graph, NodeInstance


def chain(n: int) -> Graph:
    g = Graph(name="big")
    g.add_node(NodeInstance("in", "core.input", {"shape": "64", "dtype": "float32"}))
    prev = "in"
    for i in range(n):
        dense = i % 2 == 0
        g.add_node(NodeInstance(f"n{i}", "core.dense" if dense else "core.relu",
                                {"units": 64} if dense else {}))
        g.add_edge(Edge(prev, "out", f"n{i}", "in"))
        prev = f"n{i}"
    g.add_node(NodeInstance("out", "core.output", {}))
    g.add_edge(Edge(prev, "out", "out", "in"))
    return g


def test_validation_scales_linearly():
    g = chain(2000)
    start = time.perf_counter()
    assert g.validate() == []
    shapes = g.infer_shapes()
    assert shapes["out"] == [64]
    assert time.perf_counter() - start < 0.6


def test_edge_index_lives_only_for_one_pass():
    g = chain(4)
    with g._indexed():
        assert [e.source_id for e in g.incoming("n1")] == ["n0"]
        # copies (Quick Fixes edit them) never inherit the index
        assert copy.deepcopy(g)._adjacency is None
    assert g._adjacency is None
    g.edges.append(Edge("n3", "out", "n0", "in"))      # edits after a pass are seen
    assert {e.source_id for e in g.incoming("n0")} == {"in", "n3"}


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def canvas(app, isolated_home):
    from ai_made_easy.ui.canvas.adapter import CanvasController

    controller = CanvasController()
    yield controller
    # if Python garbage-collects the graph while its undo stack holds commands, PySide
    # frees them twice (a segfault at exit): empty the stack first
    controller._graph.undo_stack().clear()


def test_canvas_opens_a_large_design(canvas, app):
    g = chain(500)
    start = time.perf_counter()
    canvas.load_ir(g)
    if canvas.has_overlaps():                    # as GraphService.load does
        canvas.auto_layout(canvas.to_ir(), undoable=False)
    app.processEvents()
    elapsed = time.perf_counter() - start
    assert len(canvas.to_ir().nodes) == 502
    names = [n.name() for n in canvas._graph.all_nodes()]
    assert len(set(names)) == len(names) and "ReLU 1" in names
    assert canvas._graph.undo_stack().count() == 0       # loading is not an edit
    assert elapsed < 15, f"opening 500 blocks took {elapsed:.1f}s"


def test_auto_layout_is_one_undo_step(canvas):
    canvas.load_ir(chain(6))                 # every block at the origin: they overlap
    stack = canvas._graph.undo_stack()
    assert stack.count() == 0 and canvas.has_overlaps()
    canvas.auto_layout(canvas.to_ir())
    assert stack.count() == 1 and not canvas.has_overlaps()
    stack.undo()                             # one Undo puts every block back
    assert canvas.has_overlaps()
