"""Hygiene: undo coverage on the real canvas (property + create paths)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=Path(__file__).parent.parent,
                          env=env, timeout=120)


def test_property_change_and_create_are_undoable():
    """The user-facing promise: Cmd+Z reverts property edits and adds."""
    code = """
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from ai_made_easy.ui.app import _ensure_qt_plugin_path
_ensure_qt_plugin_path()
from PySide6 import QtCore, QtWidgets
QtCore.QCoreApplication.setOrganizationName("aime-tests")
QtCore.QCoreApplication.setApplicationName("smoke")
app = QtWidgets.QApplication([])
QtCore.QSettings().setValue("aime/pedagogy/predict_gate", False)
from ai_made_easy.ui.context import AppContext
ctx = AppContext()
gs = ctx.graph_service

from ai_made_easy.core.graph import Graph as _G

gs.load(_G(name="fresh"))  # empty canvas (act_new re-seeds the demo)
for _ in range(60):
    app.processEvents()
assert len(gs.snapshot().nodes) == 0, "empty load should clear the canvas"

ctx.canvas.place_block("core.dense")
for _ in range(60):
    app.processEvents()
gs.settle_now()
snap = gs.snapshot()
assert len(snap.nodes) == 1, f"expected 1 node, got {len(snap.nodes)}"
nid = next(iter(snap.nodes))

canvas_node = next(n for n in ctx.canvas.node_graph.all_nodes()
                   if n.id == nid)
canvas_node.set_property("units", 99)
for _ in range(60):
    app.processEvents()
gs.settle_now()
assert gs.snapshot().nodes[nid].params["units"] == 99

stack = ctx.canvas.node_graph.undo_stack()
stack.undo()
for _ in range(60):
    app.processEvents()
gs.settle_now()
units_after_undo = gs.snapshot().nodes[nid].params["units"]
assert units_after_undo != 99, f"property undo failed: {units_after_undo}"

stack.undo()  # undo the create
for _ in range(60):
    app.processEvents()
gs.settle_now()
assert len(gs.snapshot().nodes) == 0, "create undo failed"
print("UNDO OK")
"""
    out = _run(code)
    assert "UNDO OK" in out.stdout, f"{out.stdout}\n{out.stderr}"
