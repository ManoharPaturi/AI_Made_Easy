"""Structural tests for the rebuilt UI architecture.

These enforce the ENCLOSING rules the redesign committed to:
1. OdenGraphQt may only be imported inside ui/canvas/ (the adapter boundary).
2. Every ActionSpec slot resolves on AppContext (actions-as-data integrity).
3. Stores do granular signal round-trips (ProjectStore stale-name regression).
4. The Workbench smoke: phases build, docks/menus/actions exist, renders.
5. MainWindow-era god-object cannot quietly return: shell stays layout-only.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

UI_DIR = Path(__file__).parent.parent / "ai_made_easy" / "ui"


# ---------------------------------------------------------- boundary lint

def test_odengraphqt_only_imported_inside_canvas_package():
    """The 'swappable canvas' claim must hold: enforced, not aspirational."""
    offenders = []
    for path in sorted(UI_DIR.rglob("*.py")):
        rel = path.relative_to(UI_DIR)
        if str(rel).startswith("canvas"):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "OdenGraphQt" or n.startswith("OdenGraphQt.")
                   for n in names):
                offenders.append(str(rel))
    assert not offenders, f"OdenGraphQt imported outside ui/canvas/: {offenders}"


def test_core_stays_qt_free():
    """Ryven rule: the domain layer never imports Qt."""
    core_dir = UI_DIR.parent / "core"
    offenders = []
    for path in sorted(core_dir.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] in ("PySide6", "PyQt6", "OdenGraphQt",
                                       "pyqtgraph") for n in names):
                offenders.append(str(path.relative_to(core_dir)))
    assert not offenders, f"Qt imported inside core/: {offenders}"


# ------------------------------------------------------ actions catalog

def test_every_catalog_slot_resolves_on_context():
    pytest.importorskip("PySide6")
    from ai_made_easy.ui.actions_catalog import CATALOG, MENU_ORDER

    # slots resolve without building Qt objects (name-level check)
    from ai_made_easy.ui import context as ctx_mod

    missing = [s.slot for s in CATALOG
               if not hasattr(ctx_mod.AppContext, s.slot)]
    assert not missing, f"catalog slots missing on AppContext: {missing}"
    assert [s.menu for s in CATALOG if s.menu] and set(
        s.menu for s in CATALOG) <= set(MENU_ORDER)
    ids = [s.id for s in CATALOG]
    assert len(ids) == len(set(ids)), "duplicate action ids"


# ---------------------------------------------------------------- stores

def test_project_store_name_roundtrip_no_stale_state():
    """The stale-name bug class: any write path updates every subscriber."""
    pytest.importorskip("PySide6")
    from ai_made_easy.ui.stores import ProjectStore

    store = ProjectStore()
    seen = []
    store.name_changed.connect(lambda n: seen.append(n))

    store.set_name("first")
    store.reset("second")           # new-project path
    store.set_name("third")         # open/assistant path
    assert seen == ["first", "second", "third"]
    assert store.name == "third"

    store.mark_dirty()
    assert store.dirty is True
    store.mark_clean()
    assert store.dirty is False


def test_run_store_single_writer_state_machine():
    pytest.importorskip("PySide6")
    from ai_made_easy.ui.stores import RunStore

    store = RunStore()
    transitions = []
    store.state_changed.connect(lambda s, k: transitions.append((s, k)))
    assert store.is_running is False
    store.set(RunStore.RUNNING, "train")
    assert store.is_running is True
    store.set(RunStore.FINISHED, "train")
    assert transitions == [("running", "train"), ("finished", "train")]


# ------------------------------------------------------- workbench smoke

WORKBENCH_SMOKE = """
import sys
from ai_made_easy.ui.app import _ensure_qt_plugin_path
_ensure_qt_plugin_path()
from PySide6 import QtWidgets, QtCore
app = QtWidgets.QApplication([])
from ai_made_easy.ui.theme import ThemeService
ThemeService().apply(app, "dark")
from ai_made_easy.ui.context import AppContext
from ai_made_easy.ui.workbench import Workbench
ctx = AppContext()
win = Workbench(ctx)
win.show()
def check():
    panels = [win.findChild(QtWidgets.QFrame, n)
              for n in ("panel.blocks", "panel.canvas",
                        "panel.inspector", "panel.runconsole")]
    assert all(p is not None for p in panels), panels
    assert not win.findChildren(QtWidgets.QDockWidget), "docks are gone"
    menus = [m.text() for m in win.menuBar().actions() if m.text()]
    assert menus == ["&File", "&Edit", "&View", "&Help"], menus
    assert len(win.actions) >= 16
    assert ctx.validation_store.valid or ctx.validation_store.issues is not None
    ok = win.grab().save("/tmp/aime_workbench_smoke.png")
    assert ok
    print("SMOKE-OK", flush=True)
    app.quit()
QtCore.QTimer.singleShot(1500, check)
QtCore.QTimer.singleShot(10000, app.quit)
app.exec()
"""


def test_workbench_smoke():
    """Isolated subprocess: full context + phased shell build + render."""
    pytest.importorskip("PySide6")
    result = subprocess.run(
        [sys.executable, "-c", WORKBENCH_SMOKE],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True, timeout=180,
        env={"PYTHONPATH": str(Path(__file__).parent.parent),
             "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert "SMOKE-OK" in result.stdout, result.stdout[-800:] + result.stderr[-800:]
    assert Path("/tmp/aime_workbench_smoke.png").stat().st_size > 50_000


def test_workbench_is_layout_only():
    """The shell must not grow business logic back (god-object guard)."""
    source = (UI_DIR / "workbench.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Workbench":
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
    assert methods, "Workbench class not found"
    banned = ("action_train", "run_training", "_live_validate", "export_training",
              "_write_project", "generate_code", "_on_epoch")
    assert not [m for m in methods if m in banned], methods
    assert len(methods) <= 15, f"Workbench grew: {len(methods)} methods: {methods}"
