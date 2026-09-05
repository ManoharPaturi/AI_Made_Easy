"""Elite chrome: ⌘K command palette, toasts, training progress, unified header."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_fuzzy_score_ranks_and_rejects():
    from ai_made_easy.ui.features.command_palette import fuzzy_score

    assert fuzzy_score("anything", "") == 0            # empty = browse all
    assert fuzzy_score("abc", "xyz") is None           # no match
    assert fuzzy_score("▶ Train the model", "trn") is not None
    # substring beats scattered subsequence
    assert fuzzy_score("mnist_cnn", "mnist") > fuzzy_score("mnist_cnn", "mst")
    # prefix bonus
    assert fuzzy_score("train acc", "tra") > fuzzy_score("retrain", "tra")


def test_palette_lists_actions_and_blocks():
    _app()
    from PySide6 import QtGui
    from ai_made_easy.ui.features.command_palette import CommandPalette

    act = QtGui.QAction("▶ Train the model")
    palette = CommandPalette({"run.train": act})
    palette._refill("train")
    texts = [palette.list.item(i).text()
             for i in range(palette.list.count())]
    assert any("Train the model" in t for t in texts)
    assert any("Trainer" in t for t in texts)          # blocks searched too
    assert any(t == "⚡ Actions" for t in texts)        # section headers


def test_palette_runs_block_row():
    _app()
    from PySide6 import QtCore
    from ai_made_easy.ui.features.command_palette import CommandPalette

    palette = CommandPalette({})
    palette._refill("dense")
    fired = []
    palette.place_requested.connect(fired.append)
    for i in range(palette.list.count()):
        item = palette.list.item(i)
        if item.data(QtCore.Qt.ItemDataRole.UserRole) == "block":
            palette._run_item(item)
            break
    else:
        pytest.fail("no block row matched 'dense'")
    assert fired and isinstance(fired[0], str)


def test_toast_layer_caps_and_fades():
    _app()
    from PySide6 import QtWidgets
    from ai_made_easy.ui.features.toasts import ToastLayer

    host = QtWidgets.QWidget()
    host.resize(800, 600)
    layer = ToastLayer(host)
    for i in range(5):                                 # over the cap of 3
        layer.toast(f"toast {i}")
    pills = layer.findChildren(QtWidgets.QLabel)
    assert len(pills) <= 3


def test_training_progress_lifecycle():
    _app()
    from ai_made_easy.ui.features.runconsole import TrainingPage
    from ai_made_easy.ui.stores import RunStore

    page = TrainingPage(RunStore())
    assert page.progress.isHidden()
    page._on_state(RunStore.RUNNING, "train")
    assert not page.progress.isHidden()
    assert page.progress.maximum() == 0                # busy until epochs
    page.on_epoch({"epoch": 3, "total": 8,
                   "metrics": {"train_loss": 0.4, "train_acc": 0.9}})
    assert page.progress.maximum() == 8
    assert page.progress.value() == 3
    page._on_state(RunStore.FINISHED, "train")
    assert page.progress.value() == page.progress.maximum()
    page.reset()
    assert page.progress.isHidden()


def test_workbench_wires_elite_chrome():
    """Full boot: unified titlebar, palette with core actions, toast layer."""
    import subprocess as sp
    code = """
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from ai_made_easy.ui.app import _ensure_qt_plugin_path
_ensure_qt_plugin_path()
from PySide6 import QtCore, QtWidgets
QtCore.QCoreApplication.setOrganizationName("aime-tests")
QtCore.QCoreApplication.setApplicationName("smoke")
app = QtWidgets.QApplication([])
from ai_made_easy.ui.context import AppContext
from ai_made_easy.ui.workbench import Workbench
ctx = AppContext()
window = Workbench(ctx)
window.show()
app.processEvents()
assert window.unifiedTitleAndToolBarOnMac() is True
labels = [a.text() for a in window.command_palette._actions.values()]
assert any("Train the model" in t for t in labels)
window.command_palette._refill("valid")
texts = [window.command_palette.list.item(i).text()
         for i in range(window.command_palette.list.count())]
assert any("Validate the graph" in t for t in texts), texts
window.toasts.toast("hello")
app.processEvents()
assert window.toasts.findChildren(QtWidgets.QLabel)
assert ctx.header.cmdk_btn.text().startswith("⌘K")
print("ELITE OK")
"""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    out = sp.run([sys.executable, "-c", code], capture_output=True, text=True,
                 cwd=Path(__file__).parent.parent, env=env, timeout=120)
    assert "ELITE OK" in out.stdout, f"{out.stdout}\n{out.stderr}"
