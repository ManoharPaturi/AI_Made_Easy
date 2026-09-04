"""Wave 3 — Pedagogy: predict gate + surprise, PRIMM missions + quizzes,
bias arc + fairness shortcut check, model-locked chip."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ------------------------------------------------------------- gate logic

def test_score_band_and_surprise():
    from ai_made_easy.ui.features.predict_gate import (
        is_surprising, score_band)

    assert score_band(0.97) == "great"
    assert score_band(0.72) == "okay"
    assert score_band(0.31) == "terrible"
    assert score_band(97) == "great"      # percentage form
    assert score_band(None) is None

    assert is_surprising("great", 0.31)          # two bands off → surprise
    assert is_surprising("terrible", 0.99)
    assert not is_surprising("great", 0.70)      # one band off → fine
    assert not is_surprising(None, 0.31)
    assert not is_surprising("okay", 0.31)


def test_gate_dialog_records_guess():
    _qapp()
    from PySide6 import QtWidgets

    from ai_made_easy.ui.features.predict_gate import PredictGateDialog

    dlg = PredictGateDialog(None, "cats_vs_dogs")
    buttons = dlg.findChildren(QtWidgets.QPushButton)
    assert len(buttons) == 4  # 3 guesses + skip
    dlg._pick("okay")
    assert dlg.guess == "okay"


def test_surprise_dialog_builds():
    _qapp()
    from PySide6 import QtWidgets

    from ai_made_easy.ui.features.predict_gate import SurpriseDialog

    dlg = SurpriseDialog(None, "great", 0.42)
    assert any("great" in l.text() for l in dlg.findChildren(QtWidgets.QLabel))


# ------------------------------------------------------ missions PRIMM + quiz

def test_quiz_dialog_scoring():
    _qapp()
    from ai_made_easy.ui.features.missions import MISSIONS, QuizDialog

    dlg = QuizDialog(None, MISSIONS[2])  # bias mission
    dlg._answers[0] = 1  # correct
    dlg._check()
    assert "1/3" in dlg.feedback.text()
    dlg._answers[1] = 1
    dlg._answers[2] = 0
    dlg._check()
    assert "3/3" in dlg.feedback.text()
    assert not dlg._check_btn.isEnabled()  # mastered — locked


# ------------------------------------------------------------ bias arc

def test_bias_samples_validate_and_generate():
    from ai_made_easy.core.codegen.training_gen import generate_training
    from ai_made_easy.core.graph import Graph

    root = Path(__file__).parent.parent / "samples"
    for name in ("bias_arc_biased", "bias_arc_fair"):
        g = Graph.from_dict(json.loads((root / f"{name}.json").read_text()))
        assert not [i for i in g.validate() if i.severity == "error"], name
        code = generate_training(g, "pytorch")
        assert "root = " in code and "bias_arc" in code


def test_bias_dataset_health_detects_shortcut():
    from ai_made_easy.core.dataset_health import scan_image_folder

    root = Path(__file__).parent.parent / "samples" / "bias_arc"
    biased = scan_image_folder(root / "biased")
    fair = scan_image_folder(root / "fair")
    assert any("background" in f.message for f in biased.warnings)
    assert not any("background" in f.message for f in fair.warnings)


def test_background_shortcut_pure():
    from ai_made_easy.core.dataset_health import background_shortcut

    dark = [(40, 40, 40)] * 10
    light = [(220, 220, 220)] * 10
    mixed = dark[:5] + light[5:]
    assert background_shortcut({"a": dark, "b": light}) is not None
    assert background_shortcut({"a": mixed, "b": mixed}) is None
    assert background_shortcut({"a": dark}) is None  # needs 2 classes


# --------------------------------------------- locked chip + pedagogy chain

def test_locked_chip_lifecycle():
    _qapp()
    from ai_made_easy.ui.features.runconsole import TrainingPage
    from ai_made_easy.ui.stores import RunStore

    page = TrainingPage(RunStore())
    assert page.locked_chip.text() == ""
    page.reset()
    assert page.locked_chip.text() == ""


def test_last_score():
    _qapp()
    from ai_made_easy.ui.features.runconsole import TrainingPage
    from ai_made_easy.ui.stores import RunStore

    page = TrainingPage(RunStore())
    assert page.last_score() is None
    page.on_epoch({"epoch": 1, "total": 2,
                   "metrics": {"val_acc": 0.4, "train_loss": 1.2}})
    page.on_epoch({"epoch": 2, "total": 2,
                   "metrics": {"val_acc": 0.9, "train_loss": 0.3}})
    assert page.last_score() == 0.9


def test_train_pedagogy_chain():
    """Gate → run → surprise → PRIMM events → quiz offer, all headless."""
    code = """
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from ai_made_easy.ui.app import _ensure_qt_plugin_path
_ensure_qt_plugin_path()
from PySide6 import QtCore, QtWidgets
QtCore.QCoreApplication.setOrganizationName("aime-tests")
QtCore.QCoreApplication.setApplicationName("smoke")
app = QtWidgets.QApplication([])
QtCore.QSettings().setValue("aime/pedagogy/predict_gate", True)
from ai_made_easy.ui.context import AppContext, RunStore
import ai_made_easy.ui.features.predict_gate as pg
import ai_made_easy.ui.features.missions as missions_mod

# gate auto-answers "great"; surprise + quiz recorded instead of shown
class _Auto:
    def __init__(self, guess): self.guess = guess
    def exec(self): return 1
pg.PredictGateDialog = lambda parent, name: _Auto("great")
shown = []
pg.SurpriseDialog = lambda parent, g, s: shown.append((g, s)) or _Auto(g)
missions_mod.QuizDialog = lambda parent, m: shown.append(("quiz", m["id"])) \\
    or _Auto(None)

ctx = AppContext()
calls = []
ctx.process_service.run_training = lambda g: calls.append("run")

from ai_made_easy.ui.features.missions import MISSIONS
ctx.palette.missions._select(MISSIONS[0])   # start a mission
ctx.act_train()
assert calls == ["run"], calls

# epochs arrive during the run — after reset(), before the finish
ctx.training_page.on_epoch({"epoch": 1, "total": 1,
                            "metrics": {"val_acc": 0.31}})
import pathlib, tempfile
wd = pathlib.Path(tempfile.mkdtemp(prefix="aime_w3chain_"))
(wd / "predictions.json").write_text("[]")
(wd / "net_best.pt").write_text("x")
ctx.process_service.last_workdir = str(wd)
ctx.run_store.set(RunStore.FINISHED, "train")
ctx._on_run_finished(0, "train")
kinds = [s[0] if isinstance(s[0], str) else s for s in shown]
assert any(k == "great" for k in kinds), shown      # surprise fired (0.31!)
ev = ctx.palette.missions._events                   # predict+run recorded
assert {"predict", "run"} <= ev, ev
assert ctx.training_page.locked_chip.text().startswith("🔒")
print("PEDAGOGY OK")
"""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=Path(__file__).parent.parent,
                         env=env, timeout=120)
    assert "PEDAGOGY OK" in out.stdout, f"{out.stdout}\n{out.stderr}"
