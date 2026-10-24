"""3.9 in the desktop app: Resume on stopped runs, Train More on boosted models."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from conftest import tiny_classifier_dict  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    QtCore.QCoreApplication.setApplicationName("v39-tests")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def ctx(app, isolated_home):
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    context = AppContext()
    win = Workbench(context)
    context._boot()
    app.processEvents()
    yield context
    context.experiment_service.shutdown()
    context.project_store.mark_clean()
    win.close()
    win.deleteLater()


def _boosted_graph() -> dict:
    graph = {"name": "boosted", "meta": {}, "edges": [], "nodes": [
        {"id": "data", "type": "data.csv", "params": {"path": "t.csv", "target_column": "y"},
         "position": [0, 0]},
        {"id": "model", "type": "ml.xgb_classifier", "params": {}, "position": [0, 0]}]}
    return graph


def _run(history, graph: dict, framework: str, status: str, project: str) -> str:
    rec = history.create(graph, framework=framework, project=project)
    history.finalize(rec.run_id, status, 0 if status == "finished" else 1)
    return rec.run_id


def test_resume_and_train_more_follow_the_selected_run(ctx, app):
    history, project = ctx.experiment_service.history, ctx.project_store.name
    stopped = _run(history, tiny_classifier_dict(), "pytorch", "stopped", project)
    boosted = _run(history, _boosted_graph(), "sklearn", "finished", project)
    forest = _run(history, {**_boosted_graph(), "nodes": [
        {"id": "model", "type": "ml.random_forest_classifier", "params": {},
         "position": [0, 0]}]}, "sklearn", "finished", project)
    page = ctx.experiments_page
    ctx.act_experiments()
    rows = {page.runs_table.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole): r
            for r in range(page.runs_table.rowCount())}

    def pick(run_id: str) -> None:
        page.runs_table.clearSelection()
        page.runs_table.selectRow(rows[run_id])

    pick(stopped)
    assert page.resume_btn.isEnabled() and not page.more_btn.isEnabled()
    pick(boosted)
    assert page.more_btn.isEnabled() and not page.resume_btn.isEnabled()
    assert page.boosted(boosted) and not page.boosted(forest)
    pick(forest)
    assert not page.more_btn.isEnabled()


def test_resume_explains_why_a_run_cannot(ctx, app, monkeypatch):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        lambda _parent, title, text: shown.append((title, text)))
    run_id = _run(ctx.experiment_service.history, tiny_classifier_dict(), "pytorch",
                  "stopped", ctx.project_store.name)
    ctx._resume_run(run_id)
    assert shown and "Resume" in shown[0][0] and "PyTorch" in shown[0][1]


def test_train_more_dialog_asks_for_trees_for_a_boosted_model(app):
    from ai_made_easy.ui.features.run_tools import TrainMoreDialog

    trees = TrainMoreDialog(None, "r1", trees=True)
    labels = [w.text() for w in trees.findChildren(QtWidgets.QLabel)]
    assert "More trees" in labels and "Learning-rate scale" not in labels
    assert trees.epochs.value() == 100 and trees.epochs.maximum() == 100_000
    epochs = TrainMoreDialog(None, "r2")
    assert "More epochs" in [w.text() for w in epochs.findChildren(QtWidgets.QLabel)]
    trees.close()
    epochs.close()
