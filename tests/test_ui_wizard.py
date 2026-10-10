"""Desktop New from Task wizard, the recipe notes in Properties and Explain This Design."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402
from test_recipes import churn_csv  # noqa: E402
from test_ui import app, ctx  # noqa: E402, F401 — shared fixtures


def _wait(app, condition, timeout: float = 60.0) -> None:  # noqa: F811
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise TimeoutError("the wizard did not finish in time")
        app.processEvents()
        time.sleep(0.02)


def _select_task(dialog, task_id: str) -> None:
    tree = dialog.task_tree
    for i in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(i)
        for j in range(group.childCount()):
            item = group.child(j)
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == task_id:
                tree.setCurrentItem(item)
                return
    raise KeyError(task_id)


def test_wizard_builds_a_design_from_detected_data(app, tmp_path):  # noqa: F811
    from ai_made_easy.ui.features.wizard import NewProjectWizard

    dialog = NewProjectWizard(None)
    assert not dialog.next.isEnabled()                      # pick a task first
    _select_task(dialog, "binary")
    dialog.next.click()
    dialog.use_own.setChecked(True)
    assert not dialog.next.isEnabled()                      # detect the data first
    dialog.path.setText(str(churn_csv(tmp_path)))
    dialog.detect_btn.click()
    _wait(app, lambda: dialog._facts is not None)
    assert "2 classes" in dialog.facts_label.text()
    dialog.next.click()
    dialog.limits["max_params_m"].setValue(5)
    dialog.next.click()
    _wait(app, lambda: bool(dialog._rows))
    ids = [r["recipe"]["id"] for r in dialog._rows]
    assert {"gradient_boosting", "tabular_mlp"} <= set(ids)
    assert dialog.automl.isVisibleTo(dialog) and dialog.automl.isEnabled()
    dialog.recipe_list.setCurrentRow(ids.index("tabular_mlp"))
    dialog._knob_widgets["width"].setValue(32)
    dialog.create.click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    graph = dialog.result_graph
    assert graph["meta"]["recipe"]["knobs"]["width"] == 32
    assert graph["meta"]["budget"]["max_params_m"] == 5
    assert next(n for n in graph["nodes"] if n["id"] == "input")["params"]["shape"] == "5"


def test_wizard_demo_data_and_automl_spec(app):  # noqa: F811
    from ai_made_easy.ui.features.wizard import NewProjectWizard

    dialog = NewProjectWizard(None)
    _select_task(dialog, "multiclass")
    dialog.next.click()
    assert [dialog.modality.itemText(i) for i in range(dialog.modality.count())] == \
        ["tabular", "image"]
    dialog.modality.setCurrentText("image")
    dialog.next.click()
    dialog.next.click()
    _wait(app, lambda: bool(dialog._rows))
    assert {r["recipe"]["modality"] for r in dialog._rows} == {"image"}
    dialog.trials.setValue(4)
    dialog.automl.click()
    assert dialog.automl_spec == {"task": "multiclass", "facts": {}, "modality": "image",
                                  "budget": {}, "max_trials": 4, "epochs": 0}


def test_recipe_notes_in_properties_and_explain(ctx, app, monkeypatch):  # noqa: F811
    from ai_made_easy.core import recipes

    graph = recipes.build("tabular_mlp", "regression").to_dict()
    assert ctx.open_imported(graph)
    app.processEvents()
    ir = ctx.canvas.to_ir()                                   # the canvas renames the blocks
    loss = next(n.instance_id for n in ir.nodes.values() if n.type_id == "train.loss_mse")
    assert "mean squared error" in ctx._why(loss)
    assert set(ir.meta["recipe"]["why"]) <= set(ir.nodes)
    ctx.properties.show_node(loss, ctx.canvas.block_of(loss), {}, [], why=ctx._why(loss))
    labels = [w.text() for w in ctx.properties.findChildren(QtWidgets.QLabel)]
    assert any(t.startswith("Why here: mean squared error") for t in labels)
    shown = []
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: shown.append(
        self.findChild(QtWidgets.QTextBrowser).toPlainText()) or 0)
    ctx.act_explain()
    assert "MLP recipe" in shown[0] and "mean squared error" in shown[0]
