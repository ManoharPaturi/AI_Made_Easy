"""3.5: try a finished run — its predictor, loaded once, answers from both apps."""
from __future__ import annotations

import time

import pytest
from conftest import tiny_classifier_dict

pytest.importorskip("torch")

from ai_made_easy.core import api  # noqa: E402
from ai_made_easy.core.graph import Graph  # noqa: E402
from ai_made_easy.core.runner.manager import RunManager  # noqa: E402
from ai_made_easy.core.runs.history import RunHistory  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    mgr = RunManager(RunHistory(tmp_path_factory.mktemp("try") / "runs"))
    run_id = mgr.start(Graph.from_dict(tiny_classifier_dict(epochs=1)), "auto")
    assert mgr.wait(run_id, 600)["state"] == "finished", mgr.history.get(run_id).error
    return mgr, run_id


@pytest.fixture()
def tried(trained):
    mgr, run_id = trained
    api.set_manager(mgr)
    yield run_id
    api.try_service().close()


def test_a_finished_run_describes_its_form(tried):
    info = api.try_describe(tried)
    assert info["input_kind"] == "arrays" and info["task"] == "multiclass"
    assert info["signature"]["input_shape"] == [16]
    assert info["example"] == [[0.0] * 16]


def test_predictions_come_from_the_runs_own_model(tried):
    first = api.try_run(tried, [[0.5] * 16, [-1.0] * 16])
    assert len(first["predictions"]) == 2
    pred = first["predictions"][0]
    assert set(pred) == {"label", "class_index", "confidence", "probabilities"}
    assert abs(sum(pred["probabilities"].values()) - 1) < 1e-4
    started = time.perf_counter()   # the loaded predictor answers again, without reloading
    again = api.try_run(tried, [[0.5] * 16])
    assert again["predictions"][0] == pred and time.perf_counter() - started < 5


def test_bad_inputs_are_reported_and_the_predictor_keeps_going(tried):
    with pytest.raises(api.ApiError, match="expects"):
        api.try_run(tried, [[1.0] * 3])
    with pytest.raises(api.ApiError, match="non-empty"):
        api.try_run(tried, [])
    assert len(api.try_run(tried, [[0.0] * 16])["predictions"]) == 1


def test_only_known_finished_runs_can_be_tried(tried):
    with pytest.raises(Exception):  # noqa: B017 — an unknown run id
        api.try_describe("no-such-run")
    mgr = api.manager()
    rec = mgr.history.create(tiny_classifier_dict())   # created, never trained
    with pytest.raises(api.ApiError, match="only finished runs"):
        api.try_describe(rec.run_id)


def test_the_server_tries_a_run(tried):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    api.set_manager(api.manager())
    client = TestClient(create_app())
    form = client.get(f"/api/runs/{tried}/try")
    assert form.status_code == 200 and form.json()["input_kind"] == "arrays"
    ok = client.post(f"/api/runs/{tried}/try", json={"inputs": [[0.2] * 16]})
    assert ok.status_code == 200 and ok.json()["predictions"][0]["label"]
    bad = client.post(f"/api/runs/{tried}/try", json={"inputs": [[1.0]]})
    assert bad.status_code in (400, 422) and "expects" in bad.text


def test_closing_stops_the_predictors(tried):
    api.try_describe(tried)
    workers = list(api.try_service()._workers.values())
    assert workers and all(w.alive() for w in workers)
    api.try_service().close()
    assert not any(w.alive() for w in workers)


def test_desktop_try_dialog_predicts(tried, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.ui.app import _ensure_qt_plugin_path
    from ai_made_easy.ui.features.try_dialog import TryDialog, describe_prediction

    _ensure_qt_plugin_path()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = TryDialog(None, api.manager().history, tried)

    def wait(check, seconds=60.0):
        deadline = time.time() + seconds
        while not check() and time.time() < deadline:
            app.processEvents()
            QtCore.QThread.msleep(20)
        assert check()

    wait(lambda: dialog.predict_btn.isEnabled())
    assert dialog.form_info["input_kind"] == "arrays"
    dialog.json.setPlainText("[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, "
                             "0.5, 0.5, 0.5, 0.5]]")
    dialog.predict()
    wait(lambda: "% sure" in dialog.result.toPlainText())
    dialog.json.setPlainText("[[1, 2]]")
    dialog.predict()
    wait(lambda: dialog.result.toPlainText().startswith("⚠"))
    assert "expects" in dialog.result.toPlainText()
    dialog.deleteLater()
    text = describe_prediction({"label": "b", "confidence": 0.9,
                                "probabilities": {"a": 0.1, "b": 0.9}})
    assert text.splitlines()[0].startswith("b") and "90.0%" in text
