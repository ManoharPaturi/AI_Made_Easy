"""3.6 Phase 2: which settings mattered in a sweep, and design changes between runs."""
from __future__ import annotations

import random

import pytest

from ai_made_easy.core.sweep_analysis import analyze
from ai_made_easy.core.sweeps import SweepRecord, SweepStore


def _sweep(n: int = 24, seed: int = 0, direction: str = "min") -> SweepRecord:
    """val_loss depends strongly on lr (best near 1e-2), a little on the optimizer and not at
    all on the batch size."""
    import math

    rng = random.Random(seed)
    trials = []
    for i in range(n):
        lr = 10 ** rng.uniform(-4, -1)
        opt = rng.choice(["adam", "sgd"])
        batch = rng.choice([16, 32, 64])
        loss = (math.log10(lr) + 2) ** 2 + (1.0 if opt == "sgd" else 0) + rng.gauss(0, 0.02)
        trials.append({"number": i, "values": {"opt.lr": lr, "opt.kind": opt,
                                                "trainer.batch_size": batch},
                       "run_id": f"r{i}", "state": "finished", "score": loss, "message": ""})
    trials.append({"number": n, "values": {"opt.lr": 0.5}, "run_id": "", "state": "failed",
                   "score": None, "message": "diverged"})
    best = min((t for t in trials if t["score"] is not None), key=lambda t: t["score"])
    return SweepRecord("sw-1", "lr sweep", graph={}, trials=trials,
                       spec={"metric": "val_loss", "direction": direction, "max_trials": n + 1,
                             "strategy": "random"},
                       best={"number": best["number"], "run_id": best["run_id"],
                             "score": best["score"], "values": best["values"]})


def test_the_learning_rate_matters_most_and_the_batch_size_least():
    got = analyze(_sweep())
    assert got["trials"] == 24 and got["direction"] == "min"          # the failed one left out
    order = [p["key"] for p in got["importance"]]
    assert order[0] == "opt.lr" and order[-1] == "trainer.batch_size"
    assert sum(p["importance"] for p in got["importance"]) == pytest.approx(1.0)
    assert got["importance"][0]["importance"] > 0.6
    kind = next(p for p in got["importance"] if p["key"] == "opt.kind")
    assert kind["trend"] == "best with adam"
    assert got["importance"][0]["trend"].startswith("best between")   # the optimum is inside
    axes = {a["key"]: a for a in got["axes"]}
    assert axes["opt.lr"]["kind"] == "number" and axes["opt.lr"]["log"]
    assert axes["opt.kind"]["kind"] == "choice" and set(axes["opt.kind"]["choices"]) == {"adam",
                                                                                          "sgd"}
    assert axes["opt.lr"]["label"] == "lr (opt)"
    assert sum(r["best"] for r in got["rows"]) == 1
    assert got["note"] == ""


def test_a_monotonic_effect_says_which_way_is_better():
    trials = [{"number": i, "values": {"m.units": u}, "state": "finished",
               "score": 0.5 + u / 1000, "run_id": ""} for i, u in enumerate([8, 16, 32, 64, 128,
                                                                               256])]
    got = analyze({"spec": {"metric": "accuracy", "direction": "max"}, "trials": trials,
                   "best": {"number": 5}})
    assert got["importance"][0]["trend"] == "higher values scored better"
    assert "rough guide" in got["note"]                                # fewer than 10 trials
    lower = analyze({"spec": {"metric": "val_loss"}, "trials": trials, "best": None})
    assert lower["direction"] == "min"                                 # inferred from the name
    assert lower["importance"][0]["trend"] == "lower values scored better"


def test_few_trials_give_the_chart_but_no_importance():
    got = analyze({"spec": {"metric": "accuracy"}, "trials": _sweep(3).trials[:3], "best": None})
    assert len(got["rows"]) == 3 and got["importance"] == []
    assert "at least 4 finished trials" in got["note"]
    empty = analyze({"spec": {"metric": "accuracy"}, "trials": [], "best": None})
    assert empty["rows"] == [] and empty["score_axis"] is None


def test_the_server_serves_the_analysis(isolated_home):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    SweepStore().save(_sweep())
    client = TestClient(create_app())
    got = client.get("/api/sweeps/sw-1/analysis")
    assert got.status_code == 200 and got.json()["importance"][0]["key"] == "opt.lr"
    assert client.get("/api/sweeps/nope/analysis").status_code == 404


def test_compare_reports_design_changes_against_the_first_run():
    from conftest import tiny_classifier_dict

    from ai_made_easy.core.runs.history import RunRecord, compare

    a = tiny_classifier_dict()
    b = tiny_classifier_dict(units=32)
    got = compare([RunRecord("r1", "a", graph=a), RunRecord("r2", "b", graph=b),
                   RunRecord("r3", "c", graph=a), RunRecord("r4", "d")])
    first, second, same, unsaved = got["design"]
    assert first is None and unsaved is None
    assert second["changed"] == [{"id": "d1", "type": "core.dense", "param": "units",
                                  "from": 8, "to": 32}]
    assert same["same"] and "1 parameter change" in second["summary"]


def test_the_desktop_sweeps_tab_shows_which_settings_mattered(isolated_home):
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    context = AppContext()
    win = Workbench(context)
    context._boot()
    app.processEvents()
    try:
        page = context.experiments_page
        page.set_sweeps([_sweep()])
        page.sweeps_table.selectRow(0)
        assert not page.insights.isHidden()
        assert page.importance_label.text().splitlines()[0].startswith("lr (opt)")
        assert len(page.parallel_plot.listDataItems()) == 24
        page.set_sweeps([SweepRecord("s", "plain", graph={}, spec={"metric": "accuracy"})])
        page.sweeps_table.selectRow(0)
        assert page.insights.isHidden()
    finally:
        context.experiment_service.shutdown()
        context.project_store.mark_clean()
        win.close()
        win.deleteLater()


def test_automl_trials_with_different_settings_keep_their_lines():
    trials = [{"number": 0, "values": {"recipe": "forest", "max_depth": 4}, "state": "finished",
               "score": 0.8, "run_id": ""},
              {"number": 1, "values": {"recipe": "mlp", "lr": 0.01}, "state": "finished",
               "score": 0.7, "run_id": ""}]
    got = analyze({"spec": {"metric": "accuracy"}, "trials": trials, "best": {"number": 0}})
    axes = {a["key"]: a for a in got["axes"]}
    assert axes["recipe"]["choices"] == ["forest", "mlp"]
    from ai_made_easy.core.sweep_analysis import _numeric_trend

    assert _numeric_trend({0: [1.0, 1.1], 1: [0.2], 2: [1.0]}, {0: [4, 4], 1: [16], 2: [64]},
                          "min") == "best at 16"
    assert axes["max_depth"]["kind"] == "number" and axes["lr"]["kind"] == "number"


# ------------------------------------------------------------------ 3.7: interactions

def _interacting(n: int = 40, seed: int = 0) -> dict:
    """The score depends on lr × dropout together (an XOR-like pattern: either both high
    or both low is good), a little on width on its own, and not on the seed."""
    rng = random.Random(seed)
    trials = []
    for i in range(n):
        lr, drop = rng.uniform(0, 1), rng.uniform(0, 1)
        width, seed_v = rng.choice([32, 64, 128]), rng.randint(0, 9)
        score = (1.0 if (lr > 0.5) == (drop > 0.5) else 0.0) + 0.002 * width
        trials.append({"number": i, "values": {"opt.lr": lr, "drop.p": drop,
                                                "dense.units": width, "trainer.seed": seed_v},
                       "state": "finished", "score": score, "run_id": f"r{i}"})
    dims = [{"node": "opt", "param": "lr", "kind": "float", "low": 0.0, "high": 1.0},
            {"node": "drop", "param": "p", "kind": "float", "low": 0.0, "high": 1.0},
            {"node": "dense", "param": "units", "kind": "choice", "values": [32, 64, 128]},
            {"node": "trainer", "param": "seed", "kind": "int", "low": 0, "high": 9}]
    return {"spec": {"metric": "accuracy", "direction": "max", "dimensions": dims,
                     "strategy": "random", "max_trials": n},
            "graph": {"name": "g"}, "trials": trials}


def test_the_forest_finds_a_pair_that_only_matters_together():
    pytest.importorskip("sklearn")
    got = analyze(_interacting())
    assert got["method"] == "forest"
    top = got["interactions"][0]
    assert set(top["keys"]) == {"opt.lr", "drop.p"} and top["share"] > 0.2
    others = [p["share"] for p in got["interactions"][1:]]
    assert all(s < top["share"] / 2 for s in others)
    seed = next(p for p in got["importance"] if p["key"] == "trainer.seed")
    assert seed["importance"] < 0.1
    h = got["heatmap"]
    assert {h["x_key"], h["y_key"]} == {"opt.lr", "drop.p"}
    assert len(h["z"]) == len(h["y"]) and all(len(r) == len(h["x"]) for r in h["z"])
    corners = [h["z"][0][0], h["z"][-1][-1], h["z"][0][-1], h["z"][-1][0]]
    assert min(corners[:2]) > max(corners[2:])                    # both low / both high win


def test_few_trials_keep_the_simple_reading():
    got = analyze(_sweep(n=8))
    assert got["method"] == "eta2" and got["interactions"] == [] and got["heatmap"] is None


def test_sweeping_again_narrows_to_the_best_trials():
    from ai_made_easy.core.sweep_analysis import narrowed

    rec = _sweep(n=40).to_dict()
    rec["spec"]["dimensions"] = [
        {"node": "opt", "param": "lr", "kind": "float", "low": 1e-4, "high": 1e-1, "log": True},
        {"node": "opt", "param": "kind", "kind": "choice", "values": ["adam", "sgd"]},
        {"node": "trainer", "param": "batch_size", "kind": "choice", "values": [16, 32, 64]}]
    got = narrowed(rec)
    dims = {f"{d['node']}.{d['param']}": d for d in got["spec"]["dimensions"]}
    lr = dims["opt.lr"]
    assert 1e-4 <= lr["low"] < 1e-2 < lr["high"] <= 1e-1          # around the optimum
    assert lr["high"] / lr["low"] < 1e3 / 2                       # narrower than before
    assert dims["opt.kind"]["values"] == ["adam"]                 # sgd costs 1.0
    assert got["graph"] == rec["graph"] and got["changes"]
    assert {c["key"] for c in got["changes"]} >= {"opt.lr", "opt.kind"}
    assert got["spec"]["metric"] == "val_loss"


def test_sweeping_again_needs_trials_and_ranges():
    from ai_made_easy.core.sweep_analysis import narrowed

    with pytest.raises(ValueError, match="at least"):
        narrowed(_sweep(n=2))
    with pytest.raises(ValueError, match="AutoML"):
        narrowed({"spec": {"automl": True}, "trials": []})


def test_the_server_proposes_a_narrowed_sweep(isolated_home):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    rec = _sweep(n=20)
    rec.spec["dimensions"] = [{"node": "opt", "param": "lr", "kind": "float", "low": 1e-4,
                               "high": 1e-1, "log": True}]
    SweepStore().save(rec)
    SweepStore().save(_sweep(n=2).__class__("sw-2", "tiny", graph={}, trials=[],
                                            spec={"metric": "val_loss"}))
    client = TestClient(create_app())
    got = client.get("/api/sweeps/sw-1/again")
    assert got.status_code == 200
    body = got.json()
    assert body["spec"]["dimensions"][0]["low"] > 1e-4 and body["changes"][0]["key"] == "opt.lr"
    bad = client.get("/api/sweeps/sw-2/again")
    assert bad.status_code == 400 and "at least" in bad.json()["detail"]


def test_the_desktop_shows_pairs_and_sweeps_again(isolated_home):
    pytest.importorskip("PySide6")
    pytest.importorskip("sklearn")
    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ai_made_easy.core.sweep_analysis import narrowed
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.features.experiments import SweepDialog
    from ai_made_easy.ui.workbench import Workbench

    context = AppContext()
    win = Workbench(context)
    context._boot()
    app.processEvents()
    try:
        data = _interacting()
        rec = SweepRecord("sw-x", "pairs", spec=data["spec"], graph=data["graph"],
                          trials=data["trials"], state="finished")
        page = context.experiments_page
        page.set_sweeps([rec])
        page.sweeps_table.selectRow(0)
        assert "Together: " in page.importance_label.text()
        assert not page.pair_plot.isHidden() and page.pair_image.image is not None
        assert page.sweep_again_btn.isEnabled() and not page.stop_sweep_btn.isEnabled()

        proposal = narrowed(rec)
        params = [{"key": "opt.lr", "node": "opt", "param": "lr", "block": "Adam",
                   "current": 0.5, "kind": "float", "low": 0.0, "high": 1.0, "log": False},
                  {"key": "dense.units", "node": "dense", "param": "units", "block": "Dense",
                   "current": 64, "kind": "choice", "values": [32, 64, 128]}]
        dialog = SweepDialog(win, params, ["accuracy"])
        dialog.prefill(proposal["spec"], proposal["note"])
        spec = dialog.build_spec()
        want = {f"{d['node']}.{d['param']}": d for d in proposal["spec"]["dimensions"]}
        got = {f"{d['node']}.{d['param']}": d for d in spec["dimensions"]}
        assert set(got) == {"opt.lr", "dense.units"}
        assert got["opt.lr"]["low"] == pytest.approx(want["opt.lr"]["low"])
        assert got["opt.lr"]["high"] == pytest.approx(want["opt.lr"]["high"])
        assert got["dense.units"]["values"] == want["dense.units"]["values"]
        assert spec["metric"] == "accuracy" and spec["direction"] == "max"
        dialog.close()
    finally:
        context.experiment_service.shutdown()
        context.project_store.mark_clean()
        win.close()
        win.deleteLater()
