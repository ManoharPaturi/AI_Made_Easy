"""3.8 features: find a learning rate, train more, and results by group (core, API,
web server)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import tiny_classifier_dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ai_made_easy.core import api, groups, lr_find  # noqa: E402
from ai_made_easy.core.graph import Graph  # noqa: E402
from ai_made_easy.core.runner.manager import RunManager  # noqa: E402
from ai_made_easy.core.runs.history import RunHistory  # noqa: E402


# ------------------------------------------------------------------ learning-rate finder

def test_suggestion_is_where_the_loss_falls_fastest():
    lrs = np.logspace(-7, 1, 100)
    x = np.log10(lrs)
    loss = 2.0 - 1.5 / (1 + np.exp(-3 * (x + 3)))          # falls fastest at 1e-3
    loss[x > -0.5] += (x[x > -0.5] + 0.5) ** 2 * 4         # then blows up
    out = lr_find.suggest(list(lrs), list(loss))
    assert 3e-4 < out["suggested"] < 3e-3, out
    assert out["min_loss_lr"] > out["suggested"]
    assert out["reason"] == "where the loss falls fastest"


def test_no_suggestion_from_a_handful_of_steps_or_a_flat_curve():
    assert lr_find.suggest([1e-5, 1e-4, 1e-3], [1.0, 1.0, 9.0])["suggested"] is None
    flat = lr_find.suggest(list(np.logspace(-6, 0, 30)), list(np.linspace(1.0, 1.2, 30)))
    assert flat["suggested"] == pytest.approx(1e-7, rel=0.01)   # a tenth of the minimum's rate
    assert "never fell" in flat["reason"]


def test_apply_sets_the_optimizers_learning_rate():
    graph = api.apply_learning_rate(tiny_classifier_dict(), 0.0031623)
    opt = next(n for n in graph["nodes"] if n["type"].startswith("train.") and
               "lr" in n.get("params", {}))
    assert opt["params"]["lr"] == 0.00316
    no_optimizer = {**tiny_classifier_dict(), "nodes": [
        n for n in tiny_classifier_dict()["nodes"] if "lr" not in n.get("params", {})]}
    with pytest.raises(api.ApiError, match="no optimizer"):
        api.apply_learning_rate(no_optimizer, 1e-3)


def test_the_finder_refuses_designs_it_cant_run(isolated_home, tmp_path):
    api.set_manager(RunManager(RunHistory(tmp_path / "runs")))
    no_optimizer = {**tiny_classifier_dict(), "nodes": [
        n for n in tiny_classifier_dict()["nodes"] if "lr" not in n.get("params", {})]}
    with pytest.raises(api.ApiError, match="optimizer"):
        api.find_learning_rate(no_optimizer)
    with pytest.raises(api.ApiError, match="at least 10 steps"):
        api.find_learning_rate(tiny_classifier_dict(), steps=5)


@pytest.mark.slow
def test_finder_runs_and_its_result_reads_back(isolated_home, tmp_path):
    pytest.importorskip("torch")
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    run_id = api.find_learning_rate(tiny_classifier_dict(), steps=40)["run_id"]
    assert mgr.wait(run_id, 600)["state"] == "finished"
    out = api.learning_rate_result(run_id)
    assert len(out["lrs"]) == len(out["losses"]) and len(out["lrs"]) >= 10
    assert all(b > a for a, b in zip(out["lrs"], out["lrs"][1:], strict=False))    # rising rates
    assert out["lrs"][0] == pytest.approx(lr_find.START)
    assert out["suggested"] and out["start"] <= out["suggested"] <= out["end"]
    record = mgr.history.get(run_id)
    assert record.kind == "lr_find" and "lr-find" in record.tags
    assert not list(mgr.history.path(run_id).glob("*_best.pt"))      # no model saved


# ------------------------------------------------------------------ results by group

def test_summary_flags_groups_clearly_worse_than_the_whole():
    raw = {"overall": {"count": 300, "accuracy": 0.8},
           "groups": [{"value": "a", "count": 100, "accuracy": 0.9},
                      {"value": "b", "count": 100, "accuracy": 0.6},
                      {"value": "c", "count": 4, "accuracy": 0.25}]}
    out = groups.summarize(raw)
    assert out["metric"] == "accuracy" and out["higher_is_better"]
    assert out["flagged"] == ["b"]                   # c is worse but has too few rows
    assert out["groups"][1]["gap"] == pytest.approx(0.25)
    errors = groups.summarize({"overall": {"count": 50, "mae": 1.0},
                               "groups": [{"value": "x", "count": 25, "mae": 1.5},
                                          {"value": "y", "count": 25, "mae": 0.5}]})
    assert errors["metric"] == "mae" and not errors["higher_is_better"]
    assert errors["flagged"] == ["x"]


def _region_table(path: Path) -> Path:
    rng = random.Random(0)
    rows = ["a,b,region,label"]
    for i in range(600):
        a, b = rng.random(), rng.random()
        region = ("north", "south", "east")[i % 3]
        label = int(a + b > 1)
        if region == "east" and rng.random() < 0.45:
            label = 1 - label                          # noisy labels in one region
        rows.append(f"{a:.3f},{b:.3f},{region},{label}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _network(table: Path) -> Graph:
    from verify_training import chain

    return chain("regions", "2", [("core.dense", {"units": 16}), ("core.relu", {}),
                                  ("core.dense", {"units": 2})],
                 [("data.csv", {"path": str(table), "target_column": "label"}),
                  ("prep.drop_columns", {"columns": "region"}),
                  ("train.loss_cross_entropy", {}), ("train.adam", {"lr": 0.02}),
                  ("eval.accuracy", {}),
                  ("train.trainer", {"epochs": 8, "batch_size": 32, "device": "cpu",
                                     "seed": 0})])


@pytest.mark.slow
def test_a_network_is_scored_by_group_and_trains_more(isolated_home, tmp_path):
    pytest.importorskip("torch")
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    run_id = api.start_training(_network(_region_table(tmp_path / "t.csv")))["run_id"]
    assert mgr.wait(run_id, 600)["state"] == "finished"

    cols = {c["name"]: c for c in api.group_columns(run_id)["columns"]}
    assert cols["region"]["how"] == "values" and cols["a"]["how"] == "quartiles"
    scoring = api.results_by_group(run_id, "region")["run_id"]
    assert mgr.wait(scoring, 600)["state"] == "finished"
    out = api.group_result(scoring)
    assert out["metric"] == "accuracy" and out["flagged"] == ["east"]
    assert sum(g["count"] for g in out["groups"]) == out["overall"]["count"]
    assert "by 'region'" in mgr.history.get(scoring).note

    more = api.train_more(run_id, 2)["run_id"]
    assert mgr.wait(more, 600)["state"] == "finished"
    record = mgr.history.get(more)
    assert record.epochs_done == 2 and run_id in record.note and "train-more" in record.tags
    with pytest.raises(api.ApiError, match="only a finished run"):
        api.train_more(api.start_training(_network(tmp_path / "t.csv"))["run_id"], 1)


@pytest.mark.slow
def test_a_classic_model_is_scored_by_group(isolated_home, tmp_path):
    pytest.importorskip("sklearn")
    from ai_made_easy.core import recipes

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    facts = recipes.detect(_region_table(tmp_path / "t.csv"))
    graph = recipes.build("random_forest", facts.task, facts, {"n_estimators": 50})
    run_id = mgr.start(graph)
    assert mgr.wait(run_id, 600)["state"] == "finished"
    scoring = api.results_by_group(run_id, "region")["run_id"]
    assert mgr.wait(scoring, 600)["state"] == "finished"
    assert api.group_result(scoring)["flagged"] == ["east"]
    with pytest.raises(api.ApiError, match="continues PyTorch networks"):
        api.train_more(run_id, 2)


def test_grouping_needs_a_table(isolated_home, tmp_path):
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    record = mgr.history.create(tiny_classifier_dict(), framework="pytorch")
    mgr.history.finalize(record.run_id, "finished", 0, "")
    with pytest.raises(api.ApiError, match="trained on a CSV"):
        api.results_by_group(record.run_id, "x")


# ------------------------------------------------------------------ web server

@pytest.fixture()
def client(isolated_home, tmp_path):
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    api.set_manager(RunManager(RunHistory()))
    with TestClient(create_app(token="", projects_dir=tmp_path / "projects")) as c:
        yield c
    api.manager().stop_all()
    api.set_manager(None)


def test_server_routes(client):
    graph = tiny_classifier_dict()
    applied = client.post("/api/lr-find/apply", json={"graph": graph, "lr": 0.002})
    assert applied.status_code == 200
    assert any(n.get("params", {}).get("lr") == 0.002 for n in applied.json()["graph"]["nodes"])
    assert client.post("/api/lr-find", json={"graph": graph, "steps": 3}).status_code == 400
    assert client.get("/api/runs/nope/groups/columns").status_code == 404
    assert client.post("/api/runs/nope/more", json={"epochs": 2}).status_code == 404
    assert client.post("/api/lr-find", json={}).status_code == 400
