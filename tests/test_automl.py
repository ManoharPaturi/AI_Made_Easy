"""AutoML: candidates within the budget, the baseline round, TPE over recipes and their
settings, median pruning, the leaderboard and best design, the API and the web
endpoints."""
from __future__ import annotations

import importlib.util

import pytest
from test_recipes import churn_csv

from ai_made_easy.core import api, automl, recipes
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.sweeps import SweepStore, best_graph

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_metrics_and_candidates(tmp_path):
    assert automl.AutoMLSpec("multiclass").resolved_metric() == ("accuracy", False)
    assert automl.AutoMLSpec("regression").resolved_metric() == ("rmse", True)
    assert automl.AutoMLSpec("binary", metric="f1").resolved_metric() == ("f1", False)
    assert automl.AutoMLSpec("binary", metric="f1", direction="min").resolved_metric()[1]
    with pytest.raises(automl.AutoMLError, match="does not support"):
        automl.candidates(automl.AutoMLSpec("text_generation"))
    facts = recipes.detect(churn_csv(tmp_path)).to_dict()
    ids = [r.id for r in automl.candidates(automl.AutoMLSpec("binary", facts=facts))]
    assert {"random_forest", "gradient_boosting", "tabular_mlp"} <= set(ids)
    picked = automl.candidates(automl.AutoMLSpec("binary", facts=facts,
                                                 recipes=["tabular_mlp", "random_forest"]))
    assert {r.id for r in picked} == {"tabular_mlp", "random_forest"}
    with pytest.raises(automl.AutoMLError, match="within the budget"):
        automl.candidates(automl.AutoMLSpec("binary", facts=facts, recipes=["tabular_mlp"],
                                            budget={"max_params_m": 1e-5}))


def test_trial_designs_and_pruning_rule(tmp_path):
    facts = recipes.detect(churn_csv(tmp_path)).to_dict()
    spec = {"task": "binary", "facts": facts, "epochs": 4}
    g = automl.trial_design(spec, {"recipe": "tabular_mlp", "width": 32})
    assert g.nodes["trainer"].params["epochs"] == 4
    assert g.meta["recipe"]["knobs"]["width"] == 32
    runner = automl.AutoMLRunner.__new__(automl.AutoMLRunner)
    runner.automl = automl.AutoMLSpec("binary")
    runner.spec = type("S", (), {"metric": "accuracy", "minimize": False})()
    runner._curves = {0: [0.6, 0.8, 0.9], 1: [0.5, 0.7, 0.85]}
    assert not runner._should_prune(2, 1, "accuracy", 0.1)           # warm-up
    assert runner._should_prune(2, 2, "accuracy", 0.6)               # below the median
    assert not runner._should_prune(2, 2, "accuracy", 0.8)
    runner._curves = {0: [0.6, 0.8]}
    assert not runner._should_prune(2, 2, "accuracy", 0.1)           # too few to compare


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_search_end_to_end(tmp_path, isolated_home):
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    store = SweepStore(tmp_path / "sweeps")
    facts = recipes.detect(churn_csv(tmp_path)).to_dict()
    runner = automl.start(mgr, store, {
        "task": "binary", "facts": facts, "max_trials": 5, "epochs": 3,
        "recipes": ["random_forest", "linear_baseline", "tabular_mlp"]})
    record = runner.wait(900)
    assert record.state == "finished" and len(record.trials) == 5
    baseline = [t["values"]["recipe"] for t in record.trials[:3]]
    assert sorted(baseline) == ["linear_baseline", "random_forest", "tabular_mlp"]
    assert all(t["state"] in ("finished", "pruned") for t in record.trials)
    board = automl.leaderboard(record)
    scores = [r["score"] for r in board if r["score"] is not None]
    assert scores == sorted(scores, reverse=True) and board[0]["title"]
    assert record.best["score"] == scores[0] > 0.6
    best = Graph.from_dict(best_graph(record))
    assert best.meta["recipe"]["id"] == record.best["values"]["recipe"]
    assert mgr.history.list(parent=record.sweep_id)


def test_api_and_web(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    path = str(churn_csv(tmp_path))
    with pytest.raises(api.ApiError, match="does not support"):
        api.start_automl({"task": "retrieval_qa"})
    app = create_app(token="", projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        tasks = client.get("/api/wizard/tasks").json()["tasks"]
        assert any(t["id"] == "detection" and t["data_kinds"] for t in tasks)
        facts = client.post("/api/wizard/detect", json={"path": path}).json()
        assert facts["target"] == "label" and facts["tasks"][0] == "binary"
        ranked = client.post("/api/wizard/recommend", json={
            "task": "binary", "facts": facts, "budget": {"max_params_m": 5},
            "limit": 3}).json()
        assert len(ranked["suggestions"]) == 3 and ranked["suggestions"][0]["graph"]
        built = client.post("/api/wizard/build", json={
            "recipe": "tabular_mlp", "task": "binary", "facts": facts,
            "knobs": {"width": 32}}).json()["graph"]
        assert built["meta"]["recipe"]["knobs"]["width"] == 32
        why = client.post("/api/explain", json={"graph": built}).json()
        assert why["recipe"]["id"] == "tabular_mlp"
        assert client.get("/api/recipes", params={"task": "detection"}).json()["recipes"]
        bad = client.post("/api/wizard/build", json={"recipe": "nope", "task": "binary"})
        assert bad.status_code == 400
        assert client.get("/api/automl/missing").status_code == 404
        bad = client.post("/api/automl", json={"spec": {"task": "binary",
                                                        "objectives": ["speed"]}})
        assert bad.status_code == 400 and "unknown objective" in bad.text


# ------------------------------------------------------------- multi-objective

def test_pareto_front_and_objective_checks(tmp_path):
    points = {0: [0.90, 5.0], 1: [0.85, 1.0], 2: [0.80, 2.0], 3: [0.90, 6.0], 4: [0.95, 9.0]}
    # accuracy up, latency down: 2 is beaten by 1, 3 by 0
    assert automl.pareto_front(points, [False, True]) == {0, 1, 4}
    assert automl.pareto_front({0: [1.0, 1.0], 1: [1.0, 1.0]}, [True, True]) == {0, 1}
    with pytest.raises(automl.AutoMLError, match="unknown objective"):
        automl.AutoMLSpec("binary", objectives=["speed"]).check()
    with pytest.raises(automl.AutoMLError, match="unknown device"):
        automl.AutoMLSpec("binary", objectives=["latency_ms"], device="abacus").check()
    assert automl.AutoMLSpec("binary", budget={"device": "t4"}).target_device() == "t4"
    assert automl.AutoMLSpec("binary").target_device() == automl.DEFAULT_DEVICE
    facts = recipes.detect(churn_csv(tmp_path)).to_dict()
    picked = automl.candidates(automl.AutoMLSpec("binary", facts=facts,
                                                 objectives=["latency_ms"]))
    assert picked and all(r.family in ("neural", "classic") for r in picked)  # costed only
    assert any(r.family == "classic" for r in picked)
    with pytest.raises(automl.AutoMLError, match="neural networks and classic models"):
        automl.candidates(automl.AutoMLSpec("probabilistic_inference",
                                            objectives=["latency_ms"]))


@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
def test_multi_objective_search_finds_the_trade_off(tmp_path, isolated_home):
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    pytest.importorskip("optuna")
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    store = SweepStore(tmp_path / "sweeps")
    facts = recipes.detect(churn_csv(tmp_path)).to_dict()
    runner = automl.start(mgr, store, {
        "task": "binary", "facts": facts, "max_trials": 6, "epochs": 3,
        "objectives": ["latency_ms", "params_m"], "device": "raspberry_pi_5"})
    record = runner.wait(900)
    assert record.state == "finished" and len(record.trials) == 6
    assert record.spec["objectives"] == ["latency_ms", "params_m"]
    assert record.spec["device"] == "raspberry_pi_5"
    assert not any(t["state"] == "pruned" for t in record.trials)  # no median pruning
    assert all({"latency_ms", "params_m", "model_mb"} <= set(t["objectives"])
               for t in record.trials if t["state"] == "finished")
    board = automl.leaderboard(record)
    front = [r for r in board if r["pareto"]]
    assert front and board[:len(front)] == front                 # the front comes first
    point = lambda r: (r["score"], r["objectives"]["latency_ms"], r["objectives"]["params_m"])  # noqa: E731
    for r in front:                                             # nothing beats it everywhere
        assert not any(o is not r and o["score"] is not None
                       and o["score"] >= r["score"]
                       and o["objectives"]["latency_ms"] <= r["objectives"]["latency_ms"]
                       and o["objectives"]["params_m"] <= r["objectives"]["params_m"]
                       and point(o) != point(r) for o in board)
    # the most accurate trial is always on the front
    assert max(board, key=lambda r: r["score"] or -1)["pareto"]
