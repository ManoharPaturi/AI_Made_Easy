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
        automl.candidates(automl.AutoMLSpec("probabilistic_inference"))
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
        api.start_automl({"task": "regime_detection"})
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
