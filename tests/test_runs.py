"""Run history, headless run manager and the core API run surface."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core import api
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.manager import RunManager, resolve_framework
from ai_made_easy.core.runs.history import RunHistory, compare, graph_params


def test_history_create_update_list_delete(tmp_path):
    h = RunHistory(tmp_path)
    a = h.create(tiny_classifier_dict(), project="p1")
    b = h.create(tiny_classifier_dict(lr=0.1), project="p2", tags=["x"])
    assert {r.run_id for r in h.list()} == {a.run_id, b.run_id}
    assert [r.run_id for r in h.list(project="p2")] == [b.run_id]
    h.append_epoch(a.run_id, {"type": "epoch", "epoch": 1, "total": 2,
                              "metrics": {"val_loss": 0.9, "accuracy": 0.5, "lr": 0.1}})
    h.append_epoch(a.run_id, {"type": "epoch", "epoch": 2, "total": 2,
                              "metrics": {"val_loss": 1.2, "accuracy": 0.7}})
    rec = h.get(a.run_id)
    assert rec.epochs_done == 2
    assert rec.best_metrics == {"val_loss": 0.9, "accuracy": 0.7}  # lr is not a metric
    (h.path(a.run_id) / "metrics.json").write_text(json.dumps({"accuracy": 0.66, "cm": [[1]]}))
    rec = h.finalize(a.run_id, "finished", 0)
    assert rec.final_metrics == {"accuracy": 0.66} and rec.duration is not None
    assert len(h.epochs(a.run_id)) == 2
    h.delete(b.run_id)
    assert not h.exists(b.run_id)
    with pytest.raises(KeyError):
        h.get("../etc")


def test_graph_params_and_compare(tmp_path):
    h = RunHistory(tmp_path)
    a = h.create(tiny_classifier_dict(lr=0.05))
    b = h.create(tiny_classifier_dict(lr=0.2, units=16))
    assert graph_params(tiny_classifier_dict())["opt.lr"] == 0.05
    result = compare([a, b])
    assert result["params"] == {"d1.units": [8, 16], "opt.lr": [0.05, 0.2]}


def test_mark_interrupted_only_touches_dead_processes(tmp_path):
    import os

    h = RunHistory(tmp_path)
    dead = h.create(tiny_classifier_dict())
    h.update(dead.run_id, status="running", pid=999_999)
    alive = h.create(tiny_classifier_dict())
    h.update(alive.run_id, status="running", pid=os.getpid())
    assert h.mark_interrupted() == 1
    assert h.get(dead.run_id).status == "failed"
    assert h.get(alive.run_id).status == "running"


def test_resolve_framework():
    nn = Graph.from_dict(tiny_classifier_dict())
    assert resolve_framework(nn) == "pytorch"
    assert resolve_framework(nn, "keras") == "keras"
    with pytest.raises(ValueError):
        resolve_framework(nn, "sklearn")
    classic = Graph.from_dict(api.read_sample("classic_random_forest.json"))
    assert resolve_framework(classic) == "sklearn"


def test_manager_records_a_real_pytorch_run(tmp_path):
    pytest.importorskip("torch")
    mgr = RunManager(RunHistory(tmp_path))
    seen = []
    mgr.subscribe(lambda run_id, ev: seen.append(ev["type"]))
    run_id = mgr.start(Graph.from_dict(tiny_classifier_dict()), project="demo")
    status = mgr.wait(run_id, timeout=300)
    assert status["state"] == "finished", status
    rec = mgr.history.get(run_id)
    assert rec.status == "finished" and rec.returncode == 0
    assert rec.epochs_done == 2 and len(mgr.history.epochs(run_id)) == 2
    assert "accuracy" in rec.final_metrics
    assert "torch" in rec.env["packages"]
    assert (mgr.history.path(run_id) / "metrics.json").exists()
    assert {"env", "epoch", "done"} <= set(seen)


def test_manager_records_a_classic_run(tmp_path):
    pytest.importorskip("sklearn")
    data = api.read_sample("classic_random_forest.json")
    for node in data["nodes"]:
        if node["type"] == "ml.random_forest_classifier":
            node["params"]["n_estimators"] = 10
    mgr = RunManager(RunHistory(tmp_path))
    run_id = mgr.start(Graph.from_dict(data))
    assert mgr.wait(run_id, timeout=300)["state"] == "finished"
    rec = mgr.history.get(run_id)
    assert rec.framework == "sklearn" and rec.final_metrics


def test_manager_marks_crashed_worker_failed(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import ai_made_easy.core.runner.manager as m

    monkeypatch.setattr(m, "export_training", lambda g, fw, out: _crash_script(out))
    mgr = RunManager(RunHistory(tmp_path))
    run_id = mgr.start(Graph.from_dict(tiny_classifier_dict()))
    assert mgr.wait(run_id, timeout=60)["state"] == "failed"
    rec = mgr.history.get(run_id)
    assert rec.status == "failed" and "boom" in rec.error


def _crash_script(out):
    from pathlib import Path

    path = Path(out) / "crash.py"
    path.write_text("raise RuntimeError('boom')\n")
    return path


def test_api_run_surface(tmp_path):
    pytest.importorskip("torch")
    api.set_manager(RunManager(RunHistory(tmp_path)))
    try:
        with pytest.raises(api.ApiError):
            bad = tiny_classifier_dict()
            bad["edges"] = []
            api.start_training(bad)
        a = api.start_training(tiny_classifier_dict(epochs=1), project="p")["run_id"]
        b = api.start_training(tiny_classifier_dict(epochs=1, lr=0.1), project="p")["run_id"]
        api.manager().wait(a, 300)
        api.manager().wait(b, 300)
        listing = api.list_runs("p")
        assert listing["count"] == 2 and "graph" not in listing["runs"][0]
        detail = api.get_run(a)
        assert detail["epochs"] and "metrics.json" in detail["artifacts"]
        cmp = api.compare_runs([a, b])
        assert cmp["params"]["opt.lr"] == [0.05, 0.1]
        assert api.update_run(a, tags=["best"], note="ok")["tags"] == ["best"]
        api.delete_run(b)
        assert api.list_runs("p")["count"] == 1
    finally:
        api.set_manager(None)


def test_api_validate_reports_fixes_and_shapes():
    data = tiny_classifier_dict()
    result = api.validate(data)
    assert result["valid"] and result["shapes"]["d2"] == [3]
    with pytest.raises(api.ApiError):
        api.generate(data, "nope")
    assert "class" in api.generate(data, "pytorch_model")
    assert {t["id"] for t in api.targets()["targets"]} >= {"pytorch_model", "sklearn_train"}


def test_cli_runs_lists_history(isolated_home):
    import os

    env = {**os.environ, "AIME_HOME": str(isolated_home)}
    h = RunHistory(isolated_home / "runs")
    rec = h.create(tiny_classifier_dict(), project="cli")
    out = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "runs"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0 and rec.run_id in out.stdout
    out = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "runs", "show", rec.run_id],
                         capture_output=True, text=True, env=env, timeout=120)
    assert json.loads(out.stdout)["project"] == "cli"
