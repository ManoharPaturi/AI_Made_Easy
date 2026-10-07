"""Hyperparameter sweeps: spaces, validation, and real sweeps end to end."""
from __future__ import annotations

import random

import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core import api
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.manager import RunManager
from ai_made_easy.core.runs.history import RunHistory
from ai_made_easy.core.sweeps import (
    Dimension,
    SweepError,
    SweepRunner,
    SweepSpec,
    SweepStore,
    apply_values,
    best_graph,
    sweepable_params,
)


def test_dimension_grids_and_samples():
    assert Dimension("o", "lr", low=1e-3, high=1e-1, log=True, points=3).grid() == [
        0.001, 0.01, 0.1]
    assert Dimension("o", "lr", low=0, high=1, points=3).grid() == [0.0, 0.5, 1.0]
    assert Dimension("d", "units", kind="int", low=8, high=32, step=8).grid() == [8, 16, 24, 32]
    rng = random.Random(0)
    for _ in range(50):
        v = Dimension("o", "lr", low=1e-4, high=1e-1, log=True).sample(rng)
        assert 1e-4 <= v <= 1e-1
        u = Dimension("d", "units", kind="int", low=8, high=32, step=8).sample(rng)
        assert u in (8, 16, 24, 32)
    assert Dimension("a", "b", kind="choice", values=["x"]).sample(rng) == "x"


def test_spec_checks_against_the_graph():
    g = Graph.from_dict(tiny_classifier_dict())
    ok = SweepSpec([Dimension("opt", "lr", low=0.01, high=0.1)])
    ok.check(g)
    for bad in (
        SweepSpec([]),
        SweepSpec([Dimension("nope", "lr", low=0.01, high=0.1)]),
        SweepSpec([Dimension("opt", "nope", low=0.01, high=0.1)]),
        SweepSpec([Dimension("opt", "lr", low=0.1, high=0.01)]),
        SweepSpec([Dimension("opt", "lr", low=0, high=1, log=True)]),
        SweepSpec([Dimension("opt", "lr", low=0.01, high=0.1)], strategy="magic"),
        SweepSpec([Dimension("opt", "lr", low=0.01, high=0.1)] * 2),
        SweepSpec([Dimension("d1", "units", kind="int", low=1, high=5000),
                   Dimension("opt", "lr", low=0, high=1, points=50)], strategy="grid"),
    ):
        with pytest.raises(SweepError):
            bad.check(g)
    assert SweepSpec([Dimension("opt", "lr")]).minimize is True  # val_loss default
    assert SweepSpec([Dimension("opt", "lr")], metric="accuracy").minimize is False


def test_sweepable_params_and_apply_values():
    g = Graph.from_dict(tiny_classifier_dict())
    params = {p["key"]: p for p in sweepable_params(g)}
    assert params["opt.lr"]["kind"] == "float" and params["opt.lr"]["log"] is True
    assert params["opt.lr"]["low"] < 0.05 < params["opt.lr"]["high"]
    assert params["d1.units"]["kind"] == "int"
    assert params["opt.nesterov"]["values"] == [False, True]
    data = apply_values(tiny_classifier_dict(), {"opt.lr": 0.2, "d1.units": 4})
    assert Graph.from_dict(data).nodes["opt"].params["lr"] == 0.2
    assert tiny_classifier_dict()["nodes"][6]["params"]["lr"] == 0.05  # input untouched


def test_spec_round_trips_through_dict():
    spec = SweepSpec([Dimension("opt", "lr", low=0.01, high=0.1, log=True)],
                     metric="accuracy", strategy="tpe", max_trials=4)
    assert SweepSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize("strategy", ["grid", "tpe"])
def test_real_sweep_runs_trials_and_picks_best(tmp_path, strategy):
    pytest.importorskip("torch")
    if strategy == "tpe":
        pytest.importorskip("optuna")
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    store = SweepStore(tmp_path / "sweeps")
    events = []
    dims = ([Dimension("d1", "units", kind="choice", values=[4, 0])] if strategy == "grid"
            else [Dimension("opt", "lr", low=0.01, high=0.2, log=True)])
    spec = SweepSpec(dims, metric="accuracy", strategy=strategy, max_trials=2)
    runner = SweepRunner(mgr, store, Graph.from_dict(tiny_classifier_dict(epochs=1)), spec,
                         project="sw", listener=lambda sid, ev: events.append(ev))
    runner.start()
    rec = runner.wait(600)
    assert rec.state == "finished", rec.message
    assert len(rec.trials) == 2
    states = sorted(t["state"] for t in rec.trials)
    if strategy == "grid":
        assert states == ["finished", "invalid"]  # units=0 never runs
    else:
        assert states == ["finished", "finished"]
    assert rec.best and rec.best["score"] is not None
    children = mgr.history.list(parent=rec.sweep_id)
    assert len(children) == (1 if strategy == "grid" else 2)
    assert all(c.trial for c in children)
    assert events[0]["type"] == "sweep_started" and events[-1]["type"] == "sweep_done"
    best = Graph.from_dict(best_graph(rec))
    assert best.validate() == []


def test_api_sweep_surface(tmp_path, isolated_home):
    pytest.importorskip("torch")
    api.set_manager(RunManager(RunHistory(tmp_path / "runs")))
    try:
        data = tiny_classifier_dict(epochs=1)
        assert any(p["key"] == "opt.lr" for p in api.sweepable_params(data)["params"])
        with pytest.raises(api.ApiError):
            api.start_sweep(data, {"dimensions": []})
        sid = api.start_sweep(data, {"dimensions": [
            {"node": "opt", "param": "lr", "kind": "choice", "values": [0.05, 0.1]}],
            "metric": "accuracy", "strategy": "grid", "max_trials": 5})["sweep_id"]
        rec = api.wait_sweep(sid, 600)
        assert rec["state"] == "finished" and len(rec["trials"]) == 2
        assert api.list_sweeps()["count"] == 1 and "graph" not in api.list_sweeps()["sweeps"][0]
        best = api.sweep_best_graph(sid)
        assert best["nodes"][6]["params"]["lr"] in (0.05, 0.1)
    finally:
        api.set_manager(None)


def test_cli_parse_space():
    from ai_made_easy.cli import parse_space

    assert parse_space("opt.lr=log:1e-4:1e-1") == {
        "node": "opt", "param": "lr", "kind": "float", "low": 1e-4, "high": 0.1,
        "log": True, "points": 3}
    assert parse_space("d1.units=int:8:64:8")["step"] == 8
    assert parse_space("opt.nesterov=choice:true,false")["values"] == [True, False]
    assert parse_space("act.mode=choice:a,2,0.5")["values"] == ["a", 2, 0.5]
    for bad in ("lr=log:1:2", "opt.lr=cubic:1:2", "opt.lr=float:1"):
        with pytest.raises(ValueError):
            parse_space(bad)


def test_cli_sweep_end_to_end(tmp_path, isolated_home):
    pytest.importorskip("torch")
    import json
    import os
    import subprocess
    import sys

    project = tmp_path / "p.json"
    project.write_text(json.dumps(tiny_classifier_dict(epochs=1)))
    env = {**os.environ, "AIME_HOME": str(isolated_home)}
    out = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "sweep", str(project),
                          "--list-params"], capture_output=True, text=True, env=env,
                         timeout=120)
    assert "opt.lr" in out.stdout
    out = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "sweep", str(project),
                          "-p", "opt.lr=choice:0.05,0.1", "--strategy", "grid",
                          "--metric", "accuracy"], capture_output=True, text=True, env=env,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    events = [json.loads(line) for line in out.stdout.splitlines()]
    assert [e["type"] for e in events] == ["sweep_started", "trial", "trial", "sweep_finished"]
