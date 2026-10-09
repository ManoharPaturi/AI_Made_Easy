"""State-space models: component wiring into statsmodels specifications, rules, generated
scripts that recover known components and beat a seasonal-naive forecast with calibrated
intervals, serving on new histories and deploys."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_made_easy.core import api
from ai_made_easy.core.families import family_of
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.manager import resolve_framework
from ai_made_easy.core.ssm.template import spec_of
from ai_made_easy.core.tasks import task_of

HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None
needs_statsmodels = pytest.mark.skipif(not HAS_STATSMODELS, reason="needs statsmodels")
SAMPLES = ("structural_time_series.json", "sarimax_airline.json")


def n(nid: str, type_id: str, **params) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": [0, 0]}


def e(a: str, b: str = "m/components") -> dict:
    return {"from": f"{a}/out", "to": b}


def design(parts: list, model: dict | None = None, data: dict | None = None,
           data_type: str = "data.structural_series") -> dict:
    return {"name": "ssm", "nodes": [*parts, n("m", "ssm.model", **(model or {})),
                                     n("data", data_type, **(data or {}))],
            "edges": [e(p["id"]) for p in parts]}


def messages(data: dict) -> list[str]:
    return [i.message for i in Graph.from_dict(data).validate()]


# ================================================================ family / spec

def test_family_task_and_samples():
    for sample in SAMPLES:
        graph = Graph.from_dict(api.read_sample(sample))
        assert graph.validate() == []
        assert family_of(graph).id == "ssm" and resolve_framework(graph) == "statsmodels"
        assert task_of(graph).id == "state_space_forecasting"


def test_specifications():
    graph = Graph.from_dict(design([n("t", "ssm.trend", kind="smooth trend"),
                                    n("s", "ssm.seasonal", period=7, harmonics=2,
                                      stochastic=False),
                                    n("c", "ssm.cycle", max_period=40.0),
                                    n("a", "ssm.autoregressive", order=2)],
                                   {"irregular": False}))
    kind, spec, exog, season = spec_of(graph)
    assert kind == "structural" and season == 7 and exog == []
    assert spec == {"level": True, "stochastic_level": False, "trend": True,
                    "stochastic_trend": True, "irregular": False,
                    "freq_seasonal": [{"period": 7, "harmonics": 2}],
                    "stochastic_freq_seasonal": [False], "cycle": True,
                    "stochastic_cycle": True, "damped_cycle": True,
                    "cycle_period_bounds": (1.5, 40.0), "autoregressive": 2}
    arima = Graph.from_dict(api.read_sample(SAMPLES[1]))
    assert spec_of(arima)[:2] == ("sarimax", {"order": (0, 1, 1),
                                              "seasonal_order": (0, 1, 1, 12), "trend": "n"})
    bare = Graph.from_dict(design([n("s", "ssm.seasonal")]))
    assert spec_of(bare)[1]["stochastic_level"] is False          # a fixed intercept


# ================================================================ rules

def test_rules(tmp_path):
    assert any("wire a Level / Trend" in m for m in messages(design([])))
    mixed = design([n("a", "ssm.arima"), n("t", "ssm.trend")])
    assert any("cannot be combined" in m for m in messages(mixed))
    twice = design([n("t", "ssm.trend"), n("u", "ssm.trend")])
    assert any("one Level / Trend per model" in m for m in messages(twice))
    loose = design([n("t", "ssm.trend")])
    loose["nodes"].append(n("x", "ssm.cycle"))
    assert any("Cycle is not wired" in m for m in messages(loose))
    short = design([n("t", "ssm.trend")], {"horizon": 50}, {"length": 55})
    assert any("set horizon to 11" in m for m in messages(short))
    season = design([n("t", "ssm.trend"), n("s", "ssm.seasonal", period=120)])
    assert any("two seasons of training data" in m for m in messages(season))
    harmonics = design([n("t", "ssm.trend"), n("s", "ssm.seasonal", period=6, harmonics=5)])
    assert any("set harmonics to 3" in m for m in messages(harmonics))
    reg = design([n("t", "ssm.trend"), n("r", "ssm.regression", columns="price")])
    assert any("has no explanatory columns" in m for m in messages(reg))
    many = design([n("t", "ssm.trend")], data={}, data_type="data.synthetic_series")
    assert any("fit one series" in m for m in messages(many))
    (tmp_path / "s.csv").write_text("sales,price\n" + "".join(f"{i},{i % 3}\n"
                                                              for i in range(80)))
    csv = design([n("t", "ssm.trend"), n("r", "ssm.regression", columns="price,promo")],
                 {"target_column": "units", "horizon": 8},
                 {"path": str(tmp_path / "s.csv")}, "data.timeseries_csv")
    found = messages(csv)
    assert any("'units' is not in the data" in m for m in found)
    assert any("['promo'] are not in the data" in m for m in found)


# ================================================================ scripts

def _run(data: dict, tmp_path: Path) -> tuple[Path, Path]:
    from ai_made_easy.core.codegen import export_training

    graph = Graph.from_dict(data)
    script = export_training(graph, resolve_framework(graph), tmp_path)
    proc = subprocess.run([sys.executable, script.name], cwd=tmp_path, capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    return tmp_path, script


def _infer(folder: Path, script: Path, items: list) -> list:
    code = (f"import sys, json; sys.path.insert(0, '.'); import {script.stem} as m; "
            f"m.load_predictor('.'); print('OUT ' + json.dumps(m.infer({items!r})))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True,
                          text=True, timeout=300, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert "OUT " in proc.stdout, proc.stderr[-3000:]
    return json.loads(proc.stdout.split("OUT ", 1)[1])


@needs_statsmodels
@pytest.mark.parametrize("sample", SAMPLES)
def test_forecasts_beat_seasonal_naive(sample, tmp_path):
    run, script = _run(api.read_sample(sample), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["mae"] < 0.8 * metrics["naive_mae"] and metrics["mase"] < 1
    assert metrics["coverage"] >= 0.75
    assert (run / "eval_samples" / "forecast.png").exists()
    if sample == SAMPLES[0]:
        assert metrics["trend_rmse"] < 0.5                    # the true trend is recovered
        assert (run / "eval_samples" / "components.png").exists()
    history = [10 + 0.05 * t for t in range(60)]
    out = _infer(run, script, [history, {"history": history, "horizon": 3}])
    assert len(out[0]["mean"]) == 24 and len(out[1]["mean"]) == 3
    assert out[1]["lower"][0] < out[1]["mean"][0] < out[1]["upper"][0]
    assert out[1]["mean"][0] == pytest.approx(13.0, abs=1.5)   # continues the line


@needs_statsmodels
def test_regression_component(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    price = rng.uniform(1, 3, 200)
    sales = 50 + 0.1 * np.arange(200) - 6 * price + rng.normal(0, 0.5, 200)
    (tmp_path / "sales.csv").write_text("sales,price\n" + "".join(
        f"{s:.4f},{p:.4f}\n" for s, p in zip(sales, price, strict=True)))
    data = design([n("t", "ssm.trend", kind="local linear trend"),
                   n("r", "ssm.regression", columns="price")],
                  {"horizon": 20}, {"path": str(tmp_path / "sales.csv"),
                                    "target_columns": "sales"}, "data.timeseries_csv")
    assert Graph.from_dict(data).validate() == []
    run, script = _run(data, tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["mae"] < 1.0
    state = __import__("pickle").loads((run / "ssm_ssm.pkl").read_bytes())
    beta = dict(zip(state["param_names"], state["params"], strict=True))["beta.x1"]
    assert beta == pytest.approx(-6.0, abs=0.3)                # the price effect
    out = _infer(run, script, [{"history": sales[:100].tolist(),
                                "past_covariates": price[:100, None].tolist(),
                                "future_covariates": [[1.0], [3.0]]}])
    assert out[0]["mean"][0] - out[0]["mean"][1] == pytest.approx(12.0, abs=1.5)


@needs_statsmodels
def test_ssm_run_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(api.read_sample(SAMPLES[0])))
        status = mgr.wait(run_id, 900)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        assert mgr.history.get(run_id).framework == "statsmodels"
        assert mgr.history.epochs(run_id)                    # likelihood per iteration
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["task"] == "state_space_forecasting" and meta["input_kind"] == "series"
    finally:
        api.set_manager(None)
