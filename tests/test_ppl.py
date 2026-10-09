"""Probabilistic programs (PyMC): model compilation, expressions, rules, diagnostics against
ArviZ, generated scripts that recover known parameters, serving and deploys."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_made_easy.core import api
from ai_made_easy.core.families import family_of
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.ppl.model import ModelError, compile_design, model_code, parse_expression
from ai_made_easy.core.runner.manager import resolve_framework
from ai_made_easy.core.tasks import task_of

HAS_PYMC = importlib.util.find_spec("pymc") is not None
needs_pymc = pytest.mark.skipif(not HAS_PYMC, reason="needs pymc (probabilistic extra)")


def n(nid: str, type_id: str, **params) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": [0, 0]}


def e(a: str, b: str) -> dict:
    return {"from": f"{a}/out", "to": b}


def hier(outcome: str = "gaussian", method: str = "nuts", draws: int = 300, chains: int = 2,
         extra: tuple = (), name: str = "hier") -> dict:
    lik = {"gaussian": n("y", "ppl.normal", name="y", observed="y"),
           "binary": n("y", "ppl.bernoulli", name="y", observed="y"),
           "count": n("y", "ppl.poisson", name="y", observed="y")}[outcome]
    link = {"gaussian": "a + b * x", "binary": "invlogit(a + b * x)",
            "count": "exp(a + b * x)"}[outcome]
    nodes = [n("mu_a", "ppl.normal", name="mu_a", mu=0.0, sigma=5.0),
             n("sigma_a", "ppl.half_normal", name="sigma_a"),
             n("a", "ppl.normal", name="a", group="group"),
             n("b", "ppl.normal", name="b", mu=0.0, sigma=5.0),
             n("mu", "ppl.deterministic", name="mu", expression=link), lik,
             n("data", "data.synthetic_groups", outcome=outcome),
             n("s", "ppl.sampler", method=method, draws=draws, tune=draws, chains=chains),
             *extra]
    edges = [e("mu_a", "a/mu"), e("sigma_a", "a/sigma"), e("a", "mu/inputs"),
             e("b", "mu/inputs"), e("mu", "y/p" if outcome == "binary" else "y/mu")]
    if outcome == "gaussian":
        nodes.append(n("sigma", "ppl.half_normal", name="sigma"))
        edges.append(e("sigma", "y/sigma"))
    return {"name": name, "nodes": nodes, "edges": edges}


def issues(data: dict) -> list:
    return [(i.severity, i.node_id, i.message) for i in Graph.from_dict(data).validate()]


# ================================================================ family / compilation

def test_family_task_framework_and_samples():
    graph = Graph.from_dict(hier())
    assert family_of(graph).id == "ppl" and task_of(graph).id == "bayesian_modeling"
    assert resolve_framework(graph) == "pymc"
    for sample in ("hierarchical_regression.json", "hierarchical_logistic.json"):
        g = Graph.from_dict(api.read_sample(sample))
        assert g.validate() == [] and family_of(g).id == "ppl"


def test_model_code_is_readable_pymc():
    compiled = compile_design(Graph.from_dict(hier()), {"group", "x", "y"})
    lines = model_code(compiled["vars"], compiled["info"])
    code = "\n".join(lines)
    assert 'a_offset = pm.Normal("a_offset", mu=0.0, sigma=1.0, dims="data_group")' in code
    assert 'a = pm.Deterministic("a", mu_a + sigma_a * a_offset, dims="data_group")' in code
    assert 'mu = pm.Deterministic("mu", a[data_group_idx] + b * data_x)' in code
    assert 'y = pm.Normal("y", mu=mu, sigma=sigma, observed=data[\'y\'], shape=rows.shape[0])' \
        in code
    centered = hier()
    centered["nodes"][2]["params"]["non_centered"] = False
    compiled = compile_design(Graph.from_dict(centered), {"group", "x", "y"})
    assert 'a = pm.Normal("a", mu=mu_a, sigma=sigma_a, dims="data_group")' in "\n".join(
        model_code(compiled["vars"], compiled["info"]))


def test_expressions_are_sandboxed():
    _tree, names = parse_expression("invlogit(a + b * x) ** 2")
    assert names == {"a", "b", "x"}
    for bad in ("__import__('os')", "a.b", "[a]", "a if b else c", "lambda: 1"):
        with pytest.raises(ModelError):
            parse_expression(bad)


# ================================================================ rules

def test_rules_name_the_block():
    data = hier()
    data["nodes"][4]["params"]["expression"] = "a + b * z"
    assert ("error", "mu", "mu uses 'z', which is neither a wired variable nor a data column") \
        in issues(data)
    data["nodes"][4]["params"]["expression"] = "a + c * x"
    assert any(i[1] == "mu" and "neither a wired" in i[2] for i in issues(data))
    data = hier()
    data["nodes"][0]["params"]["name"] = "pm"
    assert any(i[1] == "mu_a" and "must be an identifier" in i[2] for i in issues(data))


def test_support_and_data_rules(tmp_path):
    data = hier()
    data["edges"] = [x for x in data["edges"] if x["to"] != "y/sigma"] + [e("b", "y/sigma")]
    assert any("y.sigma must be positive but b can be real" in i[2] for i in issues(data))
    (tmp_path / "d.csv").write_text("group,x,y\n" + "".join(f"g{i % 2},{i},{i * 0.5 - 2}\n"
                                                          for i in range(20)))
    counts = hier("count")
    counts["nodes"] = [x for x in counts["nodes"] if x["type"] != "data.synthetic_groups"] + \
        [n("data", "data.csv", path=str(tmp_path / "d.csv"))]
    assert any("Poisson explains counts but 'y' has real values" in i[2] for i in issues(counts))
    nodata = hier()
    nodata["nodes"] = [x for x in nodata["nodes"] if x["type"] != "data.synthetic_groups"]
    assert any("observed variables need data" in i[2] for i in issues(nodata))
    region = hier()
    region["nodes"][2]["params"]["group"] = "region"
    assert any("group column 'region' is not in the data" in i[2] for i in issues(region))
    prior_only = hier()
    prior_only["nodes"][5]["params"]["observed"] = ""
    assert any("no variable is observed" in i[2] for i in issues(prior_only))


def test_group_mismatch_rule():
    data = hier()
    data["nodes"].append(n("z", "ppl.normal", name="z"))
    data["edges"].append(e("a", "z/mu"))         # a has one value per group, z is a scalar
    assert any("give z group = group" in i[2] for i in issues(data))


# ================================================================ diagnostics

@pytest.fixture(scope="module")
def rt():
    from ai_made_easy.core.ppl.runtime import namespace

    return namespace()


def test_diagnostics_match_arviz(rt):
    az = pytest.importorskip("arviz")
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(0)
    ar = np.zeros((4, 800))
    noise = rng.normal(size=(4, 800))
    for t in range(1, 800):
        ar[:, t] = 0.8 * ar[:, t - 1] + noise[:, t]
    drift = rng.normal(size=(4, 800)) + np.array([[0], [0], [0], [0.6]])
    for x in (rng.normal(size=(4, 800)), ar, drift):
        da = xr.DataArray(x, dims=("chain", "draw"))
        assert rt["rhat"](x) == pytest.approx(float(az.rhat(da)), rel=1e-6)
        assert rt["ess_bulk"](x) == pytest.approx(float(az.ess(da, method="bulk")), rel=1e-6)
        assert rt["ess_tail"](x) == pytest.approx(
            float(az.ess(da, method="tail", prob=(0.05, 0.95))), rel=1e-6)
    assert rt["rhat"](drift) > 1.01 > rt["rhat"](rng.normal(size=(4, 800)))


def test_synthetic_groups(rt):
    frame, truth = rt["synthetic_groups"](5, 30, 1.0, 0.5, 2.0, 0.3, "gaussian", 1)
    assert set(frame.columns) == {"group", "x", "y"} and frame["group"].nunique() == 5
    resid = frame["y"] - np.array(truth["group_effects"])[
        frame["group"].str[1:].astype(int)] - 2.0 * frame["x"]
    assert resid.std() == pytest.approx(0.3, rel=0.2)
    binary, _t = rt["synthetic_groups"](3, 50, 0.0, 0.5, 1.0, 0.1, "binary", 0)
    assert set(binary["y"].unique()) <= {0, 1}


# ================================================================ scripts

def _run(data: dict, tmp_path: Path) -> tuple[Path, str, Path]:
    from ai_made_easy.core.codegen import export_training

    graph = Graph.from_dict(data)
    script = export_training(graph, resolve_framework(graph), tmp_path)
    proc = subprocess.run([sys.executable, script.name], cwd=tmp_path, capture_output=True,
                          text=True, timeout=1200)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    return tmp_path, proc.stdout, script


def _infer(folder: Path, script: Path, items: list) -> list:
    code = (f"import sys, json; sys.path.insert(0, '.'); import {script.stem} as m; "
            f"m.load_predictor('.'); print('OUT ' + json.dumps(m.infer({items!r})))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True,
                          text=True, timeout=600, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert "OUT " in proc.stdout, proc.stderr[-3000:]
    return json.loads(proc.stdout.split("OUT ", 1)[1])


@needs_pymc
def test_hierarchical_regression_recovers_the_truth(tmp_path):
    run, _out, script = _run(hier(draws=400), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    summary = {r["name"]: r for r in json.loads((run / "summary.json").read_text())}
    assert summary["b"]["lower_94"] < 2.0 < summary["b"]["upper_94"]
    assert metrics["mean_sigma"] == pytest.approx(0.5, abs=0.08)
    assert metrics["divergences"] == 0 and metrics["max_r_hat"] < 1.05
    assert 0.85 < metrics["test_y_coverage_94"] <= 1.0
    for name in ("posterior.png", "trace.png", "ppc_y.png", "report.txt"):
        assert (run / "eval_samples" / name).exists()
    out = _infer(run, script, [{"group": "g1", "x": 0.5}])
    assert set(out[0]["y"]) == {"mean", "sd", "lower_94", "upper_94"}
    assert out[0]["y"]["lower_94"] < out[0]["y"]["mean"] < out[0]["y"]["upper_94"]


@needs_pymc
@pytest.mark.parametrize("outcome, method, key", [
    ("binary", "nuts", "test_y_accuracy"), ("count", "nuts", "test_y_rmse"),
    ("gaussian", "advi", "test_y_rmse"), ("gaussian", "map", "test_y_rmse"),
])
def test_likelihoods_and_methods(outcome, method, key, tmp_path):
    run, _o, script = _run(hier(outcome, method), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert key in metrics
    if outcome == "binary":
        assert metrics[key] > 0.7
    if method == "map":
        assert "max_r_hat" not in metrics and metrics["test_y_coverage_94"] > 0.8
        assert _infer(run, script, [{"group": "g0", "x": 0}])[0]["y"]["sd"] > 0


@needs_pymc
def test_unknown_group_level_is_reported(tmp_path):
    run, _o, script = _run(hier(method="map"), tmp_path)
    code = (f"import sys; sys.path.insert(0, '.'); import {script.stem} as m; "
            "m.load_predictor('.');\ntry:\n m.infer([{'group': 'zz', 'x': 0}])\n"
            "except ValueError as exc:\n print('ERR', exc)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=run, capture_output=True, text=True,
                          timeout=300)
    assert "unknown group level(s) ['zz']" in proc.stdout, proc.stderr[-2000:]


@needs_pymc
def test_ppl_run_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(hier(method="map", name="ppl_run")))
        status = mgr.wait(run_id, 900)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        assert mgr.history.get(run_id).framework == "pymc"
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["task"] == "bayesian_modeling" and meta["input_kind"] == "records"
        assert "pymc" in (tmp_path / "pkg/requirements.txt").read_text()
    finally:
        api.set_manager(None)
