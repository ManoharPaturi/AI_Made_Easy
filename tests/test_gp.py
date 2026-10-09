"""Gaussian processes: kernel algebra for GPyTorch and scikit-learn, rules, generated
scripts that beat a constant baseline with calibrated intervals, serving and deploys."""
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
from ai_made_easy.core.gp.kernels import gpytorch_expr, kernel_root, sklearn_expr
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.manager import resolve_framework
from ai_made_easy.core.tasks import task_of

HAS_GPYTORCH = importlib.util.find_spec("gpytorch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
needs_gpytorch = pytest.mark.skipif(not HAS_GPYTORCH, reason="needs gpytorch")


def n(nid: str, type_id: str, **params) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": [0, 0]}


def e(a: str, b: str) -> dict:
    return {"from": f"{a}/out", "to": b}


def design(kernels: list, edges: list, model: dict | None = None, data: dict | None = None,
           name: str = "gp") -> dict:
    return {"name": name, "nodes": [*kernels, n("gp", "gp.model", **(model or {})),
                                    n("data", "data.synthetic_function", **(data or {}))],
            "edges": edges}


def trend_periodic(library: str = "gpytorch", **model) -> dict:
    return design([n("lin", "gp.linear"), n("per", "gp.periodic", period=0.7), n("sum", "gp.sum")],
                  [e("lin", "sum/kernels"), e("per", "sum/kernels"), e("sum", "gp/kernel")],
                  {"library": library, **model})


def messages(data: dict) -> list[str]:
    return [i.message for i in Graph.from_dict(data).validate()]


# ================================================================ family / kernels

def test_family_task_and_library_choice():
    graph = Graph.from_dict(trend_periodic())
    assert family_of(graph).id == "gp" and task_of(graph).id == "gp_regression"
    assert resolve_framework(graph) == "gpytorch"
    assert resolve_framework(Graph.from_dict(trend_periodic("sklearn"))) == "sklearn"
    for sample in ("gp_trend_seasonality.json", "gp_classification.json"):
        assert Graph.from_dict(api.read_sample(sample)).validate() == []


def test_kernel_expressions():
    graph = Graph.from_dict(trend_periodic())
    root = kernel_root(graph)
    assert gpytorch_expr(graph, root) == (
        "(gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()) + "
        "gpytorch.kernels.ScaleKernel(init(gpytorch.kernels.PeriodicKernel(), "
        "period_length=0.7, lengthscale=1.0)))")
    assert sklearn_expr(graph, root) == (
        "(ConstantKernel(1.0) * DotProduct() + ConstantKernel(1.0) * "
        "ExpSineSquared(length_scale=1.0, periodicity=0.7))")
    ard = Graph.from_dict(design([n("k", "gp.rbf", ard=True, columns="x", scale=False)],
                                 [e("k", "gp/kernel")]))
    assert gpytorch_expr(ard, kernel_root(ard)) == (
        "init(gpytorch.kernels.RBFKernel(active_dims=dims(['x']), ard_num_dims=len(['x'])), "
        "lengthscale=1.0)")
    white = Graph.from_dict(design([n("k", "gp.rbf"), n("w", "gp.white"), n("s", "gp.sum")],
                                   [e("k", "s/kernels"), e("w", "s/kernels"), e("s", "gp/kernel")]))
    assert "WhiteKernel" not in (gpytorch_expr(white, kernel_root(white)) or "")


# ================================================================ rules

def test_rules():
    assert any("wire a kernel" in m for m in messages(design([], [])))
    sm = design([n("k", "gp.spectral_mixture")], [e("k", "gp/kernel")], {"library": "sklearn"})
    assert any("needs the gpytorch library: set library to gpytorch" in m for m in messages(sm))
    single = design([n("k", "gp.rbf"), n("s", "gp.product")],
                    [e("k", "s/kernels"), e("s", "gp/kernel")])
    assert any("combines 1 kernel(s)" in m for m in messages(single))
    orphan = design([n("k", "gp.rbf"), n("x", "gp.matern")], [e("k", "gp/kernel")])
    assert any("Matérn Kernel is not wired" in m for m in messages(orphan))
    exact_cls = design([n("k", "gp.rbf")], [e("k", "gp/kernel")], {"likelihood": "bernoulli"},
                       {"kind": "classes_2d"})
    assert any("needs a variational GP: set kind to svgp" in m for m in messages(exact_cls))
    wrong_target = design([n("k", "gp.rbf")], [e("k", "gp/kernel")], {"target_column": "z"})
    assert any("column(s) ['z'] are not in the data" in m for m in messages(wrong_target))
    big = design([n("k", "gp.rbf")], [e("k", "gp/kernel")], data={"n_points": 9000})
    assert any("O(n³)" in m and "svgp" in m for m in messages(big))
    columns = design([n("k", "gp.rbf", columns="q")], [e("k", "gp/kernel")])
    assert any("kernel column(s) ['q']" in m for m in messages(columns))


def test_data_type_rules(tmp_path):
    (tmp_path / "d.csv").write_text("x,y\n" + "".join(f"{i},{i * 0.3 - 1}\n" for i in range(30)))
    data = {"name": "g", "nodes": [n("k", "gp.rbf"),
                                   n("gp", "gp.model", likelihood="poisson", kind="svgp"),
                                   n("data", "data.csv", path=str(tmp_path / "d.csv"))],
            "edges": [e("k", "gp/kernel")]}
    assert any("poisson likelihood needs counts" in m for m in messages(data))


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


@pytest.mark.parametrize("library", [
    pytest.param("gpytorch", marks=needs_gpytorch),
    pytest.param("sklearn", marks=pytest.mark.skipif(not HAS_SKLEARN, reason="needs sklearn")),
])
def test_regression_beats_the_baseline_with_calibrated_intervals(library, tmp_path):
    run, script = _run(trend_periodic(library), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["rmse"] < 0.3 * metrics["baseline_rmse"] and metrics["r2"] > 0.9
    assert metrics["coverage_95"] >= 0.85
    assert (run / "eval_samples" / "fit.png").exists()
    out = _infer(run, script, [{"x": 12.0}])           # extrapolate the trend + season
    assert out[0]["mean"] == pytest.approx(4.8, abs=0.6)
    assert out[0]["lower_95"] < out[0]["mean"] < out[0]["upper_95"]


@needs_gpytorch
def test_sparse_variational_classification(tmp_path):
    run, script = _run(api.read_sample("gp_classification.json"), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["accuracy"] > metrics["baseline_accuracy"] + 0.2
    assert 0 <= _infer(run, script, [{"x1": 1.0, "x2": -1.0}])[0]["probability"] <= 1


@needs_gpytorch
def test_gp_run_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(api.read_sample("gp_trend_seasonality.json")))
        status = mgr.wait(run_id, 900)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        assert mgr.history.get(run_id).framework == "gpytorch"
        assert mgr.history.epochs(run_id)                    # training-loss curve
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["task"] == "gp_regression" and meta["input_kind"] == "records"
    finally:
        api.set_manager(None)
