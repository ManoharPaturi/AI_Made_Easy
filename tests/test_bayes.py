"""Bayesian deep learning and calibration: variational layers, MC dropout, mixture density
heads, the last-layer Laplace approximation, ECE / temperature scaling / conformal prediction,
the rules, generated scripts that learn, serving with uncertainty and deploys."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_made_easy.core import api
from ai_made_easy.core.bayes.codegen import probabilistic_context
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.training.spec import collect_spec

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
SAMPLES = ("uncertainty_mc_dropout.json", "bayes_by_backprop.json",
           "mixture_density_regression.json")


def n(nid: str, type_id: str, **params) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": [0, 0]}


def chain(shape: str, layers: list, extra: list, name: str = "b") -> dict:
    nodes, edges, prev = [n("in", "core.input", shape=shape)], [], "in"
    for i, (type_id, params) in enumerate(layers):
        nodes.append(n(f"l{i}", type_id, **params))
        edges.append({"from": f"{prev}/out", "to": f"l{i}/in"})
        prev = f"l{i}"
    nodes.append(n("out", "core.output"))
    edges.append({"from": f"{prev}/out", "to": "out/in"})
    return {"name": name, "nodes": nodes + extra, "edges": edges}


def moons(**data) -> dict:
    return n("data", "data.synthetic", **{"kind": "moons", "n_samples": 1200, "n_features": 2,
                                          "noise": 0.3, **data})


def messages(data: dict) -> list[str]:
    return [i.message for i in Graph.from_dict(data).validate()]


def _module(data: dict, folder: Path):
    """Import a generated training script without running main()."""
    from ai_made_easy.core.codegen import export_training

    script = export_training(Graph.from_dict(data), "pytorch", folder)
    spec = importlib.util.spec_from_file_location(f"bayes_{folder.name}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MLP_CLS = chain("2", [("core.dense", {"units": 16}), ("core.relu", {}),
                      ("core.dense", {"units": 2})],
                [moons(), n("loss", "train.loss_cross_entropy"), n("ece", "eval.ece", bins=10),
                 n("t", "eval.temperature_scaling"), n("c", "eval.conformal", alpha=0.1)])


# ================================================================ blocks / context

def test_samples_validate_and_context():
    for sample in SAMPLES:
        assert Graph.from_dict(api.read_sample(sample)).validate() == []
    ctx = probabilistic_context(Graph.from_dict(api.read_sample(SAMPLES[0])))
    assert ctx["mc_samples"] == 30 and ctx["temperature"] and ctx["conformal"] == 0.1
    assert not ctx["kl"] and ctx["mdn"] is None
    assert probabilistic_context(Graph.from_dict(api.read_sample(SAMPLES[1])))["kl"]
    mdn = Graph.from_dict(api.read_sample(SAMPLES[2]))
    assert probabilistic_context(mdn)["mdn"] == {"k": 3, "d": 1}
    assert collect_spec(mdn).output_shape == [1]            # targets, not mixture parameters
    assert probabilistic_context(Graph.from_dict(api.read_sample("iris_mlp.json"))) is None


def test_shapes_and_parameter_counts():
    g = Graph.from_dict(chain("8", [("bayes.linear", {"units": 4}), ("bayes.mdn",
                                                                    {"components": 3,
                                                                     "targets": 2})], []))
    shapes = g.infer_shapes()
    assert shapes["l0"] == [4] and shapes["l1"] == [3 * (1 + 2 * 2)]
    from ai_made_easy.core.summary import summarize

    rows = {r.type_id: r.params for r in summarize(g).layers}
    assert rows["bayes.linear"] == 2 * (8 * 4 + 4)          # a mean and a spread per weight
    assert rows["bayes.mdn"] == 4 * 15 + 15


# ================================================================ rules

def test_rules():
    mdn_middle = chain("4", [("bayes.mdn", {}), ("core.dense", {"units": 1})],
                       [n("loss", "train.loss_mse")])
    assert any("must be the last layer" in m for m in messages(mdn_middle))
    mdn_ce = chain("4", [("bayes.mdn", {})], [n("loss", "train.loss_cross_entropy")])
    assert any("predicts real-valued targets" in m for m in messages(mdn_ce))
    assert not any("real-valued" in m for m in messages(chain("4", [("bayes.mdn", {})], [])))
    lap = chain("4", [("core.dense", {"units": 2}), ("core.softmax", {})],
                [n("loss", "train.loss_cross_entropy"), n("lp", "bayes.laplace")])
    assert any("end with a Dense layer" in m for m in messages(lap))
    ece_reg = chain("4", [("core.dense", {"units": 1})],
                    [n("loss", "train.loss_mse"), n("e", "eval.ece")])
    assert any("no effect on regression" in m for m in messages(ece_reg))
    single = chain("4", [("bayes.mc_dropout", {"samples": 1}), ("core.dense", {"units": 2})],
                   [n("loss", "train.loss_cross_entropy")])
    assert any("samples = 1" in m for m in messages(single))


# ================================================================ helpers (numerics)

@needs_torch
def test_variational_layers_kl_and_stochasticity():
    import torch

    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    ns = {"torch": torch, "nn": torch.nn}
    for key in ("MCDropout", "BayesLinear", "MDNHead"):
        exec(PYTORCH_HELPERS[key], ns)  # noqa: S102 — our own source
    layer = ns["BayesLinear"](3, 2, prior_sigma=0.5)
    with torch.no_grad():
        layer.weight_rho.uniform_(-2, 0)
        layer.weight_mu.normal_()
    q = torch.distributions.Normal(layer.weight_mu, torch.nn.functional.softplus(layer.weight_rho))
    qb = torch.distributions.Normal(layer.bias_mu, torch.nn.functional.softplus(layer.bias_rho))
    prior = torch.distributions.Normal(0.0, 0.5)
    with torch.no_grad():
        want = (torch.distributions.kl_divergence(q, prior).sum()
                + torch.distributions.kl_divergence(qb, prior).sum())
        got = layer.kl_divergence()
    assert float(got) == pytest.approx(float(want), rel=1e-4)
    x = torch.randn(4, 3)
    assert not torch.equal(layer(x), layer(x))              # sampled weights
    drop = ns["MCDropout"](0.5).eval()                      # stays on in eval mode
    assert not torch.equal(drop(torch.ones(64)), drop(torch.ones(64)))


@needs_torch
def test_calibration_numerics(tmp_path):
    import numpy as np

    m = _module(MLP_CLS, tmp_path)
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, (4000, 3))
    true_p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in true_p])
    over = np.exp(logits * 2.5) / np.exp(logits * 2.5).sum(1, keepdims=True)
    assert m.fit_temperature(over, y) == pytest.approx(2.5, rel=0.15)
    assert m.ece(true_p, y, 15)[0] < 0.03 < m.ece(over, y, 15)[0]
    sets = m.conformal_sets(np.array([[0.98, 0.01, 0.01], [0.4, 0.35, 0.25]]), 0.0)
    assert sets.sum(1).tolist() == [1, 1]                   # the top class is always kept
    assert m.conformal_sets(np.array([[0.4, 0.35, 0.25]]), 0.7).sum() == 2


@needs_torch
def test_mdn_likelihood_matches_torch_distributions(tmp_path):
    import torch

    data = chain("4", [("core.dense", {"units": 8}), ("bayes.mdn", {"components": 3,
                                                                    "targets": 2})],
                 [n("data", "data.synthetic", kind="regression", n_features=4, n_samples=200),
                  n("loss", "train.loss_mse")])
    m = _module(data, tmp_path)
    out, y = torch.randn(5, 15), torch.randn(5, 2)
    logits, mu, log_sigma = m.mdn_split(out)
    mix = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.Independent(torch.distributions.Normal(mu, log_sigma.exp()), 1))
    assert float(m.mdn_nll(out, y)) == pytest.approx(float(-mix.log_prob(y).mean()), rel=1e-5)
    mean, std = m.mdn_moments(out.numpy())
    assert mean == pytest.approx(mix.mean.numpy(), abs=1e-5)
    assert std == pytest.approx(mix.stddev.numpy(), abs=1e-4)


# ================================================================ scripts

def _run(data: dict, tmp_path: Path) -> tuple[Path, Path]:
    from ai_made_easy.core.codegen import export_training

    script = export_training(Graph.from_dict(data), "pytorch", tmp_path)
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


@needs_torch
def test_mc_dropout_calibration_and_conformal_sets(tmp_path):
    data = api.read_sample(SAMPLES[0])
    for node in data["nodes"]:
        if node["type"] == "train.trainer":
            node["params"]["epochs"] = 12
    run, script = _run(data, tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["accuracy"] > 0.8
    assert metrics["calibrated_nll"] <= metrics["nll"] + 0.01
    assert 0.8 <= metrics["conformal_coverage"] and 1 <= metrics["conformal_set_size"] < 2
    assert (run / "eval_samples" / "reliability.png").exists()
    out = _infer(run, script, [[0.5, 0.25], [0.0, 1.0]])
    for row in out:
        assert 0 <= row["entropy"] <= math.log(2) + 1e-6
        assert row["label"] in row["prediction_set"]


@needs_torch
def test_bayes_by_backprop_learns(tmp_path):
    data = chain("2", [("bayes.linear", {"units": 32}), ("core.relu", {}),
                       ("bayes.linear", {"units": 2, "samples": 10})],
                 [moons(noise=0.2), n("loss", "train.loss_cross_entropy"),
                  n("opt", "train.adam", lr=0.01),
                  n("tr", "train.trainer", epochs=25, batch_size=64)])
    run, script = _run(data, tmp_path)
    assert json.loads((run / "metrics.json").read_text())["accuracy"] > 0.85
    probs = _infer(run, script, [[0.5, 0.25]])[0]["probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


@needs_torch
def test_mixture_density_intervals(tmp_path):
    data = chain("8", [("core.dense", {"units": 32}), ("core.relu", {}),
                       ("bayes.mdn", {"components": 2})],
                 [n("data", "data.synthetic", kind="regression", n_features=8, n_samples=2500,
                    noise=0.5), n("loss", "train.loss_mse"), n("opt", "train.adam", lr=0.003),
                  n("tr", "train.trainer", epochs=30, batch_size=64),
                  n("c", "eval.conformal", alpha=0.1)])
    run, script = _run(data, tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert 0.8 <= metrics["coverage_90"] <= 0.98           # the predicted spread is honest
    assert metrics["mean_std"] == pytest.approx(0.5, rel=0.4)
    assert metrics["conformal_coverage"] >= 0.8
    row = _infer(run, script, [[0.1] * 8])[0]
    assert row["interval"][0] < row["prediction"] < row["interval"][1] and row["std"] > 0


@needs_torch
@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_last_layer_laplace(kind, tmp_path):
    if kind == "classification":
        data = chain("2", [("core.dense", {"units": 32}), ("core.relu", {}),
                           ("core.dense", {"units": 2})],
                     [moons(), n("loss", "train.loss_cross_entropy"), n("lp", "bayes.laplace"),
                      n("tr", "train.trainer", epochs=10, batch_size=64)])
    else:
        data = chain("8", [("core.dense", {"units": 32}), ("core.relu", {}),
                           ("core.dense", {"units": 1})],
                     [n("data", "data.synthetic", kind="regression", n_features=8,
                        n_samples=1500, noise=0.3), n("loss", "train.loss_mse"),
                      n("lp", "bayes.laplace"), n("tr", "train.trainer", epochs=30,
                                                   batch_size=64)])
    run, script = _run(data, tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["laplace_prior_precision"] > 0
    if kind == "classification":
        assert metrics["laplace_nll"] <= metrics["nll"] + 0.05
        assert "probabilities" in _infer(run, script, [[0.5, 0.25]])[0]
    else:
        assert 0.7 <= metrics["coverage_90"] <= 1.0
        assert _infer(run, script, [[0.1] * 8])[0]["std"] > 0


@needs_torch
def test_bayes_run_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    data = api.read_sample(SAMPLES[0])
    for node in data["nodes"]:
        if node["type"] == "train.trainer":
            node["params"]["epochs"] = 3
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(data))
        status = mgr.wait(run_id, 900)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
    finally:
        api.set_manager(None)
