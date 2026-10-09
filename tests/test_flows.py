"""Normalizing flows: invertible layers with exact log-determinants, the density-estimation
task, rules, generated scripts that beat a Gaussian, serving (log-density and samples) and
deploys."""
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
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.tasks import task_of

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
SAMPLES = ("flow_checkerboard_spline.json", "flow_moons_realnvp.json")


def n(nid: str, type_id: str, **params) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": [0, 0]}


def chain(shape: str, layers: list, extra: list, name: str = "f") -> dict:
    nodes, edges, prev = [n("in", "core.input", shape=shape)], [], "in"
    for i, (type_id, params) in enumerate(layers):
        nodes.append(n(f"l{i}", type_id, **params))
        edges.append({"from": f"{prev}/out", "to": f"l{i}/in"})
        prev = f"l{i}"
    nodes.append(n("out", "core.output"))
    edges.append({"from": f"{prev}/out", "to": "out/in"})
    return {"name": name, "nodes": nodes + extra, "edges": edges}


def messages(data: dict) -> list[str]:
    return [i.message for i in Graph.from_dict(data).validate()]


def density(**params) -> dict:
    return n("data", "data.density_2d", **params)


# ================================================================ task / rules

def test_task_and_samples():
    for sample in SAMPLES:
        graph = Graph.from_dict(api.read_sample(sample))
        assert graph.validate() == []
        assert task_of(graph).id == "density_estimation"
        assert task_of(graph).trainer_kind == "flow"


def test_rules():
    dense = chain("2", [("flow.affine_coupling", {}), ("core.dense", {"units": 2})],
                  [density()])
    assert any("is not invertible" in m for m in messages(dense))
    only_perm = chain("2", [("flow.actnorm", {}), ("flow.permute", {})], [density()])
    assert any("add a coupling, spline or MAF layer" in m for m in messages(only_perm))
    wrong_dim = chain("3", [("flow.maf", {})], [density()])
    assert any("Set the Input shape to '2'" in m for m in messages(wrong_dim))
    same = chain("2", [("flow.affine_coupling", {}), ("flow.affine_coupling", {})], [density()])
    assert any("set parity to odd" in m for m in messages(same))
    permuted = chain("2", [("flow.affine_coupling", {}), ("flow.permute", {}),
                           ("flow.affine_coupling", {})], [density()])
    assert not any("parity" in m for m in messages(permuted))
    maf = chain("2", [("flow.maf", {}), ("flow.maf", {})], [density()])
    assert any("set reverse to true" in m for m in messages(maf))
    single = chain("2", [("flow.spline_coupling", {"bound": 2.0})], [density()])
    assert any("stack at least two" in m for m in messages(single))
    assert any("values beyond it pass through" in m for m in messages(single))
    one_dim = chain("1", [("flow.affine_coupling", {})], [])
    assert any("at least 2 dimensions" in m for m in messages(one_dim))


# ================================================================ layers

@needs_torch
@pytest.mark.parametrize("key,args", [
    ("AffineCoupling", (3, 16, 2, "odd")), ("MAFLayer", (3, 16, 2, True)),
    ("SplineCoupling", (3, 16, 2, 6, 3.0, "even")), ("ActNorm", (3,)),
    ("FlowPermute", (3, "random", 1)),
])
def test_layers_invert_with_exact_log_determinants(key, args):
    import torch

    from ai_made_easy.core.flows.helpers import FLOW_HELPERS

    torch.manual_seed(0)
    ns = {"torch": torch, "nn": torch.nn}
    exec(FLOW_HELPERS["FlowBase"], ns)  # noqa: S102 — our own source
    exec(FLOW_HELPERS[key], ns)  # noqa: S102
    layer = ns[key](*args).double()
    x = torch.randn(6, 3, dtype=torch.float64) * 1.5
    if key not in ("ActNorm", "FlowPermute"):
        assert torch.allclose(layer(x), x, atol=1e-10)      # starts as the identity
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0, 0.3)
    layer.eval()
    with torch.no_grad():
        z = layer(x)
        logdet = layer.logdet.clone()
        assert torch.allclose(layer.inverse(z), x, atol=1e-6)
    for i in range(len(x)):
        jac = torch.autograd.functional.jacobian(lambda v: layer(v[None])[0], x[i])
        assert float(torch.slogdet(jac)[1]) == pytest.approx(float(logdet[i]), abs=1e-6)


@needs_torch
def test_actnorm_data_dependent_init():
    import torch

    from ai_made_easy.core.flows.helpers import FLOW_HELPERS

    ns = {"torch": torch, "nn": torch.nn}
    exec(FLOW_HELPERS["FlowBase"] + FLOW_HELPERS["ActNorm"], ns)  # noqa: S102
    layer = ns["ActNorm"](4)
    x = torch.randn(512, 4) * torch.tensor([1.0, 3.0, 0.5, 10.0]) + 5
    z = layer.train()(x)
    assert z.mean(0).abs().max() < 1e-4 and (z.std(0) - 1).abs().max() < 1e-3


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


def _short(sample: str, epochs: int, n_samples: int) -> dict:
    data = api.read_sample(sample)
    for node in data["nodes"]:
        if node["type"] == "train.trainer":
            node["params"]["epochs"] = epochs
        if node["type"] == "data.density_2d":
            node["params"]["n_samples"] = n_samples
    return data


@needs_torch
def test_spline_flow_learns_the_checkerboard(tmp_path):
    run, script = _run(_short(SAMPLES[0], 25, 6000), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["nll"] < metrics["baseline_nll"] - 0.4
    assert metrics["nll"] > math.log(8) - 0.2                # cannot beat the true entropy
    assert (run / "eval_samples" / "density.png").exists()
    assert list((run / "samples").glob("epoch_*.png"))       # live pictures
    out = _infer(run, script, [[0.5, 0.5], [-0.5, 0.5], {"x1": 0.5, "x2": 0.5},
                               {"sample": 5, "seed": 1}])
    assert out[0]["log_density"] > out[1]["log_density"] + 1  # filled vs empty square
    assert out[2]["log_density"] == pytest.approx(out[0]["log_density"], abs=1e-5)
    assert len(out[3]["samples"]) == 5 and len(out[3]["samples"][0]) == 2


@needs_torch
@pytest.mark.parametrize("layers", [
    [("flow.maf", {"reverse": i % 2 == 1}) for i in range(4)],
    [("flow.affine_coupling", {"parity": "even"}), ("flow.permute", {"kind": "random"}),
     ("flow.affine_coupling", {"parity": "even"}), ("flow.permute", {"kind": "random",
                                                                     "seed": 1}),
     ("flow.affine_coupling", {"parity": "even"})],
], ids=["maf", "realnvp-permute"])
def test_csv_density(layers, tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 3000)
    y = np.sin(2 * x) + rng.normal(0, 0.1, 3000)
    z = rng.exponential(1.0, 3000)
    rows = "".join(f"{a},{b},{c}\n" for a, b, c in zip(x, y, z, strict=True))
    (tmp_path / "points.csv").write_text("x,y,z\n" + rows)
    data = chain("3", layers, [
        n("data", "data.csv", path=str(tmp_path / "points.csv"), feature_columns="x,y,z"),
        n("opt", "train.adam", lr=0.003), n("tr", "train.trainer", epochs=20, batch_size=128)])
    assert not [m for m in messages(data) if "Set the Input" in m]
    run, script = _run(data, tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["nll"] < metrics["baseline_nll"] - 0.3     # the curve and the skew
    out = _infer(run, script, [{"x": 0.0, "y": 0.0, "z": 0.5}, {"sample": 2}])
    assert math.isfinite(out[0]["log_density"]) and len(out[1]["samples"][0]) == 3


@needs_torch
def test_flow_run_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(_short(SAMPLES[1], 3, 1000)))
        status = mgr.wait(run_id, 900)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        assert mgr.history.epochs(run_id)
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["task"] == "density_estimation" and meta["input_kind"] == "density"
    finally:
        api.set_manager(None)
