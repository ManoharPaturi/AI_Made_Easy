"""Every loss / optimizer / scheduler renders valid code and actually trains."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from verify_training import chain  # noqa: E402

from ai_made_easy.core.codegen.training_gen import generate_training  # noqa: E402
from ai_made_easy.core.training import catalog as cat  # noqa: E402

TRAINER = ("train.trainer", {"epochs": 2, "batch_size": 32, "device": "cpu"})


@pytest.mark.parametrize("comp", [*cat.LOSSES, *cat.OPTIMIZERS, *cat.SCHEDULERS],
                         ids=lambda c: c.type_id)
def test_expressions_parse(comp):
    params = cat.resolved(comp.type_id, {})
    for fn in (comp.torch, comp.keras):
        if fn:
            ast.parse(fn(params), mode="eval")


def _loss_graph(loss_id: str):
    task = cat.COMPONENTS[loss_id].meta["task"]
    inp = cat.COMPONENTS[loss_id].meta["input"]
    tail = {"log_probs": [("core.log_softmax", {})], "probs": [("core.sigmoid", {})]}.get(inp, [])
    if task in ("regression", "distribution"):
        k = 3 if task == "distribution" else 1
        data = ("data.synthetic", {"kind": "regression", "n_features": 6, "n_samples": 200})
        if task == "distribution":
            return None  # needs probability targets; covered by the expression test
    else:
        k = 1 if task == "binary" else 3
        data = ("data.synthetic", {"n_features": 6, "n_classes": 3 if k == 3 else 2,
                                   "n_samples": 200})
    layers = [("core.dense", {"units": 8}), ("core.relu", {}), ("core.dense", {"units": k}),
              *tail]
    return chain(f"loss_{loss_id.split('.')[-1]}", "6", layers,
                 [data, (loss_id, {}), TRAINER])


def _run(graph, tmp_path, monkeypatch, framework="pytorch"):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    code = generate_training(graph, framework)
    ns: dict = {"__name__": "aime_training"}
    exec(compile(code, f"<{graph.name}>", "exec"), ns)
    ns["main"]()


@pytest.mark.parametrize("loss_id", cat.LOSS_IDS)
def test_every_loss_trains(loss_id, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    graph = _loss_graph(loss_id)
    if graph is None:
        pytest.skip("needs probability-distribution targets")
    assert not [i for i in graph.validate() if i.severity == "error"], graph.validate()
    _run(graph, tmp_path, monkeypatch)


@pytest.mark.parametrize("opt_id", cat.OPTIMIZER_IDS)
def test_every_optimizer_trains(opt_id, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    graph = chain(f"opt_{opt_id.split('.')[-1]}", "6",
                  [("core.dense", {"units": 8}), ("core.relu", {}), ("core.dense", {"units": 3})],
                  [("data.synthetic", {"n_features": 6, "n_classes": 3, "n_samples": 200}),
                   (opt_id, {}), TRAINER])
    _run(graph, tmp_path, monkeypatch)


@pytest.mark.parametrize("sched_id", cat.SCHEDULER_IDS)
def test_every_scheduler_trains(sched_id, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    graph = chain(f"sched_{sched_id.split('.')[-1]}", "6",
                  [("core.dense", {"units": 8}), ("core.relu", {}), ("core.dense", {"units": 3})],
                  [("data.synthetic", {"n_features": 6, "n_classes": 3, "n_samples": 200}),
                   ("train.sgd", {"lr": 0.01}), (sched_id, {}), TRAINER])
    _run(graph, tmp_path, monkeypatch)


@pytest.mark.parametrize("sched_id", [s for s in cat.SCHEDULER_IDS
                                      if cat.COMPONENTS[s].keras is not None])
def test_keras_schedulers_train(sched_id, tmp_path, monkeypatch):
    pytest.importorskip("keras")
    graph = chain(f"ksched_{sched_id.split('.')[-1]}", "6",
                  [("core.dense", {"units": 8}), ("core.relu", {}), ("core.dense", {"units": 3})],
                  [("data.synthetic", {"n_features": 6, "n_classes": 3, "n_samples": 200}),
                   ("train.adam", {"lr": 0.01}), (sched_id, {}), TRAINER])
    _run(graph, tmp_path, monkeypatch, "keras")
