"""Graph neural networks: tasks, rules, data generators, generated scripts for node
classification, graph classification and link prediction, serving and deploys."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_made_easy.core import api
from ai_made_easy.core.gnn import runtime
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.tasks import task_of

HAS_PYG = importlib.util.find_spec("torch_geometric") is not None
needs_pyg = pytest.mark.skipif(not HAS_PYG, reason="needs torch_geometric")
SAMPLES = {"graph_node_classification.json": "node_classification",
           "graph_classification_gin.json": "graph_classification",
           "graph_link_prediction.json": "link_prediction"}


def n(nid: str, type_id: str, **params) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": [0, 0]}


def chain(shape: str, layers: list, extra: list, name: str = "g") -> dict:
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


# ================================================================ tasks / rules / data

def test_samples_and_tasks():
    for sample, task in SAMPLES.items():
        graph = Graph.from_dict(api.read_sample(sample))
        assert task_of(graph).id == task and task_of(graph).trainer_kind == "graph"
        if HAS_PYG:
            assert graph.validate() == []


def test_rules(tmp_path):
    sbm = n("data", "data.synthetic_graph", feature_dim=16, communities=4)
    wrong_in = chain("8", [("graph.gcn", {"out_channels": 4})], [sbm])
    assert any("Set the Input shape to '16'" in m for m in messages(wrong_in))
    classes = chain("16", [("graph.gcn", {"out_channels": 8}), ("core.dense", {"units": 3})],
                    [sbm])
    assert any("set units to 4" in m for m in messages(classes))
    pooled = chain("16", [("graph.gcn", {"out_channels": 8}), ("graph.global_pool", {}),
                          ("core.dense", {"units": 4})], [sbm])
    assert any("is one graph: remove Global Pooling" in m for m in messages(pooled))
    motifs = n("data", "data.synthetic_graphs", feature_dim=8)
    unpooled = chain("8", [("graph.gin", {}), ("core.dense", {"units": 3})], [motifs])
    assert any("add Global Pooling" in m for m in messages(unpooled))
    late = chain("8", [("graph.gin", {}), ("graph.global_pool", {}), ("graph.gcn", {}),
                       ("core.dense", {"units": 3})], [motifs])
    assert any("comes after Global Pooling" in m for m in messages(late))
    link = chain("16", [("graph.sage", {}), ("graph.topk_pool", {})],
                 [sbm, n("dec", "graph.link_decoder")])
    assert any("link prediction needs an embedding for every node" in m
               for m in messages(link))
    plain = chain("16", [("core.dense", {"units": 4})], [sbm])
    assert any("no message-passing layer" in m for m in messages(plain))
    tabular = chain("16", [("graph.gcn", {"out_channels": 4})],
                    [n("data", "data.sklearn", dataset="iris")])
    assert any("is not a graph" in m for m in messages(tabular))
    deep = chain("16", [("graph.gcn", {"out_channels": 16})] * 7 + [("graph.gcn",
                                                                     {"out_channels": 4})],
                 [sbm])
    assert any("over-smooth" in m for m in messages(deep))
    (tmp_path / "nodes.csv").write_text("id,a,b,label\n1,0.1,1,x\n2,0.2,0,y\n3,0.3,1,x\n")
    (tmp_path / "edges.csv").write_text("source,target\n1,2\n")
    csv = chain("2", [("graph.gcn", {"out_channels": 2})],
                [n("data", "data.graph_csv", nodes_path=str(tmp_path / "nodes.csv"),
                   edges_path=str(tmp_path / "edges.csv"))])
    found = messages(csv)
    assert any("1 of 3 nodes have no edges" in m for m in found)
    assert not any("Set the Input shape" in m for m in found)


def test_generators():
    ns = runtime.namespace()
    g = ns["synthetic_graph"]({"n_nodes": 400, "communities": 4, "p_in": 0.1, "p_out": 0.005,
                               "feature_dim": 5, "feature_signal": 1.0, "seed": 1})
    ei, y = g["edge_index"], g["y"]
    assert g["x"].shape == (400, 5) and ei.shape[0] == 2
    assert {(a, b) for a, b in ei.T.tolist()} == {(b, a) for a, b in ei.T.tolist()}
    assert (y[ei[0]] == y[ei[1]]).mean() > 0.8                 # mostly within communities
    graphs, classes = ns["synthetic_graphs"]({"n_graphs": 30, "min_nodes": 10, "max_nodes": 14,
                                              "feature_dim": 6, "seed": 0})
    assert classes == ["cycle", "house", "star"] and len(graphs) == 30
    for item in graphs:
        n_nodes = len(item["x"])
        edges = item["edge_index"].shape[1] // 2
        cyclic = edges - (n_nodes - 1)                       # independent cycles
        assert cyclic == {"cycle": 1, "house": 2, "star": 0}[classes[item["y"]]]
    scores = ns["roc_auc"](runtime.namespace()["np"].array([0.9, 0.8, 0.3, 0.1]),
                           runtime.namespace()["np"].array([1, 0, 1, 0]))
    assert scores == pytest.approx(0.75)


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


def _sample(name: str, epochs: int) -> dict:
    data = api.read_sample(name)
    for node in data["nodes"]:
        if node["type"] == "train.trainer":
            node["params"]["epochs"] = epochs
    return data


@needs_pyg
def test_node_classification_uses_the_graph(tmp_path):
    run, script = _run(_sample("graph_node_classification.json", 60), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["accuracy"] > 0.8                       # features alone give ~0.4
    out = _infer(run, script, [{"nodes": [0, 5]}, {"x": [[0.0] * 16] * 3,
                                                    "edges": [[0, 1], [1, 2]]}])
    assert [r["node"] for r in out[0]["nodes"]] == [0, 5]
    assert len(out[1]["nodes"]) == 3 and out[1]["nodes"][0]["label"].startswith("community")


@needs_pyg
def test_graph_classification_finds_motifs(tmp_path):
    run, script = _run(_sample("graph_classification_gin.json", 30), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["accuracy"] > 0.9 and metrics["baseline_accuracy"] < 0.45
    ring = {"x": [[0, 0, 1, 0, 0, 0, 0, 0]] * 6, "edges": [[i, (i + 1) % 6] for i in range(6)]}
    assert _infer(run, script, [ring])[0]["label"] in ("cycle", "house", "star")


@needs_pyg
@pytest.mark.parametrize("decoder", ["dot", "mlp"])
def test_link_prediction(decoder, tmp_path):
    data = _sample("graph_link_prediction.json", 60)
    for node in data["nodes"]:
        if node["type"] == "graph.link_decoder":
            node["params"]["decoder"] = decoder
    run, script = _run(data, tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["roc_auc"] > 0.62                       # communities cap it near 0.76
    out = _infer(run, script, [{"pairs": [[0, 1], [2, 3]]}])
    assert all(0 <= p[2] <= 1 for p in out[0]["pairs"])


@needs_pyg
def test_topk_attention_pooling_and_csv(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    rows, edges = [], []
    for i in range(120):
        group = i % 2
        rows.append(f"n{i},{rng.normal(group, 1):.3f},{rng.normal(0, 1):.3f},{'ab'[group]}")
    for i in range(120):
        for j in range(i + 1, 120):
            if (i % 2 == j % 2 and rng.random() < 0.08) or rng.random() < 0.005:
                edges.append(f"n{i},n{j}")
    (tmp_path / "nodes.csv").write_text("id,f1,f2,label\n" + "\n".join(rows) + "\n")
    (tmp_path / "edges.csv").write_text("source,target\n" + "\n".join(edges) + "\n")
    data = chain("2", [("graph.gat", {"out_channels": 8, "heads": 2}), ("core.elu", {}),
                       ("graph.gat", {"out_channels": 2, "heads": 1, "concat": False})],
                 [n("data", "data.graph_csv", nodes_path=str(tmp_path / "nodes.csv"),
                    edges_path=str(tmp_path / "edges.csv")),
                  n("opt", "train.adam", lr=0.01), n("tr", "train.trainer", epochs=60, batch_size=32)])
    assert Graph.from_dict(data).validate() == []
    run, _script = _run(data, tmp_path)
    assert json.loads((run / "metrics.json").read_text())["accuracy"] > 0.75
    pooled = chain("8", [("graph.graph_conv", {"out_channels": 32}), ("core.relu", {}),
                         ("graph.sag_pool", {"ratio": 0.7}), ("graph.sage", {"out_channels": 32}),
                         ("graph.attention_pool", {}), ("core.dense", {"units": 3})],
                   [n("data", "data.synthetic_graphs", n_graphs=240),
                    n("tr", "train.trainer", epochs=15, batch_size=32)])
    run2 = tmp_path / "pooled"
    run2.mkdir()
    run2, _ = _run(pooled, run2)
    assert json.loads((run2 / "metrics.json").read_text())["accuracy"] > 0.5


@needs_pyg
def test_graph_run_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(_sample("graph_node_classification.json", 5)))
        status = mgr.wait(run_id, 900)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        assert mgr.history.epochs(run_id)
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["task"] == "node_classification" and meta["input_kind"] == "graph"
    finally:
        api.set_manager(None)
