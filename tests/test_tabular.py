"""Tabular deep learning: column layout of preprocessed tables, rules and Quick Fixes,
exact parameter counts, categorical codes kept unscaled, generated scripts that learn
category interactions, serving with unseen categories."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_made_easy.core import api
from ai_made_easy.core.fixes import fix_for_issue
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.tabular.layout import layout_of

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
SAMPLES = ("tabular_ft_transformer.json", "tabular_resnet_regression.json")


def n(nid: str, type_id: str, **params) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": [0, 0]}


def chain(shape: str, layers: list, extra: list, name: str = "t") -> dict:
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


@pytest.fixture()
def table(tmp_path) -> Path:
    path = tmp_path / "t.csv"
    path.write_text("a,color,b,size,label\n" + "".join(
        f"{i},{['red', 'blue', 'green'][i % 3]},{i * 0.5},{['s', 'm'][i % 2]},{i % 2}\n"
        for i in range(30)))
    return path


def test_layout(table):
    def layout(*steps):
        return layout_of(Graph.from_dict(chain("1", [("core.dense", {"units": 2})], [
            n("data", "data.csv", path=str(table), target_column="label"), *steps])))

    plain = layout()                                   # every text column one-hot encoded
    assert plain.width == 2 + 3 + 2 and plain.categorical == () and plain.unencoded == (
        "color", "size")
    ordinal = layout(n("o", "prep.ordinal_encode"))
    assert (ordinal.width, ordinal.categorical, ordinal.cardinalities) == (4, (1, 3), (3, 2))
    mixed = layout(n("o", "prep.ordinal_encode", columns="size"), n("d", "prep.drop_columns",
                                                                    columns="a"))
    assert (mixed.width, mixed.categorical, mixed.cardinalities) == (2 + 3, (1,), (2,))
    synthetic = layout_of(Graph.from_dict(chain("6", [("core.dense", {"units": 2})], [
        n("data", "data.synthetic_table"), n("o", "prep.ordinal_encode")])))
    assert (synthetic.width, synthetic.categorical, synthetic.cardinalities) == (
        6, (3, 4, 5), (6, 3, 4))


def test_rules_and_quick_fixes(table):
    data = chain("1", [("tab.ft_transformer", {"d": 16, "heads": 4}),
                       ("core.dense", {"units": 2})],
                 [n("data", "data.csv", path=str(table), target_column="label"),
                  n("o", "prep.ordinal_encode")])
    g = Graph.from_dict(data)
    for _ in range(3):
        issue = next((i for i in g.validate() if fix_for_issue(g, i)), None)
        if issue is None:
            break
        g = fix_for_issue(g, issue)[2]
    ft = g.nodes["l0"].params
    assert g.nodes["in"].params["shape"] == "4"
    assert (ft["categorical"], ft["cardinalities"]) == ("1, 3", "3, 2")
    assert [i for i in g.validate() if i.severity != "info"] == []
    onehot = chain("7", [("tab.ft_transformer", {"d": 16, "heads": 4})],
                   [n("data", "data.csv", path=str(table), target_column="label")])
    assert any("add Ordinal Encode" in m for m in messages(onehot))
    filtered = chain("4", [("tab.embedding", {"categorical": "1, 3", "cardinalities": "3, 2"})],
                     [n("data", "data.csv", path=str(table), target_column="label"),
                      n("o", "prep.ordinal_encode"), n("v", "prep.variance_filter")])
    assert any("Variance Filter removes columns" in m for m in messages(filtered))
    bad = chain("4", [("tab.embedding", {"categorical": "1, 9", "cardinalities": "3, 2"})], [])
    assert any("categorical positions must be 0..3" in m for m in messages(bad))
    heads = chain("4", [("tab.ft_transformer", {"d": 30, "heads": 4})], [])
    assert any("divisible by heads" in m for m in messages(heads))
    plain_tt = chain("4", [("tab.tab_transformer", {})], [])
    assert any("only an MLP" in m for m in messages(plain_tt))


@needs_torch
@pytest.mark.parametrize("type_id,params", [
    ("tab.embedding", {"categorical": "1, 3", "cardinalities": "5, 12", "dim": 0}),
    ("tab.resnet", {}),
    ("tab.ft_transformer", {"categorical": "0, 2", "cardinalities": "4, 7", "d": 32,
                            "heads": 4}),
    ("tab.tab_transformer", {"categorical": "0, 2", "cardinalities": "4, 7"}),
    ("tab.tabnet", {}),
])
def test_parameter_counts_match_pytorch(type_id, params):
    from ai_made_easy.core.codegen import class_name_for, generate
    from ai_made_easy.core.summary import summarize

    graph = Graph.from_dict(chain("9", [(type_id, params)], [], name="count"))
    ns: dict = {}
    exec(generate(graph, "pytorch"), ns)  # noqa: S102 — generated by us
    model = ns[class_name_for("count")]()
    assert summarize(graph).total_params == sum(p.numel() for p in model.parameters())


@needs_torch
def test_tabnet_sparse_masks_and_aux_loss():
    import torch

    from ai_made_easy.core.tabular.helpers import TABNET

    ns = {"torch": torch, "nn": torch.nn}
    exec(TABNET, ns)  # noqa: S102
    p = ns["sparsemax"](torch.tensor([[1.0, 0.5, -1.0], [3.0, 0.0, 0.0]]))
    assert p.tolist() == [[0.75, 0.25, 0.0], [1.0, 0.0, 0.0]]
    net = ns["TabNet"](6, steps=2)
    net(torch.randn(32, 6))
    assert float(net.aux_loss()) > 0


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


ROW = {"age": 30, "income": 30000, "tenure": 1.0, "city": "paris", "plan": "team",
       "channel": "web"}


@needs_torch
def test_ft_transformer_learns_category_interactions(tmp_path):
    for sample in SAMPLES:
        assert Graph.from_dict(api.read_sample(sample)).validate() == []
    data = api.read_sample(SAMPLES[0])
    for node in data["nodes"]:
        if node["type"] == "train.trainer":
            node["params"]["epochs"] = 15
    run, script = _run(data, tmp_path)
    source = script.read_text()
    assert "raw = flat[:, [3, 4, 5]].copy()" in source          # codes are not standardised
    assert json.loads((run / "metrics.json").read_text())["accuracy"] > 0.85
    out = _infer(run, script, [ROW, {**ROW, "city": "tokyo"}])  # an unseen category
    assert out[0]["label"] in ("churn", "stay") and out[1]["label"] in ("churn", "stay")


@needs_torch
@pytest.mark.parametrize("layers,epochs,limit", [
    ([("tab.resnet", {"d": 64}), ("core.dense", {"units": 1})], 25, 8.0),
    ([("tab.tabnet", {}), ("core.dense", {"units": 1})], 25, 15.0),
], ids=["resnet", "tabnet"])
def test_regression_beats_the_mean(layers, epochs, limit, tmp_path):
    data = chain("16", layers, [
        n("data", "data.synthetic_table", task="regression"), n("norm", "prep.normalize"),
        n("loss", "train.loss_mse"), n("opt", "train.adam", lr=0.003),
        n("tr", "train.trainer", epochs=epochs, batch_size=128)])
    assert Graph.from_dict(data).validate() == []
    run, script = _run(data, tmp_path)
    assert json.loads((run / "metrics.json").read_text())["mae"] < limit   # mean: ~15.9
    assert isinstance(_infer(run, script, [ROW])[0]["prediction"], float)
