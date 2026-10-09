"""Graphical models: network layout, CPD tables, rules, runtime scores, generated pgmpy /
hmmlearn scripts (textbook posteriors, structure learning, EM, HMMs), serving and editors."""
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
from ai_made_easy.core.fixes import fix_for_issue
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.pgm import network
from ai_made_easy.core.runner.manager import resolve_framework
from ai_made_easy.core.tasks import task_of

HAS_PGMPY = importlib.util.find_spec("pgmpy") is not None
HAS_HMM = importlib.util.find_spec("hmmlearn") is not None
needs_pgmpy = pytest.mark.skipif(not HAS_PGMPY, reason="needs pgmpy (probabilistic extra)")
needs_hmm = pytest.mark.skipif(not HAS_HMM, reason="needs hmmlearn (probabilistic extra)")
T = json.dumps
G_TABLE = [[0.3, 0.9, 0.05, 0.5], [0.4, 0.08, 0.25, 0.3], [0.3, 0.02, 0.7, 0.2]]


def var(nid: str, name: str, states: str, table=None, **params) -> dict:
    p = {"name": name, "states": states, **params}
    if table is not None:
        p["cpd"] = T(table)
    return {"id": nid, "type": "pgm.variable", "params": p, "position": [0, 0]}


def node(nid: str, type_id: str, **params) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": [0, 0]}


def student(extra: tuple = (), tables: bool = True, name: str = "student") -> dict:
    t = (lambda x: x) if tables else (lambda x: None)
    nodes = [var("d", "D", "easy, hard", t([[0.6], [0.4]])),
             var("i", "I", "low, high", t([[0.7], [0.3]])),
             var("g", "G", "A, B, C", t(G_TABLE)),
             var("l", "L", "weak, strong", t([[0.1, 0.4, 0.99], [0.9, 0.6, 0.01]])),
             var("s", "S", "low, high", t([[0.95, 0.2], [0.05, 0.8]])), *extra]
    edges = [("d", "g"), ("i", "g"), ("g", "l"), ("i", "s")]
    return {"name": name, "nodes": nodes,
            "edges": [{"from": f"{a}/out", "to": f"{b}/parents"} for a, b in edges]}


def issues(data: dict) -> list[str]:
    return [i.message for i in Graph.from_dict(data).validate()]


# ================================================================ family / layout

def test_family_task_and_framework():
    graph = Graph.from_dict(student())
    assert family_of(graph).id == "pgm" and task_of(graph).id == "probabilistic_inference"
    assert resolve_framework(graph) == "pgmpy"
    assert api.describe_design(student())["task"]["id"] == "probabilistic_inference"
    hmm = {"name": "h", "nodes": [node("h", "pgm.hmm"), node("d", "data.regime_series")],
           "edges": []}
    assert task_of(Graph.from_dict(hmm)).id == "regime_detection"


@pytest.mark.parametrize("sample", ["student_network.json", "structure_learning.json",
                                    "market_regimes_hmm.json"])
def test_samples_validate(sample):
    graph = Graph.from_dict(api.read_sample(sample))
    assert graph.validate() == [] and family_of(graph).id == "pgm"


def test_cpd_layout_orders_parents_and_fills_uniform():
    graph = Graph.from_dict(student(tables=False))
    info = network.layout(graph, "g")
    assert info["parents"] == ["D", "I"] and info["rows"] == ["A", "B", "C"]
    assert info["columns"] == ["D=easy, I=low", "D=easy, I=high", "D=hard, I=low",
                               "D=hard, I=high"]
    assert info["values"][0] == [pytest.approx(1 / 3, abs=1e-5)] * 4 and info["learned"]
    filled = network.layout(Graph.from_dict(student()), "g")
    assert filled["values"] == G_TABLE and filled["fits"]
    data = student()
    data["nodes"][0]["params"]["states"] = ""
    assert "set the states of parent 'D' first" in network.layout(Graph.from_dict(data),
                                                                  "g")["error"]


def test_table_parsing_and_normalising():
    assert network.parse_table("") is None
    assert network.parse_table("[0.2, 0.8]") == [[0.2], [0.8]]
    with pytest.raises(ValueError, match="valid JSON"):
        network.parse_table("[[0.2,")
    assert json.loads(network.normalized("[[2, 0], [2, 0]]")) == [[0.5, 0.5], [0.5, 0.5]]


# ================================================================ rules

def test_table_rules_and_normalize_fix():
    data = student()
    data["nodes"][2]["params"]["cpd"] = T([[0.3, 0.9, 0.05, 0.5], [0.4, 0.08, 0.25, 0.3],
                                           [0.4, 0.02, 0.7, 0.2]])
    graph = Graph.from_dict(data)
    issue = next(i for i in graph.validate() if "sums to 1.1" in i.message)
    fixed = fix_for_issue(graph, issue)
    assert fixed and fixed[0] == "Normalize the table"
    assert not [i for i in fixed[2].validate() if i.severity == "error"]
    data["nodes"][2]["params"]["cpd"] = T([[0.5, 0.5], [0.5, 0.5]])
    assert any("2 row(s) but G has 3" in m for m in issues(data))


def test_structure_rules():
    cyclic = student()
    cyclic["edges"].append({"from": "l/out", "to": "d/parents"})
    assert any("must be acyclic" in m and "→" in m for m in issues(cyclic))
    cyclic["nodes"].append(node("m", "pgm.model", kind="markov_network"))
    cyclic["nodes"].append(node("data", "data.network_sample"))
    assert not any("acyclic" in m or "cycle" in m for m in issues(cyclic))
    dup = student()
    dup["nodes"][1]["params"]["name"] = "D"
    assert any("two variables are named 'D'" in m for m in issues(dup))
    lonely = student((var("x", "X", "a, b", [[0.5], [0.5]]),))
    assert any("X is not connected" in m for m in issues(lonely))


def test_data_and_learning_rules(tmp_path):
    no_tables = student(tables=False)
    assert any("no probability table and there is no dataset" in m for m in issues(no_tables))
    hidden = student((node("data", "data.network_sample"),
                      node("pl", "pgm.parameter_learning", estimator="mle")), tables=False)
    hidden["nodes"][1]["params"]["latent"] = True
    msgs = issues(hidden)
    assert any("learn it with the em estimator" in m for m in msgs)
    assert any("Network Sample draws rows" in m for m in msgs)
    assert any("up to relabelling" in m for m in msgs)
    (tmp_path / "s.csv").write_text("D,I,G\neasy,low,A\n")
    csv = student((node("data", "data.csv", path=str(tmp_path / "s.csv")),), tables=False)
    assert any("no data column for L, S" in m for m in issues(csv))
    structure = student((node("sl", "pgm.structure_learning"),), tables=False)
    assert any("structure learning needs data" in m for m in issues(structure))


def test_query_rules():
    bad = student((node("q", "pgm.query", variables="I, Z", evidence="G=E, I=low"),))
    msgs = issues(bad)
    assert any("query variable 'Z'" in m for m in msgs)
    assert any("G has no state 'E'" in m for m in msgs)
    assert any("I is both queried and observed" in m for m in msgs)
    dyn = student((node("m", "pgm.model", kind="dynamic_bn"),
                   node("q", "pgm.query", variables="G", evidence="G[t-1]=A")))
    msgs = issues(dyn)
    assert any("name the time step: G[t]" in m for m in msgs)
    assert not any("evidence variable" in m for m in msgs)


def test_naive_bayes_and_hmm_rules():
    nb = student((node("m", "pgm.model", kind="naive_bayes", class_variable="Q"),))
    assert any("set class_variable" in m for m in issues(nb))
    hmm = student((node("h", "pgm.hmm"),))
    assert any("Hidden Markov Model stands alone" in m for m in issues(hmm))


# ================================================================ runtime

@pytest.fixture(scope="module")
def rt():
    from ai_made_easy.core.pgm.runtime import namespace

    return namespace()


def test_forward_sample_matches_tables(rt):
    from ai_made_easy.core.pgm.template import design

    spec = design(Graph.from_dict(student()))
    rows = rt["forward_sample"](spec["variables"], 20000, 0)
    assert (rows["I"] == "high").mean() == pytest.approx(0.3, abs=0.015)
    sub = rows[(rows["D"] == "easy") & (rows["I"] == "high")]
    assert (sub["G"] == "A").mean() == pytest.approx(0.9, abs=0.02)


def test_scores(rt):
    assert rt["shd"]([("A", "B"), ("B", "C")], [("B", "A"), ("B", "C"), ("A", "C")]) == 2
    assert rt["skeleton_errors"]([("A", "B")], [("B", "A")]) == 0
    assert rt["best_label_accuracy"]([0, 0, 1, 1], [1, 1, 0, 0], 2) == 1.0
    frame = rt["regime_series"](300, 2, 0.95, 0)
    assert set(frame.columns) == {"value", "regime"} and frame["regime"].nunique() == 2


# ================================================================ scripts

def _run(data: dict, tmp_path: Path) -> tuple[Path, str, Path]:
    from ai_made_easy.core.codegen import export_training

    graph = Graph.from_dict(data)
    script = export_training(graph, resolve_framework(graph), tmp_path)
    proc = subprocess.run([sys.executable, script.name], cwd=tmp_path, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    return tmp_path, proc.stdout, script


def _infer(folder: Path, script: Path, items: list) -> list:
    code = (f"import sys, json; sys.path.insert(0, '.'); import {script.stem} as m; "
            f"m.load_predictor('.'); print('OUT ' + json.dumps(m.infer({items!r})))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True,
                          text=True, timeout=300, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert "OUT " in proc.stdout, proc.stderr[-3000:]
    return json.loads(proc.stdout.split("OUT ", 1)[1])


@needs_pgmpy
def test_student_network_textbook_posteriors(tmp_path):
    data = student((node("q1", "pgm.query", variables="I", evidence="G=C"),
                    node("q2", "pgm.query", variables="I", evidence="G=C, S=high"),
                    node("q3", "pgm.query", variables="G, L", evidence="I=high", kind="map")))
    run, _out, script = _run(data, tmp_path)
    results = json.loads((run / "posteriors.json").read_text())
    assert results[0]["posteriors"]["I"]["high"] == pytest.approx(0.0789, abs=1e-4)
    assert results[1]["posteriors"]["I"]["high"] == pytest.approx(0.5783, abs=1e-4)
    assert results[2]["map"] == {"G": "A", "L": "strong"}
    assert (run / "eval_samples" / "network.png").exists()
    assert "P(G | D, I)" in (run / "eval_samples" / "report.txt").read_text()
    out = _infer(run, script, [{"G": "C"}, {"evidence": {"I": "high"}, "kind": "map",
                                            "variables": ["G"]}])
    assert out[0]["posteriors"]["I"]["high"] == pytest.approx(0.0789, abs=1e-4)
    assert out[1]["map"] == {"G": "A"}


@needs_pgmpy
def test_structure_learning_recovers_the_skeleton(tmp_path):
    data = student((node("rows", "data.network_sample", n_rows=5000, seed=1),
                    node("sl", "pgm.structure_learning")))
    run, stdout, _s = _run(data, tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["skeleton_errors"] == 0 and metrics["shd"] <= 1
    assert len(json.loads((run / "learned_edges.json").read_text())) == 4
    assert "learned 4 edge(s)" in stdout


@needs_pgmpy
@pytest.mark.parametrize("extra, check", [
    ((node("pl", "pgm.parameter_learning", estimator="bayesian"),
      node("inf", "pgm.inference", method="sampling", samples=20000),
      node("m", "pgm.model", class_variable="L")),
     lambda m: m["accuracy"] > m["majority_baseline"] + 0.2),
    ((node("inf", "pgm.inference", method="belief_propagation"),),
     lambda m: m["test_log_likelihood"] > -3.0),
    ((node("m", "pgm.model", kind="naive_bayes", class_variable="G"),),
     lambda m: m["accuracy"] > m["majority_baseline"]),
], ids=["bayesian_sampling", "belief_propagation", "naive_bayes"])
def test_learning_from_data(extra, check, rt, tmp_path):
    from ai_made_easy.core.pgm.template import design

    rows = rt["forward_sample"](design(Graph.from_dict(student()))["variables"], 3000, 2)
    rows.to_csv(tmp_path / "rows.csv", index=False)
    data = student((node("data", "data.csv", path=str(tmp_path / "rows.csv")), *extra),
                   tables=False)
    run, _o, _s = _run(data, tmp_path / "run")
    assert check(json.loads((run / "metrics.json").read_text()))


@needs_pgmpy
def test_markov_dynamic_gaussian_and_em(tmp_path, rt):
    import pandas as pd

    from ai_made_easy.core.pgm.template import design

    rows = rt["forward_sample"](design(Graph.from_dict(student()))["variables"], 2000, 3)
    rows.to_csv(tmp_path / "rows.csv", index=False)
    csv = node("data", "data.csv", path=str(tmp_path / "rows.csv"))
    markov = student((csv, node("m", "pgm.model", kind="markov_network"),
                      node("q", "pgm.query", variables="I", evidence="G=C")), tables=False)
    markov["edges"].append({"from": "l/out", "to": "s/parents"})       # a loop
    run, _o, _s = _run(markov, tmp_path / "mn")
    assert np.isfinite(json.loads((run / "metrics.json").read_text())["test_log_likelihood"])
    hidden = student((csv, node("pl", "pgm.parameter_learning", estimator="em",
                                iterations=50)), tables=False)
    hidden["nodes"][1]["params"]["latent"] = True
    run, _o, _s = _run(hidden, tmp_path / "em")
    assert "I" in json.loads((run / "posteriors.json").read_text())[0]["posteriors"]
    rng = np.random.default_rng(0)
    x = [0]
    for _ in range(2999):
        x.append(x[-1] if rng.random() < 0.9 else 1 - x[-1])
    x = np.array(x)
    pd.DataFrame({"X": x, "O": np.where(rng.random(3000) < 0.85, x, 1 - x)}).to_csv(
        tmp_path / "dbn.csv", index=False)
    dbn = {"name": "dbn", "nodes": [var("x", "X", "0, 1", lagged_parents="X"),
                                    var("o", "O", "0, 1"),
                                    node("data", "data.csv", path=str(tmp_path / "dbn.csv")),
                                    node("m", "pgm.model", kind="dynamic_bn"),
                                    node("q", "pgm.query", variables="X[t]",
                                         evidence="X[t-1]=1")],
           "edges": [{"from": "x/out", "to": "o/parents"}]}
    run, _o, _s = _run(dbn, tmp_path / "dbn")
    stay = json.loads((run / "posteriors.json").read_text())[0]["posteriors"]["X[t]"]["1"]
    assert stay == pytest.approx(0.9, abs=0.03)
    x = rng.normal(size=1500)
    y = 1 + 2 * x + rng.normal(scale=0.5, size=1500)
    pd.DataFrame({"x": x, "y": y}).to_csv(tmp_path / "g.csv", index=False)
    gauss = {"name": "g", "nodes": [
        {"id": "x", "type": "pgm.gaussian", "params": {"name": "x"}, "position": [0, 0]},
        {"id": "y", "type": "pgm.gaussian", "params": {"name": "y"}, "position": [0, 0]},
        node("data", "data.csv", path=str(tmp_path / "g.csv")),
        node("q", "pgm.query", variables="x", evidence="y=5")],
        "edges": [{"from": "x/out", "to": "y/parents"}]}
    run, _o, script = _run(gauss, tmp_path / "gauss")
    post = json.loads((run / "posteriors.json").read_text())[0]["posteriors"]["x"]
    # x | y=5 with x ~ N(0,1), y = 1 + 2x + N(0, .25): mean 2·4 / (4 + .25) ≈ 1.88
    assert post["mean"] == pytest.approx(8 / 4.25, abs=0.12)
    assert _infer(run, script, [{"y": 5}])[0]["posteriors"]["x"]["mean"] == post["mean"]


@needs_hmm
def test_hmm_finds_regimes_and_serves(tmp_path):
    run, stdout, script = _run(api.read_sample("market_regimes_hmm.json"), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["state_accuracy"] > 0.95 and "restart 5" in stdout
    assert (run / "eval_samples" / "regimes.png").exists()
    out = _infer(run, script, [[0.0, 2.1, 2.0, -2.0]])
    assert len(out[0]["states"]) == 4 and len(out[0]["probabilities"][0]) == 3


@needs_pgmpy
def test_pgm_run_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(api.read_sample("student_network.json")))
        status = mgr.wait(run_id, 600)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        record = mgr.history.get(run_id)
        assert record.framework == "pgmpy" and "edges" in record.final_metrics
        assert api.run_samples(run_id)["texts"]
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["input_kind"] == "evidence" and meta["framework"] == "pgmpy"
        assert "pgmpy" in (tmp_path / "pkg/requirements.txt").read_text()
    finally:
        api.set_manager(None)


# ================================================================ editors

def test_table_layout_endpoint(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    with TestClient(create_app(token="", projects_dir=tmp_path)) as client:
        body = client.post("/api/table_layout", json={"graph": student(), "node": "l"}).json()
        assert body["rows"] == ["weak", "strong"] and body["columns"] == ["G=A", "G=B", "G=C"]
        assert client.post("/api/table_layout", json={"graph": student(),
                                                      "node": "zz"}).status_code >= 400


def test_desktop_table_editor(qtbot):
    from PySide6 import QtWidgets

    from ai_made_easy.ui.features.properties import PropertyInspector
    from ai_made_easy.ui.features.table_editor import TableEditorDialog

    info = network.layout(Graph.from_dict(student()), "g")
    dialog = TableEditorDialog(None, info)
    qtbot.addWidget(dialog)
    assert "every column sums to 1" in dialog.status.text()
    dialog.table.item(0, 0).setText("0.9")
    assert "do not sum to 1" in dialog.status.text()
    dialog._normalize()
    assert "every column sums to 1" in dialog.status.text()
    dialog._save()
    saved = json.loads(dialog.result_text)
    assert len(saved) == 3 and abs(sum(r[0] for r in saved) - 1) < 1e-6
    inspector = PropertyInspector()
    qtbot.addWidget(inspector)
    from ai_made_easy.core.registry import get_registry

    with qtbot.waitSignal(inspector.table_edit_requested) as blocker:
        inspector.show_node("g", get_registry().get("pgm.variable"), {"cpd": T(G_TABLE)}, [])
        button = next(b for b in inspector.findChildren(QtWidgets.QPushButton)
                      if b.text() == "Edit table…")
        button.click()
    assert blocker.args == ["g", "cpd"]
