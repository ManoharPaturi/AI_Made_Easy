"""Tasks for every family (3.1 phase 1): classic pipelines resolve to multiclass, binary,
regression, clustering or anomaly detection; LLM workflows and pipelines have tasks too.
Clustering and anomaly-detection recipes train on tables with or without labels, the data
rules speak the design's language, and the project lists show every file's task."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from ai_made_easy.core import api, automl, recipes
from ai_made_easy.core.classic.generate import generate_classic
from ai_made_easy.core.data.lints import data_issues
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.migrate import kind_of, stamp
from ai_made_easy.core.tasks import all_tasks, get_task, task_of

pytest.importorskip("sklearn")


def _classic(estimator: str, data: dict | None = None, meta: dict | None = None) -> Graph:
    nodes = [{"id": "model", "type": estimator, "params": {}, "position": [0, 0]}]
    if data:
        nodes.append({"id": "data", "type": data.pop("type"), "params": data,
                      "position": [0, 90]})
    return Graph.from_dict({"name": "t", "nodes": nodes, "edges": [], "meta": meta or {}})


def _blobs(path, labels: str = ""):  # noqa: ANN001, ANN202
    import pandas as pd

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(np.vstack([rng.normal(0, 1, (95, 3)), rng.normal(8, 1, (5, 3))]),
                         columns=["a", "b", "c"])
    if labels == "binary":
        frame["is_fraud"] = [0] * 95 + [1] * 5
    elif labels == "grades":
        frame["grade"] = rng.integers(0, 4, len(frame))
    frame.to_csv(path, index=False)
    return path


# ================================================================ resolving tasks

def test_classic_designs_have_tasks():
    assert task_of(_classic("ml.kmeans")).id == "clustering"
    assert task_of(_classic("ml.isolation_forest")).id == "anomaly_detection"
    assert task_of(_classic("ml.ridge")).id == "regression"
    assert task_of(_classic("ml.random_forest_classifier")).id == "multiclass"
    binary = {"type": "data.sklearn", "dataset": "breast_cancer"}
    assert task_of(_classic("ml.random_forest_classifier", binary)).id == "binary"
    moons = {"type": "data.synthetic", "kind": "moons"}
    assert task_of(_classic("ml.svc", moons)).id == "binary"
    wizard = {"recipe": {"task": "binary"}}
    assert task_of(_classic("ml.logistic_regression", meta=wizard)).id == "binary"
    clustering = get_task("clustering")
    assert clustering.family == "classic" and "eval.adjusted_rand" in clustering.metrics()
    assert get_task("anomaly_detection").losses() == []


def test_every_sample_has_a_task():
    untasked = []
    for name in api.list_samples()["samples"]:
        graph = Graph.from_dict(api.read_sample(name))
        if task_of(graph) is None:
            untasked.append(name)
    # an architecture-only sample (no loss: nothing to train) is the only exception
    assert untasked == ["skip_connection_mlp.json"]
    tasks = {name: task_of(Graph.from_dict(api.read_sample(name))).id
             for name in ("llm_generation.json", "llm_lora_finetune.json",
                          "llm_rag_assistant.json", "pipeline_distill_quantize.json",
                          "classic_kmeans.json")}
    assert tasks == {"llm_generation.json": "text_generation",
                     "llm_lora_finetune.json": "llm_finetuning",
                     "llm_rag_assistant.json": "retrieval_qa",
                     "pipeline_distill_quantize.json": "multiclass",
                     "classic_kmeans.json": "clustering"}


def test_every_recipe_design_has_its_task():
    for task in all_tasks():
        for recipe in recipes.recipes_for(task.id):
            if recipe.installed:
                assert task_of(recipes.build(recipe, task.id)).id == task.id, recipe.id


# ================================================================ wizard and training

def test_tables_offer_unsupervised_tasks(tmp_path):
    facts = recipes.detect(_blobs(tmp_path / "plain.csv"))
    assert facts.tasks[0] == "regression"
    assert facts.tasks[-2:] == ["clustering", "anomaly_detection"]
    ranked = recipes.recommend("clustering", facts)
    assert [s.recipe.id for s in ranked][0] == "kmeans" and not any(s.errors for s in ranked)
    data = next(n for n in ranked[0].graph["nodes"] if n["type"] == "data.csv")
    assert data["params"]["target_column"] == ""          # no labels: every column is used
    tasks = {t["id"]: t for t in api.wizard_tasks()["tasks"]}
    assert tasks["clustering"]["family_label"] == "Classic ML pipeline"
    assert tasks["clustering"]["automl"] and tasks["anomaly_detection"]["automl"]
    assert tasks["retrieval_qa"]["family_label"] == "LLM workflow"


def _run(graph: Graph, workdir) -> dict:  # noqa: ANN001
    script = workdir / "train.py"
    script.write_text(generate_classic(graph))
    done = subprocess.run([sys.executable, script.name], cwd=workdir, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads((workdir / "metrics.json").read_text())


def test_clustering_and_anomaly_scripts_train(tmp_path):
    plain = recipes.build("kmeans", "clustering", recipes.detect(_blobs(tmp_path / "a.csv")),
                          knobs={"n_clusters": 2})
    assert _run(plain, tmp_path)["silhouette"] > 0.5
    labelled = recipes.detect(_blobs(tmp_path / "b.csv", "binary"))
    detector = recipes.build("isolation_forest", "anomaly_detection", labelled,
                             knobs={"contamination": "0.05"})
    scores = _run(detector, tmp_path)
    assert scores["recall"] == 1.0 and scores["roc_auc"] > 0.95       # labels only evaluate
    grouped = recipes.build("kmeans", "clustering", labelled)     # one cluster per class
    assert grouped.nodes["model"].params["n_clusters"] == 2
    assert _run(grouped, tmp_path)["adjusted_rand"] > 0.9


def test_automl_searches_clusterings(tmp_path, isolated_home):
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory
    from ai_made_easy.core.sweeps import SweepStore

    assert automl.AutoMLSpec("clustering").resolved_metric() == ("silhouette", False)
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    runner = automl.start(mgr, SweepStore(tmp_path / "sweeps"), {
        "task": "clustering", "max_trials": 3, "recipes": ["kmeans", "gaussian_mixture"]})
    record = runner.wait(600)
    assert record.state == "finished" and len(record.trials) == 3
    assert record.best["score"] is not None and record.best["score"] > 0


# ================================================================ rules

def test_data_rules_speak_the_designs_language(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    frame["price"] = rng.normal(size=300) * 100
    frame.to_csv(tmp_path / "prices.csv", index=False)
    facts = recipes.detect(tmp_path / "prices.csv")

    def messages(recipe: str, task: str, facts=facts) -> str:  # noqa: ANN001
        return "\n".join(i.message for i in data_issues(recipes.build(recipe, task, facts)))

    assert "uses a classifier — use a regressor" in messages("random_forest", "multiclass")
    assert "uses a classification loss — use a regression loss" in messages("tabular_mlp",
                                                                            "multiclass")
    grades = recipes.detect(_blobs(tmp_path / "grades.csv", "grades"))
    assert "only scores the clusters" in messages("kmeans", "clustering", grades)
    assert "cannot evaluate the detector" in messages("isolation_forest",
                                                      "anomaly_detection", grades)
    fraud = recipes.detect(_blobs(tmp_path / "fraud.csv", "binary"))
    issues = data_issues(recipes.build("isolation_forest", "anomaly_detection", fraud))
    assert [i.severity for i in issues if "rarer value" in i.message] == ["info"]


# ================================================================ project lists

def test_project_lists_show_classic_tasks(tmp_path):
    sample = api.read_sample("classic_random_forest.json")
    old = {**stamp(sample), "task": None}            # saved by 3.0: no classic tasks yet
    assert kind_of(old) == {"family": "Classic ML pipeline",
                            "task": "Binary classification"}
    pytest.importorskip("PySide6")
    from ai_made_easy.ui.services.project_service import recent_kind

    path = tmp_path / "forest.json"
    path.write_text(json.dumps(old))
    assert recent_kind(path) == "Classic ML pipeline · Binary classification"
    (tmp_path / "broken.json").write_text("{")
    assert recent_kind(tmp_path / "broken.json") == ""
