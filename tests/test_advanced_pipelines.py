"""Advanced pipelines (3.1 phase 4): discriminative learning rates in Fine-tune,
quantization-aware training, stacking ensembles on out-of-fold predictions, and
distillation from scikit-learn teachers, on the digits pipeline."""
from __future__ import annotations

import copy
import importlib.util

import pytest

from ai_made_easy.core import api
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.pipelines.runner import PipelineRunner, PipelineStore

DISTILL = "pipeline_distill_quantize.json"
needs_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                 reason="needs torch")
needs_sklearn = pytest.mark.skipif(importlib.util.find_spec("sklearn") is None,
                                   reason="needs scikit-learn")
FOREST = {"name": "digits_forest", "edges": [], "nodes": [
    {"id": "data", "type": "data.sklearn", "params": {"dataset": "digits"}, "position": [0, 0]},
    {"id": "forest", "type": "ml.random_forest_classifier", "params": {"n_estimators": 200},
     "position": [0, 90]},
    {"id": "acc", "type": "eval.accuracy", "params": {}, "position": [0, 180]}]}


def stage(sid: str, type_id: str, **params) -> dict:
    return {"id": sid, "type": type_id, "params": params, "position": [0, 0]}


def pipeline(nodes: list, edges: list, epochs: int | None = None) -> dict:
    meta = copy.deepcopy(api.read_sample(DISTILL)["meta"])
    meta["designs"]["forest"] = copy.deepcopy(FOREST)
    if epochs:
        for design in meta["designs"].values():
            for node in design["nodes"]:
                if node["type"] == "train.trainer":
                    node["params"]["epochs"] = epochs
    return {"name": "p", "meta": meta, "nodes": nodes,
            "edges": [{"from": f"{a}/out", "to": f"{b}/in"} for a, b in edges]}


def messages(data: dict) -> list[str]:
    return [i.message for i in Graph.from_dict(data).validate()]


def run(tmp_path, data: dict) -> dict:  # noqa: ANN001
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    runner = PipelineRunner(mgr, PipelineStore(tmp_path / "pipelines"), Graph.from_dict(data))
    runner.start()
    record = runner.wait(1200)
    assert record.state == "finished", {k: v["message"][-600:] for k, v in
                                        record.stages.items()}
    return record.stages


# ================================================================ rules

def test_rules_for_the_new_stages():
    train = stage("t", "pipeline.train", design="embedded:teacher")
    cv = [stage(f"cv{i}", "pipeline.cross_validate", design=f"embedded:{d}", k=3)
          for i, d in enumerate(("teacher", "student"))]
    stack = stage("e", "pipeline.ensemble", method="stack")
    found = messages(pipeline([*cv, stack], [("cv0", "e"), ("cv1", "e")]))
    assert any("set its refit to true" in m for m in found)
    for node in cv:
        node["params"]["refit"] = True
    assert messages(pipeline([*cv, stack], [("cv0", "e"), ("cv1", "e")])) == []
    cv[1]["params"]["k"] = 4
    assert any("same number of folds" in m for m in messages(
        pipeline([*cv, stack], [("cv0", "e"), ("cv1", "e")])))
    assert any("replace Train with a Cross-validate stage" in m for m in messages(
        pipeline([train, cv[0], stack], [("t", "e"), ("cv0", "e")])))

    tune = stage("f", "pipeline.finetune", lr_decay=0.5, freeze="all_but_last")
    issues = Graph.from_dict(pipeline([train, tune], [("t", "f")])).validate()
    assert [i.severity for i in issues if "lr_decay changes nothing" in i.message] == ["info"]
    assert messages(pipeline([train, stage("q", "pipeline.qat")], [("t", "q")])) == []


@needs_sklearn
def test_classic_teachers_are_allowed_for_distillation():
    forest = stage("t", "pipeline.train", design="embedded:forest")
    distill = stage("d", "pipeline.distill", student="embedded:student")
    assert messages(pipeline([forest, distill], [("t", "d")])) == []
    data = pipeline([forest, distill], [("t", "d")])
    data["meta"]["designs"]["forest"]["nodes"][0]["params"]["dataset"] = "iris"
    assert any("same dataset as the classic teacher" in m for m in messages(data))
    data["meta"]["designs"]["forest"]["nodes"][1]["type"] = "ml.kmeans"
    assert any("must be a classifier or a regressor" in m for m in messages(data))
    # other stages still need a PyTorch network
    assert any("supervised PyTorch network" in m for m in messages(
        pipeline([forest, stage("q", "pipeline.qat")], [("t", "q")])))


# ================================================================ end to end

@needs_torch
def test_discriminative_fine_tune_then_quantization_aware_training(tmp_path, isolated_home):
    from ai_made_easy.core.deploy.package import build_package

    stages = run(tmp_path, pipeline(
        [stage("t", "pipeline.train", design="embedded:teacher"),
         stage("f", "pipeline.finetune", lr_decay=0.3, layer_groups=2, epochs=2),
         stage("q", "pipeline.qat", epochs=2),
         stage("ev", "pipeline.evaluate")],
        [("t", "f"), ("f", "q"), ("q", "ev")], epochs=8))
    assert stages["f"]["metrics"]["accuracy"] > 0.9
    q = stages["q"]["metrics"]
    # QAT stays close to the float model (its extra epochs may even gain), at a quarter of
    # the size. The test split is 358 digits, so one point is 3–4 of them: 2 points leaves
    # room for platform numerics (macOS once landed 1.1 points under)
    assert q["accuracy"] >= q["accuracy_float"] - 0.02
    assert q["size_mb"] < 0.4 * q["size_mb_float"]
    assert {"accuracy_dynamic_int8", "size_mb_dynamic_int8"} <= set(q)
    assert stages["ev"]["metrics"]["accuracy"] == pytest.approx(q["accuracy"], abs=1e-6)
    folder = stages["q"]["output"]["model_dir"]
    result = build_package(folder, tmp_path / "pkg")
    assert result.verified, result.log[-2000:]


@needs_torch
@needs_sklearn
def test_stacking_learns_from_out_of_fold_predictions(tmp_path, isolated_home):
    stages = run(tmp_path, pipeline(
        [stage("a", "pipeline.cross_validate", design="embedded:teacher", k=3, refit=True),
         stage("b", "pipeline.cross_validate", design="embedded:student", k=3, refit=True),
         stage("e", "pipeline.ensemble", method="stack")],
        [("a", "e"), ("b", "e")], epochs=6))
    assert "cv_accuracy" in stages["a"]["metrics"] and stages["a"]["output"]["model_dir"]
    m = stages["e"]["metrics"]
    best = max(m["member0_accuracy"], m["member1_accuracy"])
    assert m["accuracy"] >= best - 0.02              # beats the best member, or matches it
    assert "average_accuracy" in m


@needs_torch
@needs_sklearn
def test_distilling_a_random_forest(tmp_path, isolated_home):
    stages = run(tmp_path, pipeline(
        [stage("t", "pipeline.train", design="embedded:forest"),
         stage("d", "pipeline.distill", student="embedded:student", alpha=0.3, epochs=8)],
        [("t", "d")]))
    m = stages["d"]["metrics"]
    assert m["teacher_coverage"] == 1.0              # every training row got soft labels
    assert m["accuracy"] > 0.9


@needs_sklearn
def test_the_advanced_sample_is_valid():
    graph = Graph.from_dict(api.read_sample("pipeline_advanced.json"))
    assert graph.validate() == []
    kinds = {n.type_id for n in graph.nodes.values()}
    assert {"pipeline.qat", "pipeline.ensemble", "pipeline.distill"} <= kinds
    designs = api.stage_designs(graph)
    assert designs["forest"]["kind"] == "Classic ML pipeline · Multi-class classification"
    assert [b["name"] for b in designs["forest"]["blocks"]][1] == "Random Forest Classifier"
