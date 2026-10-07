"""Classic ML pipelines: generated scikit-learn scripts run, and rules hold."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from verify_classic import SCENARIOS, pipeline, run_graph, sweep_graph  # noqa: E402

from ai_made_easy.core.classic import catalog as ml  # noqa: E402
from ai_made_easy.core.codegen import CodegenError, generate  # noqa: E402
from ai_made_easy.core.targets import RENDERERS  # noqa: E402

pytest.importorskip("sklearn")
pytest.importorskip("pandas")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_classic_scenarios_run(name, tmp_path):
    if name == "xgboost":
        pytest.importorskip("xgboost")
    if name == "images":
        pytest.importorskip("PIL")
    build, expect = SCENARIOS[name]
    run_graph(build(tmp_path), tmp_path, expect)


@pytest.mark.parametrize("est", ml.ESTIMATORS, ids=lambda e: e.type_id)
def test_every_estimator_fits(est, tmp_path):
    if est.package != "scikit-learn":
        pytest.importorskip(est.package)
    run_graph(sweep_graph(est), tmp_path)


def _errors(blocks):
    return [i.message for i in pipeline("x", blocks).validate() if i.severity == "error"]


@pytest.mark.parametrize("blocks, fragment", [
    ([("data.sklearn", {}), ("ml.svc", {}), ("core.dense", {})], "neural-network block"),
    ([("data.sklearn", {}), ("ml.svc", {}), ("ml.ridge", {})], "only one estimator"),
    ([("data.text_csv", {}), ("prep.normalize", {}), ("ml.multinomial_nb", {})],
     "non-negative features"),
    ([("data.sklearn", {}), ("ml.svc", {}), ("ml.hyperparameter_search", {"param_grid": "C 1"})],
     "param_grid"),
    ([("data.sklearn", {"dataset": "diabetes"}), ("ml.select_k_best", {}), ("ml.ridge", {})],
     "does not match the regression"),
    ([("data.sklearn", {}), ("ml.tfidf", {}), ("ml.svc", {})], "needs a text dataset"),
    ([("data.audio_folder", {}), ("ml.svc", {})], "not supported by classic ML"),
    ([("data.sklearn", {}), ("ml.logistic_regression", {"penalty": "l1"})], "does not support"),
])
def test_classic_rules(blocks, fragment):
    assert any(fragment in m for m in _errors(blocks)), _errors(blocks)


def test_valid_classic_pipeline_has_no_errors_and_no_input_requirement():
    assert _errors([("data.sklearn", {}), ("prep.normalize", {}),
                    ("ml.random_forest_classifier", {})]) == []


def test_classic_pipeline_exports_only_as_sklearn():
    g = pipeline("p", [("data.sklearn", {}), ("ml.random_forest_classifier", {})])
    ast.parse(RENDERERS["sklearn_train"](g))
    with pytest.raises(CodegenError, match="scikit-learn pipeline"):
        generate(g, "pytorch")
