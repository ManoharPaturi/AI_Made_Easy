"""Training-script generator: spec folding, rendering contract, rejections, live run."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai_made_easy.core.codegen import CodegenError, export_training
from ai_made_easy.core.codegen.training_gen import collect_spec, generate_training
from ai_made_easy.core.graph import Edge, Graph, NodeInstance

SAMPLES = Path(__file__).parent.parent / "samples"
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from verify_training import chain  # noqa: E402

MLP = [("core.dense", {"units": 16}), ("core.relu", {}), ("core.dense", {"units": 3})]


def load(name: str) -> Graph:
    return Graph.from_dict(json.loads((SAMPLES / name).read_text()))


def test_spec_collects_config_blocks():
    spec = collect_spec(load("mlp_mnist.json"))
    assert spec.optimizer["kind"] == "train.adam"
    assert spec.optimizer["lr"] == 0.001
    assert spec.loss["kind"] == "train.loss_cross_entropy"
    assert spec.trainer["epochs"] == 10
    assert spec.trainer["early_stopping_patience"] == 3
    assert "accuracy" in spec.metric_keys
    assert spec.task == "multiclass" and spec.modality == "tabular"  # synthetic default
    assert collect_spec(load("cifar10_cnn_augmented.json")).modality == "image"


def test_spec_defaults_when_no_config_blocks():
    g = chain("bare", "6", MLP, [])
    spec = collect_spec(g)
    assert spec.dataset["block"] == "data.synthetic"
    assert spec.dataset["n_features"] == 6 and spec.dataset["n_classes"] == 3
    assert spec.optimizer["kind"] == "train.adam"
    assert spec.metric_keys == ["accuracy"]


def test_duplicate_optimizer_rejected():
    g = chain("dup", "6", MLP, [("train.adam", {}), ("train.sgd", {})])
    with pytest.raises(CodegenError, match="at most one optimizer"):
        collect_spec(g)


def test_task_inference():
    binary = chain("b", "6", MLP[:2] + [("core.dense", {"units": 1})],
                   [("train.loss_bce_logits", {})])
    assert collect_spec(binary).task == "binary"
    multilabel = chain("ml", "6", MLP, [("train.loss_bce_logits", {})])
    assert collect_spec(multilabel).task == "multilabel"
    reg = chain("r", "6", MLP[:2] + [("core.dense", {"units": 1})], [("train.loss_mse", {})])
    spec = collect_spec(reg)
    assert spec.task == "regression" and spec.metric_keys == ["mae", "r2"]


def test_metrics_for_the_wrong_task_are_skipped_with_a_note():
    g = chain("r", "6", MLP[:2] + [("core.dense", {"units": 1})],
              [("train.loss_mse", {}), ("eval.accuracy", {}), ("eval.rmse", {})])
    spec = collect_spec(g)
    assert spec.metric_keys == ["rmse"]
    assert any("Accuracy does not apply" in w for w in spec.warnings)


def test_pytorch_script_contract():
    code = generate_training(load("cifar10_cnn_augmented.json"), "pytorch")
    ast.parse(code)
    for fragment in ("CHECKPOINT = ", "INPUT_SHAPE = (3, 32, 32)", "def make_loaders",
                     "def fit(", "def compute_metrics", "def dump_test_results",
                     'if __name__ == "__main__":', "v2.ToImage()"):
        assert fragment in code, fragment


def test_keras_script_contract():
    code = generate_training(load("regression_mlp.json"), "keras")
    ast.parse(code)
    assert "def build_model() -> keras.Model:" in code
    assert "class EpochReport(keras.callbacks.Callback)" in code
    assert "train_model.compile(" in code


def test_one_cycle_steps_per_batch():
    g = chain("oc", "6", MLP, [("train.sgd", {}), ("train.one_cycle_lr", {})])
    code = generate_training(g, "pytorch")
    batch_step = code.index("scheduler.step()")
    assert code.index("for step, (xb, yb)") < batch_step < code.index("train_loss = running")


def test_plateau_steps_on_the_monitored_loss():
    g = chain("pl", "6", MLP, [("train.plateau_lr", {})])
    assert "scheduler.step(monitored)" in generate_training(g, "pytorch")


@pytest.mark.parametrize("extra, message", [
    (("train.one_cycle_lr", {}), "One-Cycle"),
    (("train.radam", {}), "RAdam"),
    (("train.kfold", {}), "K-Fold"),
])
def test_keras_rejects_unavailable_components(extra, message):
    g = chain("kx", "6", MLP, [("data.sklearn", {"dataset": "wine"}), extra])
    g.nodes["in"].params["shape"] = "13"
    with pytest.raises(CodegenError, match=message):
        generate_training(g, "keras")


def test_preprocessing_fitted_on_training_split_only():
    g = chain("fit", "6", MLP, [("prep.normalize", {"mode": "fit"}), ("prep.impute", {})])
    code = generate_training(g, "pytorch")
    assert 'pipeline = NumericPipeline().fit(feats["train"])' in code


def test_time_series_split_is_chronological():
    g = chain("ts", "16, 1", [("core.lstm", {"return_sequences": False}),
                              ("core.dense", {"units": 1})],
              [("data.timeseries_csv", {"window": 16}), ("train.loss_mse", {}),
               ("prep.split", {"shuffle": True})])
    spec = collect_spec(g)
    assert spec.split["shuffle"] is False
    assert "(chronological)" in generate_training(g, "pytorch")


def test_irrelevant_preprocessing_is_reported():
    g = chain("aug", "6", MLP, [("prep.random_flip", {})])
    spec = collect_spec(g)
    assert any("does not apply to tabular data" in w for w in spec.warnings)


def test_training_script_actually_trains(tmp_path: Path):
    """End-to-end: tiny graph + 2 epochs of synthetic data, run as subprocess."""
    g = Graph(name="tiny")
    for nid, tid, params in (
            ("in", "core.input", {"shape": "16"}), ("d1", "core.dense", {"units": 8}),
            ("r1", "core.relu", {}), ("d2", "core.dense", {"units": 3}),
            ("out", "core.output", {}),
            ("data", "data.synthetic", {"kind": "classification", "n_samples": 240,
                                         "n_features": 16, "n_classes": 3, "seed": 7}),
            ("opt", "train.sgd", {"lr": 0.05, "momentum": 0.9}),
            ("loss", "train.loss_cross_entropy", {}),
            ("trainer", "train.trainer", {"epochs": 2, "batch_size": 32, "device": "cpu"}),
            ("f1", "eval.f1", {}), ("acc", "eval.accuracy", {})):
        g.add_node(NodeInstance(nid, tid, params))
    for a, b in (("in", "d1"), ("d1", "r1"), ("r1", "d2"), ("d2", "out")):
        g.add_edge(Edge(a, "out", b, "in"))
    assert g.validate() == []
    script = export_training(g, "pytorch", tmp_path)
    result = subprocess.run([sys.executable, script.name], capture_output=True, text=True,
                            timeout=300, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "epoch 2/2" in result.stdout and "f1=" in result.stdout
    assert (tmp_path / "tiny_best.pt").exists()
    assert (tmp_path / "predictions.json").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["accuracy"] >= 0.0
