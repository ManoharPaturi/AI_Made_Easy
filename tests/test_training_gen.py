"""Training-script generator tests: spec collection, rendering, live run."""
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


def load(name: str) -> Graph:
    return Graph.from_dict(json.loads((SAMPLES / name).read_text()))


def build(*specs: tuple[str, str, dict]) -> Graph:
    g = Graph(name="tiny")
    for nid, type_id, params in specs:
        g.add_node(NodeInstance(nid, type_id, params))
    g.add_edge(Edge("in", "out", "d1", "in"))
    g.add_edge(Edge("d1", "out", "out", "in"))
    return g


def test_spec_collects_config_blocks():
    spec = collect_spec(load("mlp_mnist.json"))
    assert spec.optimizer["kind"] == "train.adam"
    assert spec.optimizer["lr"] == 0.001
    assert spec.loss["kind"] == "train.loss_cross_entropy"
    assert spec.trainer["epochs"] == 10
    assert spec.trainer["early_stopping_patience"] == 3
    assert "accuracy" in spec.metrics
    assert not spec.is_regression
    assert spec.input_volume == 784 and spec.output_units == 10


def test_spec_defaults_when_no_config_blocks():
    """CNN sample has no config blocks: smart synthetic default + Adam/CE."""
    spec = collect_spec(load("mnist_cnn.json"))
    assert spec.dataset["block"] == "data.synthetic"
    assert spec.dataset["n_features"] == 784  # model input volume
    assert spec.dataset["n_classes"] == 10
    assert spec.optimizer["kind"] == "train.adam"


def test_duplicate_optimizer_rejected():
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 2}),
        ("out", "core.output", {}),
        ("opt1", "train.adam", {}),
        ("opt2", "train.sgd", {}),
    )
    with pytest.raises(CodegenError, match="at most one optimizer"):
        collect_spec(g)


def test_regression_mode_filters_metrics():
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 1}),
        ("out", "core.output", {}),
        ("loss", "train.loss_mse", {}),
        ("acc", "eval.accuracy", {}),
        ("mae", "eval.mae", {}),
    )
    spec = collect_spec(g)
    assert spec.is_regression
    assert "accuracy" not in spec.metrics
    assert "mae" in spec.metrics and "mse" in spec.metrics


def test_pytorch_train_script_fragments():
    code = generate_training(load("mlp_mnist.json"), "pytorch")
    ast.parse(code)
    assert "nn.CrossEntropyLoss(label_smoothing=0.0)" in code
    assert "torch.optim.Adam(model.parameters(), lr=0.001" in code
    assert "CosineAnnealingLR" in code
    assert "early stopping after" in code
    assert 'CHECKPOINT = "mnist_mlp_best.pt"' in code


def test_keras_train_script_fragments():
    code = generate_training(load("mlp_mnist.json"), "keras")
    ast.parse(code)
    assert 'loss="sparse_categorical_crossentropy"' in code
    assert "keras.callbacks.EarlyStopping" in code
    assert "LearningRateScheduler" in code


def test_one_cycle_schedules_per_batch():
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 2}),
        ("out", "core.output", {}),
        ("oc", "train.one_cycle_lr", {"max_lr": 0.01, "pct_start": 0.3}),
    )
    code = generate_training(g, "pytorch")
    assert "total_steps=EPOCHS * len(train_loader)" in code
    # per-batch step must be inside the batch loop, not the epoch tail
    batch_loop = code[code.index("for xb, yb in train_loader"):code.index("train_loss =")]
    assert "scheduler.step()" in batch_loop


def test_keras_one_cycle_rejected():
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 2}),
        ("out", "core.output", {}),
        ("oc", "train.one_cycle_lr", {"max_lr": 0.01, "pct_start": 0.3}),
    )
    with pytest.raises(CodegenError, match="cannot be exported to Keras"):
        generate_training(g, "keras")


def test_training_script_actually_trains(tmp_path: Path):
    """End-to-end: tiny graph + 2 epochs of synthetic data, run as subprocess."""
    g = build(
        ("in", "core.input", {"shape": "16"}),
        ("d1", "core.dense", {"units": 8}),
        ("r1", "core.relu", {}),
        ("d2", "core.dense", {"units": 3}),
        ("out", "core.output", {}),
        ("data", "data.synthetic", {"kind": "classification", "n_samples": 240,
                                     "n_features": 16, "n_classes": 3,
                                     "noise": 0.3, "seed": 7}),
        ("opt", "train.sgd", {"lr": 0.05, "momentum": 0.9, "nesterov": False}),
        ("loss", "train.loss_cross_entropy", {"label_smoothing": 0.0}),
        ("trainer", "train.trainer", {"epochs": 2, "batch_size": 32,
                                       "device": "cpu", "seed": 7,
                                       "early_stopping_patience": 0}),
        ("f1", "eval.f1", {"average": "macro"}),
    )
    g.add_edge(Edge("d1", "out", "r1", "in"))
    g.add_edge(Edge("r1", "out", "d2", "in"))
    assert g.validate() == []

    script = export_training(g, "pytorch", tmp_path)
    result = subprocess.run(
        [sys.executable, script.name], capture_output=True, text=True,
        timeout=300, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "epoch 1/2" in result.stdout and "epoch 2/2" in result.stdout
    assert "f1=" in result.stdout
    assert "test:" in result.stdout
    assert (tmp_path / "tiny_best.pt").exists()
