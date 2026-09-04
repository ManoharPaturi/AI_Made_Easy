"""Phase 4 tests: analytic summary vs torch, new datasets (live), new
training catalog, augmentation pipeline."""
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
from ai_made_easy.core.summary import summarize

torch = pytest.importorskip("torch", reason="torch not installed")


def build(*specs: tuple[str, str, dict]) -> Graph:
    g = Graph(name="tiny")
    for nid, type_id, params in specs:
        g.add_node(NodeInstance(nid, type_id, params))
    return g


def wire(g: Graph, *pairs: tuple[str, str]) -> None:
    for src, dst in pairs:
        g.add_edge(Edge(src, "out", dst, g.nodes[dst].definition().inputs[0].name))


# ------------------------------------------------------------- model summary

def _torch_param_count(code: str) -> int:
    ns: dict = {}
    exec(compile(code, "<gen>", "exec"), ns)  # noqa: S102
    model = ns["Tiny"]()
    return sum(p.numel() for p in model.parameters())


def test_summary_matches_torch_mlp_with_norms():
    g = build(
        ("in", "core.input", {"shape": "784"}),
        ("d1", "core.dense", {"units": 64, "bias": True}),
        ("ln", "core.layer_norm", {"epsilon": 1e-5}),
        ("r", "core.relu", {}),
        ("d2", "core.dense", {"units": 10, "bias": True}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "d1"), ("d1", "ln"), ("ln", "r"), ("r", "d2"), ("d2", "out"))
    code = generate_training(g, "pytorch")
    assert summarize(g).total_params == _torch_param_count(code)


def test_summary_matches_torch_lstm():
    g = build(
        ("in", "core.input", {"shape": "20, 32"}),
        ("e1", "core.lstm", {"hidden_size": 16, "num_layers": 2,
                              "bias": True, "bidirectional": True}),
        ("m", "core.mean_over_time", {}),
        ("d", "core.dense", {"units": 4, "bias": True}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "e1"), ("e1", "m"), ("m", "d"), ("d", "out"))
    code = generate_training(g, "pytorch")
    assert summarize(g).total_params == _torch_param_count(code)


def test_summary_matches_torch_embedding_conv_mha():
    g = build(
        ("in", "core.input", {"shape": "50"}),
        ("emb", "core.embedding", {"num_embeddings": 500, "embedding_dim": 32}),
        ("mha", "core.multihead_attention", {"embed_dim": 0, "num_heads": 4,
                                              "dropout": 0.0}),
        ("te", "core.transformer_encoder", {"nhead": 4, "dim_feedforward": 64,
                                             "dropout": 0.0}),
        ("m", "core.max_over_time", {}),
        ("d", "core.dense", {"units": 2, "bias": True}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "emb"), ("emb", "mha"), ("mha", "te"), ("te", "m"), ("m", "d"), ("d", "out"))
    code = generate_training(g, "pytorch")
    assert summarize(g).total_params == _torch_param_count(code)


# ------------------------------------------------------------------ datasets

def _train_run(g: Graph, tmp_path: Path):
    assert g.validate() == []
    script = export_training(g, "pytorch", tmp_path)
    result = subprocess.run([sys.executable, script.name], capture_output=True,
                            text=True, timeout=300, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_numpy_dataset_trains(tmp_path: Path):
    import numpy as np

    rng = np.random.default_rng(0)
    xs, ys = [], []
    for cls in range(3):
        center = rng.normal(scale=2.0, size=8)
        xs.append(center + rng.normal(0, 0.3, size=(60, 8)))
        ys.append(np.full(60, cls))
    np.savez(tmp_path / "data.npz", x=np.vstack(xs).astype("float32"),
             y=np.concatenate(ys))
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 16, "bias": True}),
        ("r", "core.relu", {}),
        ("d2", "core.dense", {"units": 3, "bias": True}),
        ("out", "core.output", {}),
        ("data", "data.numpy", {"path": "data.npz", "x_key": "x", "y_key": "y"}),
        ("opt", "train.adamax", {"lr": 0.01, "beta1": 0.9, "beta2": 0.999}),
        ("loss", "train.loss_cross_entropy", {"label_smoothing": 0.0}),
        ("trainer", "train.trainer", {"epochs": 2, "batch_size": 16,
                                       "device": "cpu", "seed": 1,
                                       "early_stopping_patience": 0}),
    )
    wire(g, ("in", "d1"), ("d1", "r"), ("r", "d2"), ("d2", "out"))
    out = _train_run(g, tmp_path)
    assert "epoch 1/2" in out and "epoch 2/2" in out


def test_json_dataset_trains(tmp_path: Path):
    import numpy as np

    rng = np.random.default_rng(1)
    with open(tmp_path / "data.jsonl", "w") as fh:
        for _ in range(120):
            y = int(rng.integers(0, 2))
            x = rng.normal(loc=y * 4.0, scale=1.0, size=6)
            fh.write(json.dumps({"x": x.round(3).tolist(), "y": y}) + "\n")
    g = build(
        ("in", "core.input", {"shape": "6"}),
        ("d1", "core.dense", {"units": 8, "bias": True}),
        ("out", "core.output", {}),
        ("data", "data.json", {"path": "data.jsonl", "x_field": "x", "y_field": "y"}),
        ("opt", "train.rmsprop", {"lr": 0.01, "alpha": 0.99, "eps": 1e-8}),
        ("loss", "train.loss_focal", {"gamma": 2.0, "alpha": 0.25}),
        ("sched", "train.exponential_lr", {"gamma": 0.9}),
        ("trainer", "train.trainer", {"epochs": 2, "batch_size": 16,
                                       "device": "cpu", "seed": 1,
                                       "early_stopping_patience": 0}),
    )
    wire(g, ("in", "d1"), ("d1", "out"))
    out = _train_run(g, tmp_path)
    assert "epoch 2/2" in out
    # focal loss class must have been emitted
    script = (tmp_path / "tiny_train_pytorch.py").read_text()
    assert "class FocalLoss(nn.Module):" in script
    assert "torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)" in script


def test_image_folder_dataset_trains(tmp_path: Path):
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(2)
    for cls, tint in ((0, 30), (1, 220)):
        d = tmp_path / "imgs" / f"class{cls}"
        d.mkdir(parents=True)
        for i in range(25):
            arr = (rng.integers(0, 40, size=(10, 10, 3)) + tint).clip(0, 255)
            Image.fromarray(arr.astype("uint8")).save(d / f"{i}.png")
    g = build(
        ("in", "core.input", {"shape": "3, 10, 10"}),
        ("c1", "core.conv2d", {"out_channels": 4, "kernel_size": 3,
                                "stride": 1, "padding": 1, "dilation": 1}),
        ("r", "core.relu", {}),
        ("gap", "core.global_avgpool2d", {}),
        ("d", "core.dense", {"units": 2, "bias": True}),
        ("out", "core.output", {}),
        ("data", "data.image_folder", {"root": "imgs", "grayscale": False}),
        ("opt", "train.nadam", {"lr": 0.005, "beta1": 0.9, "beta2": 0.999}),
        ("loss", "train.loss_cross_entropy", {"label_smoothing": 0.0}),
        ("trainer", "train.trainer", {"epochs": 1, "batch_size": 8,
                                       "device": "cpu", "seed": 1,
                                       "early_stopping_patience": 0}),
    )
    wire(g, ("in", "c1"), ("c1", "r"), ("r", "gap"), ("gap", "d"), ("d", "out"))
    out = _train_run(g, tmp_path)
    assert "epoch 1/1" in out
    assert "accuracy=" in out  # pipeline ran end-to-end over PIL-loaded images


def test_huggingface_dataset_syntax_only(tmp_path: Path):
    g = build(
        ("in", "core.input", {"shape": "784"}),
        ("d1", "core.dense", {"units": 10, "bias": True}),
        ("out", "core.output", {}),
        ("data", "data.huggingface", {"repo_id": "mnist", "split": "train",
                                       "x_field": "image", "y_field": "label"}),
    )
    wire(g, ("in", "d1"), ("d1", "out"))
    code = generate_training(g, "pytorch")
    ast.parse(code)
    assert 'load_dataset("mnist", split="train")' in code
    with pytest.raises(CodegenError, match="cannot be exported to Keras"):
        generate_training(g, "keras")


# ------------------------------------------------------- training catalog

def test_plateau_scheduler_steps_with_val_loss():
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 2, "bias": True}),
        ("out", "core.output", {}),
        ("sched", "train.plateau_lr", {"factor": 0.5, "patience": 3}),
    )
    wire(g, ("in", "d1"), ("d1", "out"))
    code = generate_training(g, "pytorch")
    assert "ReduceLROnPlateau(optimizer" in code
    assert "scheduler.step(val_loss)" in code
    assert "scheduler.step()" not in code


def test_multistep_milestones_parsed():
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 2, "bias": True}),
        ("out", "core.output", {}),
        ("sched", "train.multistep_lr", {"milestones": "30, 60", "gamma": 0.1}),
    )
    wire(g, ("in", "d1"), ("d1", "out"))
    code = generate_training(g, "pytorch")
    assert "milestones=[30, 60]" in code


def test_new_optimizer_loss_fragments():
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 4, "bias": True}),
        ("out", "core.output", {}),
        ("opt", "train.radam", {"lr": 0.001, "beta1": 0.9, "beta2": 0.999}),
        ("loss", "train.loss_smooth_l1", {"beta": 1.0}),
        ("sched", "train.linear_lr", {"start_factor": 0.1, "total_iters": 5}),
    )
    wire(g, ("in", "d1"), ("d1", "out"))
    code = generate_training(g, "pytorch")
    assert "torch.optim.RAdam(model.parameters()" in code
    assert "nn.SmoothL1Loss(beta=1.0)" in code
    assert "LinearLR(optimizer, start_factor=0.1" in code
    # regression mode: smooth_l1 counts as regression
    spec = collect_spec(g)
    assert spec.is_regression


def test_augmentation_pipeline_and_warning():
    g = build(
        ("in", "core.input", {"shape": "3, 32, 32"}),
        ("c1", "core.conv2d", {"out_channels": 8, "kernel_size": 3,
                                "stride": 1, "padding": 1, "dilation": 1}),
        ("gap", "core.global_avgpool2d", {}),
        ("d", "core.dense", {"units": 10, "bias": True}),
        ("out", "core.output", {}),
        ("data", "data.torchvision", {"dataset": "cifar10", "download": True,
                                       "data_dir": "~/.aime/data"}),
        ("aug_r", "prep.resize", {"height": 32, "width": 32}),
        ("aug_f", "prep.random_flip", {"mode": "horizontal"}),
        ("aug_j", "prep.color_jitter", {"brightness": 0.2, "contrast": 0.2,
                                         "saturation": 0.2, "hue": 0.05}),
        ("mm", "prep.minmax", {"range_min": 0.0, "range_max": 1.0}),
    )
    wire(g, ("in", "c1"), ("c1", "gap"), ("gap", "d"), ("d", "out"))
    code = generate_training(g, "pytorch")
    assert "tf.append(tv_transforms.Resize((32, 32)))" in code
    assert "tf.append(tv_transforms.RandomHorizontalFlip())" in code
    assert "tf.append(tv_transforms.ColorJitter(" in code

    # augmentations with a non-image dataset -> warning comment, skipped
    g2 = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 2, "bias": True}),
        ("out", "core.output", {}),
        ("aug_f", "prep.random_flip", {"mode": "horizontal"}),
    )
    wire(g2, ("in", "d1"), ("d1", "out"))
    code2 = generate_training(g2, "pytorch")
    assert "augmentation blocks apply to the Torchvision Dataset pipeline" in code2


def test_minmax_rendered_for_array_datasets():
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 2, "bias": True}),
        ("out", "core.output", {}),
        ("mm", "prep.minmax", {"range_min": -1.0, "range_max": 1.0}),
    )
    wire(g, ("in", "d1"), ("d1", "out"))
    code = generate_training(g, "pytorch")
    assert "* (1.0 - -1.0) + -1.0" in code
