"""(Re)generate the example projects in samples/.

Run: python scripts/build_examples.py
Every example must validate without errors (enforced by tests/test_ui.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
sys.path.insert(0, str(ROOT))

from ai_made_easy.core.graph import Graph  # noqa: E402


def flow(name, title, description, input_shape, layers, config, dtype="float32"):
    """Linear model flow + configuration blocks as a project dict."""
    nodes = [{"id": "input", "type": "core.input",
              "params": {"shape": input_shape, "dtype": dtype}, "position": [0, 0]}]
    edges = []
    prev = "input"
    for i, (tid, params) in enumerate(layers):
        nid = f"{tid.split('.')[-1]}_{i}"
        nodes.append({"id": nid, "type": tid, "params": params, "position": [0, 0]})
        edges.append({"from": f"{prev}/out", "to": f"{nid}/in"})
        prev = nid
    nodes.append({"id": "output", "type": "core.output", "params": {}, "position": [0, 0]})
    edges.append({"from": f"{prev}/out", "to": "output/in"})
    for i, (tid, params) in enumerate(config):
        nodes.append({"id": f"{tid.split('.')[-1]}_c{i}", "type": tid, "params": params,
                      "position": [0, 0]})
    return {"schema_version": 1, "name": name, "nodes": nodes, "edges": edges,
            "meta": {"title": title, "description": description}}


def pipeline(name, title, description, blocks):
    nodes = [{"id": f"{tid.split('.')[-1]}_{i}", "type": tid, "params": params,
              "position": [0, 0]} for i, (tid, params) in enumerate(blocks)]
    return {"schema_version": 1, "name": name, "nodes": nodes, "edges": [],
            "meta": {"title": title, "description": description}}


def enrich(file: str, title: str, description: str, extra: list) -> dict:
    data = json.loads((SAMPLES / file).read_text())
    present = {n["type"] for n in data["nodes"]}
    for i, (tid, params) in enumerate(extra):
        if tid not in present:
            data["nodes"].append({"id": f"{tid.split('.')[-1]}_x{i}", "type": tid,
                                  "params": params, "position": [0, 0]})
    data["meta"] = {**data.get("meta", {}), "title": title, "description": description}
    return data


def with_flatten(data: dict) -> dict:
    """Image input [1, 28, 28] followed by Flatten (for MLPs on MNIST)."""
    head = next(n for n in data["nodes"] if n["type"] == "core.input")
    if head["params"].get("shape", "").replace(" ", "") == "1,28,28":
        return data
    head["params"]["shape"] = "1, 28, 28"
    data["nodes"].append({"id": "flatten_in", "type": "core.flatten", "params": {},
                          "position": [0, 0]})
    for edge in data["edges"]:
        if edge["from"] == f"{head['id']}/out":
            edge["from"] = "flatten_in/out"
    data["edges"].append({"from": f"{head['id']}/out", "to": "flatten_in/in"})
    return data


MNIST_NORM = {"mode": "fixed", "mean": "0.1307", "std": "0.3081"}
IMAGENET_NORM = {"mode": "fixed", "mean": "0.485, 0.456, 0.406", "std": "0.229, 0.224, 0.225"}

EXAMPLES = {
    "demo_seed.json": enrich(
        "demo_seed.json", "MNIST CNN (start-up project)",
        "Two-layer CNN with global pooling trained on MNIST.",
        [("data.torchvision", {"dataset": "mnist"}), ("prep.normalize", MNIST_NORM)]),
    "mnist_cnn.json": enrich(
        "mnist_cnn.json", "MNIST — Convolutional Network",
        "Compact CNN for handwritten digits with fixed MNIST normalization.",
        [("data.torchvision", {"dataset": "mnist"}), ("prep.normalize", MNIST_NORM),
         ("eval.f1", {}), ("eval.confusion_matrix", {})]),
    "mlp_mnist.json": with_flatten(enrich(
        "mlp_mnist.json", "MNIST — Multi-Layer Perceptron",
        "784 → 128 → 10 MLP with dropout on flattened MNIST digits.",
        [("data.torchvision", {"dataset": "mnist"}), ("prep.normalize", MNIST_NORM)])),
    "cifar10_cnn_augmented.json": enrich(
        "cifar10_cnn_augmented.json", "CIFAR-10 — CNN with Augmentation",
        "BatchNorm CNN with flips, rotation and colour jitter on CIFAR-10.",
        [("prep.normalize", {"mode": "fixed", "mean": "0.4914, 0.4822, 0.4465",
                             "std": "0.247, 0.243, 0.261"})]),
    "lstm_classifier.json": enrich(
        "lstm_classifier.json", "Sequence Classifier — LSTM",
        "LSTM over 50-step sequences with a mean-over-time head (synthetic data).", []),
    "regression_mlp.json": enrich(
        "regression_mlp.json", "Tabular Regression — MLP",
        "MLP regressor with Smooth L1 loss and min-max scaling.", []),
    "skip_connection_mlp.json": enrich(
        "skip_connection_mlp.json", "Residual MLP",
        "Dense + LayerNorm block with an additive skip connection.", []),
    "llm_generation.json": enrich("llm_generation.json", "LLM — Text Generation",
                                  "Hugging Face causal language model generation script.", []),
    "llm_lora_finetune.json": enrich("llm_lora_finetune.json", "LLM — LoRA Fine-Tuning",
                                     "Parameter-efficient supervised fine-tuning with LoRA.",
                                     []),
    "llm_rag_assistant.json": enrich("llm_rag_assistant.json", "LLM — Retrieval-Augmented QA",
                                     "Embeddings, vector store and retrieval feeding a "
                                     "generator.", []),
    "iris_mlp.json": flow(
        "iris_mlp", "Iris — Tabular Neural Network",
        "Small MLP on the Iris measurements with standardization and k-fold validation.",
        "4", [("core.dense", {"units": 32}), ("core.relu", {}), ("core.dropout", {"p": 0.1}),
              ("core.dense", {"units": 3})],
        [("data.sklearn", {"dataset": "iris"}), ("prep.normalize", {"mode": "fit"}),
         ("train.loss_cross_entropy", {}), ("train.adamw", {"lr": 0.01}),
         ("train.trainer", {"epochs": 60, "batch_size": 16}),
         ("eval.accuracy", {}), ("eval.f1", {})]),
    "cifar10_transfer_resnet18.json": flow(
        "cifar10_transfer", "CIFAR-10 — Transfer Learning (ResNet-18)",
        "Frozen ImageNet ResNet-18 features with a trainable linear classifier.",
        "3, 64, 64", [("core.pretrained_backbone", {"architecture": "resnet18"}),
                      ("core.dropout", {"p": 0.2}), ("core.dense", {"units": 10})],
        [("data.torchvision", {"dataset": "cifar10"}),
         ("prep.resize", {"height": 64, "width": 64}), ("prep.random_flip", {}),
         ("prep.normalize", IMAGENET_NORM), ("train.loss_cross_entropy", {}),
         ("train.adam", {"lr": 0.001}),
         ("train.trainer", {"epochs": 5, "batch_size": 128}), ("eval.accuracy", {})]),
    "diabetes_regression_mlp.json": flow(
        "diabetes_mlp", "Diabetes — Regression Network",
        "MLP regressor on the diabetes progression dataset with Huber loss.",
        "10", [("core.dense", {"units": 64}), ("core.gelu", {}),
               ("core.dense", {"units": 32}), ("core.gelu", {}), ("core.dense", {"units": 1})],
        [("data.sklearn", {"dataset": "diabetes"}), ("prep.normalize", {"mode": "fit"}),
         ("train.loss_huber", {}), ("train.adamw", {"lr": 0.003}),
         ("train.warmup_cosine_lr", {"warmup_epochs": 5}),
         ("train.trainer", {"epochs": 120, "batch_size": 32, "early_stopping_patience": 15}),
         ("eval.mae", {}), ("eval.rmse", {}), ("eval.r2", {})]),
    "classic_random_forest.json": pipeline(
        "breast_cancer_rf", "Breast Cancer — Random Forest",
        "scikit-learn Random Forest with standardization, class weights and 5-fold CV.",
        [("data.sklearn", {"dataset": "breast_cancer"}), ("prep.normalize", {}),
         ("prep.class_balance", {}), ("ml.random_forest_classifier", {"n_estimators": 300}),
         ("train.kfold", {"k": 5}), ("eval.accuracy", {}), ("eval.f1", {}),
         ("eval.roc_auc", {}), ("eval.confusion_matrix", {})]),
    "classic_gradient_boosting_search.json": pipeline(
        "wine_boosting_search", "Wine — Gradient Boosting with Grid Search",
        "Histogram gradient boosting tuned with a cross-validated grid search.",
        [("data.sklearn", {"dataset": "wine"}), ("ml.hist_gradient_boosting_classifier", {}),
         ("ml.hyperparameter_search", {"param_grid": "learning_rate: 0.03, 0.1, 0.3\n"
                                                     "max_leaf_nodes: 15, 31", "cv": 5}),
         ("eval.accuracy", {}), ("eval.balanced_accuracy", {})]),
    "classic_kmeans.json": pipeline(
        "customer_segments", "Clustering — k-Means Segmentation",
        "Standardized features clustered with k-Means and scored by silhouette.",
        [("data.synthetic", {"n_features": 6, "n_classes": 4, "n_samples": 600}),
         ("prep.normalize", {}), ("ml.kmeans", {"n_clusters": 4}),
         ("eval.silhouette", {}), ("eval.davies_bouldin", {}), ("eval.adjusted_rand", {})]),
}


def main() -> int:
    failures = 0
    for file, data in EXAMPLES.items():
        errors = [str(i) for i in Graph.from_dict(data).validate() if i.severity == "error"]
        if errors:
            failures += 1
            print(f"INVALID {file}: {errors}")
            continue
        (SAMPLES / file).write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {file}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
