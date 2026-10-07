"""Test-session environment and shared fixtures.

Headless Qt, Keras on the PyTorch backend, and an isolated ``$AIME_HOME`` so
tests never touch the user's run history, registry or settings.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("AIME_HOME", tempfile.mkdtemp(prefix="aime_test_home_"))

import pytest  # noqa: E402


def tiny_classifier_dict(epochs: int = 2, lr: float = 0.05, units: int = 8,
                         name: str = "tiny") -> dict:
    """A small trainable MLP on synthetic data (a few seconds on CPU)."""
    nodes = [
        ("in", "core.input", {"shape": "16"}),
        ("d1", "core.dense", {"units": units}),
        ("act", "core.relu", {}),
        ("d2", "core.dense", {"units": 3}),
        ("out", "core.output", {}),
        ("data", "data.synthetic", {"kind": "classification", "n_samples": 240,
                                    "n_features": 16, "n_classes": 3,
                                    "noise": 0.3, "seed": 7}),
        ("opt", "train.sgd", {"lr": lr, "momentum": 0.9, "nesterov": False}),
        ("loss", "train.loss_cross_entropy", {"label_smoothing": 0.0}),
        ("trainer", "train.trainer", {"epochs": epochs, "batch_size": 32,
                                      "device": "cpu", "seed": 7,
                                      "early_stopping_patience": 0}),
        ("acc", "eval.accuracy", {}),
    ]
    chain = ["in", "d1", "act", "d2", "out"]
    return {
        "name": name,
        "nodes": [{"id": i, "type": t, "params": p, "position": [0, 0]} for i, t, p in nodes],
        "edges": [{"from": f"{a}/out", "to": f"{b}/in"} for a, b in zip(chain, chain[1:], strict=False)],
        "meta": {},
    }


@pytest.fixture
def tiny_classifier() -> dict:
    return tiny_classifier_dict()


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """A fresh ``$AIME_HOME`` for one test."""
    monkeypatch.setenv("AIME_HOME", str(tmp_path / "home"))
    return tmp_path / "home"
