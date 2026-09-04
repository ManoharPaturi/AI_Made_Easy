"""Golden-file tests: committed references for every generator.

Regenerate with:  UPDATE_GOLDEN=1 pytest tests/test_golden.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ai_made_easy.core.codegen import generate
from ai_made_easy.core.codegen.training_gen import generate_training
from ai_made_easy.core.graph import Graph

SAMPLES = Path(__file__).parent.parent / "samples"
GOLDEN = Path(__file__).parent / "golden"

CASES = [
    ("mlp_mnist.json", "pytorch", "model"),
    ("mlp_mnist.json", "keras", "model"),
    ("mlp_mnist.json", "pytorch", "train"),
    ("mlp_mnist.json", "keras", "train"),
    ("mnist_cnn.json", "pytorch", "model"),
    ("mnist_cnn.json", "keras", "model"),
    ("mnist_cnn.json", "pytorch", "train"),
    ("skip_connection_mlp.json", "pytorch", "model"),
    ("skip_connection_mlp.json", "keras", "model"),
    ("cifar10_cnn_augmented.json", "pytorch", "train"),
    ("lstm_classifier.json", "pytorch", "train"),
    ("regression_mlp.json", "pytorch", "train"),
]


def _render(sample: str, framework: str, kind: str) -> str:
    graph = Graph.from_dict(json.loads((SAMPLES / sample).read_text()))
    if kind == "train":
        return generate_training(graph, framework)
    return generate(graph, framework)


@pytest.mark.parametrize("sample,framework,kind", CASES)
def test_golden(sample: str, framework: str, kind: str) -> None:
    golden_path = GOLDEN / f"{sample[:-5]}_{kind}_{framework}.py"
    actual = _render(sample, framework, kind)
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.mkdir(exist_ok=True)
        golden_path.write_text(actual)
        pytest.skip("golden updated")
    assert golden_path.exists(), (
        f"missing golden {golden_path.name} — run UPDATE_GOLDEN=1 pytest "
        f"tests/test_golden.py to create it"
    )
    assert actual == golden_path.read_text(), (
        f"generated output changed vs {golden_path.name} — if intentional, "
        f"regenerate with UPDATE_GOLDEN=1"
    )
