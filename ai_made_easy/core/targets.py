"""The ONE codegen target table (renderers, preview order, labels).

export_service (UI), the MCP server and future families all read this —
adding a target (e.g. ``sklearn_train``) means registering it here once.
Pure Python, Qt-free.
"""
from __future__ import annotations

from typing import Callable

from ai_made_easy.core.codegen import generate
from ai_made_easy.core.codegen.llm_gen import generate_llm_script
from ai_made_easy.core.codegen.training_gen import generate_training
from ai_made_easy.core.graph import Graph

Renderer = Callable[[Graph], str]

RENDERERS: dict[str, Renderer] = {
    "pytorch_model": lambda g: generate(g, "pytorch"),
    "keras_model": lambda g: generate(g, "keras"),
    "pytorch_train": lambda g: generate_training(g, "pytorch"),
    "keras_train": lambda g: generate_training(g, "keras"),
    "llm": lambda g: generate_llm_script(g),
    "sklearn_train": lambda g: _sklearn(g),
}


def _sklearn(graph: Graph) -> str:
    from ai_made_easy.core.classic.generate import generate_classic

    return generate_classic(graph)

TARGET_LABELS: dict[str, str] = {
    "pytorch_model": "PyTorch model",
    "keras_model": "Keras model",
    "pytorch_train": "PyTorch training script",
    "keras_train": "Keras training script",
    "llm": "LLM workflow script",
    "sklearn_train": "scikit-learn pipeline script",
}


def register_target(target: str, renderer: Renderer, label: str) -> None:
    """Add a codegen target (used by new block families)."""
    RENDERERS[target] = renderer
    TARGET_LABELS[target] = label


def preview_targets() -> tuple[str, ...]:
    return tuple(RENDERERS)


def target_label(target: str) -> str:
    return TARGET_LABELS.get(target, target)
