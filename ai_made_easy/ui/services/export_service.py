"""ExportService: the SINGLE codegen dispatch used everywhere (files,
preview, LLM scripts) — one target table instead of parallel ones.
"""
from __future__ import annotations

from pathlib import Path

from ai_made_easy.core.codegen import export, export_training, generate
from ai_made_easy.core.codegen.llm_gen import generate_llm_script
from ai_made_easy.core.codegen.training_gen import generate_training
from ai_made_easy.core.graph import Graph

# the one dispatch table: target id -> renderer
RENDERERS = {
    "pytorch_model": lambda g: generate(g, "pytorch"),
    "keras_model": lambda g: generate(g, "keras"),
    "pytorch_train": lambda g: generate_training(g, "pytorch"),
    "keras_train": lambda g: generate_training(g, "keras"),
    "llm": lambda g: generate_llm_script(g),
}

PREVIEW_TARGETS = ("pytorch_model", "keras_model", "pytorch_train",
                   "keras_train", "llm")

_TARGET_LABELS = {
    "pytorch_model": "PyTorch model",
    "keras_model": "Keras model",
    "pytorch_train": "PyTorch training script",
    "keras_train": "Keras training script",
    "llm": "LLM workflow script",
}


def target_label(target: str) -> str:
    return _TARGET_LABELS.get(target, target)


class ExportService:
    def __init__(self, log, parent=None):
        self.log = log

    @staticmethod
    def render(graph: Graph, target: str) -> str:
        if target not in RENDERERS:
            raise ValueError(f"unknown export target {target!r}")
        return RENDERERS[target](graph)

    def write(self, graph: Graph, target: str, out_dir: Path) -> Path:
        """Write generated code to disk under the project's exports dir."""
        stem = graph.name.replace("-", "_")
        if target == "llm":
            path = Path(out_dir) / f"{stem}_llm.py"
        elif target.endswith("_train"):
            framework = target.split("_")[0]
            path = Path(out_dir) / f"{stem}_train_{framework}.py"
        else:
            framework = target.split("_")[0]
            path = Path(out_dir) / f"{stem}_{framework}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(graph, target))
        self.log.info(f"generated {target_label(target)} → {path}")
        return path

    def write_runtime_script(self, graph: Graph, kind: str,
                             out_dir: Path) -> Path:
        """ONNX / TorchScript export scripts (executed by ProcessService)."""
        from ai_made_easy.core.codegen.runtime_export import (
            generate_onnx_export,
            generate_torchscript_export,
        )

        stem = graph.name.replace("-", "_")
        if kind == "onnx":
            path = Path(out_dir) / f"{stem}_export_onnx.py"
            path.write_text(generate_onnx_export(graph, f"exports/{stem}.onnx"))
        else:
            path = Path(out_dir) / f"{stem}_export_jit.py"
            path.write_text(
                generate_torchscript_export(graph, f"exports/{stem}.torchscript.pt"))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
