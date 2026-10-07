"""ExportService: renders code targets and writes them to disk.

The target table itself lives in core/targets.py (shared with the MCP server
and the CLI).
"""
from __future__ import annotations

from pathlib import Path

from ai_made_easy.core.codegen import sanitize_identifier
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.targets import (  # noqa: F401  (re-exported)
    RENDERERS, preview_targets, target_label)

PREVIEW_TARGETS = preview_targets()

_SUFFIX = {"pytorch_model": "_pytorch.py", "keras_model": "_keras.py",
           "pytorch_train": "_train_pytorch.py", "keras_train": "_train_keras.py",
           "sklearn_train": "_train_sklearn.py", "llm": "_llm.py"}


def default_filename(graph: Graph, target: str) -> str:
    return sanitize_identifier(graph.name) + _SUFFIX.get(target, ".py")


class ExportService:
    def __init__(self, log, parent=None):
        self.log = log

    @staticmethod
    def render(graph: Graph, target: str) -> str:
        if target not in RENDERERS:
            raise ValueError(f"unknown export target {target!r}")
        return RENDERERS[target](graph)

    def write(self, graph: Graph, target: str, out: Path) -> Path:
        """Write ``target`` to ``out`` (a file path, or a directory for the default name)."""
        out = Path(out)
        path = out / default_filename(graph, target) if out.is_dir() or not out.suffix else out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(graph, target))
        self.log.info(f"exported {target_label(target)} → {path}")
        return path

    def write_runtime_script(self, graph: Graph, kind: str, output: Path) -> Path:
        """Script that serializes the model to ONNX / TorchScript at ``output``."""
        import tempfile

        from ai_made_easy.core.codegen.runtime_export import (
            generate_onnx_export,
            generate_torchscript_export,
        )

        workdir = Path(tempfile.mkdtemp(prefix=f"aime_{kind}_"))
        stem = sanitize_identifier(graph.name)
        if kind == "onnx":
            script = workdir / f"{stem}_export_onnx.py"
            script.write_text(generate_onnx_export(graph, str(output)))
        else:
            script = workdir / f"{stem}_export_torchscript.py"
            script.write_text(generate_torchscript_export(graph, str(output)))
        return script
