"""Import existing models as editable block graphs.

``import_model(kind, ...)`` runs the framework-specific importer in a
subprocess (the framework only has to exist in the training environment, not
in the designer) and returns the graph plus a fidelity report: unsupported
operations, parameter counts and — when weights can be mapped — the maximum
output difference between the original and the regenerated model.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("pytorch", "onnx", "keras")
_WORKERS = {"pytorch": "ai_made_easy.core.importers.torch_fx",
            "onnx": "ai_made_easy.core.importers.onnx_import",
            "keras": "ai_made_easy.core.importers.keras_import"}


class ModelImportError(RuntimeError):
    """The model could not be imported (message is user-facing)."""


@dataclass
class ImportResult:
    graph: dict
    unsupported: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    original: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.unsupported and bool(self.verification.get("ok"))

    def summary(self) -> str:
        if self.unsupported:
            return (f"{len(self.unsupported)} operation(s) have no block equivalent:\n  - "
                    + "\n  - ".join(self.unsupported[:12]))
        v = self.verification
        if not v.get("ok"):
            return "the imported design does not validate:\n  - " + "\n  - ".join(
                v.get("errors", [])[:8])
        lines = [f"{len(self.graph.get('nodes', []))} blocks"]
        orig = self.original.get("parameters")
        matches = v.get("outputs_match")
        if orig is not None:
            if orig == v.get("parameters"):
                note = "identical"
            elif matches:
                note = "framework conventions differ, e.g. recurrent biases / BatchNorm statistics"
            else:
                note = "DIFFERENT"
            lines.append(f"parameters: {orig:,} original, {v.get('parameters', 0):,} rebuilt "
                         f"({note})")
        if "outputs_match" in v:
            lines.append(("outputs: identical to the original model" if matches
                          else "outputs: DIFFER from the original model")
                         + f" (max |difference| {v['max_abs_diff']:.2g})")
        elif v.get("output_shape") is not None:
            lines.append(f"output shape: {v['output_shape']}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"graph": self.graph, "unsupported": self.unsupported, "warnings": self.warnings,
                "original": self.original, "verification": self.verification, "ok": self.ok,
                "summary": self.summary()}


def import_model(kind: str, *, source: str, attr: str = "", input_shape: list[int] | None = None,
                 dtype: str = "float32", kwargs: dict | None = None, name: str = "",
                 python: str | None = None, weights_out: str | None = None,
                 timeout: float = 900) -> ImportResult:
    """Import ``source`` (a .py file / module for PyTorch, an .onnx file, a .keras file)."""
    if kind not in KINDS:
        raise ModelImportError(f"unknown model kind {kind!r}; one of {KINDS}")
    if kind == "pytorch" and (not attr or not input_shape):
        raise ModelImportError("PyTorch imports need the model class / factory name and the "
                           "input shape (without the batch dimension)")
    request = {"source": source, "attr": attr, "input_shape": input_shape, "dtype": dtype,
               "kwargs": kwargs or {}, "weights_out": weights_out}
    package_root = str(Path(__file__).resolve().parents[3])
    env = {**os.environ, "PYTHONWARNINGS": "ignore",
           "PYTHONPATH": os.pathsep.join(filter(None, [package_root,
                                                       os.environ.get("PYTHONPATH", "")])),
           "KERAS_BACKEND": os.environ.get("KERAS_BACKEND", "torch")}
    with tempfile.TemporaryDirectory(prefix="aime_import_") as tmp:
        req = Path(tmp) / "request.json"
        req.write_text(json.dumps(request))
        try:
            proc = subprocess.run([python or sys.executable, "-m", _WORKERS[kind], str(req)],
                                  capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            raise ModelImportError(f"the import took longer than {timeout:.0f}s") from exc
    marker = "IMPORT-RESULT "
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(marker)), None)
    if line is None:
        tail = (proc.stderr or proc.stdout)[-2000:]
        if "No module named" in tail:
            missing = tail.split("No module named", 1)[1].strip().splitlines()[0]
            raise ModelImportError(f"the import environment lacks {missing} — install it, or set "
                               "the Python environment in Settings")
        raise ModelImportError("the importer crashed:\n" + tail)
    data = json.loads(line[len(marker):])
    if "error" in data:
        raise ModelImportError(data["error"])
    graph = data["graph"]
    if name:
        graph["name"] = name
    return ImportResult(graph=graph, unsupported=data.get("unsupported", []),
                        warnings=data.get("warnings", []), original=data.get("original", {}),
                        verification=data.get("verification", {}))
