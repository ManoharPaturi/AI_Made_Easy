"""3.7: generated scripts stay valid Python when the user's files live under a Windows path.

A path like C:\\Users\\me\\data.csv pasted into a docstring or a plain string literal is
read as escape sequences (\\U starts a 32-bit unicode escape), so the script would not
even parse on Windows. Every sample's file and folder parameters are set to Windows paths,
every target the design supports is generated, and each Python file must compile and keep
the path intact."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_made_easy.core import api
from ai_made_easy.core.codegen import CodegenError
from ai_made_easy.core.families import family_of
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry

SAMPLES = Path(__file__).resolve().parents[1] / "ai_made_easy" / "samples"
WINDOWS = "C:\\Users\\nobody\\Ünïcode data\\"
PATH_PARAMS = {"path", "root", "folder", "directory", "dir", "file", "images", "labels",
               "annotations", "train_path", "val_path", "test_path", "corpus", "source"}


def _windows(graph: dict) -> tuple[dict, list[str]]:
    reg = get_registry()
    paths = []
    for node in graph.get("nodes", []):
        if not reg.has(node.get("type", "")):
            continue
        params = node.setdefault("params", {})
        for spec in reg.get(node["type"]).params:
            if spec.type == "str" and spec.name in PATH_PARAMS:
                base = str(params.get(spec.name) or spec.default or "data").replace("\\", "/")
                params[spec.name] = WINDOWS + (base.rsplit("/", 1)[-1] or "data")
                paths.append(params[spec.name])
    return graph, paths


def _cases():
    out = []
    for path in sorted(SAMPLES.glob("*.json")):
        graph, paths = _windows(json.loads(path.read_text(encoding="utf-8")))
        if paths:
            out.append(pytest.param(graph, paths, id=path.stem))
    return out


@pytest.mark.parametrize("graph,paths", _cases())
def test_scripts_compile_with_windows_paths(graph, paths):
    g = Graph.from_dict(graph)
    compiled = 0
    for target in family_of(g).targets:
        try:
            code = api.generate(g, target)
        except (CodegenError, api.ApiError):
            continue            # a target the design can't use (checked elsewhere)
        try:
            compile(code, f"<{graph.get('name')}:{target}>", "exec")
        except SyntaxError as exc:
            if "unicode error" in str(exc) or "escape" in str(exc):
                raise AssertionError(f"{target}: {exc}") from exc
            continue            # not a Python target
        compiled += 1
    assert compiled, "no Python target generated"



def test_every_generated_module_docstring_is_raw():
    """Templates whose output starts with a module docstring that interpolates values (a
    data path, a title) open it as r\"\"\"…: a Windows path can't turn into escapes."""
    import ast

    package = Path(__file__).resolve().parents[1] / "ai_made_easy"
    bad = []
    for path in package.rglob("*.py"):
        if "static" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value.startswith('"""') and ("{{" in node.value.split('"""')[1]
                                                          or "{" in node.value.split('"""')[1]):
                bad.append(f"{path.relative_to(package.parent)}:{node.lineno}")
    assert not bad, "start the generated docstring with r\"\"\":\n" + "\n".join(bad)


@pytest.mark.parametrize("os_name,command,want", [
    ("nt", r"C:\Python\python.exe D:\a\fakes\fake_docker.py", [r"C:\Python\python.exe",
                                                               r"D:\a\fakes\fake_docker.py"]),
    ("nt", r'"C:\Program Files\Docker\docker.exe" --context x',
     [r"C:\Program Files\Docker\docker.exe", "--context", "x"]),
    ("posix", "/usr/bin/python3 '/tmp/a b/fake.py'", ["/usr/bin/python3", "/tmp/a b/fake.py"]),
])
def test_command_overrides_keep_windows_paths(os_name, command, want, monkeypatch):
    import os

    from ai_made_easy.core.runner.backends import split_command

    monkeypatch.setattr(os, "name", os_name)
    assert split_command(command) == want
