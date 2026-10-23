"""Phase 7 tests: headless RunManager, IR expand, MCP server (impl + a real
stdio client round-trip)."""
from __future__ import annotations

import os

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ai_made_easy.core.composites import expand_in_graph
from ai_made_easy.core.codegen import generate
from ai_made_easy.core.graph import Edge, Graph, NodeInstance
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.runner.manager import RunManager

SAMPLES = Path(__file__).parent.parent / "ai_made_easy" / "samples"


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    from PySide6 import QtWidgets

    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def tiny_graph(name: str = "tiny") -> Graph:
    g = Graph(name=name)
    g.add_node(NodeInstance("in", "core.input", {"shape": "8"}))
    g.add_node(NodeInstance("d1", "core.dense", {"units": 4, "bias": True}))
    g.add_node(NodeInstance("out", "core.output", {}))
    g.add_node(NodeInstance(
        "tr", "train.trainer",
        {"epochs": 2, "batch_size": 16, "device": "cpu", "seed": 1,
         "early_stopping_patience": 0}))
    g.add_edge(Edge("in", "out", "d1", "in"))
    g.add_edge(Edge("d1", "out", "out", "in"))
    return g


# ------------------------------------------------------------- run manager

def test_run_manager_trains_headlessly():
    mgr = RunManager()
    run_id = mgr.start(tiny_graph())
    status = mgr.wait(run_id, timeout=300)
    assert status["state"] == "finished"
    assert status["epochs_done"] == 2
    assert "train_loss" in status["latest_metrics"]
    assert len(mgr.metrics(run_id)) == 2


def test_run_manager_stop():
    g = tiny_graph()
    g.nodes["tr"].params["epochs"] = 100
    mgr = RunManager()
    run_id = mgr.start(g)
    status = mgr.stop(run_id)
    assert status["state"] == "stopped"


def test_run_manager_unknown_run():
    mgr = RunManager()
    with pytest.raises(KeyError):
        mgr.status("nope")


# ---------------------------------------------------------------- IR expand

def test_expand_in_graph_is_pure_and_valid():
    g = Graph(name="m")
    g.add_node(NodeInstance("in", "core.input", {"shape": "3, 16, 16"}, (-200, 0)))
    g.add_node(NodeInstance("a", "arch.resnet18",
                            {"base_channels": 4, "num_classes": 4}, (0, 0)))
    g.add_node(NodeInstance("out", "core.output", {}, (200, 0)))
    g.add_edge(Edge("in", "out", "a", "in"))
    g.add_edge(Edge("a", "out", "out", "in"))
    g2 = expand_in_graph(g, "a")
    assert len(g.nodes) == 3  # original untouched
    assert "a" not in g2.nodes
    assert g2.validate() == []
    code = generate(g2, "pytorch")
    assert "self.conv2d_" in code


# ------------------------------------------------------------- MCP server

def test_mcp_impl_functions():
    pytest.importorskip("mcp", reason="mcp SDK not installed")
    from ai_made_easy.mcp.server import (
        _impl_generate,
        _impl_list_blocks,
        _impl_list_samples,
        _impl_read_sample,
        _impl_summarize,
        _impl_validate,
    )

    catalog = json.loads(_impl_list_blocks())
    assert catalog["count"] >= 123
    archs = json.loads(_impl_list_blocks("Architectures"))
    assert archs["count"] == 8
    samples = json.loads(_impl_list_samples())
    assert any("mlp_mnist" in s for s in samples["samples"])
    graph_json = json.loads(_impl_read_sample("mlp_mnist.json"))
    assert _impl_validate(graph_json)["valid"] is True
    assert _impl_summarize(graph_json)["total_params"] > 0
    assert "class MnistMlp" in _impl_generate(graph_json, "pytorch_model")


def test_mcp_stdio_round_trip(tmp_path):
    """Real MCP client over stdio: list tools, call list_blocks + validate. The client runs
    in its own process (tests/fakes/mcp_client.py): on Windows the SDK's stdio client
    crashed the pytest worker that hosted it. Its output is read up to the RESULT line."""
    pytest.importorskip("mcp", reason="mcp SDK not installed")
    client = Path(__file__).parent / "fakes" / "mcp_client.py"
    errlog = tmp_path / "server_stderr.txt"
    proc = subprocess.Popen([sys.executable, str(client), str(errlog)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.extend(iter(proc.stdout.readline, "")),
                              daemon=True)
    reader.start()
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline and reader.is_alive() and not any(
            line.startswith("RESULT ") for line in lines):
        time.sleep(0.2)
    if proc.poll() is None:
        proc.kill()
    proc.wait(30)
    result = next((line for line in lines if line.startswith("RESULT ")), None)
    server = errlog.read_text(encoding="utf-8", errors="replace") if errlog.exists() else ""
    report = "client:\n" + "".join(lines)[-3000:] + "\nserver stderr:\n" + server[-6000:]
    if result is None and sys.platform == "win32":
        # Open (ROADMAP_3.8): on Windows the server answers initialize and list_tools but
        # not the first tool call. Kept running so the report shows where it stops.
        pytest.xfail("MCP server stalls on its first tool call on Windows\n" + report)
    assert result, report
    out = json.loads(result.split("RESULT ", 1)[1])
    assert {"list_blocks", "validate_graph", "generate_code", "start_training",
            "get_run_status", "summarize_model", "expand_architecture",
            "stop_run"} <= set(out["tools"])
    assert out["architectures"] == 8
    assert out["valid"] is True
    assert out["params"] == 101770


# --------------------------------------------------------------- assistant

def test_assistant_extract_graph():
    from ai_made_easy.core.assistant import extract_graph

    assert extract_graph("no fences here") is None
    reply = 'Sure!\n```json\n{"schema_version": 1, "nodes": [], "edges": []}\n```\ndone'
    assert extract_graph(reply) == {"schema_version": 1, "nodes": [], "edges": []}
    assert extract_graph("```json\n{broken\n```") is None


def test_assistant_system_prompt_embeds_catalog_and_graph(monkeypatch):
    from ai_made_easy.core import assistant as a

    prompt = a.build_system_prompt({"name": "g"}, [{"type_id": "core.dense",
                                                    "category": "Layers",
                                                    "params": [{"name": "units"}]}])
    assert "core.dense" in prompt and '"name": "g"' in prompt
    monkeypatch.delenv("AIME_ASSISTANT_API_KEY", raising=False)
    assert a.is_configured() is False


# ---------------------------------------------------- mined-feature tests

def test_all_samples_validate():
    """Every shipped sample opens, validates, and carries a description."""
    import json as _json

    samples = sorted((SAMPLES).glob("*.json"))
    assert len(samples) >= 9
    for path in samples:
        graph = Graph.from_dict(_json.loads(path.read_text(encoding="utf-8")))
        issues = [i for i in graph.validate() if i.severity == "error"]
        assert not issues, f"{path.name}: {issues}"
        assert graph.meta.get("description"), f"{path.name} lacks description"


def test_canvas_exports_png(tmp_path):
    """Isolated subprocess: Qt scene rendering doesn't play nice with the
    torch/MCP state accumulated earlier in the pytest process."""
    code = (
        "from PySide6 import QtWidgets, QtCore\n"
        "from ai_made_easy.ui.app import _ensure_qt_plugin_path\n"
        "_ensure_qt_plugin_path()\n"
        "app = QtWidgets.QApplication([])\n"
        "from ai_made_easy.ui.canvas import CanvasController\n"
        "canvas = CanvasController()\n"
        "canvas.widget.resize(900, 600)\n"
        "canvas.widget.show()\n"
        "canvas.seed_demo()\n"
        "def done(path, error):\n"
        "    print('done', path, error)\n"
        "    app.quit()\n"
        "QtCore.QTimer.singleShot(600, lambda: canvas.export_canvas_png(\n"
        f"    {str(tmp_path / 'canvas.png')!r}, on_done=done))\n"
        "QtCore.QTimer.singleShot(10000, app.quit)\n"
        "app.exec()\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(SAMPLES.parent.parent),
                            capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONPATH": str(SAMPLES.parent.parent)})
    assert result.returncode == 0, result.stderr[-500:]
    out = tmp_path / "canvas.png"
    assert out.exists() and out.stat().st_size > 10_000  # real image, not blank




# -------------------------------------------- restructured-UI features

def test_registry_search():
    reg = get_registry()
    hits = reg.search("conv")
    ids = [b.type_id for b in hits]
    assert "core.conv2d" in ids and "arch.resnet18" not in ids
    hits = reg.search("LSTM")
    assert {"core.lstm", "arch.lstm_classifier"} <= {b.type_id for b in hits}
    assert reg.search("") == []
    assert reg.search("zzz-no-such-block") == []
    # prefix matches rank first
    assert reg.search("dense")[0].type_id == "core.dense"


def test_themes_apply_and_canvas_colors_follow(qapp):
    from PySide6 import QtWidgets

    from ai_made_easy.ui.canvas import CanvasController
    from ai_made_easy.ui.theme import THEMES, ThemeService

    app = QtWidgets.QApplication.instance()
    service = ThemeService()
    canvas = CanvasController()
    for name in ("light", "dark"):
        service.apply(app, name)
        canvas.apply_theme(service.tokens())
        bg, _grid = service.canvas_colors()
        assert bg == THEMES[name]["CANVAS_BG"]
        assert service.active() == name
    service.apply(app, "dark")  # leave the world dark
