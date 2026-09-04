"""Codegen tests: DAG rendering, both frameworks, torch runtime checks."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from ai_made_easy.core.codegen import CodegenError, export, generate
from ai_made_easy.core.graph import Edge, Graph, NodeInstance

torch = pytest.importorskip("torch", reason="torch not installed")


def build(*specs: tuple[str, str, dict]) -> Graph:
    g = Graph(name="test model")
    for nid, type_id, params in specs:
        g.add_node(NodeInstance(nid, type_id, params))
    return g


def wire(g: Graph, *pairs: tuple[str, str]) -> None:
    for src, dst in pairs:
        g.add_edge(Edge(src, "out", dst, g.nodes[dst].definition().inputs[0].name))


def make_mlp() -> Graph:
    g = build(
        ("in", "core.input", {"shape": "784"}),
        ("d1", "core.dense", {"units": 128}),
        ("r1", "core.relu", {}),
        ("d2", "core.dense", {"units": 10}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "d1"), ("d1", "r1"), ("r1", "d2"), ("d2", "out"))
    return g


def make_cnn() -> Graph:
    g = build(
        ("in", "core.input", {"shape": "1, 28, 28"}),
        ("c1", "core.conv2d", {"out_channels": 16, "kernel_size": 3,
                                "stride": 1, "padding": 1, "dilation": 1}),
        ("r1", "core.relu", {}),
        ("p1", "core.maxpool2d", {"kernel_size": 2, "stride": 2, "padding": 0}),
        ("c2", "core.conv2d", {"out_channels": 32, "kernel_size": 3,
                                "stride": 1, "padding": 1, "dilation": 1}),
        ("r2", "core.relu", {}),
        ("g1", "core.global_avgpool2d", {}),
        ("d1", "core.dense", {"units": 10, "bias": True}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "c1"), ("c1", "r1"), ("r1", "p1"), ("p1", "c2"),
         ("c2", "r2"), ("r2", "g1"), ("g1", "d1"), ("d1", "out"))
    return g


def make_skip() -> Graph:
    """Input -> dense -> relu -> add(+input) -> out  (branch + merge)."""
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 8}),
        ("r1", "core.relu", {}),
        ("add", "core.add", {}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "d1"), ("d1", "r1"))
    g.add_edge(Edge("r1", "out", "add", "in1"))
    g.add_edge(Edge("in", "out", "add", "in2"))
    g.add_edge(Edge("add", "out", "out", "in"))
    return g


def make_lstm() -> Graph:
    g = build(
        ("in", "core.input", {"shape": "20, 32"}),
        ("l1", "core.lstm", {"hidden_size": 16, "num_layers": 1,
                              "bias": True, "bidirectional": False}),
        ("m1", "core.mean_over_time", {}),
        ("d1", "core.dense", {"units": 4}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "l1"), ("l1", "m1"), ("m1", "d1"), ("d1", "out"))
    return g


def _run_torch(graph: Graph, x: "torch.Tensor"):
    code = generate(graph, "pytorch")
    ns: dict = {}
    exec(compile(code, "<generated>", "exec"), ns)  # noqa: S102
    model = ns["TestModel"]()
    with torch.no_grad():
        return model(x), code


def _syntax_ok(code: str) -> None:
    ast.parse(code)


def test_mlp_runs_and_matches_design():
    out, code = _run_torch(make_mlp(), torch.randn(4, 784))
    assert tuple(out.shape) == (4, 10)
    assert "nn.Linear(in_features=784, out_features=128" in code
    assert "return v_dense_2" in code


def test_cnn_runs_and_matches_design():
    out, code = _run_torch(make_cnn(), torch.randn(4, 1, 28, 28))
    assert tuple(out.shape) == (4, 10)
    assert "nn.Conv2d(in_channels=1, out_channels=16" in code
    assert "nn.Conv2d(in_channels=16, out_channels=32" in code
    assert "torch.mean" in code  # functional global pool


def test_skip_connection_dag_runs():
    out, code = _run_torch(make_skip(), torch.randn(4, 8))
    assert tuple(out.shape) == (4, 8)
    assert "= v_relu_1 + x" in code  # merge consumes input var directly


def test_lstm_sequence_runs():
    out, code = _run_torch(make_lstm(), torch.randn(4, 20, 32))
    assert tuple(out.shape) == (4, 4)
    assert "input_size=32" in code
    assert "self.lstm_1({i0})" not in code  # fragment fully resolved
    assert "[0]" in code  # tuple unpack from LSTM


def test_keras_functional_syntax_all_models():
    for g in (make_mlp(), make_cnn(), make_skip(), make_lstm()):
        code = generate(g, "keras")
        _syntax_ok(code)
        assert "keras.Model(inputs=inputs" in code
    keras_code = generate(make_cnn(), "keras")
    assert "keras.Input(shape=(28, 28, 1,))" in keras_code  # NHWC translation
    assert "filters=16" in keras_code


def test_keras_unsupported_block_raises():
    g = build(
        ("in", "core.input", {"shape": "4, 16, 16"}),
        ("a1", "core.adaptive_avgpool2d", {"output_height": 2, "output_width": 2}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "a1"), ("a1", "out"))
    assert g.validate() == []
    with pytest.raises(CodegenError, match="cannot be exported to Keras"):
        generate(g, "keras")


def test_invalid_graph_refuses_codegen():
    g = make_mlp()
    g.edges = [e for e in g.edges if e.target_id != "d2"]
    with pytest.raises((ValueError, CodegenError), match="validation errors"):
        generate(g, "pytorch")


def test_generated_script_runs_as_subprocess(tmp_path: Path):
    out = export(make_mlp(), "pytorch", tmp_path)
    result = subprocess.run(
        [sys.executable, str(out)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "(1, 10)" in result.stdout


def test_export_writes_file(tmp_path: Path):
    out = export(make_mlp(), "keras", tmp_path)
    assert out.exists()
    assert "keras.Sequential" not in out.read_text()  # functional API now
    assert "build_model" in out.read_text()
