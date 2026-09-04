"""Phase 5 tests: composite architectures, runtime exports, templates."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai_made_easy.core.codegen import CodegenError, generate
from ai_made_easy.core.composites import (
    fragment_to_dict,
    wrap_fragment,
)
from ai_made_easy.core.graph import Edge, Graph, NodeInstance
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.codegen.runtime_export import (
    generate_onnx_export,
    generate_torchscript_export,
)

torch = pytest.importorskip("torch", reason="torch not installed")

ARCH_CASES = [
    ("arch.mlp", {"hidden": "32, 16", "num_classes": 4}, [12], [4]),
    ("arch.cnn", {"base_channels": 4, "num_classes": 4}, [3, 16, 16], [4]),
    ("arch.vgg", {"base_channels": 4, "blocks": 2, "num_classes": 4}, [3, 16, 16], [4]),
    ("arch.resnet18", {"base_channels": 4, "num_classes": 4}, [3, 16, 16], [4]),
    ("arch.unet", {"base_channels": 2, "out_channels": 1}, [1, 16, 16], [1, 16, 16]),
    ("arch.autoencoder", {"input_dim": 16, "hidden": "8", "latent_dim": 4}, [16], [16]),
    ("arch.lstm_classifier", {"hidden_size": 8, "num_layers": 1,
                               "dropout": 0.0, "num_classes": 3}, [10, 6], [3]),
    ("arch.transformer_classifier", {"nhead": 4,
                                      "dim_feedforward": 16, "num_layers": 2,
                                      "num_classes": 3}, [10, 8], [3]),
]


@pytest.mark.parametrize("type_id,params,input_shape,out_tail", ARCH_CASES)
def test_architecture_expands_validates_and_runs(type_id, params, input_shape, out_tail):
    block = get_registry().get(type_id)
    frag = block.builder(dict(params))
    graph = wrap_fragment(frag, input_shape)
    assert graph.validate() == [], f"{type_id} fragment has issues"
    code = generate(graph, "pytorch")
    ast.parse(code)
    ns: dict = {}
    exec(compile(code, "<gen>", "exec"), ns)  # noqa: S102
    model = ns["Fragment"]()
    x = torch.randn(2, *input_shape)
    with torch.no_grad():
        out = model(x)
    assert tuple(out.shape) == (2, *out_tail), f"{type_id} output {tuple(out.shape)}"


def test_composite_requires_expansion():
    g = Graph(name="macro")
    g.add_node(NodeInstance("in", "core.input", {"shape": "16"}))
    g.add_node(NodeInstance("m", "arch.mlp", {"hidden": "8", "num_classes": 2}))
    g.add_node(NodeInstance("out", "core.output", {}))
    g.add_edge(Edge("in", "out", "m", "in"))
    g.add_edge(Edge("m", "out", "out", "in"))
    issues = g.validate()
    assert any("Expand" in i.message for i in issues)
    with pytest.raises((CodegenError, ValueError)):
        generate(g, "pytorch")


def test_resnet18_uses_skip_connections():
    frag = get_registry().get("arch.resnet18").builder(
        {"base_channels": 4, "num_classes": 2})
    assert sum(1 for n in frag.nodes if n["type"] == "core.add") == 8  # 4 stages x 2
    assert sum(1 for n in frag.nodes if n["type"] == "core.conv2d") == 1 + 8 * 2 + 3  # stem + blocks + projections


def test_unet_uses_concatenate_skips():
    frag = get_registry().get("arch.unet").builder(
        {"base_channels": 2, "out_channels": 1})
    assert sum(1 for n in frag.nodes if n["type"] == "core.concatenate") == 2
    assert sum(1 for n in frag.nodes if n["type"] == "core.conv_transpose2d") == 2


def test_onnx_export_creates_file(tmp_path: Path):
    pytest.importorskip("onnxscript", reason="onnxscript not installed")
    g = Graph(name="tiny")
    for nid, t, p in [
        ("in", "core.input", {"shape": "3, 8, 8"}),
        ("c", "core.conv2d", {"out_channels": 4, "kernel_size": 3,
                               "stride": 1, "padding": 1, "dilation": 1}),
        ("r", "core.relu", {}),
        ("gap", "core.global_avgpool2d", {}),
        ("d", "core.dense", {"units": 2, "bias": True}),
        ("out", "core.output", {}),
    ]:
        g.add_node(NodeInstance(nid, t, p))
    for a, b in [("in", "c"), ("c", "r"), ("r", "gap"), ("gap", "d"), ("d", "out")]:
        g.add_edge(Edge(a, "out", b, g.nodes[b].definition().inputs[0].name))
    script = tmp_path / "export.py"
    onnx_path = tmp_path / "tiny.onnx"
    script.write_text(generate_onnx_export(g, str(onnx_path)))
    result = subprocess.run([sys.executable, script.name], cwd=tmp_path,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert onnx_path.exists()
    assert "onnx checker: OK" in result.stdout


def test_torchscript_round_trip(tmp_path: Path):
    g = Graph(name="tiny")
    for nid, t, p in [
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 8, "bias": True}),
        ("r", "core.relu", {}),
        ("d2", "core.dense", {"units": 2, "bias": True}),
        ("out", "core.output", {}),
    ]:
        g.add_node(NodeInstance(nid, t, p))
    for a, b in [("in", "d1"), ("d1", "r"), ("r", "d2"), ("d2", "out")]:
        g.add_edge(Edge(a, "out", b, g.nodes[b].definition().inputs[0].name))
    script = tmp_path / "export.py"
    jit_path = tmp_path / "tiny_ts.pt"
    script.write_text(generate_torchscript_export(g, str(jit_path)))
    result = subprocess.run([sys.executable, script.name], cwd=tmp_path,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert jit_path.exists()
    assert "reload smoke test OK" in result.stdout


def test_template_fragment_roundtrip(tmp_path: Path, monkeypatch):
    """Fragment serialization: build → save → load → identical wrap graph."""
    from ai_made_easy.core.composites import fragment_from_dict

    frag = get_registry().get("arch.mlp").builder(
        {"hidden": "8", "num_classes": 2})
    data = json.loads(json.dumps(fragment_to_dict(frag, "my mlp")))
    frag2 = fragment_from_dict(data)
    g1 = wrap_fragment(frag, [16])
    g2 = wrap_fragment(frag2, [16])
    assert len(g1.nodes) == len(g2.nodes)
    assert len(g1.edges) == len(g2.edges)
    assert g2.validate() == []


def test_registry_lists_composites():
    schemas = get_registry().list_blocks()
    archs = [s for s in schemas if s.get("composite")]
    assert {a["type_id"] for a in archs} >= {
        "arch.mlp", "arch.resnet18", "arch.unet", "arch.cnn"}
