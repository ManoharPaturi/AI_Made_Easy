"""Graph IR tests: DAG traversal, shape inference, validation, JSON."""
from __future__ import annotations

import pytest

from ai_made_easy.core.graph import Edge, Graph, GraphError, NodeInstance
from ai_made_easy.core.spec import parse_shape


def build(*specs: tuple[str, str, dict]) -> Graph:
    """specs: (id, type, params) — nodes only; caller wires edges."""
    g = Graph(name="test")
    for nid, type_id, params in specs:
        g.add_node(NodeInstance(nid, type_id, params))
    return g


def wire(g: Graph, *pairs: tuple[str, str]) -> None:
    for src, dst in pairs:
        g.add_edge(Edge(src, "out", dst, g.nodes[dst].definition().inputs[0].name))


def make_mlp() -> Graph:
    g = build(
        ("in", "core.input", {"shape": "784"}),
        ("d1", "core.dense", {"units": 32}),
        ("r1", "core.relu", {}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "d1"), ("d1", "r1"), ("r1", "out"))
    return g


def test_registry_has_full_catalog():
    from ai_made_easy.core.registry import get_registry

    reg = get_registry()
    # 112 built-ins; canvas-template tests may register extra custom blocks
    assert len(reg.all()) >= 112
    for cat in ("Data", "Preprocessing", "Layers", "Activations", "Attention",
                "Tensor Ops", "Normalization", "Training", "Evaluation"):
        assert cat in reg.by_category()
    schemas = reg.list_blocks()
    assert any(s["type_id"] == "core.dense" for s in schemas)


def test_topo_order_respects_edges():
    order = make_mlp().topo_order()
    assert set(order) == {"in", "d1", "r1", "out"}
    assert order.index("in") < order.index("d1") < order.index("r1") < order.index("out")


def test_cycle_is_detected():
    g = make_mlp()
    g.add_edge(Edge("r1", "out", "d1", "in"))  # feedback: relu -> dense
    with pytest.raises(GraphError, match="cycle"):
        g.topo_order()


def test_model_nodes_walks_dag_in_order():
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
    ids = [n.instance_id for n in g.model_nodes()]
    assert ids[0] == "in"
    assert ids[-1] == "out"
    assert ids.index("r1") < ids.index("add") and ids.index("d1") < ids.index("add")


def test_shapes_flow_linear_chain():
    shapes = make_mlp().infer_shapes()
    assert shapes["in"] == [784]
    assert shapes["d1"] == [32]
    assert shapes["out"] == [32]


def test_conv_pool_shapes():
    g = build(
        ("in", "core.input", {"shape": "1, 28, 28"}),
        ("c1", "core.conv2d", {"out_channels": 16, "kernel_size": 3,
                                "stride": 1, "padding": 1, "dilation": 1}),
        ("p1", "core.maxpool2d", {"kernel_size": 2, "stride": 2, "padding": 0}),
        ("g1", "core.global_avgpool2d", {}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "c1"), ("c1", "p1"), ("p1", "g1"), ("g1", "out"))
    shapes = g.infer_shapes()
    assert shapes["c1"] == [16, 28, 28]
    assert shapes["p1"] == [16, 14, 14]
    assert shapes["g1"] == [16]


def test_lstm_and_embedding_shapes():
    g = build(
        ("in", "core.input", {"shape": "20, 32"}),
        ("l1", "core.lstm", {"hidden_size": 16, "num_layers": 1,
                              "bias": True, "bidirectional": False}),
        ("m1", "core.mean_over_time", {}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "l1"), ("l1", "m1"), ("m1", "out"))
    shapes = g.infer_shapes()
    assert shapes["l1"] == [20, 16]
    assert shapes["m1"] == [16]


def test_dense_rejects_multidimensional_input():
    g = build(
        ("in", "core.input", {"shape": "28, 28"}),
        ("d1", "core.dense", {"units": 10}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "d1"), ("d1", "out"))
    issues = g.validate()
    assert any("Flatten" in i.message for i in issues)


def test_flatten_reconciles_shapes():
    g = build(
        ("in", "core.input", {"shape": "28, 28"}),
        ("f1", "core.flatten", {}),
        ("d1", "core.dense", {"units": 10}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "f1"), ("f1", "d1"), ("d1", "out"))
    assert g.validate() == []
    assert g.infer_shapes()["f1"] == [784]


def test_add_rejects_mismatched_shapes():
    g = build(
        ("in", "core.input", {"shape": "16"}),
        ("d1", "core.dense", {"units": 32}),
        ("add", "core.add", {}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "d1"))
    g.add_edge(Edge("d1", "out", "add", "in1"))
    g.add_edge(Edge("in", "out", "add", "in2"))
    g.add_edge(Edge("add", "out", "out", "in"))
    issues = g.validate()
    assert any("identical shapes" in i.message for i in issues)


def test_concat_sums_axis_dim():
    g = build(
        ("in", "core.input", {"shape": "16"}),
        ("d1", "core.dense", {"units": 32}),
        ("cat", "core.concatenate", {"axis": -1}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "d1"))
    g.add_edge(Edge("d1", "out", "cat", "in1"))
    g.add_edge(Edge("in", "out", "cat", "in2"))
    g.add_edge(Edge("cat", "out", "out", "in"))
    assert g.infer_shapes()["cat"] == [48]


def test_reshape_flattens():
    g = build(
        ("in", "core.input", {"shape": "16, 4"}),
        ("r1", "core.reshape", {"target": "64"}),
        ("out", "core.output", {}),
    )
    wire(g, ("in", "r1"), ("r1", "out"))
    assert g.infer_shapes()["r1"] == [64]


def test_floating_config_blocks_do_not_pollute_validation():
    """Training/eval blocks are canvas configs; they must not sit in tensor
    flow (no ports to wire) and must not trigger model-flow errors."""
    g = make_mlp()
    g.add_node(NodeInstance("opt", "train.adam", {"lr": 0.01}))
    g.add_node(NodeInstance("met", "eval.f1", {"average": "macro"}))
    assert g.validate() == []


def test_unconnected_input_flagged():
    g = build(
        ("in", "core.input", {"shape": "8"}),
        ("d1", "core.dense", {"units": 4}),
        ("out", "core.output", {}),
    )
    g.add_edge(Edge("in", "out", "out", "in"))
    d1_issues = [i for i in g.validate() if i.node_id == "d1"]
    assert d1_issues and "not connected" in d1_issues[0].message


def test_json_roundtrip():
    g = make_mlp()
    g2 = Graph.from_dict(g.to_dict())
    assert g2.name == "test"
    assert set(g2.nodes) == set(g.nodes)
    assert len(g2.edges) == len(g.edges)
    assert g2.validate() == []


def test_parse_shape():
    assert parse_shape("784") == [784]
    assert parse_shape("28, 28, 1") == [28, 28, 1]
    with pytest.raises(ValueError):
        parse_shape("")
    with pytest.raises(ValueError):
        parse_shape("0, 3")
