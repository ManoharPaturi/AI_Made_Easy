"""The five learner features: auto-fix, trainer progress + celebration,
dataset preview, starter missions, Predict block.
"""
from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_made_easy.core.fixes import fix_for_issue  # noqa: E402
from ai_made_easy.core.graph import Edge, Graph, NodeInstance  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402


def _mlp(extra=()):
    g = Graph(name="five")
    g.add_node(NodeInstance("i", "core.input", {"shape": "784"}, (0, 0)))
    g.add_node(NodeInstance("d", "core.dense", {"units": 8}, (200, 0)))
    g.add_node(NodeInstance("o", "core.output", {}, (400, 0)))
    for a, b in (("i", "d"), ("d", "o")):
        g.add_edge(Edge(a, "out", b, "in"))
    for nid, tid, params in extra:
        g.add_node(NodeInstance(nid, tid, params))
    return g


# ------------------------------------------------------------ auto-fix

def _rank_error_graph():
    g = Graph(name="fixme")
    g.add_node(NodeInstance("i", "core.input", {"shape": "1,28,28"}, (0, 0)))
    g.add_node(NodeInstance("c", "core.conv2d", {"out_channels": 8}, (200, 0)))
    g.add_node(NodeInstance("d", "core.dense", {"units": 10}, (400, 0)))
    g.add_node(NodeInstance("o", "core.output", {}, (600, 0)))
    for a, b in (("i", "c"), ("c", "d"), ("d", "o")):
        g.add_edge(Edge(a, "out", b, "in"))
    return g


def test_fix_inserts_flatten_for_rank_mismatch():
    g = _rank_error_graph()
    issue = next(i for i in g.validate()
                 if i.node_id == "d" and "Flatten" in i.message)
    result = fix_for_issue(g, issue)
    assert result is not None
    label, desc, fixed = result
    assert "Flatten" in label
    types = {n.type_id for n in fixed.nodes.values()}
    assert "core.flatten" in types
    # wired source -> flatten -> dense, and the fixed graph validates clean
    assert all(i.severity != "error" for i in fixed.validate())


def test_fix_clamps_out_of_range_params():
    g = _mlp()
    g.nodes["d"].params["units"] = -7
    issue = next(i for i in g.validate() if "below the minimum" in i.message)
    _, _, fixed = fix_for_issue(g, issue)
    assert fixed.nodes["d"].params["units"] == 1


def test_fix_sets_mha_embed_to_auto():
    g = _mlp()
    g.add_node(NodeInstance("m", "core.multihead_attention",
                            {"embed_dim": 100, "num_heads": 8}))
    issue = next(i for i in g.validate() if "divisible" in i.message)
    _, _, fixed = fix_for_issue(g, issue)
    assert fixed.nodes["m"].params["embed_dim"] == 0


def test_fix_halves_splitter_overlap():
    g = Graph()
    g.add_node(NodeInstance("m", "llm.model", {}))
    g.add_node(NodeInstance("s", "llm.text_splitter",
                            {"chunk_size": 100, "chunk_overlap": 100}))
    issue = next(i for i in g.validate() if "chunk_overlap" in i.message)
    _, _, fixed = fix_for_issue(g, issue)
    assert fixed.nodes["s"].params["chunk_overlap"] == 25


def test_fix_matches_input_shape_to_dataset():
    g = _mlp()
    g.add_node(NodeInstance("ds", "data.synthetic",
                            {"n_features": 64, "n_classes": 8}))
    issue = next(i for i in g.validate() if "features per sample" in i.message)
    _, _, fixed = fix_for_issue(g, issue)
    inputs = [n for n in fixed.nodes.values() if n.type_id == "core.input"]
    assert inputs[0].params["shape"] == "64"


def test_fix_removes_off_path_block():
    g = _mlp()
    g.add_node(NodeInstance("orphan", "core.relu", {}, (200, 150)))
    issue = next(i for i in g.validate()
                 if i.severity == "warning" and "is not on the path"
                 in i.message)
    _, _, fixed = fix_for_issue(g, issue)
    assert "orphan" not in fixed.nodes


def test_fix_wires_lonely_disconnected_block():
    g = Graph(name="wireme")
    g.add_node(NodeInstance("i", "core.input", {"shape": "4"}, (0, 0)))
    g.add_node(NodeInstance("r", "core.relu", {}, (200, 0)))
    g.add_node(NodeInstance("d", "core.dense", {"units": 2}, (400, 0)))
    g.add_node(NodeInstance("o", "core.output", {}, (600, 0)))
    g.add_edge(Edge("i", "out", "r", "in"))
    g.add_edge(Edge("o", "in", "d", "out")[0] if False else Edge("d", "out", "o", "in"))
    issue = next(i for i in g.validate()
                 if i.node_id == "d" and "is not connected" in i.message)
    result = fix_for_issue(g, issue)
    assert result is not None and result[0] == "Connect"
    wired = any(e.target_id == "d" for e in result[2].edges)
    assert wired


def test_fix_leaves_unknown_issues_alone():
    g = _mlp()
    from ai_made_easy.core.graph import ValidationIssue
    assert fix_for_issue(g, ValidationIssue("error", "mystery", "d")) is None


# ------------------------------------------------------ predict block

def test_predict_block_registered_and_generates():
    from ai_made_easy.core.codegen.training_gen import generate_training

    assert get_registry().has("eval.predict")
    g = _mlp([("p", "eval.predict", {"n_samples": 3})])
    for fw in ("pytorch", "keras"):
        code = generate_training(g, fw)
        import ast
        ast.parse(code)
        assert "prediction preview" in code


# ------------------------------------- trainer progress + celebration


# -------------------------------------------------------- data preview


# ------------------------------------------------------------ missions
