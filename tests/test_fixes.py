"""Regression tests for bugs found in the quality pass."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_made_easy.core.graph import Edge, Graph, NodeInstance  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402
from ai_made_easy.core.spec import ShapeError  # noqa: E402


def _mlp():
    g = Graph(name="t")
    for nid, tid, params in (("i", "core.input", {"shape": "784"}),
                             ("d", "core.dense", {"units": 32}),
                             ("o", "core.output", {})):
        g.add_node(NodeInstance(nid, tid, params))
    g.add_edge(Edge("i", "out", "d", "in"))
    g.add_edge(Edge("d", "out", "o", "in"))
    return g


def test_glu_odd_dim_raises_shape_error_not_name_error():
    glu = get_registry().get("core.glu")
    with pytest.raises(ShapeError):
        glu.shape_fn([[7]], {"dim": -1})
    assert glu.shape_fn([[8]], {"dim": -1}) == [4]


def test_kfold_does_not_hide_missing_optimizer_warning():
    g = _mlp()
    g.add_node(NodeInstance("tr", "train.trainer", {"epochs": 1}))
    g.add_node(NodeInstance("kf", "train.kfold"))
    warns = [i.message for i in g.validate()
             if i.severity == "warning" and i.node_id == "tr"]
    assert any("no optimizer configured" in m for m in warns)


def test_kfold_alone_is_not_flagged_as_orphan_scheduler():
    g = _mlp()
    g.add_node(NodeInstance("kf", "train.kfold"))
    assert not [i for i in g.validate() if i.node_id == "kf"
                and "without a Trainer" in i.message]


def test_optimizer_present_silences_warning():
    g = _mlp()
    g.add_node(NodeInstance("tr", "train.trainer", {"epochs": 1}))
    g.add_node(NodeInstance("opt", "train.adam"))
    warns = [i.message for i in g.validate() if i.node_id == "tr"]
    assert not any("optimizer" in m for m in warns)


def test_single_target_table_shared_by_ui_and_mcp():
    from ai_made_easy.core import targets
    from ai_made_easy.ui.services import export_service

    assert export_service.RENDERERS is targets.RENDERERS
    assert set(export_service.PREVIEW_TARGETS) == set(targets.RENDERERS)
    assert targets.target_label("pytorch_train") == "PyTorch training script"
    from ai_made_easy.core import api
    assert {t["id"] for t in api.targets()["targets"]} == set(targets.RENDERERS)
    pytest.importorskip("mcp")
    from ai_made_easy.mcp import server
    assert server.api is api  # MCP goes through the shared headless API
