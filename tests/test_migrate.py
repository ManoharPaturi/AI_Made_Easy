"""Older project files open with unchanged behaviour."""
from __future__ import annotations

import pytest

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.migrate import CURRENT_SCHEMA, migrate

V1 = {"name": "old", "nodes": [
    {"id": "i", "type": "core.input", "params": {"shape": "4, 16, 16"}},
    {"id": "p", "type": "core.adaptive_avgpool2d", "params": {"output_height": 2,
                                                              "output_width": 2}},
    {"id": "f", "type": "core.flatten", "params": {}},
    {"id": "n", "type": "core.layer_norm", "params": {}},
    {"id": "o", "type": "core.output", "params": {}}],
    "edges": [{"from": "i/out", "to": "p/in"}, {"from": "p/out", "to": "f/in"},
              {"from": "f/out", "to": "n/in"}, {"from": "n/out", "to": "o/in"}]}


def test_v1_projects_migrate_with_unchanged_semantics():
    data = migrate(V1)
    assert data["schema_version"] == CURRENT_SCHEMA
    params = {n["id"]: n["params"] for n in data["nodes"]}
    assert params["p"] == {"output_size": 2}
    assert params["n"]["normalized_dims"] == "all"
    assert params["i"]["dtype"] == "float32"
    assert "schema_version" not in V1, "input must not be mutated"
    g = Graph.from_dict(V1)
    assert not [i for i in g.validate() if i.severity == "error"]
    assert g.to_dict()["schema_version"] == CURRENT_SCHEMA


def test_newer_schema_is_rejected():
    with pytest.raises(ValueError, match="newer version"):
        migrate({"schema_version": CURRENT_SCHEMA + 1, "nodes": [], "edges": []})
