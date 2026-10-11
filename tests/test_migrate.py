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


def _v2_file(sample: str) -> dict:
    """A project exactly as 2.0.0 saved it: schema 2, no family / task descriptors."""
    from ai_made_easy.core import api

    return {**api.read_sample(sample), "schema_version": 2}


def test_v2_projects_load_unchanged():
    for name in ("iris_mlp.json", "classic_random_forest.json", "llm_rag_assistant.json",
                 "cifar10_transfer_resnet18.json", "skip_connection_mlp.json"):
        old = _v2_file(name)
        data = migrate(old)
        assert data["schema_version"] == CURRENT_SCHEMA == 3
        assert {k: v for k, v in data.items() if k != "schema_version"} == \
            {k: v for k, v in old.items() if k != "schema_version"}
        graph = Graph.from_dict(old)
        assert not [i for i in graph.validate() if i.severity == "error"], name
        saved = graph.to_dict()
        assert [(n["type"], n["params"]) for n in saved["nodes"]] == \
            [(n["type"], n["params"]) for n in old["nodes"]]


def test_every_sample_validates_after_migration():
    import json

    from ai_made_easy.core.api import SAMPLES_DIR

    for path in sorted(SAMPLES_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        graph = Graph.from_dict(data)
        errors = [i.message for i in graph.validate() if i.severity == "error"
                  and "not installed" not in i.message]
        assert graph.to_dict()["schema_version"] == CURRENT_SCHEMA
        # missing optional extras are warnings; anything else would be a broken sample
        assert not errors, (path.name, errors[:2])


def test_stamp_records_family_and_task():
    from ai_made_easy.core import api
    from ai_made_easy.core.migrate import kind_of, stamp

    cases = {"iris_mlp.json": ("neural", "multiclass"),
             "student_network.json": ("pgm", "probabilistic_inference"),
             "shapes_detection.json": ("neural", "detection"),
             "pipeline_distill_quantize.json": ("pipeline", None),
             "classic_random_forest.json": ("classic", None)}
    for name, (family, task) in cases.items():
        data = stamp(api.read_sample(name))
        assert (data["schema_version"], data["family"], data["task"]) == (3, family, task)
        assert Graph.from_dict(data).to_dict()["nodes"] == \
            Graph.from_dict(api.read_sample(name)).to_dict()["nodes"]
    assert kind_of(_v2_file("iris_mlp.json")) == \
        {"family": "Neural network", "task": "Multi-class classification"}
    # an unfinished design has a family but no task yet
    assert stamp({"name": "x", "nodes": [], "edges": []})["task"] is None


def test_server_lists_family_and_task(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import json

    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    projects = tmp_path / "projects"
    projects.mkdir()
    (projects / "old.json").write_text(json.dumps(_v2_file("iris_mlp.json")))
    (projects / "broken.json").write_text("{not json")
    with TestClient(create_app(token="", projects_dir=projects)) as client:
        from ai_made_easy.core import api

        assert client.put("/api/projects/bn", json=api.read_sample("student_network.json")) \
            .status_code == 200
        saved = json.loads((projects / "bn.json").read_text())
        assert (saved["schema_version"], saved["family"]) == (3, "pgm")
        rows = {r["name"]: r for r in client.get("/api/projects").json()["projects"]}
    assert (rows["old"]["family"], rows["old"]["task"]) == \
        ("Neural network", "Multi-class classification")
    assert rows["bn"]["family"] == "Graphical model"
    assert rows["broken"]["family"] is None
