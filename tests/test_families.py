"""V3 foundations: families, tasks, port roles, optional requirements, trainer kinds, zoo."""
from __future__ import annotations

import importlib.util
import json

import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core import api
from ai_made_easy.core.codegen import CodegenError
from ai_made_easy.core.families import Family, family_of, get_family, register_family
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ROLES, BlockDefinition, PortSpec, roles_compatible
from ai_made_easy.core.tasks import Task, all_tasks, get_task, register_task, resolve_task
from ai_made_easy.core.training import catalog as cat


@pytest.fixture()
def temp_blocks():
    """Register throw-away blocks for one test."""
    reg = get_registry()
    added: list[str] = []

    def add(defn: BlockDefinition) -> BlockDefinition:
        reg.register(defn)
        added.append(defn.type_id)
        return defn

    yield add
    for type_id in added:
        reg.unregister(type_id)


# ================================================================ families

@pytest.mark.parametrize("sample, family", [
    ("mnist_cnn.json", "neural"), ("iris_mlp.json", "neural"),
    ("classic_random_forest.json", "classic"), ("classic_kmeans.json", "classic"),
    ("llm_rag_assistant.json", "llm"), ("llm_generation.json", "llm")])
def test_family_of_samples(sample, family):
    data = api.read_sample(sample)
    assert family_of(data).id == family
    assert family_of(Graph.from_dict(data)).id == family
    assert api.validate(data)["valid"]


def test_family_labels_and_frameworks():
    from ai_made_easy.core.runner.manager import resolve_framework
    from ai_made_easy.ui.services.project_service import project_kind

    assert project_kind(api.read_sample("classic_kmeans.json")) == "Classic ML pipeline"
    assert project_kind(api.read_sample("llm_generation.json")) == "LLM workflow"
    assert project_kind(tiny_classifier_dict()) == "Neural network"
    with pytest.raises(ValueError, match="not trained in the app"):
        resolve_framework(Graph.from_dict(api.read_sample("llm_generation.json")))
    with pytest.raises(ValueError, match="train with pytorch or keras"):
        resolve_framework(Graph.from_dict(tiny_classifier_dict()), "sklearn")
    families = {f["id"]: f for f in api.list_families()["families"]}
    assert families["neural"]["frameworks"] == ["pytorch", "keras"]
    assert families["llm"]["trainable"] is False


def test_custom_family_validation(temp_blocks):
    temp_blocks(BlockDefinition("zz.node", "Thing", "Testing"))
    calls = []

    def validate(graph):
        calls.append(graph)
        from ai_made_easy.core.graph import ValidationIssue

        return [ValidationIssue("warning", "zz rule fired")]

    register_family(Family("zz", "ZZ models", detect=lambda t: "zz.node" in t, priority=50,
                           validate=validate, trainable=False))
    try:
        data = {"name": "z", "nodes": [{"id": "a", "type": "zz.node", "params": {},
                                         "position": [0, 0]}], "edges": []}
        issues = api.validate(data)["issues"]
        assert calls and [i["message"] for i in issues] == ["zz rule fired"]
        assert api.describe_design(data)["family"]["id"] == "zz"
    finally:
        from ai_made_easy.core import families

        families._FAMILIES.pop("zz")
    with pytest.raises(KeyError):
        get_family("zz")


# ================================================================ tasks

def test_builtin_tasks_cover_the_catalog():
    ids = {t.id for t in all_tasks()}
    assert {"multiclass", "binary", "multilabel", "regression", "distribution"} <= ids
    for loss_id in cat.LOSS_IDS:  # every loss trains a registered task
        resolve_task(cat.COMPONENTS[loss_id].meta["task"])
    for metric_id in cat.METRIC_IDS:
        assert set(cat.COMPONENTS[metric_id].meta["tasks"]) <= ids
    for task in all_tasks():
        assert set(task.default_metrics) <= set(task.metrics()), task.id
        if task.classification:
            assert task.losses() and task.metrics()
    assert resolve_task("binary", 3).id == "multilabel"
    assert resolve_task("binary", 1).id == "binary"
    with pytest.raises(KeyError):
        get_task("nope")


def test_task_of_designs():
    from ai_made_easy.core.tasks import task_of

    assert task_of(Graph.from_dict(tiny_classifier_dict())).id == "multiclass"
    regression = Graph.from_dict(api.read_sample("diabetes_regression_mlp.json"))
    assert task_of(regression).id == "regression"
    described = api.describe_design(tiny_classifier_dict())
    assert described["task"]["label"] == "Multi-class classification"
    assert "train.loss_cross_entropy" in described["task"]["losses"]


def test_unknown_trainer_kind_is_reported():
    from ai_made_easy.core.training.generate import generate_training

    with pytest.raises(ValueError, match="trainer kind"):
        register_task(Task("zz_task", "ZZ", trainer_kind="telepathy"))
    original = get_task("multiclass")
    register_task(Task("multiclass", original.label, trainer_kind="adversarial",
                       classification=True, default_metrics=original.default_metrics))
    try:
        with pytest.raises(CodegenError, match="adversarial trainer"):
            generate_training(Graph.from_dict(tiny_classifier_dict()), "pytorch")
    finally:
        register_task(original)
    assert "class " in generate_training(Graph.from_dict(tiny_classifier_dict()), "pytorch")


# ================================================================ ports

def test_port_roles(temp_blocks):
    assert roles_compatible("tensor", "boxes") and roles_compatible("masks", "tensor")
    assert not roles_compatible("boxes", "masks")
    with pytest.raises(ValueError, match="unknown role"):
        PortSpec("x", role="banana")
    assert set(ROLES) >= {"logits", "boxes", "masks", "variable", "graph"}
    temp_blocks(BlockDefinition("zz.boxes", "Boxes", "Testing",
                                outputs=(PortSpec("out", role="boxes"),)))
    temp_blocks(BlockDefinition("zz.masks", "Masks", "Testing",
                                inputs=(PortSpec("in", role="masks"),)))
    temp_blocks(BlockDefinition("zz.any", "Any", "Testing", inputs=(PortSpec("in"),)))
    graph = {"name": "r", "nodes": [
        {"id": "b", "type": "zz.boxes", "params": {}, "position": [0, 0]},
        {"id": "m", "type": "zz.masks", "params": {}, "position": [0, 0]},
        {"id": "a", "type": "zz.any", "params": {}, "position": [0, 0]}],
        "edges": [{"from": "b/out", "to": "m/in"}, {"from": "b/out", "to": "a/in"}]}
    messages = [i.message for i in Graph.from_dict(graph)._role_issues()]
    assert messages == ["'in' expects masks but receives boxes from Boxes"]
    schema = next(b for b in api.list_blocks()["blocks"] if b["type_id"] == "zz.boxes")
    assert schema["outputs"][0]["role"] == "boxes"


def test_missing_requirements_warn(temp_blocks):
    temp_blocks(BlockDefinition("zz.needs", "Needs Things", "Testing",
                                requires=("json", "surely_not_installed_pkg"),
                                extra="vision-tasks"))
    graph = Graph.from_dict({"name": "n", "nodes": [
        {"id": "x", "type": "zz.needs", "params": {}, "position": [0, 0]}], "edges": []})
    messages = [i.message for i in graph._requirement_issues()]
    assert len(messages) == 1 and "surely_not_installed_pkg" in messages[0]
    assert "ai-made-easy[vision-tasks]" in messages[0] and "json" not in messages[0][:30]
    row = next(b for b in api.list_blocks()["blocks"] if b["type_id"] == "zz.needs")
    assert row["missing"] == ["surely_not_installed_pkg"] and row["extra"] == "vision-tasks"


# ================================================================ zoo

def test_zoo_catalog_feeds_the_backbone_block():
    from ai_made_easy.core.blocks.pretrained import BACKBONES
    from ai_made_easy.core.zoo import catalog, catalogs, entry

    assert "torchvision_image" in catalogs()
    rows = catalog("torchvision_image")
    assert len(rows) >= 35 and len({r["name"] for r in rows}) == len(rows)
    for row in rows:
        assert row["feature_dim"] > 0 and row["params_m"] > 0 and row["license"]
    assert BACKBONES["resnet18"] == (512, None, 32, None)
    assert entry("torchvision_image", "vit_b_16")["fixed_side"] == 224
    options = get_registry().get("core.pretrained_backbone").params[0].options
    assert set(options) == {r["name"] for r in rows}


@pytest.mark.skipif(importlib.util.find_spec("keras") is None, reason="needs keras")
def test_zoo_keras_names_exist():
    import keras

    from ai_made_easy.core.zoo import catalog

    missing = [r["keras"] for r in catalog("torchvision_image")
               if r["keras"] and not hasattr(keras.applications, r["keras"])]
    assert missing == []


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="needs torchvision")
@pytest.mark.parametrize("arch, side", [("swin_t", 224), ("maxvit_t", 224),
                                        ("shufflenet_v2_x1_0", 64), ("efficientnet_v2_s", 64),
                                        ("resnext50_32x4d", 64), ("densenet169", 64)])
def test_new_backbones_generate_and_run(arch, side):
    import torch

    from ai_made_easy.core.codegen import generate

    data = {"name": "bb", "nodes": [
        {"id": "in", "type": "core.input", "params": {"shape": f"3, {side}, {side}"},
         "position": [0, 0]},
        {"id": "bb", "type": "core.pretrained_backbone",
         "params": {"architecture": arch, "weights": "none", "freeze": True},
         "position": [0, 0]},
        {"id": "out", "type": "core.output", "params": {}, "position": [0, 0]}],
        "edges": [{"from": "in/out", "to": "bb/in"}, {"from": "bb/out", "to": "out/in"}]}
    graph = Graph.from_dict(data)
    assert api.validate(data)["valid"], api.validate(data)["issues"]
    expected = graph.infer_shapes()["bb"]
    namespace: dict = {"__name__": "zoo_test"}
    exec(compile(generate(graph, "pytorch"), "<gen>", "exec"), namespace)  # noqa: S102
    with torch.no_grad():
        out = namespace["build_model"]().eval()(torch.zeros(1, 3, side, side))
    assert list(out.shape[1:]) == expected


# ================================================================ server

def test_server_family_endpoints(isolated_home, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    with TestClient(create_app(token="", projects_dir=tmp_path)) as client:
        tasks = client.get("/api/tasks").json()["tasks"]
        assert {"multiclass", "regression"} <= {t["id"] for t in tasks}
        assert client.get("/api/tasks", params={"family": "classic"}).json()["tasks"] == []
        families = client.get("/api/families").json()["families"]
        assert [f["id"] for f in families][-1] == "neural"
        described = client.post("/api/describe", json={"graph": tiny_classifier_dict()}).json()
        assert described["task"]["id"] == "multiclass"
        blocks = client.get("/api/blocks").json()["blocks"]
        assert all("missing" in b and "role" in b["outputs"][0] for b in blocks if b["outputs"])
    json.dumps(tasks)
