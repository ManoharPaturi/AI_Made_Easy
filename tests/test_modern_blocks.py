"""3.3 Phase 1: modern network parts, new optimizers and weight averaging."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from verify_codegen import block_cases, run_keras, run_torch  # noqa: E402
from verify_training import chain  # noqa: E402

from ai_made_easy.core.blocks import modern  # noqa: E402
from ai_made_easy.core.codegen import class_name_for, generate  # noqa: E402
from ai_made_easy.core.codegen.training_gen import generate_training  # noqa: E402
from ai_made_easy.core.graph import Graph  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402

IDS = sorted(d.type_id for d in modern._blocks())
VARIANTS = {"vision.swin_block": [{"shift": True}],
            "core.gqa_attention": [{"heads": 4, "kv_heads": 1}],
            "core.mixture_of_experts": [{"experts": 2, "top_k": 1}]}


def _cases():
    return [c for c in block_cases(VARIANTS) if c[0] in IDS]


def test_every_new_block_has_a_harness_case():
    assert sorted({c[0] for c in _cases()}) == IDS


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c[0]}-{c[2]}")
def test_runtime_shapes_and_parameter_counts_match(case):
    pytest.importorskip("torch")
    type_id, _shape, _params, graph = case
    shapes = graph.infer_shapes()
    assert run_torch(graph) == tuple(shapes["blk"])
    node = graph.nodes["blk"]
    defn = node.definition()
    if defn.supports("keras"):
        pytest.importorskip("keras")
        got, want = run_keras(graph)
        assert got == want
    ns: dict = {"__name__": "aime_generated"}
    exec(generate(graph, "pytorch"), ns)
    real = sum(p.numel() for p in ns[class_name_for(graph.name)]().parameters())
    ins = [shapes["inp"]] * len(defn.inputs)
    assert defn.param_fn(ins, node.resolved_params()) == real
    assert defn.meta["cost"](ins, node.resolved_params())[0] == real


def test_shape_errors_explain_the_fix():
    reg = get_registry()
    with pytest.raises(Exception, match="divide heads"):
        reg.get("core.gqa_attention").shape_fn([[8, 16]], {"heads": 4, "kv_heads": 3})
    with pytest.raises(Exception, match="not divisible into"):
        reg.get("vision.patch_embedding").shape_fn([[3, 30, 30]], {"patch": 4, "dim": 8,
                                                                   "cls_token": True})
    with pytest.raises(Exception, match="more than one window"):
        reg.get("vision.swin_block").shape_fn([[4, 8, 8]], {"heads": 2, "window": 8,
                                                            "shift": True, "mlp_ratio": 4.0})
    with pytest.raises(Exception, match="odd"):
        reg.get("seq.conformer").shape_fn([[8, 16]], {"heads": 4, "kernel_size": 4})


def test_new_categories_are_colour_pure():
    colours = {}
    for d in modern._blocks():
        colours.setdefault(d.category, set()).add(d.color)
    assert all(len(c) == 1 for c in colours.values())


# ------------------------------------------------------------------ training

TRAINER = ("train.trainer", {"epochs": 2, "batch_size": 32, "device": "cpu"})
DATA = ("data.synthetic", {"n_features": 6, "n_classes": 3, "n_samples": 200})
MLP = [("core.dense", {"units": 8}), ("core.relu", {}), ("core.dense", {"units": 3})]


def _run(graph, tmp_path, monkeypatch, framework="pytorch") -> str:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    code = generate_training(graph, framework)
    ns: dict = {"__name__": "aime_training"}
    exec(compile(code, f"<{graph.name}>", "exec"), ns)
    ns["main"]()
    return code


def test_weight_averaging_trains_and_keeps_the_average(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    trainer = ("train.trainer", {**TRAINER[1], "ema_decay": 0.9})
    graph = chain("ema", "6", MLP, [DATA, ("train.adamw", {}), trainer])
    code = _run(graph, tmp_path, monkeypatch)
    assert "AveragedModel" in code and "evaluate(ema.module" in code
    plain = generate_training(chain("plain", "6", MLP, [DATA, ("train.adamw", {}), TRAINER]),
                              "pytorch")
    assert "EMA_DECAY" not in plain and "ema" not in plain.split("def fit")[1].split("\n")[0]


def test_keras_weight_averaging(tmp_path, monkeypatch):
    pytest.importorskip("keras")
    trainer = ("train.trainer", {**TRAINER[1], "ema_decay": 0.9})
    graph = chain("kema", "6", MLP, [DATA, ("train.adam", {}), trainer])
    code = _run(graph, tmp_path, monkeypatch, "keras")
    assert "use_ema=True" in code


@pytest.mark.parametrize("opt_id", ["train.lion", "train.lamb"])
def test_keras_lion_and_lamb(opt_id, tmp_path, monkeypatch):
    pytest.importorskip("keras")
    graph = chain(f"k_{opt_id.split('.')[-1]}", "6", MLP, [DATA, (opt_id, {}), TRAINER])
    _run(graph, tmp_path, monkeypatch, "keras")


def _llama_design() -> dict:
    nodes = [("in", "core.input", {"shape": "32", "dtype": "int64"}),
             ("emb", "core.embedding", {"num_embeddings": 256, "embedding_dim": 32}),
             ("b1", "core.llama_block", {"heads": 4, "kv_heads": 2}),
             ("b2", "core.llama_block", {"heads": 4, "kv_heads": 2}),
             ("norm", "core.rms_norm", {}),
             ("head", "core.dense", {"units": 256}),
             ("out", "core.output", {}),
             ("text", "data.synthetic_text", {"n_sentences": 300, "seed": 0}),
             ("opt", "train.adamw", {"lr": 0.003}),
             ("ppl", "eval.perplexity", {}),
             ("trainer", "train.trainer", {"epochs": 1, "batch_size": 32, "device": "cpu"})]
    chain_ids = ["in", "emb", "b1", "b2", "norm", "head", "out"]
    return {"name": "tiny_llama",
            "nodes": [{"id": i, "type": t, "params": p, "position": [0, 0]}
                      for i, t, p in nodes],
            "edges": [{"from": f"{a}/out", "to": f"{b}/in"}
                      for a, b in zip(chain_ids, chain_ids[1:], strict=False)]}


def test_a_llama_style_decoder_trains_as_a_language_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from ai_made_easy.core.tasks import task_of

    graph = Graph.from_dict(_llama_design())
    assert not [i for i in graph.validate() if i.severity == "error"], graph.validate()
    assert task_of(graph).trainer_kind == "language_model"
    _run(graph, tmp_path, monkeypatch)


def test_a_vision_transformer_trains_from_scratch(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    graph = chain("tiny_vit", "768",
                  [("core.reshape", {"target": "3,16,16"}),
                   ("vision.patch_embedding", {"patch": 4, "dim": 32}),
                   ("core.transformer_encoder", {"nhead": 4, "dim_feedforward": 64,
                                                 "norm_first": True}),
                   ("core.layer_norm", {}),
                   ("core.reduce", {"op": "mean", "dim": 0}),
                   ("core.dense", {"units": 3})],
                  [("data.synthetic", {"n_features": 768, "n_classes": 3, "n_samples": 96}),
                   ("train.adamw", {}), TRAINER])
    assert not [i for i in graph.validate() if i.severity == "error"], graph.validate()
    _run(graph, tmp_path, monkeypatch)


def test_mixture_of_experts_trains_with_its_balancing_loss(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    graph = chain("moe", "6", [("core.dense", {"units": 16}),
                               ("core.mixture_of_experts", {"experts": 4, "top_k": 2,
                                                            "hidden": 16}),
                               ("core.dense", {"units": 3})],
                  [DATA, ("train.lion", {}), TRAINER])
    code = _run(graph, tmp_path, monkeypatch)
    assert "m.aux_loss()" in code and "class Lion" in code


def test_tiny_llama_sample_is_a_valid_language_model():
    from ai_made_easy.core import api
    from ai_made_easy.core.tasks import task_of

    graph = Graph.from_dict(api.read_sample("tiny_llama.json"))
    assert not [i for i in graph.validate() if i.severity == "error"]
    assert task_of(graph).id == "language_modeling"


# ------------------------------------------------------------------ classic ML

@pytest.mark.parametrize("tid, params", [("ml.isomap", {"n_components": 3}),
                                         ("ml.nystroem", {"n_components": 40}),
                                         ("ml.umap", {"n_components": 2, "n_neighbors": 8})])
def test_new_classic_transformers_fit(tid, params, tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    if tid == "ml.umap":
        pytest.importorskip("umap")
    from verify_classic import pipeline, run_graph

    graph = pipeline(f"classic_{tid.split('.')[-1]}",
                     [("data.sklearn", {"dataset": "iris"}), ("prep.normalize", {}),
                      (tid, params), ("ml.logistic_regression", {})])
    assert not [i for i in graph.validate() if i.severity == "error"]
    run_graph(graph, tmp_path, ("accuracy",))


def test_umap_scripts_import_it_and_list_its_package():
    pytest.importorskip("sklearn")
    from verify_classic import pipeline

    defn = get_registry().get("ml.umap")
    assert defn.extra == "umap" and defn.requires == ("umap",)
    graph = pipeline("classic_umap", [("data.sklearn", {"dataset": "iris"}),
                                      ("ml.umap", {}), ("ml.kmeans", {"n_clusters": 3})])
    from ai_made_easy.core.targets import RENDERERS

    code = RENDERERS["sklearn_train"](graph)
    assert "from umap import UMAP" in code and "umap-learn" in code
