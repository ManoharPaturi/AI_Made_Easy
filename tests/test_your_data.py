"""Your data everywhere (3.1 phase 2): image folders for VAE / GAN / diffusion, tables for
flows, grouped tables for PyMC, series for the state-space models and the HMM, user / item
feature tables beside interactions, and graph-level CSV datasets. Detection offers the
matching tasks, the wizard's recipes swap the user's data in, and each one trains."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_made_easy.core import recipes
from ai_made_easy.core.graph import Graph

pd = pytest.importorskip("pandas")


# ================================================================ tiny fixtures

def _images(root: Path, classes: tuple[str, ...] = (), n: int = 12) -> Path:
    from PIL import Image

    rng = np.random.default_rng(0)
    for i in range(n):
        folder = root / classes[i % len(classes)] if classes else root
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray((rng.random((20, 20, 3)) * 255).astype("uint8")).save(folder / f"{i}.png")
    return root


def _grouped(path: Path, binary: bool = False) -> Path:
    rng = np.random.default_rng(0)
    school = np.repeat([f"s{i}" for i in range(6)], 20)
    hours = rng.normal(size=120)
    outcome = (hours + rng.normal(0, 1, 120) > 0).astype(int) if binary \
        else 50 + 5 * hours + rng.normal(0, 3, 120)
    pd.DataFrame({"school": school, "hours": hours,
                  "passed" if binary else "score": outcome}).to_csv(path, index=False)
    return path


def _series(path: Path, steps: int = 72) -> Path:
    t = np.arange(steps)
    pd.DataFrame({"month": pd.date_range("2018-01-01", periods=steps, freq="MS"),
                  "sales": 100 + t + 10 * np.sin(t * 2 * np.pi / 12)
                  + np.random.default_rng(0).normal(0, 1, steps)}).to_csv(path, index=False)
    return path


def _interactions(root: Path) -> Path:
    rng = np.random.default_rng(0)
    root.mkdir()
    pd.DataFrame({"user": rng.integers(0, 30, 400), "item": rng.integers(0, 20, 400)}) \
        .drop_duplicates().to_csv(root / "interactions.csv", index=False)
    pd.DataFrame({"user": range(30), "age": rng.integers(18, 70, 30),
                  "tenure": rng.random(30)}).to_csv(root / "users.csv", index=False)
    pd.DataFrame({"item": range(20), "price": rng.random(20)}).to_csv(root / "items.csv",
                                                                       index=False)
    return root / "interactions.csv"


def _graphs(root: Path, labels: bool = True) -> Path:
    """Chains and rings: a ring closes its chain, and its nodes carry a hint."""
    rng = np.random.default_rng(0)
    root.mkdir()
    nodes, edges, graphs = [], [], []
    for g in range(40):
        n, ring = int(rng.integers(5, 9)), g % 2
        nodes += [{"graph_id": g, "id": v, "f1": ring + rng.random(), "f2": rng.random()}
                  for v in range(n)]
        edges += [{"graph_id": g, "source": v, "target": v - 1} for v in range(1, n)]
        edges += [{"graph_id": g, "source": 0, "target": n - 1}] if ring else []
        graphs.append({"graph_id": g, "label": "ring" if ring else "chain"})
    pd.DataFrame(nodes).to_csv(root / "nodes.csv", index=False)
    pd.DataFrame(edges).to_csv(root / "edges.csv", index=False)
    if labels:
        pd.DataFrame(graphs).to_csv(root / "graphs.csv", index=False)
    return root


def _shrink(graph: Graph) -> Graph:
    for node in graph.nodes.values():
        p = node.params
        if node.type_id == "train.trainer":
            p["epochs"] = 1
        elif node.type_id == "ppl.sampler":
            p.update(draws=100, tune=100, chains=1)
        elif node.type_id == "eval.fid":
            p["samples"] = 16
        elif node.type_id == "pgm.hmm":
            p.update(restarts=1, iterations=10)
    return graph


def _train(graph: Graph, workdir: Path) -> dict:
    from ai_made_easy.core.codegen import export_training
    from ai_made_easy.core.runner.manager import resolve_framework

    workdir.mkdir(exist_ok=True)
    script = export_training(graph, resolve_framework(graph), workdir)
    done = subprocess.run([sys.executable, script.name], cwd=workdir, capture_output=True,
                          text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-2500:]
    return json.loads((workdir / "metrics.json").read_text(encoding="utf-8"))


def _build(recipe: str, task: str, facts) -> Graph:  # noqa: ANN001
    graph = recipes.build(recipe, task, facts)
    errors = [i.message for i in graph.validate() if i.severity == "error"]
    assert not errors, errors
    return graph


# ================================================================ detection

def test_image_folders_offer_generative_tasks(tmp_path):
    flat = recipes.detect(_images(tmp_path / "flat"))
    assert flat.kind == "image_folder" and flat.classes == [] and flat.rows == 12
    assert flat.tasks == ["vae_generation", "gan_generation", "diffusion_generation"]
    assert flat.summary().startswith("image folder · 12 images")
    labelled = recipes.detect(_images(tmp_path / "pets", ("cat", "dog")))
    assert labelled.tasks[:2] == ["binary", "multiclass"]
    assert "diffusion_generation" in labelled.tasks


def test_tables_series_and_graphs_are_detected(tmp_path):
    grouped = recipes.detect(_grouped(tmp_path / "scores.csv"))
    assert grouped.columns["group"] == "school"
    assert {"bayesian_modeling", "density_estimation"} <= set(grouped.tasks)
    assert "bayesian_modeling" in recipes.detect(_grouped(tmp_path / "p.csv", True)).tasks
    series = recipes.detect(_series(tmp_path / "sales.csv"))
    assert series.kind == "series" and series.period == 12
    assert series.tasks == ["forecasting", "state_space_forecasting", "regime_detection"]
    rec = recipes.detect(_interactions(tmp_path / "shop"))
    assert set(rec.paths) == {"user_features", "item_features"}
    many = recipes.detect(_graphs(tmp_path / "mols"))
    assert (many.kind, many.rows, many.classes) == ("graphs", 40, ["chain", "ring"])
    assert many.columns["graph"] == "graph_id" and many.tasks == ["graph_classification"]
    assert recipes.detect(tmp_path / "mols" / "edges.csv").kind == "graphs"
    unlabelled = recipes.detect(_graphs(tmp_path / "bare", labels=False))
    assert "add a graph table" in unlabelled.error


# ================================================================ swaps

def test_recipes_swap_the_data_in(tmp_path):
    flat = recipes.detect(_images(tmp_path / "faces"))
    vae = _build("vae", "vae_generation", flat)
    assert vae.nodes["input"].params["shape"] == "3, 32, 32"
    assert vae.nodes["dec2"].params["out_channels"] == 3
    assert _build("dcgan", "gan_generation", flat).nodes["generator"].params["channels"] == 3
    ddpm = _build("ddpm", "diffusion_generation", flat)
    assert ddpm.nodes["denoiser"].params["num_classes"] == 0         # no class folders
    two = recipes.detect(_images(tmp_path / "pets", ("cat", "dog")))
    assert _build("ddpm", "diffusion_generation", two).nodes["denoiser"].params[
        "num_classes"] == 2

    flow = _build("realnvp", "density_estimation", recipes.detect(_grouped(tmp_path / "a.csv")))
    data = next(n for n in flow.nodes.values() if n.type_id == "data.csv")
    assert data.params["feature_columns"] == "hours, score"           # the numeric columns

    ppl = _build("hierarchical_regression", "bayesian_modeling",
                 recipes.detect(_grouped(tmp_path / "b.csv")))
    assert ppl.nodes["a"].params["group"] == "school"
    assert ppl.nodes["mu"].params["expression"] == "a + b * hours"
    assert ppl.nodes["y"].params["observed"] == "score"
    assert ppl.nodes["mu_a"].params["mu"] == pytest.approx(50, abs=2)   # priors on the data's scale
    wrong = recipes.recommend("bayesian_modeling", recipes.detect(tmp_path / "b.csv"))
    assert "must hold 0 / 1" in next(s for s in wrong
                                     if s.recipe.id == "hierarchical_logistic").errors[0]

    series = recipes.detect(_series(tmp_path / "sales.csv"))
    sts = _build("structural_time_series", "state_space_forecasting", series)
    assert sts.nodes["season"].params["period"] == 12
    assert sts.nodes["model"].params["target_column"] == "sales"
    hourly = recipes.detect(_series(tmp_path / "short.csv", 20))
    short = _build("structural_time_series", "state_space_forecasting", hourly)
    assert "season" not in short.nodes                     # too short for a seasonal component
    assert short.nodes["model"].params["horizon"] == 4

    dlrm = _build("dlrm", "recommendation", recipes.detect(_interactions(tmp_path / "shop")))
    assert dlrm.nodes["input"].params["shape"] == "5"      # user, item, 2 + 1 features
    gin = _build("gin", "graph_classification", recipes.detect(_graphs(tmp_path / "mols")))
    assert next(n for n in gin.nodes.values() if n.type_id == "data.graphs_csv")


def test_recommender_feature_tables_are_checked(tmp_path):
    path = _interactions(tmp_path / "shop")
    graph = _build("dlrm", "recommendation", recipes.detect(path))
    data = next(n for n in graph.nodes.values() if n.type_id == "data.interactions_csv")
    data.params["user_features"] = str(tmp_path / "missing.csv")
    assert any("user features: missing.csv" in i.message for i in graph.validate())


# ================================================================ training

def test_generative_and_flow_recipes_train_on_your_data(tmp_path):
    pytest.importorskip("torch")
    vae = _shrink(_build("vae", "vae_generation", recipes.detect(_images(tmp_path / "faces"))))
    assert _train(vae, tmp_path / "vae")["fid"] >= 0
    assert Path(tmp_path / "vae" / "eval_samples" / "samples.png").is_file()
    flow = _shrink(_build("realnvp", "density_estimation",
                          recipes.detect(_grouped(tmp_path / "a.csv"))))
    assert np.isfinite(_train(flow, tmp_path / "flow")["nll"])


def test_series_recipes_train_on_your_data(tmp_path):
    pytest.importorskip("statsmodels")
    series = recipes.detect(_series(tmp_path / "sales.csv"))
    metrics = _train(_build("structural_time_series", "state_space_forecasting", series),
                     tmp_path / "sts")
    assert metrics["mase"] < 1                     # beats the seasonal-naive forecast
    pytest.importorskip("hmmlearn")
    hmm = _shrink(_build("hmm", "regime_detection", series))
    assert np.isfinite(_train(hmm, tmp_path / "hmm")["test_log_likelihood"])


def test_grouped_table_trains_a_hierarchical_model(tmp_path):
    pytest.importorskip("pymc")
    graph = _shrink(_build("hierarchical_regression", "bayesian_modeling",
                           recipes.detect(_grouped(tmp_path / "scores.csv"))))
    metrics = _train(graph, tmp_path / "ppl")
    assert metrics["mean_b"] == pytest.approx(5, abs=1.5)    # the true slope


def test_recommender_and_graph_recipes_train_on_your_data(tmp_path):
    pytest.importorskip("torch")
    dlrm = _shrink(_build("dlrm", "recommendation",
                          recipes.detect(_interactions(tmp_path / "shop"))))
    assert _train(dlrm, tmp_path / "rec")["recall_at_10"] >= 0
    pytest.importorskip("torch_geometric")
    gin = _build("gin", "graph_classification", recipes.detect(_graphs(tmp_path / "mols")))
    next(n for n in gin.nodes.values() if n.type_id == "train.trainer").params["epochs"] = 30
    assert _train(gin, tmp_path / "gin")["accuracy"] > 0.7
