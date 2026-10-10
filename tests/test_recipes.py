"""Recipes and the new-project wizard: data detection, a valid design for every task,
designs fitted to the user's data and budget, explanations, the CLI and quick training
runs of the built recipes."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_made_easy.core import api, recipes
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.tasks import all_tasks, task_of

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch")


# ----------------------------------------------------------------- tiny datasets

def churn_csv(folder: Path, n: int = 300) -> Path:
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    city = rng.choice(["paris", "rome", "oslo"], size=n)
    label = np.where(x1 + (city == "rome") * 1.5 + rng.normal(scale=0.5, size=n) > 0.5,
                     "yes", "no")
    frame = pd.DataFrame({"x1": x1, "x2": x2, "city": city, "label": label})
    frame.loc[::17, "x2"] = np.nan
    path = folder / "churn.csv"
    frame.to_csv(path, index=False)
    return path


def reviews_csv(folder: Path, n: int = 120) -> Path:
    rng = np.random.default_rng(1)
    words = {"pos": "great lovely excellent superb fine wonderful".split(),
             "neg": "awful terrible bad poor horrible dull".split()}
    rows = [{"review": " ".join(rng.choice(words[k], 8)) + f" and then story number {i}",
             "sentiment": k} for i, k in enumerate(rng.choice(["pos", "neg"], n))]
    path = folder / "reviews.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def image_folder(folder: Path, per_class: int = 8) -> Path:
    from PIL import Image

    rng = np.random.default_rng(2)
    for name, colour in (("red", (220, 30, 30)), ("blue", (30, 30, 220))):
        (folder / "pics" / name).mkdir(parents=True)
        for i in range(per_class):
            pixels = np.clip(np.array(colour) + rng.normal(0, 20, (24, 24, 3)), 0, 255)
            Image.fromarray(pixels.astype("uint8")).save(folder / "pics" / name / f"{i}.png")
    return folder / "pics"


# ----------------------------------------------------------------- detection

def test_detect_tables(tmp_path):
    facts = recipes.detect(churn_csv(tmp_path))
    assert (facts.kind, facts.target, facts.tasks[0]) == ("table", "label", "binary")
    assert facts.categorical == ["city"] and facts.cardinalities == [3] and facts.missing
    assert sorted(facts.classes) == ["no", "yes"]
    house = tmp_path / "house.csv"
    pd.DataFrame({"a": np.arange(50.0), "price": np.arange(50.0) * 2}).to_csv(house,
                                                                               index=False)
    facts = recipes.detect(house, target="price")
    assert facts.target == "price" and facts.tasks[0] == "regression"
    series = tmp_path / "sales.csv"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60), "sales": np.arange(60.0)}
                 ).to_csv(series, index=False)
    facts = recipes.detect(series)
    assert (facts.kind, facts.time_column, facts.target) == ("series", "date", "sales")
    clicks = tmp_path / "clicks.csv"
    pd.DataFrame({"user_id": [1, 2, 3], "item_id": [4, 5, 6], "rating": [5, 3, 1]}).to_csv(
        clicks, index=False)
    facts = recipes.detect(clicks)
    assert facts.kind == "interactions" and facts.columns["rating"] == "rating"
    facts = recipes.detect(reviews_csv(tmp_path))
    assert (facts.kind, facts.text_column, facts.target) == ("text_table", "review",
                                                             "sentiment")
    (tmp_path / "book.txt").write_text("once upon a time " * 20)
    assert recipes.detect(tmp_path / "book.txt").task == "language_modeling"
    assert recipes.detect(tmp_path / "nope.csv").error
    assert recipes.detect(None).demo


@pytest.mark.skipif(not HAS_PIL, reason="needs Pillow")
def test_detect_folders(tmp_path):
    facts = recipes.detect(image_folder(tmp_path))
    assert facts.kind == "image_folder" and facts.classes == ["blue", "red"]
    assert (facts.image_size, facts.channels, facts.rows) == (24, 3, 16)
    coco = tmp_path / "coco"
    (coco / "images").mkdir(parents=True)
    (coco / "ann.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.png"}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]}],
        "categories": [{"id": 1, "name": "cat"}]}))
    facts = recipes.detect(coco)
    assert facts.kind == "coco" and facts.classes == ["cat"] and facts.task == "detection"
    seg = tmp_path / "seg"
    for sub in ("images", "masks"):
        (seg / sub).mkdir(parents=True)
    from PIL import Image

    Image.fromarray(np.zeros((8, 8, 3), "uint8")).save(seg / "images" / "a.png")
    Image.fromarray(np.array([[0, 2], [1, 255]], "uint8")).save(seg / "masks" / "a.png")
    facts = recipes.detect(seg)
    assert facts.kind == "mask_folder" and facts.n_classes == 3


# ----------------------------------------------------------------- recipes

def test_every_task_has_a_valid_design():
    covered = {t for r in recipes.all_recipes() for t in r.tasks}
    assert {t.id for t in all_tasks()} <= covered
    for task in all_tasks():
        ranked = recipes.recommend(task.id)
        assert ranked, task.id
        for sug in ranked:
            assert not (sug.recipe.installed and sug.errors), (sug.recipe.id, sug.errors)
            graph = Graph.from_dict(sug.graph)
            if sug.recipe.installed and sug.recipe.family == "neural":
                assert task_of(graph).id == task.id, sug.recipe.id
            assert graph.meta["recipe"]["id"] == sug.recipe.id


def test_designs_fit_the_users_data(tmp_path):
    facts = recipes.detect(churn_csv(tmp_path))
    ranked = recipes.recommend("binary", facts)
    ids = [s.recipe.id for s in ranked]
    assert ids[0] in ("gradient_boosting", "random_forest")      # trees lead on 300 rows
    assert not any(s.errors for s in ranked)
    mlp = Graph.from_dict(next(s.graph for s in ranked if s.recipe.id == "tabular_mlp"))
    assert mlp.nodes["input"].params["shape"] == "5"             # 2 numbers + 3 cities
    ft = next(s for s in ranked if s.recipe.id == "tabular_ft_transformer")
    assert any("may overfit" in c for c in ft.cautions)
    assert Graph.from_dict(ft.graph).nodes["ft_transformer"].params["cardinalities"] == "3"
    assert {"data", "impute", "encode", "normalize", "loss", "optimizer", "trainer"} <= \
        set(mlp.nodes)
    text = recipes.recommend("binary", recipes.detect(reviews_csv(tmp_path)))
    assert {s.recipe.id for s in text} == {"text_bag_of_embeddings", "text_lstm",
                                           "text_transformer"}
    assert not any(s.errors for s in text)


def test_modalities_and_demo_tasks():
    assert recipes.modalities_for("multiclass") == ["tabular", "image"]
    image = recipes.recommend("multiclass", modality="image")
    assert {s.recipe.id for s in image} == {"image_cnn", "image_cnn_augmented",
                                            "image_transfer"}
    assert all(Graph.from_dict(s.graph).nodes["dense"].params["units"] == 10 for s in image)
    assert not recipes.recommend("binary", modality="image")   # the image demo has 10 classes


def test_budget_ranks_and_flags():
    loose = recipes.recommend("multiclass", modality="tabular")
    tight = recipes.recommend("multiclass", budget={"max_params_m": 0.001},
                              modality="tabular")
    ft = next(s for s in tight if s.recipe.id == "tabular_ft_transformer")
    assert ft.over_budget and not ft.ready
    assert all(s.recipe.family == "classic" or s.over_budget or s.estimate["params"] <= 1000
               for s in tight)
    assert [s.recipe.id for s in loose] != [s.recipe.id for s in tight]
    roomy = recipes.recommend("multiclass", budget={"max_params_m": 5}, modality="tabular")
    assert all(s.ready for s in roomy)
    assert any("fits the budget" in r for s in roomy for r in s.reasons)


def test_knobs_and_errors(tmp_path):
    g = recipes.build("tabular_mlp", "regression", knobs={"width": 128, "depth": 3})
    dense = [n for n in g.nodes.values() if n.type_id == "core.dense"]
    assert [n.params["units"] for n in dense] == [128, 64, 32, 1]
    assert g.meta["recipe"]["knobs"]["width"] == 128
    with pytest.raises(recipes.RecipeError, match="within"):
        recipes.build("tabular_mlp", "regression", knobs={"width": 1})
    with pytest.raises(recipes.RecipeError, match="no setting"):
        recipes.build("tabular_mlp", "regression", knobs={"colour": 1})
    with pytest.raises(recipes.RecipeError, match="does not train"):
        recipes.build("image_cnn", "regression")
    with pytest.raises(recipes.RecipeError, match="cannot read"):
        recipes.build("image_cnn", "multiclass", recipes.detect(churn_csv(tmp_path)))
    ppo = recipes.build("ppo", knobs={"lr": 1e-3, "total_timesteps": 5000})
    assert ppo.nodes["algo"].params["total_timesteps"] == 5000


def test_starters_swap_in_the_users_data(tmp_path):
    clicks = tmp_path / "clicks.csv"
    pd.DataFrame({"user": np.arange(40) % 7, "item": np.arange(40) % 11}).to_csv(
        clicks, index=False)
    g = recipes.build("matrix_factorization", "recommendation", recipes.detect(clicks))
    data = g.nodes["data"]
    assert data.type_id == "data.interactions_csv" and data.params["user_column"] == "user"
    assert g.nodes["model"].params["n_users"] == 7               # fitted by a Quick Fix
    series = tmp_path / "sales.csv"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=200),
                  "sales": np.sin(np.arange(200.0))}).to_csv(series, index=False)
    g = recipes.build("forecast_patchtst", "forecasting", recipes.detect(series))
    assert next(n for n in g.nodes.values() if n.type_id.startswith("data.")).type_id == \
        "data.forecast_csv"
    assert not [i for i in g.validate() if i.severity == "error"]


def test_explain():
    g = recipes.build("tabular_mlp", "multiclass")
    out = recipes.explain(g)
    assert out["recipe"]["id"] == "tabular_mlp" and out["task"]["id"] == "multiclass"
    by_id = {b["id"]: b for b in out["blocks"]}
    assert by_id["loss"]["source"] == "recipe" and "cross-entropy" in by_id["loss"]["why"]
    assert by_id["output"]["source"] == "block"                  # falls back to the block
    sample = recipes.explain(Graph.from_dict(api.read_sample("iris_mlp.json")))
    assert sample["recipe"] is None and all(b["why"] for b in sample["blocks"])


def test_input_width_rule_for_any_network(tmp_path):
    path = churn_csv(tmp_path)
    data = {"name": "t", "nodes": [
        {"id": "in", "type": "core.input", "params": {"shape": "9"}},
        {"id": "d", "type": "core.dense", "params": {"units": 2}},
        {"id": "out", "type": "core.output", "params": {}},
        {"id": "data", "type": "data.csv", "params": {"path": str(path),
                                                      "target_column": "label"}},
        {"id": "hot", "type": "prep.one_hot", "params": {}}],
        "edges": [{"from": "in/out", "to": "d/in"}, {"from": "d/out", "to": "out/in"}]}
    issues = Graph.from_dict(data).validate()
    assert any("Set the Input shape to '5'" in i.message for i in issues)
    fitted, applied = recipes.autofit(Graph.from_dict(data))
    assert fitted.nodes["in"].params["shape"] == "5" and applied


def test_api_and_cli(tmp_path):
    path = churn_csv(tmp_path)
    tasks = {t["id"]: t for t in api.wizard_tasks()["tasks"]}
    assert tasks["multiclass"]["demo_modalities"] == ["tabular", "image"]
    assert {"text", "audio"} <= set(tasks["binary"]["modalities"])
    assert tasks["binary"]["automl"] and not tasks["probabilistic_inference"]["automl"]
    out = api.recommend_recipes("binary", path=str(path), limit=2)
    assert out["facts"]["target"] == "label" and len(out["suggestions"]) == 2
    with pytest.raises(api.ApiError):
        api.recommend_recipes("binary", path=str(tmp_path / "missing.csv"))
    graph = api.build_recipe("random_forest", "binary", path=str(path))
    assert graph["meta"]["recipe"]["id"] == "random_forest"
    project = tmp_path / "churn.json"
    proc = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "new", "--task", "binary",
                           "--data", str(path), "-o", str(project)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "using Gradient Boosting" in proc.stderr or "using Random Forest" in proc.stderr
    saved = Graph.from_dict(json.loads(project.read_text()))
    assert not [i for i in saved.validate() if i.severity == "error"]
    proc = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "recipes", "--task",
                           "regression"], capture_output=True, text=True, timeout=300)
    assert "tabular_resnet" in proc.stdout and "image_cnn" not in proc.stdout


# ----------------------------------------------------------------- training

@needs_torch
def test_built_recipes_train(tmp_path, isolated_home):
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    jobs = [(churn_csv(tmp_path), "tabular_mlp", {"epochs": 3}, "accuracy"),
            (churn_csv(tmp_path), "random_forest", {"n_estimators": 50}, "accuracy"),
            (reviews_csv(tmp_path), "text_bag_of_embeddings", {"epochs": 2}, "accuracy")]
    if HAS_PIL:
        jobs.append((image_folder(tmp_path), "image_cnn", {"epochs": 2, "image_size": 32},
                     "accuracy"))
    jobs += [(None, "tabular_mlp", {"epochs": 3}, "multilabel"),
             (None, "tabular_mlp", {"epochs": 3}, "distribution")]
    for path, recipe_id, knobs, metric in jobs:
        facts = recipes.detect(path)
        task = metric if path is None else facts.task
        graph = recipes.build(recipe_id, task, facts, knobs)
        run_id = mgr.start(graph)
        status = mgr.wait(run_id, 600)
        rec = mgr.history.get(run_id)
        assert status["state"] == "finished", (recipe_id, rec.error[-2000:])
        if path is not None:
            assert metric in rec.final_metrics, (recipe_id, rec.final_metrics)
        else:                                   # demo multi-label / soft-label targets
            losses = [e["metrics"]["val_loss"] for e in mgr.history.epochs(run_id)]
            assert losses[-1] < losses[0], (task, losses)


@pytest.mark.parametrize("code,returncode", [("'the data does not fit'", 1), ("3", 3),
                                              ("None", 0)])
def test_worker_exit_codes(tmp_path, code, returncode):
    """A script that stops with SystemExit("message") is a failed run (Python's rule)."""
    from ai_made_easy.core.runner.manager import worker_script_path

    script = tmp_path / "train.py"
    script.write_text(f"raise SystemExit({code})\n")
    proc = subprocess.run([sys.executable, str(worker_script_path()), str(script)],
                          capture_output=True, text=True, timeout=120)
    from ai_made_easy.core.runner.protocol import parse_event

    events = [e for e in map(parse_event, proc.stdout.splitlines()) if e]
    done = next(e for e in events if e["type"] == "done")
    assert done["returncode"] == returncode
    if returncode == 1:
        assert any(e["type"] == "error" and "does not fit" in e["traceback"] for e in events)
