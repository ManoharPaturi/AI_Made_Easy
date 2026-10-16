"""3.3 — the rest of the planned blocks: metric-learning losses (ArcFace, CosFace, triplet,
supervised contrastive), SAM, stochastic weight averaging, Integrated Gradients, target
encoding and SMOTE, and the ArcFace recipe. Tiny synthetic data, a few epochs each."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_training import chain  # noqa: E402

from ai_made_easy.core.codegen.training_gen import generate_training  # noqa: E402
from ai_made_easy.core.codegen import CodegenError  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402

TRAINER = ("train.trainer", {"epochs": 6, "batch_size": 32, "device": "cpu", "seed": 0})
DATA = ("data.synthetic", {"n_features": 6, "n_classes": 3, "n_samples": 300, "seed": 1})
EMBED = [("core.dense", {"units": 32}), ("core.relu", {}), ("core.dense", {"units": 8})]
MLP = [("core.dense", {"units": 16}), ("core.relu", {}), ("core.dense", {"units": 3})]


def _run(graph, tmp_path, monkeypatch) -> tuple[str, dict]:
    monkeypatch.chdir(tmp_path)
    code = generate_training(graph, "pytorch")
    ns: dict = {"__name__": "aime_training"}
    exec(compile(code, f"<{graph.name}>", "exec"), ns)
    ns["main"]()
    return code, ns


def _final_accuracy(capsys) -> float:
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "accuracy=" in line]
    assert lines, out[-2000:]
    return float(lines[-1].split("accuracy=")[1].split()[0])


def test_new_blocks_are_registered_with_their_libraries():
    reg = get_registry()
    for tid in ("train.loss_arcface", "train.loss_cosface", "train.loss_triplet",
                "train.loss_supcon", "train.sam", "train.schedule_free_adamw",
                "eval.integrated_gradients",
                "ml.target_encode", "ml.smote"):
        assert reg.get(tid).description, tid
    smote = reg.get("ml.smote")
    assert smote.extra == "imbalance" and smote.requires == ("imblearn",)


@pytest.mark.parametrize("loss", [("train.loss_arcface", {"classes": 3}),
                                  ("train.loss_cosface", {"classes": 3}),
                                  ("train.loss_triplet", {"classes": 3}),
                                  ("train.loss_supcon", {"classes": 3})])
def test_metric_losses_learn_embeddings_that_classify(loss, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    graph = chain(f"metric_{loss[0].split('_')[-1]}", "6", EMBED,
                  [DATA, loss, ("train.adamw", {"lr": 0.01}), TRAINER,
                   ("eval.accuracy", {})])
    assert not [i for i in graph.validate() if i.severity == "error"]
    code, ns = _run(graph, tmp_path, monkeypatch)
    assert "EMBED_DIM = 8" in code and "NUM_CLASSES = 3" in code
    assert _final_accuracy(capsys) > 0.6          # chance is 1/3
    if loss[0] in ("train.loss_arcface", "train.loss_cosface"):
        assert "optimizer.add_param_group" in code   # the class centres train too


def test_metric_loss_classes_must_match_the_data():
    graph = chain("metric_mismatch", "6", EMBED,
                  [DATA, ("train.loss_arcface", {"classes": 5}), ("train.adamw", {}), TRAINER])
    errors = [i.message for i in graph.validate() if i.severity == "error"]
    assert any("3 classes" in m and "Set its classes to 3" in m for m in errors), errors
    # a squashed embedding gets a warning
    softmax = chain("metric_softmax", "6", [*EMBED, ("core.softmax", {})],
                    [DATA, ("train.loss_triplet", {"classes": 3}), ("train.adamw", {}), TRAINER])
    assert any("Dense layer" in i.message for i in softmax.validate()
               if i.severity == "warning")


def test_metric_losses_have_no_keras_form():
    graph = chain("metric_keras", "6", EMBED,
                  [DATA, ("train.loss_arcface", {"classes": 3}), ("train.adam", {}), TRAINER])
    with pytest.raises(CodegenError, match="ArcFace"):
        generate_training(graph, "keras")


def test_sam_trains_in_two_passes(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    graph = chain("sam", "6", MLP, [DATA, ("train.sam", {"lr": 0.05}), TRAINER,
                                    ("eval.accuracy", {})])
    code, _ = _run(graph, tmp_path, monkeypatch)
    assert "optimizer.first_step()" in code and "use_amp = False" in code
    assert "scaler.scale" not in code.split("def fit")[1].split("def ")[0]
    assert _final_accuracy(capsys) > 0.6


def test_sam_with_adamw_and_a_scheduler(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    graph = chain("sam_adamw", "6", MLP,
                  [DATA, ("train.sam", {"base": "adamw", "lr": 0.01, "adaptive": True}),
                   ("train.cosine_annealing_lr", {"T_max": 6}), TRAINER])
    _run(graph, tmp_path, monkeypatch)


def test_schedule_free_adamw_evaluates_its_average(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    graph = chain("sfree", "6", MLP, [DATA, ("train.schedule_free_adamw", {"lr": 0.01}),
                                      TRAINER, ("eval.accuracy", {})])
    code, _ = _run(graph, tmp_path, monkeypatch)
    assert "optimizer.eval()" in code and "optimizer.train()" in code
    assert _final_accuracy(capsys) > 0.6
    from ai_made_easy.core.training.spec import collect_spec

    with_schedule = chain("sfree_sched", "6", MLP,
                          [DATA, ("train.schedule_free_adamw", {}),
                           ("train.cosine_annealing_lr", {}), TRAINER])
    assert any("Schedule-Free" in w for w in collect_spec(with_schedule).warnings)


def test_swa_averages_from_its_start_epoch(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    layers = [("core.dense", {"units": 16}), ("core.batch_norm1d", {}), ("core.relu", {}),
              ("core.dense", {"units": 3})]
    trainer = ("train.trainer", {**TRAINER[1], "swa_start": 3})
    code, _ = _run(chain("swa", "6", layers, [DATA, ("train.adamw", {}), trainer]),
                   tmp_path, monkeypatch)
    assert "SWA_START = 3" in code and "update_bn(train_loader, swa, device)" in code
    assert "evaluate(scored" in code


def test_swa_and_ema_together_are_refused():
    trainer = ("train.trainer", {**TRAINER[1], "swa_start": 2, "ema_decay": 0.9})
    graph = chain("both", "6", MLP, [DATA, ("train.adamw", {}), trainer])
    assert any("EMA or SWA" in i.message for i in graph.validate() if i.severity == "error")
    with pytest.raises(CodegenError, match="EMA or SWA"):
        generate_training(graph, "pytorch")
    with pytest.raises(CodegenError, match="SWA"):
        generate_training(chain("kswa", "6", MLP, [DATA, ("train.adam", {}), (
            "train.trainer", {**TRAINER[1], "swa_start": 2})]), "keras")


def test_plain_scripts_are_unchanged_by_the_new_options():
    code = generate_training(chain("plain", "6", MLP, [DATA, ("train.adamw", {}), TRAINER]),
                             "pytorch")
    for marker in ("EMBED_DIM", "SCORER", "SWA_START", "first_step", "add_param_group",
                   "optimizer.eval()"):
        assert marker not in code, marker


def test_integrated_gradients_finds_the_informative_feature(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    import numpy as np

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"signal": rng.normal(size=400), "noise": rng.normal(size=400),
                          "other": rng.normal(size=400)})
    frame["label"] = (frame["signal"] > 0).astype(int)
    csv = tmp_path / "table.csv"
    frame.to_csv(csv, index=False)
    graph = chain("ig", "3", [("core.dense", {"units": 16}), ("core.relu", {}),
                              ("core.dense", {"units": 2})],
                  [("data.csv", {"path": str(csv), "target_column": "label"}),
                   ("train.adam", {"lr": 0.02}),
                   ("train.trainer", {**TRAINER[1], "epochs": 10}),
                   ("eval.integrated_gradients", {"steps": 16})])
    code, _ = _run(graph, tmp_path, monkeypatch)
    assert "torch.autograd.grad" in code
    explained = json.loads((tmp_path / "explain.json").read_text())
    assert explained["method"] == "integrated_gradients"
    assert explained["features"][0]["name"] == "signal"


def test_one_explanation_per_design():
    graph = chain("two_explain", "6", MLP,
                  [DATA, ("train.adamw", {}), TRAINER, ("eval.permutation_importance", {}),
                   ("eval.integrated_gradients", {})])
    with pytest.raises(CodegenError, match="at most one"):
        generate_training(graph, "pytorch")


# ------------------------------------------------------------------ classic


def _imbalanced_table(tmp: Path) -> Path:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 400
    label = (rng.random(n) < 0.12).astype(int)              # 12 % positives
    frame = pd.DataFrame({
        "x1": rng.normal(size=n) + 2.0 * label, "x2": rng.normal(size=n),
        "city": rng.choice([f"c{i}" for i in range(30)], size=n), "label": label})
    path = tmp / "imbalanced.csv"
    frame.to_csv(path, index=False)
    return path


def test_target_encoding_and_smote_in_a_classic_pipeline(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    pytest.importorskip("imblearn")
    from verify_classic import pipeline, run_graph

    path = _imbalanced_table(tmp_path)
    graph = pipeline("classic_smote", [
        ("data.csv", {"path": str(path), "target_column": "label"}),
        ("prep.normalize", {}), ("ml.target_encode", {"cv": 3}),
        ("ml.smote", {"k_neighbors": 3}), ("ml.logistic_regression", {})])
    code = generate_training(graph, "sklearn")
    assert "from imblearn.pipeline import Pipeline" in code
    assert "TargetEncoder(smooth=\"auto\", cv=3, random_state=SEED)" in code
    assert '("smote", SMOTE(' in code and "imbalanced-learn" in code
    run_graph(graph, tmp_path, ("accuracy",))


def test_smote_needs_a_classifier_and_one_way_to_oversample():
    pytest.importorskip("sklearn")
    from verify_classic import pipeline

    regress = pipeline("smote_regress", [("data.sklearn", {"dataset": "diabetes"}),
                                         ("ml.smote", {}), ("ml.linear_regression", {})])
    with pytest.raises(CodegenError, match="classification"):
        generate_training(regress, "sklearn")
    both = pipeline("smote_both", [("data.sklearn", {"dataset": "iris"}), ("ml.smote", {}),
                                   ("prep.class_balance", {"strategy": "oversample"}),
                                   ("ml.logistic_regression", {})])
    with pytest.raises(CodegenError, match="one way to oversample"):
        generate_training(both, "sklearn")


def test_plain_classic_scripts_keep_sklearns_pipeline():
    pytest.importorskip("sklearn")
    from verify_classic import pipeline

    code = generate_training(pipeline("plain_classic", [("data.sklearn", {"dataset": "iris"}),
                                                        ("ml.logistic_regression", {})]),
                             "sklearn")
    assert "from sklearn.pipeline import Pipeline" in code and "imblearn" not in code


# ------------------------------------------------------------------ recipe


def test_arcface_recipe_builds_a_valid_embedding_model():
    from ai_made_easy.core.recipes import build

    graph = build("image_arcface", task="multiclass")
    types = {n.type_id for n in graph.nodes.values()}
    assert "train.loss_arcface" in types
    errors = [i for i in graph.validate() if i.severity == "error"]
    assert not errors, errors
    code = generate_training(graph, "pytorch")
    assert "MarginSoftmaxLoss(EMBED_DIM, NUM_CLASSES" in code and "NUM_CLASSES = 10" in code
