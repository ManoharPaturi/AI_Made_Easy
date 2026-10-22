"""Pretrained recipes (3.1 phase 3): text, audio and image classifiers on the Hugging Face,
timm and DINOv2 encoder blocks, each as a linear probe and a fine-tune. The wizard shows
what they download (size and licence from core.zoo) and hides them when they do not fit
the budget. Offline, the encoders run with random weights (weights = "none"); the nightly
job fine-tunes one real checkpoint per modality (tests/test_online.py)."""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from ai_made_easy.core import recipes, zoo
from ai_made_easy.core.budget import estimate
from ai_made_easy.core.graph import Graph

PRETRAINED = ("text_encoder_probe", "text_encoder_finetune", "audio_encoder_probe",
              "audio_encoder_finetune", "timm_probe", "timm_finetune", "dinov2_probe",
              "dinov2_finetune")
WORDS = ("great", "awful", "lovely", "broken", "superb", "useless", "fine", "slow")


# ================================================================ tiny fixtures
# (tests/test_online.py fine-tunes real checkpoints on the same fixtures)

def _reviews(path: Path) -> Path:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    rows = [{"review": f"review {i}: the product was " + " and ".join(
        rng.choice(WORDS[i % 2::2], 4)) + " according to the buyer", "label": i % 2}
            for i in range(32)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _clips(root: Path) -> Path:
    for label, freq in (("low", 220.0), ("high", 880.0)):
        (root / label).mkdir(parents=True)
        for i in range(6):
            t = np.arange(16000) / 16000
            tone = 0.5 * np.sin(2 * np.pi * (freq + 10 * i) * t)
            with wave.open(str(root / label / f"{i}.wav"), "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(16000)
                fh.writeframes((tone * 32767).astype(np.int16).tobytes())
    return root


def _photos(root: Path) -> Path:
    from PIL import Image

    rng = np.random.default_rng(0)
    for label, colour in (("red", (200, 30, 30)), ("blue", (30, 30, 200))):
        (root / label).mkdir(parents=True)
        for i in range(8):
            pixels = np.clip(rng.normal(colour, 30, (40, 40, 3)), 0, 255).astype("uint8")
            Image.fromarray(pixels).save(root / label / f"{i}.png")
    return root


def _offline(graph: Graph) -> Graph:
    """Random encoder weights, a word tokenizer and one short epoch: nothing downloads."""
    for node in graph.nodes.values():
        p = node.params
        if "weights" in p:
            p["weights"] = "none"
        if node.type_id == "prep.tokenize":
            p["method"] = "word"
        if node.type_id == "train.trainer":
            p.update(epochs=1, batch_size=4)
    return graph


def _train(graph: Graph, workdir: Path) -> dict:
    from ai_made_easy.core.codegen import export_training

    workdir.mkdir()
    script = export_training(graph, "pytorch", workdir)
    done = subprocess.run([sys.executable, script.name], cwd=workdir, capture_output=True,
                          text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-2500:]
    return json.loads((workdir / "metrics.json").read_text(encoding="utf-8"))


# ================================================================ the wizard

def test_wizard_shows_downloads_and_licences():
    assert zoo.download("vision.hf_image_encoder", {"model_id": "facebook/dinov2-small",
                                                    "weights": "pretrained"}) == {
        "model": "facebook/dinov2-small", "download_mb": 88, "license": "apache-2.0",
        "params_m": 22.1}
    assert zoo.download("vision.timm_backbone", {"model": "convnext_atto",
                                                 "weights": "none"}) is None
    assert zoo.download("core.hf_text_encoder", {"model_id": "roberta-base"})["license"] \
        == "mit"
    ranked = {s.recipe.id: s for s in recipes.recommend("keyword_spotting")}
    probe = ranked["audio_encoder_probe"]
    assert probe.downloads[0]["model"] == "facebook/wav2vec2-base"
    assert probe.downloads[0]["label"] == "facebook/wav2vec2-base · ~380 MB · apache-2.0 licence"
    assert probe.to_dict()["downloads"] == probe.downloads
    assert ranked["keyword_cnn"].downloads == []


def test_pretrained_recipes_are_offered_only_within_the_budget():
    roomy = [s.recipe.id for s in recipes.recommend("multiclass", modality="image")]
    assert {"timm_probe", "dinov2_probe", "dinov2_finetune"} <= set(roomy)
    small = [s.recipe.id for s in recipes.recommend("multiclass", modality="image",
                                                    budget={"max_params_m": 5})]
    assert "timm_probe" in small                    # ConvNeXt-atto: 3.4M parameters
    assert "dinov2_probe" not in small and "image_cnn" in small


def test_probes_freeze_the_encoder_and_fine_tunes_train_it():
    probe = estimate(recipes.build("dinov2_probe", "multiclass"))
    tune = estimate(recipes.build("dinov2_finetune", "multiclass"))
    assert probe.params == tune.params > 22e6
    assert probe.trainable_params < 1e5 < 22e6 < tune.trainable_params
    assert probe.train_memory_bytes() < tune.train_memory_bytes()
    audio = recipes.build("audio_encoder_probe", "keyword_spotting",
                          knobs={"model": "facebook/hubert-base-ls960"})
    assert audio.nodes["hf_encoder"].params["model_id"] == "facebook/hubert-base-ls960"


def test_every_pretrained_recipe_builds_for_your_data(tmp_path):
    pytest.importorskip("PIL")
    facts = {"text": recipes.detect(_reviews(tmp_path / "reviews.csv")),
             "audio": recipes.detect(_clips(tmp_path / "clips")),
             "image": recipes.detect(_photos(tmp_path / "photos"))}
    assert facts["audio"].tasks[0] == "binary"     # a folder of clips trains a classifier
    for recipe_id in PRETRAINED:
        recipe = recipes.get_recipe(recipe_id)
        graph = recipes.build(recipe, "binary", facts[recipe.modality])
        errors = [i.message for i in graph.validate() if i.severity == "error"]
        assert not errors, (recipe_id, errors)
        data = next(n for n in graph.nodes.values() if n.type_id.startswith("data."))
        assert data.type_id in ("data.text_csv", "data.audio_folder", "data.image_folder")
        assert graph.meta["recipe"]["tier"] == "pretrained"


# ================================================================ training offline

def test_text_encoder_trains_offline(tmp_path):
    pytest.importorskip("transformers")
    graph = recipes.build("text_encoder_probe", "binary", recipes.detect(
        _reviews(tmp_path / "reviews.csv")), knobs={
        "model": "sentence-transformers/all-MiniLM-L6-v2", "max_length": 64})
    assert graph.nodes["tokenize"].params["hf_tokenizer"] == \
        "sentence-transformers/all-MiniLM-L6-v2"
    assert "accuracy" in _train(_offline(graph), tmp_path / "run")


def test_audio_encoder_trains_offline(tmp_path):
    pytest.importorskip("transformers")
    graph = recipes.build("audio_encoder_finetune", "binary",
                          recipes.detect(_clips(tmp_path / "clips")))
    assert "accuracy" in _train(_offline(graph), tmp_path / "run")


def test_image_encoders_train_offline(tmp_path):
    pytest.importorskip("PIL")
    facts = recipes.detect(_photos(tmp_path / "photos"))
    if importlib_ok("timm"):
        timm = recipes.build("timm_probe", "binary", facts, knobs={"image_size": 64})
        assert "accuracy" in _train(_offline(timm), tmp_path / "timm")
    pytest.importorskip("transformers")
    dino = recipes.build("dinov2_finetune", "binary", facts, knobs={"image_size": 112})
    assert "accuracy" in _train(_offline(dino), tmp_path / "dino")


def importlib_ok(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None
