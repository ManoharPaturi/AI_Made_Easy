"""Generative models: VAE, GAN, diffusion, language models and seq2seq — helpers, data,
metrics, rules, profiles, training, serving and live samples."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_made_easy.core import api, budget
from ai_made_easy.core.fixes import fix_for_issue
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.tasks import task_of

torch = pytest.importorskip("torch")

VAE = [("core.conv2d", {"out_channels": 8, "kernel_size": 4, "stride": 2, "padding": 1}),
       ("core.relu", {}), ("core.flatten", {}), ("core.dense", {"units": 16}),
       ("gen.reparameterize", {}), ("core.dense", {"units": 512}), ("core.relu", {}),
       ("core.reshape", {"target": "8, 8, 8"}),
       ("core.conv_transpose2d", {"out_channels": 1, "kernel_size": 4, "stride": 2,
                                  "padding": 1}), ("core.sigmoid", {})]
IMAGES = ("data.synthetic_images", {"n_images": 120, "image_size": 16})
GAN = [("gen.dcgan_generator", {"channels": 1, "image_size": 16, "width": 8})]
UNET = [("gen.diffusion_unet", {"base_channels": 8, "depth": 2, "num_classes": 3})]
GPT = [("gen.gpt", {"context": 32, "d_model": 32, "layers": 1, "heads": 2})]
S2S = [("gen.seq2seq_transformer", {"max_target_len": 10, "d_model": 32, "encoder_layers": 1,
                                    "decoder_layers": 1, "heads": 2})]


@pytest.fixture(scope="module")
def rt():
    from ai_made_easy.core.generative.runtime import namespace

    return namespace()


def design(shape: str, layers: list, extra: tuple = (), *, epochs: int = 1,
           dtype: str = "float32", name: str = "gen_case") -> dict:
    nodes = [{"id": "in", "type": "core.input", "params": {"shape": shape, "dtype": dtype},
              "position": [0, 0]}]
    edges, prev = [], "in"
    for i, (type_id, params) in enumerate(layers):
        nodes.append({"id": f"l{i}", "type": type_id, "params": params, "position": [0, 0]})
        edges.append({"from": f"{prev}/out", "to": f"l{i}/in"})
        prev = f"l{i}"
    nodes += [{"id": "out", "type": "core.output", "params": {}, "position": [0, 0]},
              {"id": "tr", "type": "train.trainer",
               "params": {"epochs": epochs, "batch_size": 32, "device": "cpu"},
               "position": [0, 0]}]
    nodes += [{"id": f"x{j}", "type": t, "params": p, "position": [0, 0]}
              for j, (t, p) in enumerate(extra)]
    edges.append({"from": f"{prev}/out", "to": "out/in"})
    return {"name": name, "nodes": nodes, "edges": edges}


CASES = {
    "vae": design("1, 16, 16", VAE, (IMAGES,)),
    "gan": design("16", GAN, (IMAGES, ("gen.discriminator", {"width": 8}))),
    "diffusion": design("1, 16, 16", UNET, (IMAGES, ("gen.noise_scheduler", {
        "timesteps": 50, "sample_steps": 5, "sample_every": 1}))),
    "lm": design("32", GPT, (("data.synthetic_text", {"n_sentences": 300}),), dtype="int64"),
    "s2s": design("8", S2S, (("data.synthetic_pairs", {"n_pairs": 200}),), dtype="int64"),
}
TASKS = {"vae": "vae_generation", "gan": "gan_generation", "diffusion": "diffusion_generation",
         "lm": "language_modeling", "s2s": "sequence_to_sequence"}


def issues(data: dict) -> list:
    return [i for i in Graph.from_dict(data).validate() if "optimizer" not in i.message]


# ================================================================ tasks / samples

@pytest.mark.parametrize("key", list(CASES))
def test_designs_resolve_and_validate(key):
    graph = Graph.from_dict(CASES[key])
    assert task_of(graph).id == TASKS[key]
    assert not [i for i in graph.validate() if i.severity in ("error", "warning")
                and "optimizer" not in i.message]


@pytest.mark.parametrize("sample, task", [
    ("shapes_vae.json", "vae_generation"), ("shapes_gan.json", "gan_generation"),
    ("shapes_diffusion.json", "diffusion_generation"),
    ("tiny_gpt.json", "language_modeling"), ("reverse_seq2seq.json", "sequence_to_sequence")])
def test_samples_validate_and_resolve(sample, task):
    graph = Graph.from_dict(api.read_sample(sample))
    assert graph.validate() == []
    assert task_of(graph).id == task


# ================================================================ helpers

def _module(name: str, *args):
    from ai_made_easy.core.generative.helpers import GENERATIVE_HELPERS

    ns = {"torch": torch, "nn": torch.nn}
    for code in GENERATIVE_HELPERS.values():
        exec(code, ns)  # noqa: S102
    return ns[name](*args)


@pytest.mark.parametrize("key", ["gan", "diffusion", "lm", "s2s"])
def test_model_costs_match_torch(key):
    from ai_made_easy.core.codegen import class_name_for, generate

    graph = Graph.from_dict(CASES[key])
    ns: dict = {"__name__": "aime_generated"}
    exec(compile(generate(graph, "pytorch"), "<gen>", "exec"), ns)  # noqa: S102
    net = ns[class_name_for(graph.name)]().eval()
    shape = graph.infer_shapes()["in"]
    x = torch.randint(0, 256, (2, *shape)) if key in ("lm", "s2s") else torch.randn(2, *shape)
    with torch.no_grad():
        out = net(x)
    assert list(out.shape[1:]) == list(graph.infer_shapes()["l0"])
    assert budget.estimate(graph).params == sum(p.numel() for p in net.parameters())


def test_reparameterize_kl_and_prior():
    rep = _module("Reparameterize", 1.0).train()
    x = torch.zeros(4, 8)          # mean 0, log-variance 0: the prior itself
    rep(x)
    assert float(rep.kl) == pytest.approx(0.0, abs=1e-6)
    x[:, :4] = 1.0                  # mean 1: KL = 0.5 * 4 * 1
    rep(x)
    assert float(rep.kl) == pytest.approx(2.0, abs=1e-5)
    rep.prior_z = torch.ones(3, 4)
    assert torch.equal(rep(torch.zeros(5, 8)), torch.ones(3, 4))
    assert tuple(_module("Reparameterize", 1.0)(torch.zeros(2, 6, 4, 4)).shape) == (2, 3, 4, 4)


def test_gpt_is_causal():
    gpt = _module("GPT", 50, 16, 32, 2, 4, 0.0, True).eval()
    x = torch.randint(0, 50, (1, 16))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 50
    with torch.no_grad():
        a, b = gpt(x), gpt(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:])
    assert gpt.head.weight is gpt.tok.weight     # tied


def test_seq2seq_ignores_padding_and_generates():
    s2s = _module("Seq2SeqTransformer", 259, 8, 6, 32, 1, 1, 2, 0.0).eval()
    src = torch.tensor([[97, 98, 99, 256, 256, 256, 256, 256]])
    other = src.clone()
    other[0, 4:] = 256
    with torch.no_grad():
        torch.testing.assert_close(s2s(src), s2s(other))
        out = s2s.generate(src)
    assert out.shape[0] == 1 and out.shape[1] <= 6


def test_discriminator_handles_odd_sizes():
    for size in (16, 28, 33):
        disc = _module("Discriminator", 3, size, size, 8, "layer", True)
        assert tuple(disc(torch.randn(2, 3, size, size)).shape) == (2,)


# ================================================================ data / metrics

def test_synthetic_data(rt):
    x, y, names = rt["synthetic_images"](30, 16, 3, "shapes", 0)
    assert x.shape == (30, 3, 16, 16) and 0 <= x.min() and x.max() <= 1
    assert names == ["circle", "square", "triangle"] and set(y) <= {0, 1, 2}
    again, _, _ = rt["synthetic_images"](30, 16, 3, "shapes", 0)
    np.testing.assert_array_equal(x, again)
    text = rt["synthetic_text"](5, 0)
    assert text.count("\n") == 5 and all(line.endswith(".") for line in text.splitlines())
    pairs = rt["synthetic_pairs"]("reverse", 10, 6, 0)
    assert all(t == s[::-1] for s, t in pairs)
    assert rt["synthetic_pairs"]("digits", 3, 10, 1)[0][1].split()[0] in (
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def test_byte_codec_and_padding(rt):
    ids = rt["encode_bytes"]("héllo")
    assert len(ids) == 6 and rt["decode_bytes"](ids) == "héllo"
    padded = rt["encode_pair"]("abc", 7, True)
    assert padded.tolist() == [257, 97, 98, 99, 258, 256, 256]
    assert rt["encode_pair"]("abcdefgh", 4, False).tolist() == [97, 98, 99, 100]


def test_read_image_folder(rt, tmp_path):
    from PIL import Image

    for cls in ("cat", "dog"):
        (tmp_path / cls).mkdir()
        Image.new("RGB", (20, 10), (255, 0, 0)).save(tmp_path / cls / "a.png")
    x, y, names = rt["read_image_folder"](tmp_path, 1, 8, 8)
    assert x.shape == (2, 1, 8, 8) and names == ["cat", "dog"] and y.tolist() == [0, 1]


def test_image_metrics(rt):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(400, 6)), rng.normal(size=(400, 6))
    shifted = rng.normal(1.0, 1.0, size=(400, 6))
    assert rt["frechet_distance"](a, b) < 0.3
    assert rt["frechet_distance"](a, shifted) == pytest.approx(6.0, rel=0.15)  # |Δμ|² = 6
    assert abs(rt["kernel_distance"](a, b)) < 0.05 < rt["kernel_distance"](a, shifted)
    imgs = rng.random((3, 1, 32, 32)).astype(np.float32)
    feats = rt["pixel_features"](imgs, 4)
    np.testing.assert_allclose(feats[0, 0], imgs[0, 0, :8, :8].mean(), rtol=1e-5)


def test_text_metrics(rt):
    # values match sacrebleu (BLEU with add-k=1 smoothing, chrF defaults)
    refs, hyps = ["hello world", "the quick brown fox"], ["hello word", "the quick brown dog"]
    assert rt["chrf"](refs, hyps) == pytest.approx(77.563, abs=1e-3)
    assert rt["bleu"](refs, hyps) == pytest.approx(60.428, abs=1e-3)
    assert rt["bleu"](["a b c d"], ["a b c d"]) == pytest.approx(100.0)
    assert rt["chrf"](["hello"], ["hello"]) == pytest.approx(100.0)
    assert rt["rouge_l"](["a b c d"], ["a c d"]) == pytest.approx(100 * 6 / 7)
    assert rt["bleu"](["x"], [""]) == 0.0


# ================================================================ rules

def test_shape_rules_and_fixes():
    vae = design("1, 32, 32", VAE, (IMAGES,))
    issue = next(i for i in issues(vae) if "Set the Input shape" in i.message)
    assert fix_for_issue(Graph.from_dict(vae), issue)[2].nodes["in"].params["shape"] == \
        "1, 16, 16"
    gan = design("16", [("gen.dcgan_generator", {"image_size": 32})],
                 (IMAGES, ("gen.discriminator", {})))
    issue = next(i for i in issues(gan) if "set image_size to 16" in i.message)
    assert fix_for_issue(Graph.from_dict(gan), issue)[2].nodes["l0"].params["image_size"] == 16


def test_diffusion_rules():
    bad = design("1, 16, 16", [("gen.diffusion_unet", {"num_classes": 5})],
                 (IMAGES, ("gen.noise_scheduler", {})))
    assert any("set num_classes to 3" in i.message for i in issues(bad))
    deep = design("1, 12, 12", [("gen.diffusion_unet", {"depth": 4})],
                  (("data.synthetic_images", {"image_size": 12}),))
    issue = next(i for i in issues(deep) if "set depth to 3" in i.message)
    assert fix_for_issue(Graph.from_dict(deep), issue)[2].nodes["l0"].params["depth"] == 3


def test_language_model_rules():
    leaky = design("32", [("core.embedding", {"num_embeddings": 256, "embedding_dim": 16}),
                          ("core.positional_encoding", {}),
                          ("core.transformer_encoder", {"nhead": 2}),
                          ("core.dense", {"units": 100})],
                   (("data.synthetic_text", {}),), dtype="int64")
    msgs = [i.message for i in issues(leaky)]
    assert any("see later tokens" in m for m in msgs)
    assert any("set units to 256" in m for m in msgs)
    causal = design("32", [("core.embedding", {"num_embeddings": 256, "embedding_dim": 16}),
                           ("core.positional_encoding", {}),
                           ("gen.causal_transformer", {"heads": 2}),
                           ("core.dense", {"units": 256})],
                    (("data.synthetic_text", {}),), dtype="int64")
    assert not [i for i in issues(causal) if i.severity == "error"]
    floats = design("32", GPT, (("data.synthetic_text", {}),))
    assert any("Input dtype to int64" in i.message for i in issues(floats))
    long = design("64", GPT, (("data.synthetic_text", {}),), dtype="int64")
    assert any("set context to 64" in i.message for i in issues(long))


def test_seq2seq_rules():
    data = design("4", [("gen.seq2seq_transformer", {"vocab_size": 256,
                                                     "max_target_len": 5})],
                  (("data.synthetic_pairs", {"max_length": 8}),), dtype="int64")
    msgs = [i.message for i in issues(data)]
    assert any("set vocab_size to 259" in m for m in msgs)
    assert any("Set the Input shape to '8'" in m for m in msgs)
    assert any("set max_target_len to 9" in m for m in msgs)


def test_conflicts_and_orphan_images():
    both = design("1, 16, 16", VAE, (IMAGES, ("gen.discriminator", {})))
    msgs = [i.message for i in issues(both)]
    assert any("one kind of generative model" in m for m in msgs) and len(
        [m for m in msgs if "generat" in m or "VAE" in m]) == 1
    alone = design("1, 16, 16", [("core.conv2d", {})], (IMAGES,))
    assert any("Synthetic Images trains generative models" in i.message
               for i in issues(alone))


def test_vae_design_has_no_classification_lints():
    msgs = [i.message for i in issues(CASES["vae"])]
    assert not any("CrossEntropyLoss" in m or "loss function" in m for m in msgs)


# ================================================================ profile

def test_profiles(tmp_path):
    from ai_made_easy.core.data.profile import profile_dataset
    from ai_made_easy.core.registry import get_registry

    def defaults(type_id):
        return {p.name: p.default for p in get_registry().get(type_id).params}

    p = profile_dataset("data.synthetic_images", {**defaults("data.synthetic_images"),
                                                  "n_images": 60})
    assert len(p.classes) == 3 and "Input [1, 32, 32]" in p.details[0]
    (tmp_path / "c.txt").write_text("hello world\n" * 20)
    p = profile_dataset("data.text_corpus", {**defaults("data.text_corpus"), "path": "c.txt"},
                        tmp_path)
    assert p.rows == 240 and any("240 bytes" in f.message for f in p.findings)
    (tmp_path / "p.csv").write_text("source,target\nabc,cba\nhello,olleh\n,x\n")
    p = profile_dataset("data.text_pairs", {**defaults("data.text_pairs"), "path": "p.csv"},
                        tmp_path)
    assert "max_target_len 6" in p.details[1]
    assert any("empty source or target" in f.message for f in p.findings)
    missing = profile_dataset("data.text_corpus", {**defaults("data.text_corpus"),
                                                   "path": "nope.txt"}, tmp_path)
    assert "does not exist" in missing.error


# ================================================================ training / serving

def _train(data: dict, tmp_path: Path) -> tuple[Path, str]:
    from ai_made_easy.core.training.generate import generate_training

    (tmp_path / "train.py").write_text(generate_training(Graph.from_dict(data), "pytorch"))
    proc = subprocess.run([sys.executable, "train.py"], cwd=tmp_path, capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    assert "resources: peak_memory_mb" in proc.stdout
    return tmp_path, proc.stdout


def _infer(folder: Path, items: list) -> list:
    code = ("import sys, json; sys.path.insert(0, '.'); import train; "
            "train.load_predictor('.'); "
            f"print('OUT ' + json.dumps(train.infer({items!r})))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True,
                          text=True, timeout=300, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert "OUT " in proc.stdout, proc.stderr[-3000:]
    return json.loads(proc.stdout.split("OUT ", 1)[1])


@pytest.mark.parametrize("key, request_items, expect", [
    ("vae", [2], {"vae": ["recon", "kl", "elbo", "fid"]}),
    ("gan", [{"n": 2, "seed": 1}], {"gan": ["fid"]}),
    ("diffusion", [{"n": 2, "class": "square", "steps": 3}], {"diffusion": ["fid"]}),
    ("lm", ["the cat", {"prompt": "a", "max_new_tokens": 5}],
     {"lm": ["loss", "perplexity", "bits_per_byte"]}),
    ("s2s", ["abcd"], {"s2s": ["bleu", "chrf", "rouge_l", "exact"]}),
])
def test_generative_kinds_train_and_serve(key, request_items, expect, tmp_path):
    from PIL import Image

    run, stdout = _train(CASES[key], tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(expect[key]) <= set(metrics)
    live = [ln.split(" ", 1)[1] for ln in stdout.splitlines() if ln.startswith("samples: ")]
    assert live and (run / live[-1]).exists()
    out = _infer(run, request_items)
    assert len(out) == len(request_items)
    if key in ("vae", "gan", "diffusion"):
        images = out[0]["images"]
        assert len(images) == (2 if key != "gan" else 2)
        import io

        img = Image.open(io.BytesIO(base64.b64decode(images[0])))
        assert img.size == (16, 16)
        assert (run / "eval_samples" / "samples.png").exists()
    else:
        assert isinstance(out[0]["text"], str)
        assert (run / "eval_samples" / "samples.txt").exists()
    if key == "lm":
        assert len(out[1]["text"].encode()) <= 5 * 4


def test_worker_emits_sample_events(tmp_path):
    from ai_made_easy.core.runner.protocol import parse_event, worker_script_path

    script = tmp_path / "train.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path('samples').mkdir()\n"
        "Path('samples/epoch_002.txt').write_text('hello')\n"
        "print('samples: samples/epoch_002.txt')\n"
        "print('samples: missing.png')\n")
    proc = subprocess.run([sys.executable, str(worker_script_path()), str(script)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)
    events = [parse_event(line) for line in proc.stdout.splitlines()]
    sample = next(e for e in events if e and e["type"] == "samples")
    assert sample["text"] == "hello" and sample["epoch"] == 2
    assert any(e and e["type"] == "log" and "missing.png" in e["line"] for e in events)


def test_generative_run_deploys_with_generate_route(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict({**CASES["lm"], "name": "tiny_lm"}))
        status = mgr.wait(run_id, 600)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        assert mgr.latest_samples(run_id)["text"]
        samples = api.run_samples(run_id)
        assert samples["texts"] and samples["per_class_metric"] == ""
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["task"] == "language_modeling" and meta["input_kind"] == "prompt"
        assert "/generate" in (tmp_path / "pkg/README.md").read_text()
        sys.path.insert(0, str(tmp_path / "pkg"))
        try:
            import importlib

            app_module = importlib.import_module("app")
            with TestClient(app_module.app) as client:
                body = client.post("/generate", json={"inputs": [{"prompt": "the",
                                                                  "max_new_tokens": 4}]})
                assert body.status_code == 200, body.text
                assert isinstance(body.json()["predictions"][0]["text"], str)
        finally:
            sys.path.remove(str(tmp_path / "pkg"))
            sys.modules.pop("app", None)
            sys.modules.pop("model_def", None)
    finally:
        api.set_manager(None)
