"""Generative blocks: VAE bottleneck, GAN / diffusion / language-model / seq2seq models,
their training configuration blocks, datasets and metrics."""
from __future__ import annotations

import math
from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.generative import helpers as h
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition

LAYERS = "Generative"
MODELS = "Generative Models"
CONFIG = "Generative Training"
BYTE_VOCAB = 256
SEQ2SEQ_VOCAB = 259            # 256 bytes + PAD, BOS, EOS
GENERATIVE_DATA: dict[str, str] = {}      # type id -> "images" | "text" | "pairs"
GENERATIVE_METRICS: dict[str, dict] = {}
CONFIG_BLOCKS = ("gen.discriminator", "gen.noise_scheduler")


def _with_cost(defn: BlockDefinition, cost, **meta) -> BlockDefinition:
    return replace(defn, meta={**(defn.meta or {}), "cost": cost, **meta})


def _size(shape: list[int]) -> int:
    return int(math.prod(shape))


# ------------------------------------------------------------------ layers

def _reparam_shape(in_shapes, p):
    s = list(in_shapes[0])
    if len(s) not in (1, 3):
        raise ShapeError(f"Reparameterize reads [2 × latent] or [2 × channels, H, W], got {s}")
    if s[0] % 2:
        raise ShapeError(f"Reparameterize splits its input into mean and log-variance halves: "
                         f"the first dimension must be even, got {s[0]}")
    return [s[0] // 2, *s[1:]]


def _causal_shape(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 2:
        raise ShapeError(f"Causal Transformer reads token features [L, D], got {s}")
    if s[1] % int(p["heads"]):
        raise ShapeError(f"the feature size {s[1]} must be divisible by heads ({p['heads']})")
    return list(s)


def _causal_cost(in_shapes, p):
    s = in_shapes[0]
    params, macs = h.causal_cost(s[1], int(p["layers"]), s[0])
    return params, 2 * macs


def _layers() -> list[BlockDefinition]:
    reparam = nn_block(
        "gen.reparameterize", "Reparameterize (VAE)", LAYERS, family="model",
        params=(P("beta", "float", 1.0, lo=0.0, hi=100.0,
                  help="KL weight (β-VAE: > 1 gives more disentangled latents)"),
                P("kl_warmup", "int", 0, lo=0, hi=1000,
                  help="Epochs over which the KL weight ramps up from 0 (avoids collapse)")),
        shape=_reparam_shape, torch=lambda c: f"Reparameterize({float(c['beta'])})",
        torch_helpers=("Reparameterize",),
        desc="The VAE bottleneck: splits its input into mean and log-variance, samples the "
             "latent z and adds β·KL to the loss. Put the encoder before it, the decoder after.")
    causal = nn_block(
        "gen.causal_transformer", "Causal Transformer", LAYERS, family="model",
        params=(P("layers", "int", 4, lo=1, hi=48), P("heads", "int", 4, lo=1),
                P("dropout", "float", 0.1, lo=0.0, hi=1.0)),
        shape=_causal_shape, param_fn=lambda s, p: _causal_cost(s, p)[0],
        torch=lambda c: (f"CausalTransformer({c['input_shape'][1]}, {int(c['layers'])}, "
                         f"{int(c['heads'])}, {float(c['dropout'])})"),
        torch_helpers=("CausalTransformer",),
        desc="Transformer blocks with causal self-attention ([L, D] → [L, D]): each position "
             "sees only earlier ones, as language models need.")
    return [_with_cost(reparam, lambda s, p: (0, _size(s[0]) * 4)),
            _with_cost(causal, _causal_cost)]


# ------------------------------------------------------------------ models

def _dcgan_shape(in_shapes, p):
    s = in_shapes[0]
    if len(s) not in (1, 3) or (len(s) == 3 and s[1:] != [1, 1]):
        raise ShapeError(f"the generator reads a latent vector [Z], got {s}")
    size = int(p["image_size"])
    if size < 8 or size & (size - 1):
        raise ShapeError(f"image_size must be a power of two ≥ 8, got {size}")
    return [int(p["channels"]), size, size]


def _dcgan_cost(in_shapes, p):
    params, macs = h.dcgan_cost(_size(in_shapes[0]), int(p["channels"]), int(p["image_size"]),
                                int(p["width"]))
    return params, 2 * macs


def _unet_shape(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 3:
        raise ShapeError(f"the diffusion U-Net denoises images [C, H, W], got {s}")
    if int(p["base_channels"]) % 2:
        raise ShapeError("base_channels must be even (sine / cosine time embedding)")
    depth = int(p["depth"])
    step = 2 ** (depth - 1)
    if s[1] % step or s[2] % step:
        fits = depth
        while fits > 1 and (s[1] % 2 ** (fits - 1) or s[2] % 2 ** (fits - 1)):
            fits -= 1
        raise ShapeError(f"a depth-{depth} U-Net halves the image {depth - 1} times, so H and "
                         f"W must be multiples of {step} (got {s[1]} × {s[2]}): set depth to "
                         f"{fits} or resize the images")
    return list(s)


def _unet_cost(in_shapes, p):
    s = in_shapes[0]
    params, macs = h.unet_cost(s[0], s[1], s[2], int(p["base_channels"]), int(p["depth"]),
                               bool(p["attention"]), int(p["num_classes"]))
    return params, 2 * macs


def _gpt_shape(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 1:
        raise ShapeError(f"GPT reads a token sequence [L], got {s}")
    if s[0] > int(p["context"]):
        raise ShapeError(f"the input has {s[0]} tokens but the context is {p['context']}: "
                         f"set context to {s[0]}")
    if int(p["d_model"]) % int(p["heads"]):
        raise ShapeError(f"d_model {p['d_model']} must be divisible by heads ({p['heads']})")
    return [s[0], int(p["vocab_size"])]


def _gpt_cost(in_shapes, p):
    params, macs = h.gpt_cost(int(p["vocab_size"]), int(p["context"]), int(p["d_model"]),
                              int(p["layers"]), in_shapes[0][0], bool(p["tie_weights"]))
    return params, 2 * macs


def _seq2seq_shape(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 1:
        raise ShapeError(f"the seq2seq transformer reads source tokens [S], got {s}")
    if int(p["d_model"]) % int(p["heads"]):
        raise ShapeError(f"d_model {p['d_model']} must be divisible by heads ({p['heads']})")
    if int(p["vocab_size"]) < 4:
        raise ShapeError("vocab_size must leave room for PAD, BOS and EOS")
    return [int(p["max_target_len"]), int(p["vocab_size"])]


def _seq2seq_cost(in_shapes, p):
    params, macs = h.seq2seq_cost(int(p["vocab_size"]), in_shapes[0][0],
                                  int(p["max_target_len"]), int(p["d_model"]),
                                  int(p["encoder_layers"]), int(p["decoder_layers"]))
    return params, 2 * macs


def _models() -> list[BlockDefinition]:
    dcgan = nn_block(
        "gen.dcgan_generator", "DCGAN Generator", MODELS, family="model",
        params=(P("channels", "int", 1, lo=1, hi=4), P("image_size", "int", 32, lo=8, hi=256),
                P("width", "int", 64, lo=4, help="Feature maps in the last layer (doubles "
                                                 "towards the latent)")),
        shape=_dcgan_shape, param_fn=lambda s, p: _dcgan_cost(s, p)[0],
        torch=lambda c: (f"DCGANGenerator({_size(c['input_shape'])}, {int(c['channels'])}, "
                         f"{int(c['image_size'])}, {int(c['width'])})"),
        torch_helpers=("DCGANGenerator",),
        desc="Latent [Z] → image [C, S, S] in [-1, 1] through transposed convolutions "
             "(DCGAN). Pair with a Discriminator block to train it as a GAN.")
    unet = nn_block(
        "gen.diffusion_unet", "Diffusion U-Net", MODELS, family="model",
        params=(P("base_channels", "int", 32, lo=4, hi=512),
                P("depth", "int", 3, lo=1, hi=6, help="Resolution levels"),
                P("attention", "bool", True, help="Self-attention at the lowest resolution"),
                P("num_classes", "int", 0, lo=0, hi=10000,
                  help="Class-conditional generation with classifier-free guidance (0 = "
                       "unconditional)"),
                P("dropout", "float", 0.1, lo=0.0, hi=1.0)),
        shape=_unet_shape, param_fn=lambda s, p: _unet_cost(s, p)[0],
        torch=lambda c: (f"DiffusionUNet({c['input_shape'][0]}, {int(c['base_channels'])}, "
                         f"{int(c['depth'])}, {bool(c['attention'])}, "
                         f"{int(c['num_classes'])}, {float(c['dropout'])})"),
        torch_helpers=("DiffusionUNet",),
        desc="Predicts the noise in a noisy image given the timestep (and class): the "
             "denoiser of a diffusion model. Configure noise and sampling with a Noise "
             "Scheduler block.")
    gpt = nn_block(
        "gen.gpt", "GPT (Decoder-only Transformer)", MODELS, family="model",
        params=(P("vocab_size", "int", BYTE_VOCAB, lo=2, help="256 for byte-level text"),
                P("context", "int", 128, lo=2, help="Longest sequence the model attends over"),
                P("d_model", "int", 128, lo=8), P("layers", "int", 4, lo=1, hi=48),
                P("heads", "int", 4, lo=1), P("dropout", "float", 0.1, lo=0.0, hi=1.0),
                P("tie_weights", "bool", True,
                  help="Share the token embedding with the output layer")),
        shape=_gpt_shape, param_fn=lambda s, p: _gpt_cost(s, p)[0],
        torch=lambda c: (f"GPT({int(c['vocab_size'])}, {int(c['context'])}, "
                         f"{int(c['d_model'])}, {int(c['layers'])}, {int(c['heads'])}, "
                         f"{float(c['dropout'])}, {bool(c['tie_weights'])})"),
        torch_helpers=("CausalTransformer", "GPT"), input_dtype="int",
        desc="Tokens [L] → next-token logits [L, vocab]: a language model trained from "
             "scratch (GPT-style, learned positions, pre-norm).")
    seq2seq = nn_block(
        "gen.seq2seq_transformer", "Seq2Seq Transformer", MODELS, family="model",
        params=(P("vocab_size", "int", SEQ2SEQ_VOCAB, lo=4,
                  help="259 for bytes + PAD / BOS / EOS"),
                P("max_target_len", "int", 32, lo=2, help="Longest output, incl. EOS"),
                P("d_model", "int", 128, lo=8), P("encoder_layers", "int", 3, lo=1, hi=24),
                P("decoder_layers", "int", 3, lo=1, hi=24), P("heads", "int", 4, lo=1),
                P("dropout", "float", 0.1, lo=0.0, hi=1.0)),
        shape=_seq2seq_shape, param_fn=lambda s, p: _seq2seq_cost(s, p)[0],
        torch=lambda c: (f"Seq2SeqTransformer({int(c['vocab_size'])}, {c['input_shape'][0]}, "
                         f"{int(c['max_target_len'])}, {int(c['d_model'])}, "
                         f"{int(c['encoder_layers'])}, {int(c['decoder_layers'])}, "
                         f"{int(c['heads'])}, {float(c['dropout'])})"),
        torch_helpers=("Seq2SeqTransformer",), input_dtype="int",
        desc="Source tokens [S] → target logits [T, vocab]: an encoder-decoder transformer "
             "for translation-style tasks, decoded greedily.")
    return [_with_cost(dcgan, _dcgan_cost), _with_cost(unet, _unet_cost),
            _with_cost(gpt, _gpt_cost), _with_cost(seq2seq, _seq2seq_cost)]


# ------------------------------------------------------------------ configuration

def _disc_checks(p: dict) -> list:
    out = []
    if p["loss"] == "wgan_gp" and p["norm"] == "batch":
        out.append(("warning", "batch norm breaks the WGAN-GP gradient penalty (it mixes "
                               "samples): use layer or no normalization"))
    return out


def _scheduler_checks(p: dict) -> list:
    out = []
    if int(p["sample_steps"]) > int(p["timesteps"]):
        out.append(("error", f"sample_steps ({p['sample_steps']}) cannot exceed timesteps "
                             f"({p['timesteps']})"))
    if p["sampler"] == "ddpm" and int(p["sample_steps"]) != int(p["timesteps"]):
        out.append(("info", "the DDPM sampler always runs every timestep; sample_steps only "
                            "applies to DDIM"))
    return out


def _config() -> list[BlockDefinition]:
    color = family_color("training")
    return [
        BlockDefinition(
            type_id="gen.discriminator", display_name="Discriminator (GAN)", category=CONFIG,
            color=color, library="PyTorch", checks_fn=_disc_checks,
            params=(P("width", "int", 64, lo=4, help="Feature maps in the first layer"),
                    P("norm", "enum", "batch", options=("batch", "layer", "none")),
                    P("spectral_norm", "bool", False, help="Stabilises training (SN-GAN)"),
                    P("loss", "enum", "bce", options=("bce", "hinge", "wgan_gp"),
                      help="bce: original GAN; hinge: SN-GAN / BigGAN; wgan_gp: Wasserstein "
                           "with gradient penalty"),
                    P("gp_weight", "float", 10.0, lo=0.0, hi=1000.0),
                    P("d_steps", "int", 1, lo=1, hi=10,
                      help="Discriminator updates per generator update"),
                    P("lr", "float", 0.0002, lo=1e-7, hi=1.0),
                    P("beta1", "float", 0.5, lo=0.0, hi=0.999)),
            description="Turns the design into a GAN: the canvas model is the generator "
                        "(latent → image) and this block builds the critic that judges real "
                        "vs generated images.",
            meta={"kind": "config", "generative": True}),
        BlockDefinition(
            type_id="gen.noise_scheduler", display_name="Noise Scheduler (Diffusion)",
            category=CONFIG, color=color, library="PyTorch", checks_fn=_scheduler_checks,
            params=(P("timesteps", "int", 1000, lo=10, hi=10000),
                    P("schedule", "enum", "cosine", options=("linear", "cosine")),
                    P("sampler", "enum", "ddim", options=("ddim", "ddpm"),
                      help="DDIM: fast deterministic sampling in sample_steps; DDPM: every "
                           "timestep, stochastic"),
                    P("sample_steps", "int", 50, lo=1, hi=10000),
                    P("guidance", "float", 2.0, lo=0.0, hi=50.0,
                      help="Classifier-free guidance scale (needs num_classes > 0)"),
                    P("cond_dropout", "float", 0.1, lo=0.0, hi=1.0,
                      help="Share of training steps without the class (for guidance)"),
                    P("ema", "float", 0.999, lo=0.0, hi=0.99999,
                      help="Exponential moving average of the weights used to sample (0 = "
                           "off)"),
                    P("sample_every", "int", 5, lo=1, hi=1000,
                      help="Epochs between sample grids / FID")),
            description="Turns the design into a diffusion model (DDPM training) and sets "
                        "how images are sampled: DDIM / DDPM, steps, guidance and EMA.",
            meta={"kind": "config", "generative": True}),
    ]


# ------------------------------------------------------------------ data

def _split_checks(p: dict) -> list:
    total = float(p.get("val_fraction", 0)) + float(p.get("test_fraction", 0))
    if total >= 0.9:
        return [("error", "validation + test take almost everything: lower the fractions")]
    return []


def _data(type_id: str, name: str, kind: str, params: tuple, desc: str,
          test: bool = False) -> BlockDefinition:
    GENERATIVE_DATA[type_id] = kind
    split = (P("val_fraction", "float", 0.1, lo=0.0, hi=0.5),
             *((P("test_fraction", "float", 0.1, lo=0.0, hi=0.5),) if test else ()))
    return BlockDefinition(
        type_id=type_id, display_name=name, category="Data", color=family_color("data"),
        params=(*params, *split), description=desc, library="PyTorch",
        checks_fn=_split_checks,
        meta={"modality": "image" if kind == "images" else "text", "generative": kind})


def _datasets() -> list[BlockDefinition]:
    return [
        _data("data.synthetic_images", "Synthetic Images", "images",
              (P("kind", "enum", "shapes", options=("shapes", "blobs"),
                 help="shapes: circles / squares / triangles (3 classes); blobs: soft "
                      "Gaussian blobs"),
               P("n_images", "int", 2000, lo=16), P("image_size", "int", 32, lo=8, hi=256),
               P("channels", "int", 1, lo=1, hi=3), P("seed", "int", 0, lo=0)),
              "Generated images for trying VAEs, GANs and diffusion models without data "
              "files."),
        _data("data.text_corpus", "Text Corpus", "text",
              (P("path", "str", "corpus.txt", help="A .txt file or a folder of .txt files"),),
              "Plain text for training a language model (byte-level: any language, no "
              "tokenizer files)."),
        _data("data.synthetic_text", "Synthetic Text", "text",
              (P("n_sentences", "int", 4000, lo=10), P("seed", "int", 0, lo=0)),
              "Generated toy sentences (\"the cat sees a red ball in the park.\") for trying "
              "language models."),
        _data("data.text_pairs", "Text Pairs", "pairs",
              (P("path", "str", "pairs.csv"), P("source_column", "str", "source"),
               P("target_column", "str", "target")),
              "Source / target text pairs in a CSV or TSV file (translation, summarisation, "
              "rewriting).", test=True),
        _data("data.synthetic_pairs", "Synthetic Pairs", "pairs",
              (P("task", "enum", "reverse", options=("reverse", "sort", "copy", "digits"),
                 help="reverse / sort / copy letters, or spell digits as words"),
               P("n_pairs", "int", 4000, lo=10), P("max_length", "int", 8, lo=2, hi=200),
               P("seed", "int", 0, lo=0)),
              "Generated sequence-to-sequence tasks for trying encoder-decoders.", test=True),
    ]


# ------------------------------------------------------------------ metrics

def _metric(type_id: str, name: str, key: str, tasks: tuple, desc: str,
            params: tuple = ()) -> BlockDefinition:
    GENERATIVE_METRICS[type_id] = {"key": key, "tasks": tasks}
    return BlockDefinition(
        type_id=type_id, display_name=name, category="Metrics",
        color=family_color("evaluation"), params=params, description=desc, library="PyTorch",
        meta={"kind": "metrics", "key": key, "tasks": list(tasks), "generative": True})


IMAGE_TASKS = ("vae_generation", "gan_generation", "diffusion_generation")
FEATURES = (P("features", "enum", "pixels", options=("pixels", "inception"),
              help="pixels: offline, compares downsampled images; inception: the standard "
                   "Inception-v3 features (downloads torchvision weights)"),
            P("samples", "int", 500, lo=10, hi=50000, help="Generated images compared"))


def _metrics() -> list[BlockDefinition]:
    return [
        _metric("eval.fid", "FID", "fid", IMAGE_TASKS,
                "Fréchet distance between real and generated image features (lower is "
                "better).", FEATURES),
        _metric("eval.kid", "KID", "kid", IMAGE_TASKS,
                "Kernel (MMD) distance between real and generated features; unbiased for "
                "small sample counts.", FEATURES),
        _metric("eval.perplexity", "Perplexity", "perplexity", ("language_modeling",),
                "exp(cross-entropy) of held-out text: the average branching factor."),
        _metric("eval.bleu", "BLEU", "bleu", ("sequence_to_sequence",),
                "Corpus BLEU-4 n-gram overlap with the references (0-100)."),
        _metric("eval.chrf", "chrF", "chrf", ("sequence_to_sequence",),
                "Character n-gram F-score (0-100); robust for morphology and short outputs."),
        _metric("eval.rouge_l", "ROUGE-L", "rouge_l", ("sequence_to_sequence",),
                "Longest-common-subsequence F1 with the references (0-100)."),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in (*_layers(), *_models(), *_config(), *_datasets(), *_metrics()):
        reg.register(defn)
