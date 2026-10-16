"""Recipes built from the shipped sample projects: one or more per remaining task
(vision, forecasting, speech, generative, sequence, probabilistic, graph, recommender,
reinforcement-learning and LLM designs). With the user's data the sample's demo dataset
block is swapped for one that reads it; the rest of the design is fitted by
:func:`autofit`."""
from __future__ import annotations

import copy
import keyword
import re
from typing import Callable

from ai_made_easy.core.recipes import (
    Context,
    Draft,
    Knob,
    Recipe,
    RecipeError,
    register_recipe,
)
from ai_made_easy.core.recipes.facts import DataFacts, density_columns

Swap = Callable[[DataFacts], tuple[str, dict, str]]   # -> (type_id, params, why)
Fit = Callable[[dict, DataFacts], None]               # sizes the sample's blocks to the data


def _sample(name: str) -> dict:
    from ai_made_easy.core import api

    return api.read_sample(f"{name}.json")


def _data_node(graph: dict) -> dict | None:
    return next((n for n in graph["nodes"] if n["type"].startswith("data.")), None)


# ------------------------------------------------------------------ data swaps

def _vision(facts: DataFacts) -> tuple[str, dict, str]:
    if facts.kind == "coco":
        return ("data.coco", {"images_dir": facts.paths["images_dir"],
                              "annotations": facts.paths["annotations"]},
                f"your COCO dataset ({len(facts.classes)} classes)")
    if facts.kind == "yolo":
        return ("data.yolo", {"images_dir": facts.paths["images_dir"],
                              "labels_dir": facts.paths["labels_dir"],
                              "classes": ", ".join(facts.classes)}, "your YOLO dataset")
    return "data.voc", {"root": facts.paths["root"]}, "your Pascal VOC dataset"


def _masks(facts: DataFacts) -> tuple[str, dict, str]:
    return ("data.mask_folder", {"images_dir": facts.paths["images_dir"],
                                 "masks_dir": facts.paths["masks_dir"],
                                 "num_classes": max(facts.n_classes, 2)},
            "your images and their per-pixel class masks")


def _series(facts: DataFacts) -> tuple[str, dict, str]:
    return ("data.forecast_csv", {"path": facts.source, "format": facts.format or "csv",
                                  "time_column": facts.time_column,
                                  "target_column": facts.target,
                                  "id_column": facts.columns.get("id", "")},
            f"your series: '{facts.target}' over '{facts.time_column}'")


def _corpus(facts: DataFacts) -> tuple[str, dict, str]:
    return "data.text_corpus", {"path": facts.source}, "your text"


def _pairs(facts: DataFacts) -> tuple[str, dict, str]:
    return ("data.text_pairs", {"path": facts.source,
                                "source_column": facts.columns["source"],
                                "target_column": facts.columns["target"]},
            "your source / target text pairs")


def _graph(facts: DataFacts) -> tuple[str, dict, str]:
    return ("data.graph_csv", {"nodes_path": facts.paths["nodes_path"],
                               "edges_path": facts.paths["edges_path"],
                               "id_column": facts.columns.get("id", "id"),
                               "label_column": facts.columns.get("label", ""),
                               "source_column": facts.columns["source"],
                               "target_column": facts.columns["target"]},
            "your graph: a node table and an edge list")


def _graphs(facts: DataFacts) -> tuple[str, dict, str]:
    return ("data.graphs_csv", {"nodes_path": facts.paths["nodes_path"],
                                "edges_path": facts.paths["edges_path"],
                                "graphs_path": facts.paths["graphs_path"],
                                "graph_column": facts.columns["graph"],
                                "id_column": facts.columns["id"],
                                "label_column": facts.columns["label"],
                                "source_column": facts.columns["source"],
                                "target_column": facts.columns["target"]},
            f"your {facts.rows:,} graphs: node and edge tables keyed by "
            f"'{facts.columns['graph']}', labels in the graph table")


def _interactions(facts: DataFacts) -> tuple[str, dict, str]:
    sides = [s for s in ("user", "item") if f"{s}_features" in facts.paths]
    return ("data.interactions_csv", {"path": facts.source,
                                      "user_column": facts.columns["user"],
                                      "item_column": facts.columns["item"],
                                      "rating_column": facts.columns.get("rating", ""),
                                      "user_features": facts.paths.get("user_features", ""),
                                      "item_features": facts.paths.get("item_features", "")},
            "your user-item interactions" + (f" with {' and '.join(sides)} features"
                                             if sides else ""))


def _images(facts: DataFacts) -> tuple[str, dict, str]:
    classes = f", {facts.n_classes} classes" if facts.n_classes >= 2 else ""
    return ("data.image_folder", {"root": facts.source, "grayscale": facts.channels == 1},
            f"your {facts.rows:,} images{classes}, resized to the model's size")


def _fit_images(graph: dict, facts: DataFacts) -> None:
    """Colour or grey like the user's images; class-conditional only with class folders."""
    channels = 1 if facts.channels == 1 else 3
    types = {n["type"] for n in graph["nodes"]}
    for node in graph["nodes"]:
        p = node["params"]
        if node["type"] == "core.input" and str(p.get("shape", "")).count(",") == 2:
            p["shape"] = ", ".join([str(channels), *(d.strip() for d in
                                                     str(p["shape"]).split(",")[1:])])
        elif node["type"] == "gen.dcgan_generator":
            p["channels"] = channels
        elif node["type"] == "gen.diffusion_unet":
            p["num_classes"] = facts.n_classes if facts.n_classes >= 2 else 0
        elif node["type"] == "gen.noise_scheduler" and facts.n_classes < 2:
            p["guidance"] = 1.0
    if "gen.reparameterize" in types:   # the VAE decoder's last layer makes the pixels
        last = [n for n in graph["nodes"] if n["type"] in ("core.conv_transpose2d",
                                                           "core.conv2d")][-1]
        last["params"]["out_channels"] = channels


def _density(facts: DataFacts) -> tuple[str, dict, str]:
    if facts.format not in ("csv", "tsv"):
        raise RecipeError("flows read .csv / .tsv tables")
    columns = density_columns(facts)
    return ("data.csv", {"path": facts.source, "format": facts.format, "target_column": "",
                         "feature_columns": ", ".join(columns)},
            f"your table's {len(columns)} numeric columns: the flow learns their joint density")


def _grouped(facts: DataFacts) -> tuple[str, dict, str]:
    if facts.format not in ("csv", "tsv", "parquet"):
        raise RecipeError("PyMC models read .csv / .tsv / .parquet tables")
    if not facts.columns.get("group"):
        raise RecipeError("a hierarchical model needs a group column (school, store, region, "
                          "...): a column with a few repeated values")
    return ("data.csv", {"path": facts.source, "format": facts.format, "target_column": "",
                         "feature_columns": ""},
            f"your table: '{facts.target}' per row, grouped by '{facts.columns['group']}'")


def _fit_groups(graph: dict, facts: DataFacts) -> None:
    """Point the model at the group, predictor and outcome columns; scale the priors."""
    from ai_made_easy.core.data.profile import read_table

    names = {n["params"].get("name") for n in graph["nodes"]}
    group = facts.columns["group"]
    predictor = next((c for c in facts.numeric if c != group and c.isidentifier()
                      and not keyword.iskeyword(c) and c not in names), "")
    if not predictor:
        raise RecipeError("needs a numeric predictor column whose name is a Python identifier "
                          "(letters, digits, _)")
    logistic = any(n["type"] == "ppl.bernoulli" for n in graph["nodes"])
    frame = read_table(facts.source, facts.format, nrows=20_000)
    y = frame[facts.target]
    if logistic and sorted(map(str, y.dropna().unique())) not in (["0", "1"], ["0.0", "1.0"]):
        raise RecipeError(f"'{facts.target}' must hold 0 / 1 for a logistic model")
    if not logistic and (facts.classes or str(y.dtype)[:3] not in ("int", "flo")):
        raise RecipeError(f"'{facts.target}' is not numeric: use hierarchical logistic "
                          "regression for a 0 / 1 outcome")
    sd_x = float(frame[predictor].std() or 1.0)
    sd_y = 1.0 if logistic else float(y.std() or 1.0)
    priors = {"mu_a": {"mu": 0.0 if logistic else round(float(y.mean()), 3),
                       "sigma": round((5.0 if logistic else 2 * sd_y), 3)},
              "sigma_a": {"sigma": round(sd_y, 3)}, "sigma": {"sigma": round(sd_y, 3)},
              "b": {"sigma": round((2.5 if logistic else 2 * sd_y) / sd_x, 4)}}
    for node in graph["nodes"]:
        p = node["params"]
        if p.get("group") == "group":
            p["group"] = group
        if p.get("observed") == "y":
            p["observed"] = facts.target
        if node["type"] == "ppl.deterministic":
            p["expression"] = re.sub(r"\bx\b", predictor, p["expression"])
        p.update(priors.get(p.get("name"), {}))


def _ssm(facts: DataFacts) -> tuple[str, dict, str]:
    if facts.format != "csv":
        raise RecipeError("state-space models read .csv series")
    note = (f"; one series (the rows are not split by '{facts.columns['id']}')"
            if facts.columns.get("id") else "")
    return ("data.timeseries_csv", {"path": facts.source, "target_columns": facts.target,
                                    "feature_columns": ""},
            f"your series '{facts.target}', in file order{note}")


def _fit_season(graph: dict, facts: DataFacts) -> None:
    """The series' column, a horizon it can hold and its season length (none if unknown)."""
    period = facts.period if facts.period >= 2 and facts.rows >= 3 * facts.period else 0
    drop = set()
    for node in graph["nodes"]:
        p = node["params"]
        if node["type"] == "ssm.model":
            p["target_column"] = facts.target
            p["horizon"] = max(1, min(int(p["horizon"]), facts.rows // 5))
        elif node["type"] == "ssm.seasonal":
            if period:
                p["period"] = period
            else:
                drop.add(node["id"])
        elif node["type"] == "ssm.arima":
            p["season"] = period
            if not period:
                p.update(seasonal_p=0, seasonal_d=0, seasonal_q=0)
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] not in drop]
    graph["edges"] = [e for e in graph["edges"]
                      if e["from"].split("/")[0] not in drop and e["to"].split("/")[0] not in drop]


def _regimes(facts: DataFacts) -> tuple[str, dict, str]:
    return ("data.csv", {"path": facts.source, "format": facts.format or "csv",
                         "target_column": "", "feature_columns": ""},
            f"your series '{facts.target}': the HMM finds its hidden regimes")


def _fit_regimes(graph: dict, facts: DataFacts) -> None:
    for node in graph["nodes"]:
        if node["type"] == "pgm.hmm":
            node["params"]["observations"] = facts.target


# ------------------------------------------------------------------ starter recipes

def starter(recipe_id: str, sample: str, task: str, tier: str, modality: str, title: str,
            description: str, *, swaps: dict[str, Swap] | None = None,
            knobs: tuple[Knob, ...] = (), edit: Callable[[dict], None] | None = None,
            requires: tuple[str, ...] = (), extra: str = "", strengths: str = "",
            demo_note: str = "", rows: tuple[int, int] = (0, 10**12),
            priority: float = 0.0, family: str = "neural", fit: Fit | None = None) -> Recipe:
    swaps = swaps or {}

    def builder(ctx: Context) -> Draft:
        graph = copy.deepcopy(_sample(sample))
        if edit is not None:
            edit(graph)
        d = Draft(graph.get("name", recipe_id), title, description)
        d.nodes, d.edges = graph["nodes"], graph["edges"]
        data = _data_node(graph)
        if data is not None:
            if ctx.facts.kind in swaps:
                type_id, params, why = swaps[ctx.facts.kind](ctx.facts)
                data["type"], data["params"] = type_id, params
                d.why[data["id"]] = why
                if fit is not None:
                    fit(graph, ctx.facts)
                    d.nodes, d.edges = graph["nodes"], graph["edges"]
            else:
                d.why[data["id"]] = demo_note or "demo data generated on the fly: replace " \
                                                 "it with your own dataset"
        return d

    return register_recipe(Recipe(
        recipe_id, title, (task,), tier, modality, description, builder,
        data_kinds=("demo", *swaps), knobs=knobs, requires=requires, extra=extra,
        family=family, strengths=strengths, rows=rows, priority=priority))


def lr(target: str, default: float) -> Knob:
    return Knob("lr", "float", default, 1e-6, 0.5, log=True, help="Learning rate",
                target=target)


def epochs(target: str, default: int) -> Knob:
    return Knob("epochs", "int", default, 1, 1000, help="Passes over the data", target=target)


def _swap_head(type_id: str, arch: str, metrics: tuple[tuple[str, str], ...],
               **params) -> Callable[[dict], None]:
    def edit(graph: dict) -> None:
        for node in graph["nodes"]:
            if node["id"] == "detector":
                node["type"] = type_id
                node["params"] = {**node["params"], "arch": arch, **params}
        graph["nodes"] = [n for n in graph["nodes"] if not n["type"].startswith("eval.")]
        graph["nodes"] += [{"id": nid, "type": t, "params": {}, "position": [360, 900 + 80 * i]}
                           for i, (nid, t) in enumerate(metrics)]
    return edit


def _speech_tags(graph: dict) -> None:
    for node in graph["nodes"]:
        if node["type"] == "data.synthetic_speech":
            node["params"]["task"] = "tags"
    graph["nodes"] = [n for n in graph["nodes"] if n["type"] not in
                      ("train.loss_cross_entropy", "eval.accuracy")]
    graph["nodes"] += [{"id": "bce", "type": "train.loss_bce_logits", "params": {},
                        "position": [360, 900]},
                       {"id": "tag_map", "type": "eval.tag_map", "params": {},
                        "position": [360, 980]}]


VISION = {"coco": _vision, "yolo": _vision, "voc": _vision}
VISION_EXTRA = dict(requires=("torchvision",), extra="vision-tasks")
TRAIN_KNOBS = lambda lr_default, n: (lr("optimizer.lr", lr_default),  # noqa: E731
                                     epochs("trainer.epochs", n))

# vision
starter("faster_rcnn", "shapes_detection", "detection", "pretrained", "image",
        "Faster R-CNN (MobileNetV3)", "A two-stage torchvision detector on an ImageNet "
        "backbone: proposes regions, then classifies and refines each box.",
        swaps=VISION, knobs=TRAIN_KNOBS(3e-4, 15), **VISION_EXTRA,
        strengths="accurate boxes from a few hundred images")
starter("mask_rcnn", "shapes_detection", "instance_segmentation", "pretrained", "image",
        "Mask R-CNN", "Faster R-CNN plus a mask head: a box and a pixel mask per object.",
        swaps={"coco": _vision, "yolo": _vision}, knobs=TRAIN_KNOBS(3e-4, 15),
        edit=_swap_head("vision.instance_segmenter", "maskrcnn_resnet50_fpn",
                        (("map", "eval.map"), ("mask_map", "eval.mask_map"))),
        **VISION_EXTRA, strengths="the standard instance-segmentation model")
starter("keypoint_rcnn", "shapes_detection", "keypoints", "pretrained", "image",
        "Keypoint R-CNN", "Faster R-CNN plus a keypoint head: a box and its keypoints "
        "(e.g. a person's joints) per object.", swaps={"coco": _vision, "yolo": _vision},
        knobs=TRAIN_KNOBS(3e-4, 15),
        edit=_swap_head("vision.keypoint_detector", "keypointrcnn_resnet50_fpn",
                        (("map", "eval.map"), ("pck", "eval.pck")), num_keypoints=4),
        **VISION_EXTRA, strengths="pose and landmark estimation")
starter("unet", "shapes_segmentation", "semantic_segmentation", "medium", "image", "U-Net",
        "An encoder-decoder with skip connections that labels every pixel.",
        swaps={"mask_folder": _masks}, knobs=TRAIN_KNOBS(1e-3, 20),
        strengths="trains from scratch on small datasets; sharp boundaries")
# sequences, audio
starter("forecast_patchtst", "demand_forecasting", "forecasting", "medium", "timeseries",
        "Forecasting network", "A deep forecasting model over sliding windows with "
        "covariates, compared with a seasonal-naive baseline.", swaps={"series": _series},
        knobs=TRAIN_KNOBS(1e-3, 30), strengths="many related series and covariates")
starter("keyword_cnn", "keyword_spotting", "keyword_spotting", "small", "audio",
        "Keyword CNN", "A CNN over mel spectrograms that recognises spoken commands.",
        knobs=TRAIN_KNOBS(1e-3, 25),
        strengths="small enough for a microcontroller-sized budget")
starter("audio_tagger", "keyword_spotting", "audio_tagging", "small", "audio",
        "Audio tagger", "The mel CNN with one yes/no output per tag: several sounds can be "
        "present at once.", knobs=TRAIN_KNOBS(1e-3, 25), edit=_speech_tags,
        strengths="multi-label sound events")
starter("ctc_lstm", "speech_recognition_ctc", "speech_recognition", "medium", "audio",
        "BiLSTM + CTC", "Mel features, a bidirectional LSTM and CTC: transcribes speech "
        "letter by letter.", knobs=TRAIN_KNOBS(3e-3, 40),
        strengths="learns to transcribe without aligned labels")
# generative
IMAGES = dict(swaps={"image_folder": _images}, fit=_fit_images)
starter("vae", "shapes_vae", "vae_generation", "small", "image", "Variational autoencoder",
        "An encoder to a latent space and a decoder back: smooth, controllable generation.",
        knobs=TRAIN_KNOBS(1e-3, 30), **IMAGES, strengths="stable training, a meaningful latent space")
starter("dcgan", "shapes_gan", "gan_generation", "medium", "image", "DCGAN",
        "A generator and a critic trained against each other.",
        knobs=(lr("optimizer.lr", 2e-4), epochs("trainer.epochs", 40)), **IMAGES,
        strengths="sharp samples")
starter("ddpm", "shapes_diffusion", "diffusion_generation", "large", "image",
        "Diffusion (DDPM)", "A U-Net learns to remove noise step by step.",
        knobs=TRAIN_KNOBS(1e-3, 30), **IMAGES, strengths="the highest-quality, most diverse samples")
starter("sd_lora", "stable_diffusion_lora", "text_to_image", "pretrained", "image",
        "Stable Diffusion LoRA", "Low-rank adapters teach a pretrained Stable Diffusion a "
        "subject or a style from a few dozen captioned images.",
        swaps={"image_folder": _images}, knobs=(lr("optimizer.lr", 1e-4),
                                                epochs("trainer.epochs", 10)),
        requires=("diffusers", "peft", "transformers"), extra="generative",
        family="text_to_image", strengths="photorealistic images from text prompts")
starter("tiny_gpt", "tiny_gpt", "language_modeling", "small", "text", "Tiny GPT",
        "A small causal transformer that writes text byte by byte.",
        swaps={"corpus": _corpus}, knobs=TRAIN_KNOBS(1e-3, 30),
        strengths="any language, no tokenizer files")
starter("tiny_llama", "tiny_llama", "language_modeling", "small", "text", "Tiny Llama",
        "A small Llama-style decoder: RMSNorm, grouped-query attention with rotary "
        "positions and SwiGLU feed-forward blocks.",
        swaps={"corpus": _corpus}, knobs=TRAIN_KNOBS(1e-3, 30), priority=-0.1,
        strengths="the architecture of today's open LLMs, small enough to read")
starter("seq2seq_transformer", "reverse_seq2seq", "sequence_to_sequence", "medium", "text",
        "Encoder-decoder transformer", "Reads the source sequence and writes the target "
        "(translation, summarisation, rewriting).", swaps={"pairs": _pairs},
        knobs=TRAIN_KNOBS(5e-4, 30), strengths="maps one sequence to another")
starter("realnvp", "flow_moons_realnvp", "density_estimation", "small", "tabular",
        "RealNVP flow", "Affine coupling layers: exact likelihoods and fast sampling.",
        knobs=(lr("opt.lr", 3e-3), epochs("trainer.epochs", 40)),
        swaps={"table": _density}, strengths="exact densities")
starter("spline_flow", "flow_checkerboard_spline", "density_estimation", "medium", "tabular",
        "Neural spline flow", "Rational-quadratic spline couplings for sharp, multi-modal "
        "densities.", knobs=(lr("opt.lr", 3e-3), epochs("trainer.epochs", 60)),
        swaps={"table": _density},
        strengths="complex, multi-modal densities")
starter("glow", "glow_shapes", "density_estimation", "large", "image", "Glow",
        "A multi-scale image flow: squeeze, ActNorm, invertible 1×1 convolutions and "
        "convolutional couplings. Exact likelihoods in bits per dimension, and new images.",
        knobs=(lr("opt.lr", 1e-3), epochs("trainer.epochs", 15)),
        swaps={"image_folder": _images}, fit=_fit_images,
        strengths="exact image likelihoods and sampling in one network")
# graphs, recommenders, RL
GRAPH_EXTRA = dict(requires=("torch_geometric",), extra="graph")
starter("gcn", "graph_node_classification", "node_classification", "small", "graph", "GCN",
        "Graph convolutions average each node's neighbourhood.", swaps={"graph": _graph},
        knobs=(lr("opt.lr", 1e-2), epochs("trainer.epochs", 100)), **GRAPH_EXTRA,
        strengths="the classic baseline for labelling nodes")
starter("gin", "graph_classification_gin", "graph_classification", "medium", "graph", "GIN",
        "Graph isomorphism network with global pooling: one prediction per graph.",
        knobs=(lr("opt.lr", 1e-2), epochs("trainer.epochs", 60)), **GRAPH_EXTRA,
        swaps={"graphs": _graphs},
        strengths="the most expressive simple message-passing network")
starter("graphsage_links", "graph_link_prediction", "link_prediction", "small", "graph",
        "GraphSAGE link predictor", "Node embeddings scored pairwise: which edges are "
        "missing?", swaps={"graph": _graph},
        knobs=(lr("opt.lr", 1e-2), epochs("trainer.epochs", 100)), **GRAPH_EXTRA,
        strengths="scales to large graphs by sampling neighbours")
starter("matrix_factorization", "recommender_matrix_factorization", "recommendation",
        "small", "interactions", "Matrix factorization", "A vector per user and per item; "
        "their dot product is the score.", swaps={"interactions": _interactions},
        knobs=(lr("opt.lr", 1e-2), epochs("trainer.epochs", 20)),
        strengths="the strong, simple collaborative-filtering baseline")
starter("dlrm", "recommender_dlrm", "recommendation", "medium", "interactions", "DLRM",
        "Embeddings plus user and item features with an MLP on top.",
        swaps={"interactions": _interactions},
        knobs=(lr("opt.lr", 1e-2), epochs("trainer.epochs", 20)),
        strengths="uses side features as well as ids")
RL_EXTRA = dict(requires=("gymnasium", "stable_baselines3"), extra="rl")
starter("ppo", "rl_cartpole_ppo", "reinforcement_learning", "small", "environment", "PPO",
        "Proximal policy optimisation on a classic-control environment.",
        knobs=(lr("algo.learning_rate", 3e-4),
               Knob("total_timesteps", "int", 40_000, 1_000, 10**8,
                    help="Environment steps", target="algo.total_timesteps")),
        **RL_EXTRA, strengths="the reliable default for discrete or continuous actions")
starter("sac", "rl_pendulum_sac", "reinforcement_learning", "medium", "environment", "SAC",
        "Soft actor-critic for continuous control.",
        knobs=(lr("algo.learning_rate", 1e-3),
               Knob("total_timesteps", "int", 20_000, 1_000, 10**8,
                    help="Environment steps", target="algo.total_timesteps")),
        **RL_EXTRA, strengths="sample-efficient continuous control")
starter("atari_ppo", "rl_atari_pong_ppo", "reinforcement_learning", "large", "environment",
        "Atari PPO (Nature CNN)", "PPO with the Nature-DQN convolutional policy on stacked "
        "84×84 grayscale frames of an Atari game.",
        knobs=(lr("algo.learning_rate", 2.5e-4),
               Knob("total_timesteps", "int", 2_000_000, 10_000, 10**9,
                    help="Environment steps", target="algo.total_timesteps")),
        requires=("gymnasium", "stable_baselines3", "ale_py"), extra="atari",
        strengths="the standard pixel-based benchmark; needs a GPU and millions of steps")
# probabilistic models
PROB = dict(extra="probabilistic")
GROUPS = dict(swaps={"table": _grouped}, fit=_fit_groups)
SERIES = dict(swaps={"series": _ssm}, fit=_fit_season)
starter("bayesian_network", "student_network", "probabilistic_inference", "small", "tabular",
        "Bayesian network", "Variables, their dependencies and probability tables; ask "
        "questions given evidence.", requires=("pgmpy",), family="pgm", **PROB,
        strengths="explainable reasoning under uncertainty")
starter("structure_learning", "structure_learning", "probabilistic_inference", "medium",
        "tabular", "Learned Bayesian network", "Learns the dependency graph and the tables "
        "from data.", requires=("pgmpy",), family="pgm", **PROB,
        knobs=(Knob("method", "choice", "ges", values=("ges", "hill_climb", "mmhc", "tree"),
                    help="Search", target="structure.method"),
               Knob("score", "choice", "bic", values=("bic", "bdeu", "k2"),
                    help="Structure score", target="structure.score"),
               Knob("equivalent_sample_size", "float", 5.0, 0.5, 100.0, log=True,
                    help="Dirichlet prior strength", target="parameters.equivalent_sample_size")),
        strengths="discovers which variables depend on which")
starter("hmm", "market_regimes_hmm", "regime_detection", "small", "timeseries",
        "Hidden Markov model", "Hidden regimes that switch over time.",
        requires=("hmmlearn",), family="pgm", **PROB,
        swaps={"series": _regimes}, fit=_fit_regimes,
        knobs=(Knob("iterations", "int", 100, 10, 10_000, help="EM iterations",
                    target="hmm.iterations"),
               Knob("n_states", "int", 3, 2, 6, help="Hidden states", target="hmm.n_states"),
               Knob("covariance", "choice", "diag", values=("diag", "full", "spherical"),
                    help="Emission covariance", target="hmm.covariance")),
        strengths="finds hidden states in a sequence")
starter("hierarchical_regression", "hierarchical_regression", "bayesian_modeling", "small",
        "tabular", "Hierarchical regression", "A Bayesian model with partial pooling across "
        "groups, sampled with NUTS.", requires=("pymc",), family="ppl", **PROB, **GROUPS,
        knobs=(Knob("draws", "int", 1000, 100, 100_000, help="Posterior draws",
                    target="sampler.draws"),
               Knob("non_centered", "choice", True, values=(True, False),
                    help="Non-centred group effects", target="a.non_centered"),
               Knob("group_scale", "float", 1.0, 0.1, 10.0, log=True,
                    help="Prior scale of the group effects", target="a.sigma")),
        strengths="honest uncertainty with few rows per group")
starter("hierarchical_logistic", "hierarchical_logistic", "bayesian_modeling", "medium",
        "tabular", "Hierarchical logistic regression", "Partial pooling for a yes/no "
        "outcome.", requires=("pymc",), family="ppl", **PROB, **GROUPS,
        knobs=(Knob("draws", "int", 1000, 100, 100_000, help="Posterior draws",
                    target="sampler.draws"),
               Knob("non_centered", "choice", True, values=(True, False),
                    help="Non-centred group effects", target="a.non_centered"),
               Knob("group_scale", "float", 1.0, 0.1, 10.0, log=True,
                    help="Prior scale of the group effects", target="a.sigma")),
        strengths="binary outcomes across groups")
starter("gp_regression", "gp_trend_seasonality", "gp_regression", "small", "tabular",
        "Gaussian process", "A kernel GP: trend plus seasonality with uncertainty bands.",
        family="gp", knobs=(lr("gp.lr", 0.05),
                            Knob("iterations", "int", 200, 10, 10_000,
                                 help="Optimisation steps", target="gp.iterations")),
        strengths="calibrated uncertainty from small data")
starter("structural_time_series", "structural_time_series", "state_space_forecasting",
        "small", "timeseries", "Structural time series", "Level, trend and seasonal "
        "components estimated with a Kalman filter.", requires=("statsmodels",),
        family="ssm", **PROB, **SERIES,
        knobs=(Knob("trend", "choice", "local linear trend",
                    values=("local level", "local linear trend", "smooth trend",
                            "random walk with drift"), help="Trend component",
                    target="trend.kind"),
               Knob("ar_order", "int", 1, 1, 4, help="Autoregressive order",
                    target="ar.order")),
        strengths="interpretable components and intervals")
starter("sarimax", "sarimax_airline", "state_space_forecasting", "medium", "timeseries",
        "SARIMAX", "Seasonal ARIMA with exogenous regressors.", requires=("statsmodels",),
        family="ssm", **PROB, **SERIES,
        knobs=(Knob("p", "int", 0, 0, 3, help="AR order", target="arima.p"),
               Knob("q", "int", 1, 0, 3, help="MA order", target="arima.q"),
               Knob("seasonal_q", "int", 1, 0, 2, help="Seasonal MA order",
                    target="arima.seasonal_q")),
        strengths="the classical statistical forecasting model")
# LLM workflows (exported as scripts, not trained in the app)
LLM = dict(family="llm", extra="llm")
starter("llm_prompting", "llm_generation", "text_generation", "pretrained", "text",
        "Prompted LLM", "A pretrained instruction model answers a prompt template.",
        requires=("transformers",), **LLM,
        strengths="no training: describe the task in the prompt")
starter("llm_lora", "llm_lora_finetune", "llm_finetuning", "pretrained", "text",
        "LoRA fine-tuning", "Supervised fine-tuning of a small LLM with LoRA adapters.",
        requires=("transformers", "peft", "datasets"), **LLM,
        strengths="adapts a model to your examples on modest hardware")
starter("rag_assistant", "llm_rag_assistant", "retrieval_qa", "pretrained", "text",
        "RAG assistant", "Embeds your documents, retrieves the relevant passages and answers "
        "from them.", requires=("transformers",), **LLM,
        strengths="answers grounded in your own documents")
