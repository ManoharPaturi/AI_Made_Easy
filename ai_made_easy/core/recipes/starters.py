"""Recipes built from the shipped sample projects: one or more per remaining task
(vision, forecasting, speech, generative, sequence, probabilistic, graph, recommender and
reinforcement-learning designs). With the user's data the sample's demo dataset block is
swapped for one that reads it; the rest of the design is fitted by :func:`autofit`."""
from __future__ import annotations

import copy
from typing import Callable

from ai_made_easy.core.recipes import Context, Draft, Knob, Recipe, register_recipe
from ai_made_easy.core.recipes.facts import DataFacts

Swap = Callable[[DataFacts], tuple[str, dict, str]]   # -> (type_id, params, why)


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
                               "source_column": facts.columns["source"],
                               "target_column": facts.columns["target"]},
            "your graph: a node table and an edge list")


def _interactions(facts: DataFacts) -> tuple[str, dict, str]:
    return ("data.interactions_csv", {"path": facts.source,
                                      "user_column": facts.columns["user"],
                                      "item_column": facts.columns["item"],
                                      "rating_column": facts.columns.get("rating", "")},
            "your user-item interactions")


# ------------------------------------------------------------------ starter recipes

def starter(recipe_id: str, sample: str, task: str, tier: str, modality: str, title: str,
            description: str, *, swaps: dict[str, Swap] | None = None,
            knobs: tuple[Knob, ...] = (), edit: Callable[[dict], None] | None = None,
            requires: tuple[str, ...] = (), extra: str = "", strengths: str = "",
            demo_note: str = "", rows: tuple[int, int] = (0, 10**12),
            priority: float = 0.0, family: str = "neural") -> Recipe:
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
        knobs=TRAIN_KNOBS(1e-3, 25), requires=("torchaudio",), extra="audio",
        strengths="small enough for a microcontroller-sized budget")
starter("audio_tagger", "keyword_spotting", "audio_tagging", "small", "audio",
        "Audio tagger", "The mel CNN with one yes/no output per tag: several sounds can be "
        "present at once.", knobs=TRAIN_KNOBS(1e-3, 25), edit=_speech_tags,
        requires=("torchaudio",), extra="audio", strengths="multi-label sound events")
starter("ctc_lstm", "speech_recognition_ctc", "speech_recognition", "medium", "audio",
        "BiLSTM + CTC", "Mel features, a bidirectional LSTM and CTC: transcribes speech "
        "letter by letter.", knobs=TRAIN_KNOBS(3e-3, 40), requires=("torchaudio",),
        extra="audio", strengths="learns to transcribe without aligned labels")
# generative
starter("vae", "shapes_vae", "vae_generation", "small", "image", "Variational autoencoder",
        "An encoder to a latent space and a decoder back: smooth, controllable generation.",
        knobs=TRAIN_KNOBS(1e-3, 30), strengths="stable training, a meaningful latent space")
starter("dcgan", "shapes_gan", "gan_generation", "medium", "image", "DCGAN",
        "A generator and a critic trained against each other.",
        knobs=(lr("optimizer.lr", 2e-4), epochs("trainer.epochs", 40)),
        strengths="sharp samples")
starter("ddpm", "shapes_diffusion", "diffusion_generation", "large", "image",
        "Diffusion (DDPM)", "A U-Net learns to remove noise step by step.",
        knobs=TRAIN_KNOBS(1e-3, 30), strengths="the highest-quality, most diverse samples")
starter("tiny_gpt", "tiny_gpt", "language_modeling", "small", "text", "Tiny GPT",
        "A small causal transformer that writes text byte by byte.",
        swaps={"corpus": _corpus}, knobs=TRAIN_KNOBS(1e-3, 30),
        strengths="any language, no tokenizer files")
starter("seq2seq_transformer", "reverse_seq2seq", "sequence_to_sequence", "medium", "text",
        "Encoder-decoder transformer", "Reads the source sequence and writes the target "
        "(translation, summarisation, rewriting).", swaps={"pairs": _pairs},
        knobs=TRAIN_KNOBS(5e-4, 30), strengths="maps one sequence to another")
starter("realnvp", "flow_moons_realnvp", "density_estimation", "small", "tabular",
        "RealNVP flow", "Affine coupling layers: exact likelihoods and fast sampling.",
        knobs=(lr("opt.lr", 3e-3), epochs("trainer.epochs", 40)),
        strengths="exact densities")
starter("spline_flow", "flow_checkerboard_spline", "density_estimation", "medium", "tabular",
        "Neural spline flow", "Rational-quadratic spline couplings for sharp, multi-modal "
        "densities.", knobs=(lr("opt.lr", 3e-3), epochs("trainer.epochs", 60)),
        strengths="complex, multi-modal densities")
# graphs, recommenders, RL
GRAPH_EXTRA = dict(requires=("torch_geometric",), extra="graph")
starter("gcn", "graph_node_classification", "node_classification", "small", "graph", "GCN",
        "Graph convolutions average each node's neighbourhood.", swaps={"graph": _graph},
        knobs=(lr("opt.lr", 1e-2), epochs("trainer.epochs", 100)), **GRAPH_EXTRA,
        strengths="the classic baseline for labelling nodes")
starter("gin", "graph_classification_gin", "graph_classification", "medium", "graph", "GIN",
        "Graph isomorphism network with global pooling: one prediction per graph.",
        knobs=(lr("opt.lr", 1e-2), epochs("trainer.epochs", 60)), **GRAPH_EXTRA,
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
# probabilistic models
PROB = dict(extra="probabilistic")
starter("bayesian_network", "student_network", "probabilistic_inference", "small", "tabular",
        "Bayesian network", "Variables, their dependencies and probability tables; ask "
        "questions given evidence.", requires=("pgmpy",), family="pgm", **PROB,
        strengths="explainable reasoning under uncertainty")
starter("structure_learning", "structure_learning", "probabilistic_inference", "medium",
        "tabular", "Learned Bayesian network", "Learns the dependency graph and the tables "
        "from data.", requires=("pgmpy",), family="pgm", **PROB,
        strengths="discovers which variables depend on which")
starter("hmm", "market_regimes_hmm", "regime_detection", "small", "timeseries",
        "Hidden Markov model", "Hidden regimes that switch over time.",
        requires=("hmmlearn",), family="pgm", **PROB,
        knobs=(Knob("iterations", "int", 100, 10, 10_000, help="EM iterations",
                    target="hmm.iterations"),),
        strengths="finds hidden states in a sequence")
starter("hierarchical_regression", "hierarchical_regression", "bayesian_modeling", "small",
        "tabular", "Hierarchical regression", "A Bayesian model with partial pooling across "
        "groups, sampled with NUTS.", requires=("pymc",), family="ppl", **PROB,
        knobs=(Knob("draws", "int", 1000, 100, 100_000, help="Posterior draws",
                    target="sampler.draws"),),
        strengths="honest uncertainty with few rows per group")
starter("hierarchical_logistic", "hierarchical_logistic", "bayesian_modeling", "medium",
        "tabular", "Hierarchical logistic regression", "Partial pooling for a yes/no "
        "outcome.", requires=("pymc",), family="ppl", **PROB,
        knobs=(Knob("draws", "int", 1000, 100, 100_000, help="Posterior draws",
                    target="sampler.draws"),),
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
        family="ssm", **PROB, strengths="interpretable components and intervals")
starter("sarimax", "sarimax_airline", "state_space_forecasting", "medium", "timeseries",
        "SARIMAX", "Seasonal ARIMA with exogenous regressors.", requires=("statsmodels",),
        family="ssm", **PROB, strengths="the classical statistical forecasting model")
