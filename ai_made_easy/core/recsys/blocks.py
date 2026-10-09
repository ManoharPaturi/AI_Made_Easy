"""Recommender blocks: matrix factorisation, neural collaborative filtering, two-tower
retrieval and DLRM-lite; the training objective; interaction datasets.

The Input holds one row per (user, item) pair: ``[user, item, *dense features]`` with ids
as numbers (the training script maps raw ids to 0..n-1). Models output one score.
"""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition

CATEGORY = "Recommenders"
MODELS = ("rec.mf", "rec.ncf", "rec.two_tower", "rec.dlrm")
REC_DATA = ("data.synthetic_interactions", "data.interactions_csv")
USERS = P("n_users", "int", 1000, lo=1, hi=100_000_000,
          help="Size of the user embedding table (at least the number of users)")
ITEMS = P("n_items", "int", 1000, lo=1, hi=100_000_000,
          help="Size of the item embedding table (at least the number of items)")
DIM = P("dim", "int", 32, lo=1, hi=4096, help="Embedding width")


def ints(text) -> list[int]:  # noqa: ANN001
    return [int(float(v)) for v in str(text or "").split(",") if v.strip()]


def _pairs(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 1 or s[0] < 2:
        raise ShapeError(f"recommenders read rows [user, item, ...features], got {s}")
    return [1]


def _layers(text, name: str) -> list[int]:
    sizes = ints(text)
    if not sizes or any(v < 1 for v in sizes):
        raise ShapeError(f"{name}: give layer widths like 64, 32")
    return sizes


def _ncf_shape(in_shapes, p):
    _layers(p["hidden"], "hidden")
    return _pairs(in_shapes, p)


def _dlrm_shape(in_shapes, p):
    _layers(p["bottom"], "bottom")
    _layers(p["top"], "top")
    return _pairs(in_shapes, p)


def _mlp_params(sizes: list[int]) -> int:
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:], strict=False))


def _mf_params(s, p) -> int:
    u, i, d = int(p["n_users"]), int(p["n_items"]), int(p["dim"])
    return (u + i) * d + ((u + i + 1) if p["biases"] else 0)


def _ncf_params(s, p) -> int:
    u, i, d = int(p["n_users"]), int(p["n_items"]), int(p["dim"])
    hidden = ints(p["hidden"])
    return 2 * (u + i) * d + _mlp_params([2 * d, *hidden]) + (d + hidden[-1]) + 1


def _tower_params(s, p) -> int:
    u, i, d, h = int(p["n_users"]), int(p["n_items"]), int(p["dim"]), int(p["hidden"])
    return (u + i) * h + 2 * _mlp_params([h, h, d])


def _dlrm_params(s, p) -> int:
    u, i, d = int(p["n_users"]), int(p["n_items"]), int(p["dim"])
    dense = s[0][0] - 2
    bottom, top = ints(p["bottom"]), ints(p["top"])
    n_vec = 3 if dense else 2
    pairs = n_vec * (n_vec - 1) // 2
    return ((u + i) * d + (_mlp_params([dense, *bottom, d]) if dense else 0)
            + _mlp_params([d + pairs, *top]) + top[-1] + 1)


def _models() -> list[BlockDefinition]:
    blocks = [
        nn_block("rec.mf", "Matrix Factorization", CATEGORY, family="model",
                 params=(USERS, ITEMS, DIM, P("biases", "bool", True,
                                               help="User, item and global biases")),
                 shape=_pairs, param_fn=_mf_params,
                 torch=lambda c: (f"MatrixFactorization({int(c['n_users'])}, "
                                  f"{int(c['n_items'])}, {int(c['dim'])}, {bool(c['biases'])})"),
                 torch_helpers=("RecIds", "MatrixFactorization",),
                 desc="Collaborative filtering: users and items as vectors; the score is "
                      "their dot product (plus biases)."),
        nn_block("rec.ncf", "Neural Collaborative Filtering", CATEGORY, family="model",
                 params=(USERS, ITEMS, DIM, P("hidden", "str", "64, 32",
                                               help="MLP path widths")),
                 shape=_ncf_shape, param_fn=_ncf_params,
                 torch=lambda c: (f"NeuralCF({int(c['n_users'])}, {int(c['n_items'])}, "
                                  f"{int(c['dim'])}, {ints(c['hidden'])})"),
                 torch_helpers=("RecIds", "NeuralCF",),
                 desc="NCF: a matrix-factorisation path and an MLP path over user and item "
                      "embeddings (He et al., 2017)."),
        nn_block("rec.two_tower", "Two-Tower Retrieval", CATEGORY, family="model",
                 params=(USERS, ITEMS, DIM, P("hidden", "int", 64, lo=1, hi=8192),
                         P("temperature", "float", 0.1, lo=0.001, hi=10.0,
                           help="Scales the cosine similarity")),
                 shape=_pairs, param_fn=_tower_params,
                 torch=lambda c: (f"TwoTower({int(c['n_users'])}, {int(c['n_items'])}, "
                                  f"{int(c['dim'])}, {int(c['hidden'])}, "
                                  f"{float(c['temperature'])})"),
                 torch_helpers=("RecIds", "TwoTower",),
                 desc="User and item towers produce vectors compared by cosine similarity: "
                      "item vectors can be indexed for nearest-neighbour retrieval."),
        nn_block("rec.dlrm", "DLRM", CATEGORY, family="model",
                 params=(USERS, ITEMS, P("dim", "int", 16, lo=1, hi=1024),
                         P("bottom", "str", "64", help="Bottom MLP widths (dense features)"),
                         P("top", "str", "64, 32", help="Top MLP widths")),
                 shape=_dlrm_shape, param_fn=_dlrm_params,
                 torch=lambda c: (f"DLRM({int(c['n_users'])}, {int(c['n_items'])}, "
                                  f"{c['input_shape'][0] - 2}, {int(c['dim'])}, "
                                  f"{ints(c['bottom'])}, {ints(c['top'])})"),
                 torch_helpers=("RecIds", "NeuralCF", "DLRM"),
                 desc="Deep learning recommendation model (lite): dense user / item features "
                      "and id embeddings interact through dot products (Naumov et al., 2019)."),
    ]
    return [replace(b, meta={**(b.meta or {}), "recsys": True}) for b in blocks]


def _config() -> list[BlockDefinition]:
    return [BlockDefinition(
        type_id="rec.objective", display_name="Recommender Objective", category=CATEGORY,
        color=family_color("model"), library="PyTorch",
        params=(P("objective", "enum", "auto", options=("auto", "bpr", "bce", "mse"),
                  help="auto: BPR for implicit feedback, MSE for ratings; bpr: rank a seen "
                       "item above an unseen one; bce: classify seen vs sampled unseen"),
                P("negatives", "int", 4, lo=1, hi=100,
                  help="Unseen items sampled per interaction (bce)"),
                P("top_k", "int", 10, lo=1, hi=1000, help="K for Recall@K and NDCG@K")),
        description="How the recommender learns (pairwise ranking, pointwise "
                    "classification or rating regression) and K for the ranking metrics.",
        meta={"recsys": "objective"})]


def _data() -> list[BlockDefinition]:
    split = (P("val_fraction", "float", 0.1, lo=0.0, hi=0.5),
             P("test_fraction", "float", 0.2, lo=0.05, hi=0.5))

    def block(type_id, name, params, desc):  # noqa: ANN001, ANN202
        return BlockDefinition(type_id=type_id, display_name=name, category="Data",
                               color=family_color("data"), library="PyTorch",
                               params=(*params, *split), description=desc,
                               meta={"modality": "interactions"})

    return [
        block("data.synthetic_interactions", "Synthetic Interactions",
              (P("n_users", "int", 600, lo=10, hi=10_000_000),
               P("n_items", "int", 400, lo=10, hi=10_000_000),
               P("per_user", "int", 30, lo=2, hi=10000, help="Average interactions per user"),
               P("feedback", "enum", "implicit", options=("implicit", "ratings")),
               P("features", "int", 4, lo=0, hi=256,
                 help="Dense features per user and per item (noisy views of their tastes)"),
               P("seed", "int", 0, lo=0)),
              "Users and items with hidden tastes (latent factors) and a popularity skew: "
              "who interacted with what, or 1–5 star ratings."),
        block("data.interactions_csv", "Interactions CSV",
              (P("path", "str", "interactions.csv"), P("user_column", "str", "user"),
               P("item_column", "str", "item"),
               P("rating_column", "str", "", help="Empty: implicit feedback (every row is an "
                                                  "interaction)")),
              "Your own interaction log: one row per user–item interaction (optionally with "
              "a rating)."),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in (*_models(), *_config(), *_data()):
        reg.register(defn)
