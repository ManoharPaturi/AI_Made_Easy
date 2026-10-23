"""Deep learning for tables: ResNet-MLP, FT-Transformer, TabTransformer, TabNet and
categorical entity embeddings.

Models read the preprocessed feature vector [F]. Categorical columns must be ordinal
encoded (Ordinal Encode block); their positions in the vector and their number of
categories are the ``categorical`` / ``cardinalities`` parameters, filled from the data by
a Quick Fix. Backbones return a representation: add a Dense head for the targets.
"""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition

CATEGORY = "Tabular Models"
CATEGORICAL_BLOCKS = ("tab.embedding", "tab.ft_transformer", "tab.tab_transformer")
TAB_BLOCKS = (*CATEGORICAL_BLOCKS, "tab.resnet", "tab.tabnet", "tab.tabm")

CAT = P("categorical", "str", "",
        help="Positions of the ordinal-encoded categorical columns in the input (e.g. 0, 3); "
             "the Quick Fix fills them from the data")
CARDS = P("cardinalities", "str", "", help="Number of categories of each of those columns")


def ints(text) -> list[int]:  # noqa: ANN001
    return [int(float(v)) for v in str(text or "").replace(";", ",").split(",") if v.strip()]


def auto_dim(card: int) -> int:
    """Embedding width from the number of categories (fast.ai's rule of thumb)."""
    return max(1, min(50, round(1.6 * (card + 1) ** 0.56)))


def columns(in_shapes, p) -> tuple[int, list[int], list[int]]:
    s = in_shapes[0]
    if len(s) != 1:
        raise ShapeError(f"tabular models read feature vectors [F], got {s}")
    f = int(s[0])
    if "categorical" not in p:
        return f, [], []
    cat, cards = ints(p["categorical"]), ints(p["cardinalities"])
    if len(cat) != len(cards):
        raise ShapeError(f"{len(cat)} categorical position(s) but {len(cards)} "
                         "cardinalities: give one per column")
    if any(i < 0 or i >= f for i in cat):
        raise ShapeError(f"categorical positions must be 0..{f - 1}")
    if len(set(cat)) != len(cat):
        raise ShapeError("a categorical position is listed twice")
    if any(c < 1 for c in cards):
        raise ShapeError("cardinalities must be at least 1")
    return f, cat, cards


def _dims(p, cards: list[int]) -> list[int]:
    d = int(p["dim"])
    return [d if d > 0 else auto_dim(c) for c in cards]


def _embed_shape(in_shapes, p):
    f, cat, cards = columns(in_shapes, p)
    return [f - len(cat) + sum(_dims(p, cards))]


def _encoder_params(d: int, ff: int, layers: int) -> int:
    per = (3 * d * d + 3 * d) + (d * d + d) + (d * ff + ff) + (ff * d + d) + 4 * d
    return per * layers


def _ft_params(in_shapes, p) -> int:
    f, cat, cards = columns(in_shapes, p)
    d = int(p["d"])
    return (2 * (f - len(cat)) * d + sum((c + 1) * d for c in cards) + d
            + _encoder_params(d, int(d * 4 / 3), int(p["layers"])) + 2 * d)


def _tt_params(in_shapes, p) -> int:
    f, cat, cards = columns(in_shapes, p)
    d, h, n_num = int(p["d"]), int(p["hidden"]), f - len(cat)
    width = len(cat) * d + n_num
    return (sum((c + 1) * d for c in cards) + _encoder_params(d, 4 * d, int(p["layers"]))
            + (2 * n_num if n_num else 0) + width * h + h + h * h + h)


def _resnet_params(in_shapes, p) -> int:
    f = columns(in_shapes, p)[0]
    d = int(p["d"])
    h = max(1, int(d * float(p["hidden_factor"])))
    return f * d + d + int(p["blocks"]) * (2 * d + d * h + h + h * d + d) + 2 * d


def _tabm_params(in_shapes, p) -> int:
    f = columns(in_shapes, p)[0]
    d, blocks = int(p["d"]), int(p["blocks"])
    return int(p["k"]) * f + (f * d + d) + (blocks - 1) * (d * d + d)


def _tabnet_params(in_shapes, p) -> int:
    f = columns(in_shapes, p)[0]
    n_d, n_a, steps = int(p["n_d"]), int(p["n_a"]), int(p["steps"])
    w = n_d + n_a

    def glu(n_in: int) -> int:
        return n_in * 2 * w + 4 * w

    return 2 * f + glu(f) + glu(w) + (steps + 1) * 2 * glu(w) + steps * (n_a * f + 2 * f)


def _encoder_flops(tokens: int, d: int, ff: int, layers: int) -> int:
    """Forward FLOPs of ``layers`` transformer layers over ``tokens`` tokens: the
    projections and feed-forward per token, plus attention scores and mixing."""
    per_token = 2 * (4 * d * d + 2 * d * ff) + 4 * tokens * d
    return layers * tokens * per_token


def _ft_cost(in_shapes, p) -> tuple[int, int]:
    f, _cat, _cards = columns(in_shapes, p)
    d, layers = int(p["d"]), int(p["layers"])
    tokens = f + 1                                   # one per column plus [CLS]
    return _ft_params(in_shapes, p), 2 * f * d + _encoder_flops(tokens, d, d * 4 // 3,
                                                                  layers) + 4 * d


def _tt_cost(in_shapes, p) -> tuple[int, int]:
    f, cat, _cards = columns(in_shapes, p)
    d, h = int(p["d"]), int(p["hidden"])
    width = len(cat) * d + f - len(cat)
    return (_tt_params(in_shapes, p),
            _encoder_flops(len(cat), d, 4 * d, int(p["layers"])) + 2 * (width * h + h * h))


# kernel launches per forward pass: attention layers run ~10 kernels (projections,
# scores, softmax, mixing, norms, feed-forward, residuals); ResNet blocks ~6
KERNELS = {
    "tab.ft_transformer": lambda p: 3 + 10 * int(p["layers"]),
    "tab.tab_transformer": lambda p: 6 + 10 * int(p["layers"]),
    "tab.resnet": lambda p: 3 + 6 * int(p["blocks"]),
    "tab.tabnet": lambda p: 4 + 12 * int(p["steps"]),
    "tab.tabm": lambda p: 2 + 3 * int(p["blocks"]),
}
COSTS = {"tab.ft_transformer": _ft_cost, "tab.tab_transformer": _tt_cost}


def _tt_shape(in_shapes, p):
    f, cat, _cards = columns(in_shapes, p)
    if not cat and f == 0:
        raise ShapeError("no input columns")
    return [int(p["hidden"])]


def _heads(in_shapes, p):
    columns(in_shapes, p)
    if int(p["d"]) % int(p["heads"]):
        raise ShapeError(f"d ({p['d']}) must be divisible by heads ({p['heads']})")


def _blocks() -> list[BlockDefinition]:
    def cat_args(c) -> str:
        return f"{ints(c['categorical'])}, {ints(c['cardinalities'])}"

    def ft_shape(s, p):
        _heads(s, p)
        return [int(p["d"])]

    def tt_shape(s, p):
        _heads(s, p)
        return _tt_shape(s, p)

    dropout = P("dropout", "float", 0.1, lo=0.0, hi=0.9)
    blocks = [
        nn_block("tab.embedding", "Categorical Embedding", CATEGORY, family="model",
                 params=(CAT, CARDS, P("dim", "int", 0, lo=0, hi=1024,
                                       help="Width per column (0: from its number of "
                                            "categories)")),
                 shape=_embed_shape,
                 param_fn=lambda s, p: sum((c + 1) * d for c, d in
                                           zip(columns(s, p)[2], _dims(p, columns(s, p)[2]),
                                               strict=True)),
                 torch=lambda c: (f"CategoricalEmbedding({c['input_shape'][0]}, {cat_args(c)}, "
                                  f"{_dims(c, ints(c['cardinalities']))})"),
                 torch_helpers=("TabularSplit", "CategoricalEmbedding"),
                 desc="Learned vectors for categorical codes (entity embeddings): numbers "
                      "pass through, each category column becomes a small dense vector."),
        nn_block("tab.resnet", "Tabular ResNet", CATEGORY, family="model",
                 params=(P("d", "int", 128, lo=1, hi=8192), P("blocks", "int", 3, lo=1, hi=64),
                         P("hidden_factor", "float", 2.0, lo=0.25, hi=16.0), dropout),
                 shape=lambda s, p: (columns(s, p), [int(p["d"])])[1], param_fn=_resnet_params,
                 torch=lambda c: (f"TabularResNet({c['input_shape'][0]}, {int(c['d'])}, "
                                  f"{int(c['blocks'])}, {float(c['hidden_factor'])}, "
                                  f"{float(c['dropout'])})"),
                 torch_helpers=("TabularResNet",),
                 desc="A strong simple baseline for tables: residual MLP blocks with batch "
                      "normalisation (Gorishniy et al., 2021)."),
        nn_block("tab.ft_transformer", "FT-Transformer", CATEGORY, family="model",
                 params=(CAT, CARDS, P("d", "int", 64, lo=8, hi=2048),
                         P("layers", "int", 3, lo=1, hi=24), P("heads", "int", 8, lo=1, hi=64),
                         dropout),
                 shape=ft_shape, param_fn=_ft_params,
                 torch=lambda c: (f"FTTransformer({c['input_shape'][0]}, {cat_args(c)}, "
                                  f"{int(c['d'])}, {int(c['layers'])}, {int(c['heads'])}, "
                                  f"{float(c['dropout'])})"),
                 torch_helpers=("TabularSplit", "FTTransformer"),
                 desc="Feature tokenizer + transformer: every column becomes a token and "
                      "attention learns interactions (Gorishniy et al., 2021)."),
        nn_block("tab.tab_transformer", "TabTransformer", CATEGORY, family="model",
                 params=(CAT, CARDS, P("d", "int", 32, lo=8, hi=1024),
                         P("layers", "int", 3, lo=1, hi=24), P("heads", "int", 4, lo=1, hi=64),
                         P("hidden", "int", 128, lo=1, hi=8192), dropout),
                 shape=tt_shape, param_fn=_tt_params,
                 torch=lambda c: (f"TabTransformer({c['input_shape'][0]}, {cat_args(c)}, "
                                  f"{int(c['d'])}, {int(c['layers'])}, {int(c['heads'])}, "
                                  f"{int(c['hidden'])}, {float(c['dropout'])})"),
                 torch_helpers=("TabularSplit", "TabTransformer"),
                 desc="Self-attention over the categorical columns' embeddings, joined with "
                      "the numbers in an MLP (Huang et al., 2020)."),
        nn_block("tab.tabnet", "TabNet", CATEGORY, family="model",
                 params=(P("n_d", "int", 16, lo=1, hi=1024, help="Decision width"),
                         P("n_a", "int", 16, lo=1, hi=1024, help="Attention width"),
                         P("steps", "int", 3, lo=1, hi=20, help="Decision steps"),
                         P("gamma", "float", 1.3, lo=1.0, hi=3.0,
                           help="How much a feature may be reused across steps (1: never)"),
                         P("sparsity", "float", 1e-3, lo=0.0, hi=1.0,
                           help="Weight of the mask-entropy penalty added to the loss")),
                 shape=lambda s, p: (columns(s, p), [int(p["n_d"])])[1],
                 param_fn=_tabnet_params,
                 torch=lambda c: (f"TabNet({c['input_shape'][0]}, {int(c['n_d'])}, "
                                  f"{int(c['n_a'])}, {int(c['steps'])}, {float(c['gamma'])}, "
                                  f"{float(c['sparsity'])})"),
                 torch_helpers=("TabNet",),
                 desc="Sequential attention picks a sparse set of columns at each decision "
                      "step — interpretable feature selection (Arik & Pfister, 2021). Needs "
                      "large tables and long training; on small ones start with Tabular "
                      "ResNet."),
        nn_block("tab.tabm", "TabM", CATEGORY, family="model",
                 params=(P("d", "int", 256, lo=1, hi=8192), P("blocks", "int", 3, lo=1, hi=16),
                         P("k", "int", 16, lo=1, hi=64, help="Ensemble members"), dropout),
                 shape=lambda s, p: (columns(s, p), [int(p["d"])])[1], param_fn=_tabm_params,
                 torch=lambda c: (f"TabM({c['input_shape'][0]}, {int(c['d'])}, "
                                  f"{int(c['blocks'])}, {int(c['k'])}, {float(c['dropout'])})"),
                 torch_helpers=("TabM",),
                 desc="An ensemble of MLPs that share their weights, each member seeing the "
                      "columns through its own learned scaling (TabM-mini, Gorishniy et al., "
                      "2025): among the strongest deep models for tables, and fast."),
    ]
    out = []
    for b in blocks:
        meta = {**(b.meta or {}), "tabular": True}
        if b.type_id in KERNELS:
            meta["kernels"] = KERNELS[b.type_id]
        if b.type_id in COSTS:
            meta["cost"] = COSTS[b.type_id]
        if b.type_id == "tab.tabnet":
            meta["aux_loss"] = True
        out.append(replace(b, meta=meta))
    return out


def register_all() -> None:
    reg = get_registry()
    for defn in _blocks():
        reg.register(defn)
