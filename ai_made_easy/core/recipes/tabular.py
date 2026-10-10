"""Recipes for tables: neural networks (MLP, Tabular ResNet, FT-Transformer) and
scikit-learn models (random forest, gradient boosting, linear) for classification,
regression, multi-label and distribution targets."""
from __future__ import annotations

from ai_made_easy.core.recipes import Context, Draft, Knob, Recipe, register_recipe
from ai_made_easy.core.recipes.facts import DataFacts

CLASSIFY = ("multiclass", "binary")
NEURAL_TASKS = (*CLASSIFY, "regression", "multilabel", "distribution")
# demo datasets bundled with scikit-learn (no download): name, features, classes
DEMO = {"multiclass": ("wine", 13, 3), "binary": ("breast_cancer", 30, 2),
        "regression": ("diabetes", 10, 0)}
MULTILABEL_DEMO = {"kind": "classification", "n_samples": 2000, "n_features": 20,
                   "n_classes": 4, "noise": 0.1, "seed": 42}
EPOCHS = lambda v: Knob("epochs", "int", v, 1, 500, help="Passes over the data")  # noqa: E731


# ------------------------------------------------------------------ shared pieces

def outputs(ctx: Context) -> int:
    """Output units of the last layer for the task."""
    facts = ctx.facts
    if ctx.task in ("regression", "binary"):
        return 1
    if ctx.task == "multilabel":
        return MULTILABEL_DEMO["n_classes"]
    if facts.classes:
        return max(facts.n_classes, 2)
    return DEMO["multiclass"][2] if ctx.task == "multiclass" else 3


def data_blocks(d: Draft, ctx: Context, *, ordinal: bool = False,
                demo: tuple[str, dict] | None = None) -> int:
    """Dataset + cleaning + encoding + scaling; returns the input width."""
    facts, task = ctx.facts, ctx.task
    if facts.demo:
        if demo is not None:
            d.add(demo[0], "demo data with numeric and categorical columns: replace it "
                           "with your own table", nid="data", **demo[1])
            if ordinal:
                d.add("prep.ordinal_encode", "turns the categories into integer codes the "
                                             "embeddings look up", nid="encode")
            d.add("prep.normalize", "puts every numeric column on the same scale",
                  nid="normalize", mode="fit")
            return 0
        if task in ("multilabel", "distribution"):
            params = dict(MULTILABEL_DEMO, n_classes=3) if task == "distribution" \
                else MULTILABEL_DEMO
            d.add("data.synthetic", "generated demo data: replace it with your own table",
                  nid="data", **params)
            d.add("prep.normalize", "puts every column on the same scale (training is "
                                    "faster and steadier)", nid="normalize", mode="fit")
            return int(params["n_features"])
        name, width, _classes = DEMO["regression" if task == "regression" else
                                     "binary" if task == "binary" else "multiclass"]
        d.add("data.sklearn", f"the {name} demo table bundled with scikit-learn: replace "
                              "it with your own data", nid="data", dataset=name)
        d.add("prep.normalize", "puts every column on the same scale (training is faster "
                                "and steadier)", nid="normalize", mode="fit")
        return width
    d.add("data.csv", f"your table; '{facts.target}' is the column to predict", nid="data",
          path=facts.source, format=facts.format or "csv", target_column=facts.target)
    if facts.missing:
        d.add("prep.impute", "some values are missing: fill them with the column median",
              nid="impute", strategy="median")
    if facts.categorical:
        if ordinal:
            d.add("prep.ordinal_encode", f"{len(facts.categorical)} text column(s) become "
                                         "integer codes the embeddings look up", nid="encode")
        else:
            d.add("prep.one_hot", f"{len(facts.categorical)} text column(s) become one 0/1 "
                                  "column per category", nid="encode")
    d.add("prep.normalize", "puts every numeric column on the same scale (training is faster "
                            "and steadier)", nid="normalize", mode="fit")
    if ordinal or not facts.categorical:
        return len(facts.features)
    return len(facts.numeric) + sum(facts.cardinalities)


def head(ctx: Context) -> list[tuple]:
    """Final layer (+ activation) for the task."""
    n = outputs(ctx)
    what = {"regression": "one number: the predicted value",
            "binary": "one score: the log-odds of the positive class",
            "multilabel": f"{n} scores: one yes/no per label",
            "distribution": f"{n} log-probabilities: a predicted distribution"}.get(
        ctx.task, f"{n} scores, one per class")
    steps = [("core.dense", {"units": n}, f"outputs {what}")]
    if ctx.task == "distribution":
        steps.append(("core.log_softmax", {}, "turns the scores into log-probabilities, "
                                              "which the KL-divergence loss expects"))
    return steps


def training(d: Draft, ctx: Context, *, optimizer: str = "train.adamw", epochs: int,
             batch_size: int = 64) -> None:
    task = ctx.task
    loss, note = {
        "regression": ("train.loss_mse", "mean squared error: penalises large misses most"),
        "binary": ("train.loss_bce_logits", "binary cross-entropy on the log-odds"),
        "multilabel": ("train.loss_bce_logits", "binary cross-entropy for every label"),
        "distribution": ("train.loss_kl_div", "KL divergence between the predicted and the "
                                              "true distribution"),
    }.get(task, ("train.loss_cross_entropy", "cross-entropy: the standard loss for picking "
                                             "one class"))
    d.add(loss, note, nid="loss")
    d.add(optimizer, "AdamW: Adam with decoupled weight decay, a robust default"
          if optimizer == "train.adamw" else "", nid="optimizer", lr=ctx["lr"])
    d.add("train.trainer", "early stopping ends training when the validation loss stops "
                           "improving", nid="trainer", epochs=ctx["epochs"] or epochs,
          batch_size=batch_size, early_stopping_patience=max(5, (ctx["epochs"] or epochs) // 6))
    metrics(d, task)


def metrics(d: Draft, task: str) -> None:
    if task == "regression":
        d.add("eval.rmse", "root mean squared error, in the target's units", nid="rmse")
        d.add("eval.mae", "mean absolute error: the typical miss", nid="mae")
        d.add("eval.r2", "R²: the share of the variance explained (1 is perfect)", nid="r2")
    elif task in ("multilabel", "distribution"):
        d.add("eval.f1", "F1 balances precision and recall", nid="f1")
    else:
        d.add("eval.accuracy", "the share of rows classified correctly", nid="accuracy")
        d.add("eval.f1", "macro F1: every class counts equally, so rare classes matter",
              nid="f1")
        if task == "binary":
            d.add("eval.roc_auc", "ROC AUC: how well the scores rank positives above "
                                  "negatives", nid="roc_auc")


# ------------------------------------------------------------------ neural recipes

def _mlp(ctx: Context) -> Draft:
    d = Draft("tabular_mlp", "Tabular — MLP", "A small fully-connected network.")
    width = data_blocks(d, ctx)
    steps: list[tuple] = [("core.input", {"shape": str(width or 1)},
                           "one row of the table: one value per (encoded) feature")]
    units = int(ctx["width"])
    for layer in range(int(ctx["depth"])):
        steps += [("core.dense", {"units": units}, "a fully-connected layer mixes all the "
                                                   "features" if layer == 0 else
                   "a narrower layer combines the first layer's features"),
                  ("core.relu", {}, "ReLU lets the network learn non-linear patterns")]
        if ctx["dropout"] > 0:
            steps.append(("core.dropout", {"p": ctx["dropout"]},
                          "dropout switches off random units so the network does not "
                          "memorise the rows"))
        units = max(units // 2, 8)
    d.chain(*steps, *head(ctx), ("core.output", {}, ""))
    training(d, ctx, epochs=80)
    return d


def _resnet(ctx: Context) -> Draft:
    d = Draft("tabular_resnet", "Tabular — ResNet", "Residual MLP blocks for tables.")
    width = data_blocks(d, ctx)
    d.chain(("core.input", {"shape": str(width or 1)}, "one row of the table"),
            ("tab.resnet", {"d": int(ctx["d"]), "blocks": int(ctx["blocks"]),
                            "dropout": ctx["dropout"]},
             "residual blocks (norm → linear → ReLU → dropout → linear + skip): a strong, "
             "easy-to-train baseline for tables"),
            *head(ctx), ("core.output", {}, ""))
    training(d, ctx, epochs=60, batch_size=128)
    return d


def _ft(ctx: Context) -> Draft:
    d = Draft("tabular_ft_transformer", "Tabular — FT-Transformer",
              "Feature tokens and self-attention.")
    task = "classification" if ctx.task in CLASSIFY else "regression"
    width = data_blocks(d, ctx, ordinal=True,
                        demo=("data.synthetic_table", {"task": task, "n_samples": 3000})
                        if ctx.facts.demo else None)
    heads = 4 if int(ctx["d"]) % 4 == 0 else 2
    d.chain(("core.input", {"shape": str(width or 6)}, "one row: numeric values and "
                                                       "category codes"),
            ("tab.ft_transformer", {"d": int(ctx["d"]), "layers": int(ctx["layers"]),
                                    "heads": heads, "dropout": 0.1},
             "turns every feature into a token and lets attention find interactions "
             "between them; categories get their own embeddings"),
            *head(ctx), ("core.output", {}, ""))
    training(d, ctx, epochs=40, batch_size=128)
    return d


NEURAL_KINDS = ("demo", "table")
register_recipe(Recipe(
    "tabular_mlp", "MLP", NEURAL_TASKS, "small", "tabular",
    "A few fully-connected layers with dropout: quick to train on any table.", _mlp,
    data_kinds=NEURAL_KINDS, rows=(0, 200_000),
    knobs=(Knob("lr", "float", 3e-3, 1e-5, 0.1, log=True, help="Optimizer learning rate"),
           Knob("width", "int", 64, 8, 1024, help="Units in the first layer"),
           Knob("depth", "int", 2, 1, 6, help="Hidden layers"),
           Knob("dropout", "float", 0.1, 0.0, 0.6),
           EPOCHS(80)),
    strengths="fast and simple: a solid first neural baseline"))
register_recipe(Recipe(
    "tabular_resnet", "Tabular ResNet", (*CLASSIFY, "regression"), "medium", "tabular",
    "Residual MLP blocks with normalization (Gorishniy et al. 2021): trains reliably as it "
    "gets deeper.", _resnet, data_kinds=NEURAL_KINDS, rows=(1_000, 10**7),
    knobs=(Knob("lr", "float", 1e-3, 1e-5, 0.1, log=True, help="Optimizer learning rate"),
           Knob("d", "int", 128, 16, 1024, help="Width of the blocks"),
           Knob("blocks", "int", 2, 1, 8, help="Residual blocks"),
           Knob("dropout", "float", 0.1, 0.0, 0.5), EPOCHS(60)),
    strengths="deep networks that still train smoothly on tables"))
register_recipe(Recipe(
    "tabular_ft_transformer", "FT-Transformer", (*CLASSIFY, "regression"), "large",
    "tabular", "Feature-tokenizer transformer: embeds every column and models interactions "
               "with attention.", _ft, data_kinds=NEURAL_KINDS, rows=(5_000, 10**8),
    knobs=(Knob("lr", "float", 1e-3, 1e-5, 0.05, log=True, help="Optimizer learning rate"),
           Knob("d", "choice", 32, values=(16, 32, 64, 128), help="Token width"),
           Knob("layers", "int", 2, 1, 6, help="Transformer layers"), EPOCHS(40)),
    strengths="learns interactions between categorical columns"))


# ------------------------------------------------------------------ scikit-learn recipes

def _classic(model: str, params: dict, note: str):  # noqa: ANN202
    def builder(ctx: Context) -> Draft:
        regression = ctx.task == "regression"
        type_id = model.format(kind="regressor" if regression else "classifier")
        d = Draft(type_id.split(".", 1)[1], f"Tabular — {type_id.split('.', 1)[1]}", note)
        data_blocks(d, ctx)
        if ctx.task in CLASSIFY and not ctx.facts.demo and _imbalanced(ctx.facts):
            d.add("prep.class_balance", "the classes are imbalanced: weight rare classes up",
                  nid="balance")
        d.add(type_id, note, nid="model", **{k: ctx.knobs.get(k, v) for k, v in params.items()})
        d.add("train.kfold", "5-fold cross-validation: every row is used for both training "
                             "and validation", nid="kfold", k=5, stratified=not regression)
        metrics(d, ctx.task)
        return d
    return builder


def _imbalanced(facts: DataFacts) -> bool:
    return any("imbalanc" in w.lower() for w in facts.warnings)


CLASSIC_TASKS = (*CLASSIFY, "regression")
register_recipe(Recipe(
    "random_forest", "Random Forest", CLASSIC_TASKS, "small", "tabular",
    "Hundreds of decision trees vote: hard to overfit, little tuning, works on raw features.",
    _classic("ml.random_forest_{kind}", {"n_estimators": 300, "max_depth": 0},
             "a random forest: many decision trees trained on random subsets, averaged"),
    data_kinds=NEURAL_KINDS, family="classic", rows=(0, 500_000), priority=0.4,
    knobs=(Knob("n_estimators", "int", 300, 50, 2000, help="Trees"),
           Knob("max_depth", "int", 0, 0, 64, help="Tree depth (0: unlimited)")),
    strengths="trees are hard to beat on small and medium tables"))
register_recipe(Recipe(
    "gradient_boosting", "Gradient Boosting", CLASSIC_TASKS, "medium", "tabular",
    "Histogram gradient boosting: trees added one at a time to fix the previous ones' "
    "mistakes; the usual winner on tabular benchmarks.",
    _classic("ml.hist_gradient_boosting_{kind}", {"learning_rate": 0.1, "max_iter": 300,
                                                  "max_leaf_nodes": 31},
             "gradient-boosted trees: each new tree corrects the errors of the ones before"),
    data_kinds=NEURAL_KINDS, family="classic", rows=(500, 10**8), priority=0.6,
    knobs=(Knob("learning_rate", "float", 0.1, 0.005, 0.5, log=True),
           Knob("max_iter", "int", 300, 50, 3000, help="Boosting rounds"),
           Knob("max_leaf_nodes", "int", 31, 4, 256)),
    strengths="state of the art on most tables, handles missing values"))
register_recipe(Recipe(
    "linear_baseline", "Linear Baseline", CLASSIC_TASKS, "small", "tabular",
    "Logistic regression or ridge regression: the baseline every other model must beat.",
    lambda ctx: _classic("ml.ridge" if ctx.task == "regression" else
                         "ml.logistic_regression", {"C": 1.0} if ctx.task != "regression"
                         else {"alpha": 1.0},
                         "a linear model: one weight per feature, easy to interpret")(ctx),
    data_kinds=NEURAL_KINDS, family="classic", rows=(0, 10**8), priority=-0.4,
    strengths="a quick, interpretable baseline"))
