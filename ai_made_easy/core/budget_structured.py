"""Cost estimates for designs without layers: classic ML, graphical models, PyMC
programs, Gaussian processes and state-space models.

These models are sized by their data, so the estimates read the dataset block (rows,
features, classes; files are profiled once and cached) and fall back to a documented
default. What each family stores and computes:

* trees / forests / boosting: ``nodes = estimators × min(2^(depth+1) − 1, 2·rows/leaf)``
  values of about five numbers each; predicting walks ``depth`` nodes per tree
* linear models, naive Bayes, LDA: ``features × classes`` weights
* kNN, LOF, SVMs: the training rows (or about a third of them as support vectors)
* sklearn MLPs: their weight matrices
* Bayesian networks: every CPD table, ``card × ∏ parent cards``; HMMs: transitions and
  emissions
* PyMC: the posterior draws (``chains × draws × parameter values``) plus the pointwise
  log-likelihood PSIS-LOO keeps (``chains × draws × rows``)
* exact GPs: the ``n × n`` kernel matrix; sparse GPs: ``m × m`` and ``n × m``
* state-space models: a handful of parameters and the ``state × state`` covariance

Values are 8-byte floats. Latency uses the device's FLOPs at a low efficiency
(``BRANCHY``) for tree walks and the usual efficiency for dense algebra. Measured costs
replace these after a trial trains (``costs.json``, see ``core.automl``).
"""
from __future__ import annotations

import math

from ai_made_easy.core.budget import COMPUTE_EFFICIENCY, Estimate, LayerCost

PREFIXES = ("ml.", "pgm.", "ppl.", "gp.", "ssm.")
DEFAULT_ROWS, DEFAULT_FEATURES, DEFAULT_CLASSES, DEFAULT_GROUPS = 10_000, 32, 2, 10
BRANCHY = 0.002          # share of peak FLOPs a tree walk reaches (branches, cache misses)
VALUES_PER_TREE_NODE = 5
TREES = {"ml.decision_tree_classifier": 1, "ml.decision_tree_regressor": 1}
FORESTS = ("ml.random_forest_classifier", "ml.random_forest_regressor",
           "ml.extra_trees_classifier", "ml.extra_trees_regressor",
           "ml.gradient_boosting_classifier", "ml.gradient_boosting_regressor",
           "ml.adaboost_classifier", "ml.adaboost_regressor", "ml.bagging_classifier",
           "ml.xgb_classifier", "ml.xgb_regressor", "ml.isolation_forest")


def applies(graph) -> bool:  # noqa: ANN001
    return any(n.type_id.startswith(PREFIXES) for n in graph.nodes.values())


# ------------------------------------------------------------------- data size

def data_shape(graph) -> tuple[int, int, int, bool]:  # noqa: ANN001
    """(rows, features, classes, known): from the dataset block, else defaults."""
    from ai_made_easy.core.training import data_catalog as dcat

    node = next((n for n in graph.nodes.values() if n.type_id.startswith("data.")), None)
    if node is None:
        return DEFAULT_ROWS, DEFAULT_FEATURES, DEFAULT_CLASSES, False
    p = node.resolved_params()
    if node.type_id == "data.synthetic":
        return (int(p.get("n_samples") or DEFAULT_ROWS), int(p.get("n_features") or 1),
                int(p.get("n_classes") or 2), True)
    if node.type_id == "data.sklearn" and p.get("dataset") in dcat.SKLEARN:
        _loader, features, classes = dcat.SKLEARN[p["dataset"]]
        rows = {"iris": 150, "wine": 178, "breast_cancer": 569, "digits": 1797,
                "diabetes": 442, "california_housing": 20640}.get(p["dataset"], DEFAULT_ROWS)
        return rows, features, max(classes, 1), True
    if node.type_id == "data.synthetic_table":
        return int(p.get("n_samples") or DEFAULT_ROWS), 6, 2, True
    try:
        from ai_made_easy.core.data.profile import profile_dataset

        prof = profile_dataset(node.type_id, dict(p))
        if prof.ok and prof.rows:
            features = max(len(prof.columns) - 1, 1) if prof.columns else DEFAULT_FEATURES
            return prof.rows, features, max(len(prof.classes), 1) or DEFAULT_CLASSES, True
    except Exception:  # noqa: BLE001, S110 — an unreadable file falls back to defaults
        pass
    rows = int(p.get("n_samples") or DEFAULT_ROWS)
    return rows, int(p.get("n_features") or DEFAULT_FEATURES), DEFAULT_CLASSES, False


# ------------------------------------------------------------------- per family

def _depth(p: dict, rows: int, leaf: int = 1) -> int:
    depth = int(p.get("max_depth") or p.get("depth") or 0)
    natural = max(1, math.ceil(math.log2(max(rows / max(leaf, 1), 2))))
    return min(depth, natural) if depth > 0 else natural


def _tree_nodes(depth: int, rows: int, leaf: int) -> int:
    return int(min(2 ** (depth + 1) - 1, 2 * rows / max(leaf, 1)))


def classic_costs(type_id: str, p: dict, rows: int, features: int,
                  classes: int) -> tuple[int, int, float, int]:
    """(stored values, predict FLOPs per row, efficiency, training workspace values)."""
    outputs = classes if classes > 2 else 1
    data = rows * features
    leaf = int(p.get("min_samples_leaf") or 1)
    if type_id in TREES or type_id in FORESTS:
        trees = int(p.get("n_estimators") or 100) if type_id in FORESTS else 1
        if type_id == "ml.isolation_forest":
            depth, nodes = 8, 255           # 256-row subsamples
        else:
            depth = _depth(p, rows, leaf)
            nodes = _tree_nodes(depth, rows, leaf)
        per_node = VALUES_PER_TREE_NODE + (outputs if "classifier" in type_id else 1)
        return trees * nodes * per_node, 2 * trees * depth, BRANCHY, data
    if type_id.startswith(("ml.hist_gradient_boosting", "ml.lgbm")):
        trees = int(p.get("max_iter") or p.get("n_estimators") or 100) * outputs
        leaves = int(p.get("max_leaf_nodes") or p.get("num_leaves") or 31)
        depth = max(1, math.ceil(math.log2(max(leaves, 2))))
        return trees * (2 * leaves - 1) * VALUES_PER_TREE_NODE, 2 * trees * depth, BRANCHY, \
            data
    if type_id.startswith("ml.catboost"):   # oblivious trees: one split per level
        trees, depth = int(p.get("iterations") or 1000), int(p.get("depth") or 6)
        return trees * (2 ** depth + 2 * depth), 2 * trees * depth, BRANCHY, data
    if type_id.startswith(("ml.knn", "ml.local_outlier")):
        return data, 3 * data, COMPUTE_EFFICIENCY, data
    if type_id in ("ml.svc", "ml.svr", "ml.one_class_svm"):
        support = max(1, rows // 3)
        return support * (features + 1), 4 * support * features, COMPUTE_EFFICIENCY, \
            data + min(rows, 20_000) ** 2 // 4
    if type_id.startswith("ml.mlp"):
        sizes = [int(s) for s in str(p.get("hidden_layer_sizes") or "100").replace(
            "(", "").replace(")", "").split(",") if s.strip()]
        widths = [features, *sizes, outputs]
        weights = sum(a * b + b for a, b in zip(widths, widths[1:], strict=False))
        return weights, 2 * weights, COMPUTE_EFFICIENCY, data + 3 * weights
    if type_id in ("ml.kmeans", "ml.minibatch_kmeans", "ml.birch", "ml.agglomerative",
                   "ml.spectral"):
        k = int(p.get("n_clusters") or 8)
        return k * features, 3 * k * features, COMPUTE_EFFICIENCY, data
    if type_id == "ml.gaussian_mixture":
        k = int(p.get("n_components") or 1)
        cov = features * features if p.get("covariance_type", "full") == "full" else features
        return k * (features + cov + 1), 2 * k * cov, COMPUTE_EFFICIENCY, data
    if type_id in ("ml.qda",):
        return classes * features * (features + 1), 2 * classes * features ** 2, \
            COMPUTE_EFFICIENCY, data
    if type_id in ("ml.pca", "ml.truncated_svd", "ml.kernel_pca", "ml.fast_ica"):
        comps = int(p.get("n_components") or min(features, 2))
        return features * comps, 2 * features * comps, COMPUTE_EFFICIENCY, data
    if type_id in ("ml.dbscan", "ml.hdbscan"):
        return data, 3 * data, COMPUTE_EFFICIENCY, data
    # linear models, naive Bayes, LDA, transformers of columns
    return (features + 1) * outputs * (2 if "nb" in type_id else 1), \
        2 * features * outputs, COMPUTE_EFFICIENCY, data


def _states(node) -> int:  # noqa: ANN001
    raw = node.resolved_params().get("states") or ""
    return max(len([s for s in str(raw).split(",") if s.strip()]), 2)


def pgm_costs(graph, rows: int, features: int) -> tuple[int, int, int]:  # noqa: ANN001
    """(stored values, inference FLOPs per query, workspace values)."""
    variables = {n.instance_id: n for n in graph.nodes.values()
                 if n.type_id in ("pgm.variable", "pgm.gaussian")}
    hmm = next((n for n in graph.nodes.values() if n.type_id == "pgm.hmm"), None)
    if hmm is not None:
        p = hmm.resolved_params()
        k, mix = int(p.get("n_states") or 3), int(p.get("mixtures") or 1)
        dim = max(len(str(p.get("observations") or "x").split(",")), 1)
        cov = dim * dim if p.get("covariance") == "full" else dim
        values = k * k + k + k * mix * (dim + cov + 1)
        return values, 2 * k * k + k * mix * cov, rows * k * 3
    if not variables:
        return 0, 0, 0
    cards = {nid: (1 if n.type_id == "pgm.gaussian" else _states(n))
             for nid, n in variables.items()}
    indegree = 0
    for n in graph.nodes.values():
        if n.type_id == "pgm.structure_learning":
            indegree = int(n.resolved_params().get("max_indegree") or 0) or 3
    values = largest = 0
    for nid in variables:
        parents = [e.source_id for e in graph.edges if e.target_id == nid
                   and e.source_id in variables]
        table = cards[nid] * math.prod(cards[p] for p in parents)
        if indegree and len(parents) < indegree:   # learned structure: room for parents
            table *= max(cards.values()) ** (indegree - len(parents))
        values += table
        largest = max(largest, table)
    return values, 4 * values, largest * 2 + rows * len(variables)


def ppl_costs(graph, rows: int) -> tuple[int, int, int]:  # noqa: ANN001
    """(posterior values kept, predict FLOPs per row, workspace values)."""
    sampler = next((n.resolved_params() for n in graph.nodes.values()
                    if n.type_id == "ppl.sampler"), {})
    method = sampler.get("method", "nuts")
    chains = 1 if method in ("map", "advi") else int(sampler.get("chains") or 4)
    draws = 1 if method == "map" else int(sampler.get("draws") or 1000)
    size = 0
    for n in graph.nodes.values():
        if not n.type_id.startswith("ppl.") or n.type_id in ("ppl.sampler",
                                                              "ppl.deterministic"):
            continue
        p = n.resolved_params()
        if p.get("observed"):
            continue
        width = int(p.get("size") or 1)
        size += width * (DEFAULT_GROUPS if p.get("group") else 1)
    posterior = chains * draws * max(size, 1)
    return posterior, 2 * min(chains * draws, 400) * max(size, 1), chains * draws * rows


def gp_costs(graph, rows: int, features: int) -> tuple[int, int, int]:  # noqa: ANN001
    """(stored values, predict FLOPs per point, training workspace values)."""
    model = next((n.resolved_params() for n in graph.nodes.values()
                  if n.type_id == "gp.model"), {})
    kernels = sum(1 for n in graph.nodes.values() if n.type_id.startswith("gp.")
                  and n.type_id not in ("gp.model", "gp.sum", "gp.product"))
    if "svgp" in str(model.get("kind", "")) or "sparse" in str(model.get("kind", "")):
        m = int(model.get("inducing_points") or 64)
        return m * (features + m + 1), 2 * m * (features + m) * max(kernels, 1), \
            m * m + rows * m
    n = rows
    return n * (features + 1), 2 * n * features * max(kernels, 1), n * n


def ssm_costs(graph, rows: int) -> tuple[int, int, int]:  # noqa: ANN001
    state = 1
    for n in graph.nodes.values():
        p = n.resolved_params()
        if n.type_id == "ssm.seasonal":
            state += int(p.get("period") or 12) - 1
        elif n.type_id == "ssm.trend":
            state += 2
        elif n.type_id == "ssm.autoregressive":
            state += int(p.get("order") or 1)
        elif n.type_id == "ssm.arima":
            state += max(int(p.get("p") or 0), int(p.get("q") or 0) + 1) + \
                int(p.get("season") or 0) * (int(p.get("seasonal_p") or 0)
                                            + int(p.get("seasonal_q") or 0))
    params = 3 * state
    return params + state * state, 2 * state * state, rows * state * state


# ------------------------------------------------------------------- estimate

def estimate(graph) -> Estimate:  # noqa: ANN001
    """An :class:`Estimate` for a structured design: ``params`` are stored values (8
    bytes each), ``flops`` the predict cost of one row, ``work_values`` training memory."""
    rows, features, classes, known = data_shape(graph)
    est = Estimate(value_bytes=8, optimizer_states=0, family="structured",
                   assumed_data="" if known else
                   f"assumed {rows:,} rows × {features} features (no dataset size)")
    efficiency = COMPUTE_EFFICIENCY
    layers: list[tuple] = []
    for node in graph.nodes.values():
        if node.type_id.startswith("ml."):
            values, flops, eff, work = classic_costs(node.type_id, node.resolved_params(),
                                                     rows, features, classes)
            layers.append((node, values, flops, work))
            efficiency = min(efficiency, eff)
    prefixes = {n.type_id.split(".")[0] for n in graph.nodes.values()}
    family_cost = None
    if "pgm" in prefixes:
        family_cost = pgm_costs(graph, rows, features)
    elif "ppl" in prefixes:
        family_cost = ppl_costs(graph, rows)
    elif "gp" in prefixes:
        family_cost = gp_costs(graph, rows, features)
    elif "ssm" in prefixes:
        family_cost = ssm_costs(graph, rows)
    if family_cost is not None:
        anchor = next(n for n in graph.nodes.values()
                      if n.type_id.split(".")[0] in ("pgm", "ppl", "gp", "ssm"))
        layers.append((anchor, *family_cost))
    for node, values, flops, work in layers:
        defn = node.definition()
        est.layers.append(LayerCost(node.instance_id, defn.display_name, node.type_id, [],
                                    int(values), int(flops), 0, False, 1))
        est.params += int(values)
        est.flops += int(flops)
        est.work_values += int(work)
    est.efficiency = efficiency
    return est
