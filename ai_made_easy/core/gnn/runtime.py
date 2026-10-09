"""Runtime code shared by generated graph scripts and the app (numpy only; PyG for the
benchmark downloads). A graph is a dict: ``x`` [N, F] float32, ``edge_index`` [2, E] int64
(both directions for undirected graphs), ``y`` (node classes [N], or one graph class)."""
from __future__ import annotations

DATA_CODE = r'''
# ======================================================================== graph data
def _undirected(edges: np.ndarray) -> np.ndarray:
    """[2, E] -> both directions, no self-loops or duplicates."""
    if edges.size == 0:
        return np.zeros((2, 0), np.int64)
    both = np.concatenate([edges, edges[::-1]], axis=1)
    both = both[:, both[0] != both[1]]
    return np.unique(both, axis=1).astype(np.int64)


def synthetic_graph(d: dict) -> dict:
    """Stochastic block model: communities are the node classes."""
    rng = np.random.default_rng(int(d["seed"]))
    n, k = int(d["n_nodes"]), int(d["communities"])
    y = rng.integers(0, k, n)
    members = [np.flatnonzero(y == c) for c in range(k)]
    src, dst = [], []
    for a in range(k):                  # sample edge counts per pair of communities: O(E)
        for b in range(a, k):
            na, nb = len(members[a]), len(members[b])
            pairs = na * (na - 1) // 2 if a == b else na * nb
            count = rng.binomial(pairs, float(d["p_in"] if a == b else d["p_out"])) \
                if pairs else 0
            if count:
                src.append(rng.choice(members[a], count))
                dst.append(rng.choice(members[b], count))
    src = np.concatenate(src) if src else np.zeros(0, np.int64)
    dst = np.concatenate(dst) if dst else np.zeros(0, np.int64)
    centres = rng.normal(0, 1, (k, int(d["feature_dim"])))
    x = float(d["feature_signal"]) * centres[y] + rng.normal(0, 1, (n, int(d["feature_dim"])))
    return {"x": x.astype(np.float32), "edge_index": _undirected(np.stack([src, dst])),
            "y": y.astype(np.int64), "classes": [f"community {i}" for i in range(k)]}


MOTIFS = ("cycle", "house", "star")


def _motif_graph(rng, n_tree: int, motif: str, feature_dim: int) -> dict:
    edges = [(i, int(rng.integers(0, i))) for i in range(1, n_tree)]   # random tree
    base = n_tree
    if motif == "cycle":
        nodes = list(range(base, base + 6))
        edges += [(nodes[i], nodes[(i + 1) % 6]) for i in range(6)]
    elif motif == "house":
        nodes = list(range(base, base + 5))
        a, b, c, e, roof = nodes
        edges += [(a, b), (b, c), (c, e), (e, a), (a, roof), (b, roof)]
    else:
        nodes = list(range(base, base + 6))
        edges += [(nodes[0], v) for v in nodes[1:]]
    edges.append((nodes[0], int(rng.integers(0, n_tree))))           # attach to the tree
    n = base + len(nodes)
    ei = _undirected(np.array(edges, dtype=np.int64).T)
    degree = np.bincount(ei[0], minlength=n)
    x = np.zeros((n, feature_dim), np.float32)
    x[np.arange(n), np.minimum(degree, feature_dim - 1)] = 1.0
    return {"x": x, "edge_index": ei}


def synthetic_graphs(d: dict) -> tuple[list, list]:
    """Random trees with a planted cycle, house or star: graph classes."""
    rng = np.random.default_rng(int(d["seed"]))
    lo, hi = int(d["min_nodes"]), max(int(d["min_nodes"]), int(d["max_nodes"]))
    graphs = []
    for i in range(int(d["n_graphs"])):
        label = i % len(MOTIFS)
        g = _motif_graph(rng, max(2, int(rng.integers(lo, hi + 1)) - 5), MOTIFS[label],
                         int(d["feature_dim"]))
        g["y"] = label
        graphs.append(g)
    order = rng.permutation(len(graphs))
    return [graphs[i] for i in order], list(MOTIFS)


def graph_csv(d: dict) -> dict:
    import pandas as pd

    nodes = pd.read_csv(Path(d["nodes_path"]).expanduser())
    edges = pd.read_csv(Path(d["edges_path"]).expanduser())
    id_col, label = d["id_column"], str(d.get("label_column") or "")
    index = {v: i for i, v in enumerate(nodes[id_col].tolist())}
    wanted = [c.strip() for c in str(d.get("feature_columns") or "").split(",") if c.strip()]
    cols = wanted or [c for c in nodes.select_dtypes("number").columns
                      if c not in (id_col, label)]
    if not cols:
        raise SystemExit("the node table has no numeric feature columns")
    src = edges[d["source_column"]].map(index)
    dst = edges[d["target_column"]].map(index)
    keep = src.notna() & dst.notna()
    if (~keep).any():
        print(f"warning: {int((~keep).sum())} edges point at unknown nodes and are skipped")
    pairs = np.stack([src[keep].astype(int).to_numpy(), dst[keep].astype(int).to_numpy()])
    edge_index = pairs.astype(np.int64) if d.get("directed") else _undirected(pairs)
    out = {"x": nodes[cols].to_numpy(np.float32), "edge_index": edge_index,
           "columns": cols, "ids": [str(v) for v in nodes[id_col].tolist()]}
    if label and label in nodes.columns:
        classes = sorted(nodes[label].astype(str).unique().tolist())
        lookup = {c: i for i, c in enumerate(classes)}
        out["y"] = nodes[label].astype(str).map(lookup).to_numpy(np.int64)
        out["classes"] = classes
    return out


def planetoid(d: dict) -> dict:
    from torch_geometric.datasets import Planetoid

    data = Planetoid(str(Path(d["root"]).expanduser()), d["name"])[0]
    return {"x": data.x.numpy(), "edge_index": data.edge_index.numpy(),
            "y": data.y.numpy(), "classes": [str(i) for i in range(int(data.y.max()) + 1)],
            "masks": {"train": data.train_mask.numpy(), "val": data.val_mask.numpy(),
                      "test": data.test_mask.numpy()}}


def tudataset(d: dict) -> tuple[list, list]:
    from torch_geometric.datasets import TUDataset

    ds = TUDataset(str(Path(d["root"]).expanduser()), d["name"])
    graphs = [{"x": g.x.numpy() if g.x is not None else np.ones((g.num_nodes, 1), np.float32),
               "edge_index": g.edge_index.numpy(), "y": int(g.y)} for g in ds]
    order = np.random.default_rng(0).permutation(len(graphs))
    return [graphs[i] for i in order], [str(i) for i in range(ds.num_classes)]


def load_graph_data(d: dict):
    """One graph (dict) for node-level tasks, or (graphs, classes) for graph-level ones."""
    block = d["block"]
    if block == "data.synthetic_graph":
        return synthetic_graph(d)
    if block == "data.synthetic_graphs":
        return synthetic_graphs(d)
    if block == "data.graph_csv":
        return graph_csv(d)
    if block == "data.planetoid":
        return planetoid(d)
    if block == "data.tudataset":
        return tudataset(d)
    raise SystemExit(f"unsupported graph dataset {block}")


def split_ids(n: int, val: float, test: float, seed: int) -> dict:
    order = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(round(n * test)), int(round(n * val))
    return {"test": order[:n_test], "val": order[n_test:n_test + n_val],
            "train": order[n_test + n_val:]}


def graph_stats(g: dict) -> dict:
    n = len(g["x"])
    degree = np.bincount(g["edge_index"][0], minlength=n) if g["edge_index"].size else \
        np.zeros(n, int)
    return {"nodes": n, "edges": int(g["edge_index"].shape[1]),
            "isolated": int((degree == 0).sum()), "mean_degree": float(degree.mean())}
'''

METRICS_CODE = r'''
# ======================================================================== metrics
def accuracy_f1(pred: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp, fn = np.sum((pred == c) & (y != c)), np.sum((pred != c) & (y == c))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"accuracy": float(np.mean(pred == y)) if len(y) else float("nan"),
            "macro_f1": float(np.mean(f1s)) if f1s else float("nan")}


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (rank formulation, ties averaged)."""
    from scipy.stats import rankdata

    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    hits = labels[order] == 1
    if not hits.any():
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())
'''


def namespace() -> dict:
    import json
    from pathlib import Path

    import numpy as np

    ns: dict = {"np": np, "json": json, "Path": Path}
    for code in (DATA_CODE, METRICS_CODE):
        exec(compile(code, "<gnn-runtime>", "exec"), ns)  # noqa: S102 — our own source
    return ns
