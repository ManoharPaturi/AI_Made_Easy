"""Runtime code shared by generated recommender scripts and the app (numpy / pandas).

Interactions are ``(users, items, ratings)`` arrays of indices 0..n-1 (ratings None for
implicit feedback) plus optional dense user / item feature tables.
"""
from __future__ import annotations

DATA_CODE = r'''
# ======================================================================== interactions
def synthetic_interactions(d: dict) -> dict:
    """Latent tastes + popularity: implicit interactions or 1-5 star ratings."""
    rng = np.random.default_rng(int(d["seed"]))
    n_u, n_i, k = int(d["n_users"]), int(d["n_items"]), 8
    pu, qi = rng.normal(0, 1, (n_u, k)), rng.normal(0, 1, (n_i, k))
    popularity = rng.normal(0, 1, n_i)
    users, items = [], []
    for u in range(n_u):
        m = max(2, int(rng.poisson(int(d["per_user"]))))
        logits = pu[u] @ qi.T / np.sqrt(k) * 2.0 + popularity
        p = np.exp(logits - logits.max())
        chosen = rng.choice(n_i, size=min(m, n_i), replace=False, p=p / p.sum())
        users += [u] * len(chosen)
        items += chosen.tolist()
    users, items = np.array(users), np.array(items)
    ratings = None
    if d["feedback"] == "ratings":
        raw = (pu[users] * qi[items]).sum(1) / np.sqrt(k) + 0.3 * popularity[items]
        ratings = np.clip(np.round(3 + 1.2 * raw + rng.normal(0, 0.5, len(raw))), 1, 5)
    f = int(d["features"])
    proj_u, proj_i = rng.normal(0, 1, (k, f)), rng.normal(0, 1, (k, f))
    return {"users": users, "items": items, "ratings": ratings, "n_users": n_u, "n_items": n_i,
            "user_features": (pu @ proj_u + rng.normal(0, 0.5, (n_u, f))).astype(np.float32),
            "item_features": (qi @ proj_i + rng.normal(0, 0.5, (n_i, f))).astype(np.float32),
            "user_ids": [str(u) for u in range(n_u)], "item_ids": [str(i) for i in range(n_i)]}


def interactions_csv(d: dict) -> dict:
    import pandas as pd

    frame = pd.read_csv(Path(d["path"]).expanduser())
    for col in (d["user_column"], d["item_column"]):
        if col not in frame.columns:
            raise SystemExit(f"column {col!r} is not in {d['path']}: {list(frame.columns)}")
    user_ids = sorted(frame[d["user_column"]].astype(str).unique())
    item_ids = sorted(frame[d["item_column"]].astype(str).unique())
    u_index = {v: i for i, v in enumerate(user_ids)}
    i_index = {v: i for i, v in enumerate(item_ids)}
    rating = str(d.get("rating_column") or "")
    return {"users": frame[d["user_column"]].astype(str).map(u_index).to_numpy(),
            "items": frame[d["item_column"]].astype(str).map(i_index).to_numpy(),
            "ratings": frame[rating].to_numpy(np.float32) if rating else None,
            "n_users": len(user_ids), "n_items": len(item_ids),
            "user_features": np.zeros((len(user_ids), 0), np.float32),
            "item_features": np.zeros((len(item_ids), 0), np.float32),
            "user_ids": user_ids, "item_ids": item_ids}


def load_interactions(d: dict) -> dict:
    if d["block"] == "data.synthetic_interactions":
        return synthetic_interactions(d)
    return interactions_csv(d)


def split_per_user(users: np.ndarray, val: float, test: float, seed: int) -> dict:
    """Hold out a fraction of every user's interactions (users keep at least one in
    training)."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    order = np.argsort(users, kind="stable")
    bounds = np.flatnonzero(np.diff(users[order])) + 1
    for group in np.split(order, bounds):
        group = rng.permutation(group)
        n = len(group)
        n_test = int(round(n * test)) if n > 2 else 0
        n_val = int(round(n * val)) if n > 3 else 0
        n_test = min(n_test, n - 1)
        n_val = min(n_val, n - 1 - n_test)
        parts["test"] += group[:n_test].tolist()
        parts["val"] += group[n_test:n_test + n_val].tolist()
        parts["train"] += group[n_test + n_val:].tolist()
    return {k: np.array(v, dtype=np.int64) for k, v in parts.items()}
'''

METRICS_CODE = r'''
# ======================================================================== ranking metrics
def ranking_metrics(scores: np.ndarray, relevant: list, k: int) -> dict:
    """scores [users, items] with seen training items set to -inf; relevant: held-out item
    sets per user. Recall@K, NDCG@K, MRR (users without held-out items are skipped)."""
    recall, ndcg, mrr = [], [], []
    top = np.argsort(-scores, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for row, rel in zip(top, relevant):
        if not rel:
            continue
        hits = np.isin(row[:k], list(rel))
        recall.append(hits.sum() / min(len(rel), k))
        ideal = discounts[:min(len(rel), k)].sum()
        ndcg.append((hits * discounts).sum() / ideal)
        first = np.flatnonzero(np.isin(row, list(rel)))
        mrr.append(1.0 / (first[0] + 1) if len(first) else 0.0)
    if not recall:
        return {f"recall_at_{k}": float("nan"), f"ndcg_at_{k}": float("nan"), "mrr": float("nan")}
    return {f"recall_at_{k}": float(np.mean(recall)), f"ndcg_at_{k}": float(np.mean(ndcg)),
            "mrr": float(np.mean(mrr))}
'''


def namespace() -> dict:
    import json
    from pathlib import Path

    import numpy as np

    ns: dict = {"np": np, "json": json, "Path": Path}
    for code in (DATA_CODE, METRICS_CODE):
        exec(compile(code, "<recsys-runtime>", "exec"), ns)  # noqa: S102 — our own source
    return ns
