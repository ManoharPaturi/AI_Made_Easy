"""Design rules for recommenders ("set <param> to <value>" and "Set the Input shape to '…'"
are Quick Fixes)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule
from ai_made_easy.core.recsys.blocks import MODELS
from ai_made_easy.core.recsys.tasks import dataset_of, recsys_task


@lru_cache(maxsize=8)
def _csv_counts(path: str, mtime: float, user: str, item: str, rating: str) -> dict:
    import pandas as pd

    frame = pd.read_csv(path)
    out = {"columns": list(frame.columns)}
    if user in frame.columns and item in frame.columns:
        out.update(users=int(frame[user].nunique()), items=int(frame[item].nunique()),
                   features=0, ratings=bool(rating))
    return out


def data_facts(node) -> dict:  # noqa: ANN001
    """Users, items, dense features per side and whether there are ratings."""
    if node is None:
        return {}
    p = node.resolved_params()
    if node.type_id == "data.synthetic_interactions":
        return {"users": int(p["n_users"]), "items": int(p["n_items"]),
                "features": int(p["features"]), "ratings": p["feedback"] == "ratings"}
    path = Path(str(p["path"])).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return {}
    try:
        return _csv_counts(str(path), path.stat().st_mtime, str(p["user_column"]),
                           str(p["item_column"]), str(p["rating_column"] or ""))
    except Exception:  # noqa: BLE001 — unreadable files are reported when training
        return {}


def recsys_rules(ctx: LintContext) -> list:
    if recsys_task(ctx.graph) is None:
        return []
    out = []
    models = [n for n in ctx.chain if n.type_id in MODELS]
    data = dataset_of(ctx.graph)
    if not models:
        anchor = data or (ctx.chain[0] if ctx.chain else None)
        return [_issue("error", "add a recommender model (Matrix Factorization, NCF, Two-Tower "
                                "or DLRM) between the Input and the Output",
                       anchor.instance_id if anchor else None)]
    if len(models) > 1:
        out.append(_issue("error", "one recommender model per design", models[1].instance_id))
    model = models[0]
    facts = data_facts(data)
    if data is not None and data.type_id == "data.interactions_csv" and "columns" in facts \
            and "users" not in facts:
        p = data.resolved_params()
        out.append(_issue("error", f"columns {p['user_column']!r} / {p['item_column']!r} are not "
                                   f"in the file: {facts['columns'][:8]}", data.instance_id))
    p = model.resolved_params()
    for key, label in (("users", "n_users"), ("items", "n_items")):
        if key in facts and int(p[label]) < facts[key]:
            out.append(_issue("error", f"the data has {facts[key]:,} {key}: set {label} to "
                                       f"{facts[key]}", model.instance_id))
        elif key in facts and int(p[label]) > 4 * facts[key] + 100:
            out.append(_issue("info", f"{label} = {int(p[label]):,} for {facts[key]:,} {key}: "
                                      f"set {label} to {facts[key]} to save memory",
                              model.instance_id))
    head = ctx.chain[0] if ctx.chain else None
    width = ctx.shapes.get(head.instance_id, [0])[0] if head is not None else 0
    if "features" in facts:
        full = 2 + 2 * facts["features"]
        if width not in (2, full):
            out.append(_issue("error", f"rows are [user, item] or [user, item, features] "
                                       f"({full} values): Set the Input shape to '2'",
                              head.instance_id))
        elif model.type_id == "rec.dlrm" and width == 2 and facts["features"]:
            out.append(_issue("info", f"DLRM can read the {2 * facts['features']} user / item "
                                      f"features: Set the Input shape to '{full}'",
                              head.instance_id))
        elif model.type_id != "rec.dlrm" and width > 2:
            out.append(_issue("info", f"{_name(model)} uses only the ids: the features are "
                                      "ignored (DLRM reads them)", model.instance_id))
    last = ctx.last_compute()
    if last is not None and last.instance_id in ctx.shapes and ctx.shapes[last.instance_id] != [1]:
        out.append(_issue("error", "a recommender outputs one score per (user, item) row",
                          last.instance_id))
    for node in ctx.nodes_of("rec.objective"):
        objective = node.resolved_params()["objective"]
        if objective == "mse" and facts and not facts.get("ratings"):
            out.append(_issue("error", "MSE regresses ratings but the data is implicit "
                                       "feedback: set objective to bpr", node.instance_id))
        if objective in ("bpr", "bce") and facts.get("ratings"):
            out.append(_issue("warning", "ratings are treated as plain interactions (every "
                                         "rating counts as positive): set objective to mse to "
                                         "predict them", node.instance_id))
    return out


register_rule(recsys_rules)
