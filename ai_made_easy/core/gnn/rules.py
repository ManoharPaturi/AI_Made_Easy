"""Design rules for graph neural networks ("set <param> to <value>" and "Set the Input shape
to '…'" phrasing are one-click Quick Fixes)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ai_made_easy.core.gnn.blocks import (
    CONV_LAYERS,
    GLOBAL_POOLS,
    GRAPH_BLOCKS,
    GRAPH_DATA,
    NODE_POOLS,
    PLANETOID,
    TU_DATASETS,
)
from ai_made_easy.core.gnn.tasks import dataset_of, graph_kind
from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule

DEEP = 6


@lru_cache(maxsize=16)
def _csv_graph(nodes: str, edges: str, mtime: float, id_col: str, label: str, src: str,
               dst: str, features: str) -> dict | None:
    import pandas as pd

    n = pd.read_csv(nodes)
    e = pd.read_csv(edges, usecols=[src, dst])
    wanted = [c.strip() for c in features.split(",") if c.strip()]
    cols = wanted or [c for c in n.select_dtypes("number").columns if c not in (id_col, label)]
    linked = set(e[src].tolist()) | set(e[dst].tolist())
    return {"features": len(cols), "classes": int(n[label].nunique()) if label in n else None,
            "isolated": int((~n[id_col].isin(linked)).sum()), "nodes": len(n),
            "has_label": label in n.columns}


def _resolve(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def data_facts(node) -> dict:  # noqa: ANN001
    """Feature width, class count and isolated nodes a graph dataset will produce (keys
    missing when unknown)."""
    if node is None:
        return {}
    p = node.resolved_params()
    t = node.type_id
    if t == "data.synthetic_graph":
        return {"features": int(p["feature_dim"]), "classes": int(p["communities"])}
    if t == "data.synthetic_graphs":
        return {"features": int(p["feature_dim"]), "classes": 3}
    if t == "data.planetoid":
        f, c = PLANETOID[p["name"]]
        return {"features": f, "classes": c}
    if t == "data.tudataset":
        f, c = TU_DATASETS[p["name"]]
        return {"features": f, "classes": c}
    nodes, edges = _resolve(str(p["nodes_path"])), _resolve(str(p["edges_path"]))
    if not nodes.is_file() or not edges.is_file():
        return {}
    try:
        facts = _csv_graph(str(nodes), str(edges), nodes.stat().st_mtime + edges.stat().st_mtime,
                           str(p["id_column"]), str(p["label_column"] or ""),
                           str(p["source_column"]), str(p["target_column"]),
                           str(p["feature_columns"] or ""))
    except Exception:  # noqa: BLE001 — unreadable files are reported when training
        return {}
    return {k: v for k, v in (facts or {}).items() if v is not None}


def gnn_rules(ctx: LintContext) -> list:
    kind = graph_kind(ctx.graph)
    if kind is None:
        return []
    out = []
    data = dataset_of(ctx.graph)
    layers = [n for n in ctx.chain if n.type_id not in ("core.input", "core.output")]
    convs = [n for n in layers if n.type_id in CONV_LAYERS]
    other_data = next((n for n in ctx.graph.nodes.values()
                       if n.type_id.startswith("data.") and n.type_id not in GRAPH_DATA), None)
    if other_data is not None:
        out.append(_issue("error", f"{_name(other_data)} is not a graph: graph layers need a "
                                   "graph dataset (Synthetic Graph, Graph CSV, Planetoid, TU "
                                   "benchmarks)", other_data.instance_id))
    if not convs:
        anchor = data or (layers[0] if layers else None)
        out.append(_issue("warning", "no message-passing layer: add a GCN, GAT or GraphSAGE "
                                     "layer, otherwise the edges are ignored",
                          anchor.instance_id if anchor else None))
    facts = data_facts(data)
    head = ctx.chain[0] if ctx.chain else None
    if head is not None and "features" in facts and head.instance_id in ctx.shapes:
        if list(ctx.shapes[head.instance_id]) != [facts["features"]]:
            out.append(_issue("error", f"nodes have {facts['features']} features: Set the "
                                       f"Input shape to '{facts['features']}'",
                              head.instance_id))
    data_level = GRAPH_DATA[data.type_id] if data is not None else None
    pools = [n for n in layers if n.type_id in GLOBAL_POOLS]
    if kind == "graph_classification" and data_level == "nodes":
        for node in pools:
            out.append(_issue("error", f"{_name(data) if data else 'The data'} is one graph: "
                                       f"remove {_name(node)} (node classification predicts "
                                       "every node)", node.instance_id))
    if data_level == "graphs":
        if not pools:
            out.append(_issue("error", "graph classification needs one row per graph: add "
                                       "Global Pooling after the graph layers",
                              data.instance_id))
        decoder = ctx.nodes_of("graph.link_decoder")
        for node in decoder:
            out.append(_issue("error", "link prediction works on one graph: use a node-level "
                                       "dataset or remove the Link Predictor",
                              node.instance_id))
    if kind == "link_prediction":
        for node in pools + [n for n in layers if n.type_id in NODE_POOLS]:
            out.append(_issue("error", f"{_name(node)} removes nodes: link prediction needs an "
                                       "embedding for every node", node.instance_id))
        last = ctx.last_compute()
        if last is not None and last.type_id in ("core.softmax", "core.log_softmax"):
            out.append(_issue("warning", "node embeddings for link prediction should not end "
                                         "with a softmax", last.instance_id))
    if pools:
        after = False
        for node in layers:
            if node.type_id in GLOBAL_POOLS:
                after = True
            elif after and node.type_id in GRAPH_BLOCKS:
                out.append(_issue("error", f"{_name(node)} comes after Global Pooling: there "
                                           "are no edges between graphs", node.instance_id))
    if kind != "link_prediction" and "classes" in facts:
        last = ctx.last_compute()
        if last is not None and last.type_id == "core.dense" and \
                int(last.resolved_params()["units"]) != facts["classes"]:
            out.append(_issue("error", f"the data has {facts['classes']} classes: set units to "
                                       f"{facts['classes']}", last.instance_id))
    if kind == "node_classification" and data is not None and \
            data.type_id == "data.graph_csv" and facts.get("has_label") is False:
        out.append(_issue("error", "the node table has no label column: set label_column, or "
                                   "add a Link Predictor for link prediction",
                          data.instance_id))
    if facts.get("isolated"):
        out.append(_issue("warning", f"{facts['isolated']} of {facts['nodes']} nodes have no "
                                     "edges: they are predicted from their own features only",
                          data.instance_id))
    if len(convs) > DEEP:
        out.append(_issue("info", f"{len(convs)} message-passing layers: deep GNNs over-smooth "
                                  "(node features become alike); 2-4 layers or skip "
                                  "connections usually work better", convs[DEEP].instance_id))
    return out


register_rule(gnn_rules)
