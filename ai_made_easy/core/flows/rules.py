"""Design rules for normalizing flows ("set <param> to <value>" phrasing is a Quick Fix)."""
from __future__ import annotations

from ai_made_easy.core.flows.blocks import COUPLINGS, FLOW_LAYERS
from ai_made_easy.core.flows.tasks import dataset_of, is_flow
from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule

TRANSFORMS = ("flow.affine_coupling", "flow.spline_coupling", "flow.maf")


def _data_dims(node) -> int | None:  # noqa: ANN001
    if node is None:
        return None
    if node.type_id == "data.density_2d":
        return 2
    cols = [c for c in str(node.resolved_params().get("feature_columns") or "").split(",")
            if c.strip()]
    return len(cols) or None


def flow_rules(ctx: LintContext) -> list:
    if not is_flow(ctx.graph):
        return []
    out = []
    try:
        chain = ctx.graph.model_nodes()
    except Exception:  # noqa: BLE001 — incomplete designs are reported elsewhere
        return []
    layers = [n for n in chain if n.type_id not in ("core.input", "core.output")]
    for node in layers:
        if node.type_id not in FLOW_LAYERS:
            out.append(_issue("error", f"{_name(node)} is not invertible: a flow may only "
                                       "contain flow layers (couplings, MAF, ActNorm, "
                                       "permutations)", node.instance_id))
    if not any(n.type_id in TRANSFORMS for n in layers):
        out.append(_issue("error", "add a coupling, spline or MAF layer: permutations and "
                                   "ActNorm alone cannot learn a density",
                          layers[0].instance_id if layers else None))
    head = chain[0] if chain else None
    data = dataset_of(ctx.graph)
    dims = _data_dims(data)
    if head is not None and dims is not None:
        try:
            shape = ctx.graph.infer_shapes()[head.instance_id]
        except Exception:  # noqa: BLE001
            shape = None
        if shape is not None and list(shape) != [dims]:
            out.append(_issue("error", f"the data has {dims} columns: Set the Input shape to "
                                       f"'{dims}'", head.instance_id))
    last: dict[str, object] = {}
    for node in layers:
        if node.type_id == "flow.permute":
            last.clear()
            continue
        p = node.resolved_params()
        if node.type_id in COUPLINGS:
            prev = last.get("coupling")
            if prev is not None and prev == p["parity"]:
                other = "odd" if p["parity"] == "even" else "even"
                out.append(_issue("warning", f"{_name(node)} transforms the same half as the "
                                             f"coupling before it: set parity to {other}",
                                  node.instance_id))
            last["coupling"] = p["parity"]
        elif node.type_id == "flow.maf":
            prev = last.get("maf")
            if prev is not None and bool(prev) == bool(p["reverse"]):
                out.append(_issue("warning", f"{_name(node)} uses the same order as the MAF "
                                             f"layer before it: set reverse to "
                                             f"{str(not bool(p['reverse'])).lower()}",
                                  node.instance_id))
            last["maf"] = p["reverse"]
    if sum(n.type_id in TRANSFORMS for n in layers) == 1 and dims and dims > 1:
        node = next(n for n in layers if n.type_id in TRANSFORMS)
        if node.type_id in COUPLINGS:
            out.append(_issue("info", "one coupling layer leaves half of the dimensions "
                                      "unchanged: stack at least two with alternating parity",
                              node.instance_id))
    for node in ctx.nodes_of("flow.spline_coupling"):
        if float(node.resolved_params()["bound"]) < 3:
            out.append(_issue("info", f"{_name(node)} only bends [-bound, bound] of the "
                                      "standardised data: values beyond it pass through "
                                      "linearly", node.instance_id))
    for node in ctx.graph.nodes.values():
        if node.type_id.startswith("train.loss"):
            out.append(_issue("info", "flows train by maximum likelihood: the loss block is "
                                      "ignored", node.instance_id))
    return out


register_rule(flow_rules)
