"""Design rules for Gaussian processes (the ``gp`` family's validator)."""
from __future__ import annotations

from ai_made_easy.core.gp.blocks import COMBINERS, KERNELS
from ai_made_easy.core.gp.kernels import KernelError, _names, kernel_root, leaves, model_node
from ai_made_easy.core.graph import Graph, ValidationIssue
from ai_made_easy.core.ppl.rules import _column_kind, data_frame

EXACT_LIMIT = 5000


def _issue(severity: str, message: str, node_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity, message, node_id)


def gp_issues(graph: Graph) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    models = [n for n in graph.nodes.values() if n.type_id == "gp.model"]
    if not models:
        anchor = next((n for n in graph.nodes.values() if n.type_id.startswith("gp.")), None)
        return [_issue("error", "add a Gaussian Process block and wire a kernel into it",
                       anchor.instance_id if anchor else None)]
    if len(models) > 1:
        out.append(_issue("error", "one Gaussian Process per design", models[1].instance_id))
    model = model_node(graph)
    p = model.resolved_params()
    root = kernel_root(graph)
    if root is None:
        return out + [_issue("error", "wire a kernel (e.g. RBF) into the Gaussian Process",
                             model.instance_id)]
    try:
        used = leaves(graph, root)
    except KernelError as exc:
        return out + [_issue("error", str(exc), root.instance_id)]
    used_ids = {n.instance_id for n in used}
    for node in graph.nodes.values():
        if node.type_id in KERNELS and node.instance_id not in used_ids:
            out.append(_issue("warning", f"{node.definition().display_name} is not wired into "
                                         "the model's kernel", node.instance_id))
        if node.type_id in COMBINERS:
            count = len(graph.incoming(node.instance_id))
            if count < 2:
                out.append(_issue("warning" if count else "error",
                                  f"{node.definition().display_name} combines "
                                  f"{count} kernel(s): wire at least two", node.instance_id))
    library = p["library"]
    for leaf in used:
        libs = KERNELS[leaf.type_id][3]
        if library not in libs:
            if leaf.type_id == "gp.white":
                out.append(_issue("info", "GPyTorch learns the noise in its Gaussian "
                                          "likelihood: the White Noise kernel is ignored",
                                  leaf.instance_id))
            else:
                out.append(_issue("error", f"{leaf.definition().display_name} needs the "
                                           f"{' or '.join(libs)} library: set library to "
                                           f"{libs[0]}", leaf.instance_id))
        if library == "sklearn" and _names(leaf.resolved_params().get("columns")):
            out.append(_issue("error", "kernels restricted to some columns need the gpytorch "
                                       "library", leaf.instance_id))
    data = next((n for n in graph.nodes.values()
                 if n.type_id in ("data.csv", "data.synthetic_function")), None)
    if data is None:
        out.append(_issue("error", "a Gaussian process learns from data: add a table or "
                                   "Synthetic Function", model.instance_id))
        return out
    frame = data_frame(data)
    columns = set(frame.columns) if frame is not None else (
        {"x1", "x2", "y"} if data.type_id == "data.synthetic_function"
        and data.resolved_params()["kind"] == "classes_2d" else
        {"x", "y"} if data.type_id == "data.synthetic_function" else None)
    target = str(p["target_column"])
    features = _names(p["feature_columns"])
    if columns is not None:
        missing = [c for c in [target, *features] if c not in columns]
        if missing:
            out.append(_issue("error", f"column(s) {missing} are not in the data (columns: "
                                       f"{', '.join(sorted(columns)[:8])})", model.instance_id))
        for leaf in used:
            bad = [c for c in _names(leaf.resolved_params().get("columns"))
                   if c not in columns or c == target]
            if bad:
                out.append(_issue("error", f"kernel column(s) {bad} are not input columns",
                                  leaf.instance_id))
    if frame is not None and target in frame.columns:
        kind = _column_kind(frame[target])
        if p["likelihood"] == "bernoulli" and kind != "binary":
            out.append(_issue("error", f"the bernoulli likelihood needs 0 / 1 targets but "
                                       f"{target!r} has {kind} values", model.instance_id))
        if p["likelihood"] == "poisson" and kind not in ("count", "binary"):
            out.append(_issue("error", f"the poisson likelihood needs counts but {target!r} has "
                                       f"{kind} values", model.instance_id))
    rows = len(frame) if frame is not None else (
        int(data.resolved_params()["n_points"]) if data.type_id == "data.synthetic_function"
        else 0)
    if p["kind"] == "exact" and p["library"] == "gpytorch" and rows > EXACT_LIMIT:
        out.append(_issue("warning", f"an exact GP costs O(n³) in the {rows:,} rows: set kind "
                                     "to svgp", model.instance_id))
    synthetic_classes = data.type_id == "data.synthetic_function" and \
        data.resolved_params()["kind"] == "classes_2d"
    if synthetic_classes and p["likelihood"] != "bernoulli":
        out.append(_issue("warning", "classes_2d has 0 / 1 targets: use the bernoulli "
                                     "likelihood", model.instance_id))
    return out
