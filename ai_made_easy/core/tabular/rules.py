"""Design rules for tabular deep learning ("Use the data's categories" and "Set the Input
shape to '…'" are one-click Quick Fixes)."""
from __future__ import annotations

from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule
from ai_made_easy.core.tabular.blocks import CATEGORICAL_BLOCKS, TAB_BLOCKS, ints
from ai_made_easy.core.tabular.layout import layout_of

FIX = "Use the data's categories"
HUGE_CARDINALITY = 10_000


def tabular_rules(ctx: LintContext) -> list:
    blocks = ctx.nodes_of(*TAB_BLOCKS)
    if not blocks:
        return []
    out = []
    cat_blocks = [n for n in blocks if n.type_id in CATEGORICAL_BLOCKS]
    if cat_blocks and ctx.nodes_of("prep.variance_filter"):
        node = ctx.nodes_of("prep.variance_filter")[0]
        out.append(_issue("error", "Variance Filter removes columns and shifts the categorical "
                                   "positions the tabular model embeds: remove it",
                          node.instance_id))
    layout = layout_of(ctx.graph)
    if layout is not None:
        head = ctx.chain[0] if ctx.chain else None
        if head is not None and head.instance_id in ctx.shapes and \
                list(ctx.shapes[head.instance_id]) != [layout.width]:
            out.append(_issue("error", f"the preprocessed table has {layout.width} features: "
                                       f"Set the Input shape to '{layout.width}'",
                              head.instance_id))
        for node in cat_blocks:
            p = node.resolved_params()
            have = (tuple(ints(p["categorical"])), tuple(ints(p["cardinalities"])))
            if layout.unencoded and not layout.categorical:
                out.append(_issue("warning", f"text column(s) {list(layout.unencoded)[:5]} are "
                                             f"one-hot encoded: add Ordinal Encode so "
                                             f"{_name(node)} embeds them", node.instance_id))
            elif have != (layout.categorical, layout.cardinalities):
                cols = ", ".join(f"{c} ({k})" for c, k in
                                 zip(layout.names, layout.cardinalities, strict=True)) \
                    or "none"
                out.append(_issue("warning", f"{_name(node)} does not match the data's "
                                             f"categorical columns ({cols}): {FIX}",
                                  node.instance_id))
            if any(c > HUGE_CARDINALITY for c in layout.cardinalities):
                out.append(_issue("info", "a categorical column has more than "
                                          f"{HUGE_CARDINALITY:,} categories: its embedding "
                                          "table dominates the parameters", node.instance_id))
    for node in ctx.nodes_of("tab.tab_transformer"):
        if not ints(node.resolved_params()["categorical"]):
            out.append(_issue("warning", "TabTransformer without categorical columns is only "
                                         "an MLP: use FT-Transformer or Tabular ResNet",
                              node.instance_id))
    return out


def fix_categories(graph, node) -> bool:  # noqa: ANN001
    """Quick Fix: copy the data's categorical positions and cardinalities to the block."""
    layout = layout_of(graph)
    if layout is None:
        return False
    node.params["categorical"] = ", ".join(map(str, layout.categorical))
    node.params["cardinalities"] = ", ".join(map(str, layout.cardinalities))
    return True


register_rule(tabular_rules)
