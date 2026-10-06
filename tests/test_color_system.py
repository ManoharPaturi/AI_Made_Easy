"""Colour-system rules: one accent colour per functional family."""
from __future__ import annotations

from ai_made_easy.core.blocks._palette import FAMILY_COLORS
from ai_made_easy.core.registry import get_registry


def _registry_blocks():
    return list(get_registry().all())


def test_categories_use_their_family_colour():
    """Each layer category is drawn in its own family accent."""
    expected = {"Linear": "model", "Convolution": "conv", "Pooling": "pool",
                "Recurrent": "recurrent", "Attention": "attention",
                "Normalization": "normalization", "Regularization": "regularization",
                "Merge": "merge", "Tensor Ops": "tensor", "Input / Output": "io"}
    for block in _registry_blocks():
        family = expected.get(block.category)
        if family:
            assert block.color == FAMILY_COLORS[family], block.type_id


def test_activations_share_one_colour():
    acts = [b for b in _registry_blocks() if b.category == "Activations"]
    assert acts, "activation blocks missing"
    assert {b.color for b in acts} == {FAMILY_COLORS["activation"]}


def test_every_block_uses_a_family_colour():
    allowed = set(FAMILY_COLORS.values())
    off = [b.type_id for b in _registry_blocks() if b.color not in allowed]
    assert not off, f"blocks off-palette: {off}"


def test_categories_are_colour_pure():
    """No category mixes colours; each category maps to exactly one family."""
    by_cat: dict[str, set[str]] = {}
    for b in _registry_blocks():
        by_cat.setdefault(b.category, set()).add(b.color)
    mixed = {cat: cols for cat, cols in by_cat.items() if len(cols) > 1}
    assert not mixed, mixed


def test_family_colours_distinct():
    vals = list(FAMILY_COLORS.values())
    assert len(vals) == len(set(vals)), "family colours must not collide"


def test_ink_reads_on_every_family_colour():
    """Flat pastel fills with dark ink text: luminance contrast check."""
    from PySide6 import QtGui

    for hexcol in FAMILY_COLORS.values():
        c = QtGui.QColor(hexcol)
        lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        assert lum > 140, f"{hexcol} too dark for ink text (lum={lum:.0f})"


def test_flat_painter_installed():
    from OdenGraphQt.qgraphics.node_base import NodeItem

    from ai_made_easy.ui.canvas import painter

    painter.install_flat_node_style()
    assert getattr(NodeItem, "_aime_flat", False)
    assert NodeItem.paint is painter._flat_paint
    painter.install_flat_node_style()  # idempotent
    assert NodeItem.paint is painter._flat_paint
