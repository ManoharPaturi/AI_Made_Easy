"""User-template store: reusable Custom blocks saved under $AIME_HOME/templates.

All filesystem IO for templates lives here; the adapter and graph service
only call these functions.
"""
from __future__ import annotations

import json
from pathlib import Path

from ai_made_easy.core.block_packs import (
    custom_block_definition,
    register_custom_blocks,
    save_block,
    slugify,
    templates_dir,
)
from ai_made_easy.core.composites import Fragment, fragment_from_dict, fragment_to_dict
from ai_made_easy.core.spec import BlockDefinition

__all__ = ["slugify", "save_template", "load_template", "template_block",
           "register_user_templates"]


def save_template(frag: Fragment, name: str) -> Path:
    return save_block(fragment_to_dict(frag, name), name)


def load_template(name_or_path: str) -> Fragment:
    path = templates_dir() / f"{slugify(name_or_path)}.json"
    return fragment_from_dict(json.loads(path.read_text()))


def template_block(path: Path) -> BlockDefinition:
    """Build the Custom-block definition for one saved template file."""
    return custom_block_definition(path)


def register_user_templates(register, make_node) -> None:  # noqa: ANN001
    """Register every saved template that isn't known yet and build its node class.

    ``register`` is kept for callers; blocks go into the core registry.
    """
    for block in register_custom_blocks():
        make_node(block)
