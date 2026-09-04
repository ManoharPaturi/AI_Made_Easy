"""User-template store: reusable Custom blocks saved under ~/.aime/templates.

All filesystem IO for templates lives here; the adapter and graph service
only call these functions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ai_made_easy.core.composites import (
    Fragment,
    fragment_from_dict,
    fragment_to_dict,
)
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, PortSpec
from ai_made_easy.core.blocks._palette import family_color

TEMPLATES_DIR = Path.home() / ".aime" / "templates"

_CUSTOM_COLOR = family_color("custom")


def slugify(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    return slug or "template"


def save_template(frag: Fragment, name: str) -> Path:
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMPLATES_DIR / f"{slugify(name)}.json"
    path.write_text(json.dumps(fragment_to_dict(frag, name), indent=2))
    return path


def load_template(name_or_path: str) -> Fragment:
    path = TEMPLATES_DIR / f"{slugify(name_or_path)}.json"
    return fragment_from_dict(json.loads(path.read_text()))


def template_block(path: Path) -> BlockDefinition:
    """Build the Custom-block definition for one saved template file."""
    def _builder(_params, p=path) -> Fragment:
        return fragment_from_dict(json.loads(p.read_text()))

    return BlockDefinition(
        type_id=f"custom.{slugify(path.stem)}",
        display_name=path.stem.replace("_", " ").title(),
        category="Custom",
        color=_CUSTOM_COLOR,
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        builder=_builder,
    )


def register_user_templates(register, make_node) -> None:
    """Register every saved template that isn't known yet.

    `register` is BlockRegistry.register, `make_node` is the node factory.
    """
    reg = get_registry()
    if not TEMPLATES_DIR.is_dir():
        return
    for path in sorted(TEMPLATES_DIR.glob("*.json")):
        block = template_block(path)
        if reg.has(block.type_id):
            continue
        reg.register(block)
        make_node(block)
