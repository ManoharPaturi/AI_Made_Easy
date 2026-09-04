"""Node-class factory: BlockDefinitions -> dynamic OdenGraphQt node classes.

Owns the bidirectional type maps so lookups are O(1) both ways (the old
reverse scans are gone). Only this package imports OdenGraphQt.
"""
from __future__ import annotations

import re
from typing import Type

from OdenGraphQt import BaseNode
from OdenGraphQt.constants import NodePropWidgetEnum
from PySide6 import QtGui

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition
from ai_made_easy.ui.canvas.painter import TEXT_COLOR, install_flat_node_style

_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]+")

# canvas node type <-> core block type_id (bidirectional, O(1) both ways)
NODE_TYPE_TO_BLOCK: dict[str, str] = {}
BLOCK_TO_NODE_TYPE: dict[str, str] = {}


def _class_name(block: BlockDefinition) -> str:
    parts = _IDENT_RE.split(block.display_name)
    camel = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return f"{camel}Node"


def _identifier_for(block: BlockDefinition) -> str:
    """Namespace doubles as palette grouping: aim.data, aim.layers, ..."""
    slug = _IDENT_RE.sub("", block.category).lower() or "misc"
    return f"aim.{slug}"


def _widget_type(param) -> tuple[int, dict]:
    """Map a ParamSpec to (NodePropWidgetEnum value, extra kwargs)."""
    ptype = param.type
    if ptype == "int":
        default = int(param.default) if param.default is not None else 0
        lo = int(param.minimum) if param.minimum is not None else min(0, default)
        hi = int(param.maximum) if param.maximum is not None else max(100_000, default)
        return NodePropWidgetEnum.QSPIN_BOX.value, {"range": (lo, hi)}
    if ptype == "float":
        default = float(param.default) if param.default is not None else 0.0
        lo = (float(param.minimum) if param.minimum is not None
              else min(0.0, default))
        hi = (float(param.maximum) if param.maximum is not None
              else max(1_000_000_000.0, default))
        return NodePropWidgetEnum.QDOUBLESPIN_BOX.value, {"range": (lo, hi)}
    if ptype == "bool":
        return NodePropWidgetEnum.QCHECK_BOX.value, {}
    if ptype == "enum":
        return NodePropWidgetEnum.QCOMBO_BOX.value, {"items": list(param.options)}
    return NodePropWidgetEnum.QLINE_EDIT.value, {}


def node_type_for(block_type_id: str) -> str | None:
    return BLOCK_TO_NODE_TYPE.get(block_type_id)


def make_node_class(block: BlockDefinition) -> Type[BaseNode]:
    """Build a BaseNode subclass mirroring the BlockDefinition."""
    install_flat_node_style()
    attrs = {
        "__identifier__": _identifier_for(block),
        "NODE_NAME": block.display_name,
    }

    def _init(self, block=block) -> None:
        super(type(self), self).__init__()
        # OdenGraphQt paints with QColor(*color): RGB ints, never hex/tuples.
        qcolor = QtGui.QColor(block.color)
        rgba = qcolor.getRgb()
        port_rgba = qcolor.darker(135).getRgb()  # deeper family shade for dots/wires
        self.set_color(rgba[0], rgba[1], rgba[2])  # NodeObject wants r, g, b
        self.view.text_color = TEXT_COLOR        # dark ink on the pastel fill
        self.view.border_color = rgba            # flat: border == fill
        for port in block.inputs:
            self.add_input(port.name, color=port_rgba, display_name=False)
        for port in block.outputs:
            self.add_output(port.name, color=port_rgba, display_name=False)
        for param in block.params:
            widget_type, extra = _widget_type(param)
            kwargs = {"widget_type": widget_type, **extra}
            if param.help:
                kwargs["widget_tooltip"] = param.help
            self.create_property(param.name, param.default, **kwargs)

    attrs["__init__"] = _init
    cls = type(_class_name(block), (BaseNode,), attrs)
    node_type = f"{_identifier_for(block)}.{cls.__name__}"  # BaseNode.type_
    NODE_TYPE_TO_BLOCK[node_type] = block.type_id
    BLOCK_TO_NODE_TYPE[block.type_id] = node_type
    return cls
