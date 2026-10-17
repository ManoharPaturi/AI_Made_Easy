"""Sticky notes and group frames on the desktop canvas (3.3).

Both are OdenGraphQt backdrops: a group wraps its blocks and moves them with it; a note
is a backdrop with text and no blocks. They are saved in the design's ``meta.notes`` /
``meta.groups`` (``core.canvas``), which code generation ignores.
"""
from __future__ import annotations

from OdenGraphQt import BackdropNode


class NoteNode(BackdropNode):
    __identifier__ = "aime.canvas"
    NODE_NAME = "Note"

    def __init__(self) -> None:
        super().__init__()
        self.model.color = (255, 236, 153, 255)


class GroupNode(BackdropNode):
    __identifier__ = "aime.canvas"
    NODE_NAME = "Group"


NOTE_TYPE = f"{NoteNode.__identifier__}.{NoteNode.__name__}"
GROUP_TYPE = f"{GroupNode.__identifier__}.{GroupNode.__name__}"


def hex_to_rgba(color: str) -> tuple[int, int, int, int]:
    c = color.lstrip("#")
    if len(c) != 6:
        return (145, 167, 255, 255)
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255)


def rgba_to_hex(rgba) -> str:  # noqa: ANN001
    r, g, b = (int(v) for v in list(rgba)[:3])
    return f"#{r:02X}{g:02X}{b:02X}"
