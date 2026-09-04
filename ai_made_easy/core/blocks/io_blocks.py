"""I/O blocks: dataset/model entry and exit points."""
from __future__ import annotations

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, ParamSpec, PortSpec
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_IO_COLOR = family_color("data")

reg.register(
    BlockDefinition(
        type_id="core.input",
        display_name="Input",
        category="Data",
        color=_IO_COLOR,
        params=(
            ParamSpec(
                name="shape",
                type="str",
                default="784",
                help="Tensor shape per sample, no batch dim, e.g. '784' or '28,28,1'",
            ),
        ),
        outputs=(PortSpec("out"),),
    )
)

reg.register(
    BlockDefinition(
        type_id="core.output",
        display_name="Output",
        category="Data",
        color=_IO_COLOR,
        inputs=(PortSpec("in"),),
    )
)
