"""Model entry and exit points."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P
from ai_made_easy.core.blocks._palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, PortSpec

reg = get_registry()

reg.register(
    BlockDefinition(
        type_id="core.input",
        display_name="Input",
        category="Input / Output",
        color=family_color("io"),
        params=(
            P("shape", "str", "784",
              help="Per-sample shape without the batch dimension, channels-first: "
                   "'784' (vector), '16,128' (sequence [L, C] or signal [C, L]), "
                   "'3,224,224' (image [C, H, W])"),
            P("dtype", "enum", "float32", options=("float32", "int64"),
              help="int64 for token/category indices feeding an Embedding"),
        ),
        outputs=(PortSpec("out"),),
        description="Model input tensor. Shapes are channels-first and exclude the batch.",
        library="PyTorch · Keras",
    )
)

reg.register(
    BlockDefinition(
        type_id="core.output",
        display_name="Output",
        category="Input / Output",
        color=family_color("io"),
        inputs=(PortSpec("in"),),
        description="Model output tensor (logits for classification).",
        library="PyTorch · Keras",
    )
)
