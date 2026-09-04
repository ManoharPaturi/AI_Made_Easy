"""Built-in block library. Importing submodules registers them globally.

Add new blocks by dropping a module here and importing it below — no core
code changes needed (this is also how Phase 5 'custom blocks' will plug in).
"""
from ai_made_easy.core.blocks import (  # noqa: F401
    activations,
    architectures,
    attention,
    data_blocks,
    evaluation,
    io_blocks,
    layers,
    llm_blocks,
    normalization,
    preprocessing,
    tensor_ops,
    training,
)
