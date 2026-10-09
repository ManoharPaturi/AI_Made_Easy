"""Built-in block library. Importing submodules registers them globally.

Add new blocks by dropping a module here and importing it below.
"""
from ai_made_easy.core.blocks import (  # noqa: F401
    io_blocks,
    linear,
    convolution,
    pooling,
    resizing,
    recurrent,
    attention,
    activations,
    normalization,
    merge,
    tensor_ops,
    pretrained,
    architectures,
    data_blocks,
    preprocessing,
    training,
    evaluation,
    llm_blocks,
    classic,
    vision,
    sequence,
    forecast,
    speech,
    generative,
    pgm,
    ppl,
    gp,
    bayes,
)
