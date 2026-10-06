"""Block colour system: one accent colour per functional family.

Every category maps to exactly one family. The canvas uses the colour as the
block's accent (header strip, ports, wires); the library uses it for swatches.
"""
from __future__ import annotations

FAMILY_COLORS: dict[str, str] = {
    "io": "#8CE99A",             # model Input / Output
    "model": "#F5D547",          # linear layers, embeddings, architectures
    "conv": "#FFC078",           # convolutions
    "pool": "#FFD8A8",           # pooling
    "resize": "#E599F7",         # padding, upsampling, cropping
    "recurrent": "#B197FC",      # RNN / LSTM / GRU
    "attention": "#91A7FF",      # attention + transformer blocks
    "activation": "#FF8787",     # activation functions
    "normalization": "#74C0FC",  # batch/layer/group/instance norms
    "regularization": "#A5D8FF", # dropout + noise
    "merge": "#63E6BE",          # add / concat / multiply ...
    "tensor": "#96F2D7",         # reshape / permute / reductions
    "data": "#B2F2BB",           # dataset sources
    "preprocess": "#D8F5A2",     # preprocessing & augmentation
    "training": "#D0BFFF",       # optimizers, losses, schedulers, trainer
    "evaluation": "#99E9F2",     # metrics
    "llm": "#F783AC",            # tokenizers, LoRA, RAG, generation
    "classic": "#FCC2D7",        # scikit-learn & gradient boosting
    "custom": "#FFE8CC",         # user-saved templates
}


def family_color(family: str) -> str:
    return FAMILY_COLORS[family]
