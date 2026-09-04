"""One colour per functional family — the block colour system.

Conv / transformer / architecture blocks are all "model compute" and share
ONE colour (the user's explicit rule); activations share another; every
functional family maps to exactly one flat pastel that dark ink text reads
on. The canvas paints blocks with these fills (see ui/canvas/painter.py).
"""
from __future__ import annotations

FAMILY_COLORS: dict[str, str] = {
    "model": "#F5D547",        # layers + attention + architectures (conv, transformer, ...)
    "activation": "#FF8787",   # activation functions
    "normalization": "#74C0FC",
    "tensor": "#63E6BE",       # reshapes, concatenation, splits
    "data": "#8CE99A",         # inputs/outputs + dataset sources
    "preprocess": "#D8F5A2",   # augmentation & preprocessing
    "training": "#D0BFFF",     # optimizers, losses, schedulers, trainer
    "evaluation": "#99E9F2",   # metrics
    "llm": "#F783AC",          # tokenizers, LoRA, RAG, generation
    "custom": "#FFE8CC",       # user-saved templates
}


def family_color(family: str) -> str:
    return FAMILY_COLORS[family]
