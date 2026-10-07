"""Test-session environment: headless Qt and Keras on the PyTorch backend."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KERAS_BACKEND", "torch")
