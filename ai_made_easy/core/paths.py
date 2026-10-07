"""Per-user data locations shared by the desktop app, CLI, MCP and web server.

``$AIME_HOME`` overrides the default ``~/.aime`` (tests and servers use it).
"""
from __future__ import annotations

import os
from pathlib import Path


def aime_home() -> Path:
    path = Path(os.environ.get("AIME_HOME") or Path.home() / ".aime").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def subdir(name: str) -> Path:
    path = aime_home() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
