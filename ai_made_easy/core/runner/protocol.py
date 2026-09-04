"""Worker event protocol: decode one stdout line from the training worker.

Tolerant by contract — training processes emit warnings, banners, and other
garbage; anything undecodable becomes a plain log event. Never raises.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_event(line: str) -> dict[str, Any] | None:
    """Decode one worker line into an event dict; None for blank lines."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return {"type": "log", "line": line}
    if isinstance(obj, dict) and isinstance(obj.get("type"), str):
        return obj
    return {"type": "log", "line": line}


def worker_script_path() -> Path:
    """Absolute path to the standalone worker script (ships with the app)."""
    import ai_made_easy.worker

    return Path(ai_made_easy.worker.__file__).parent / "train_worker.py"
