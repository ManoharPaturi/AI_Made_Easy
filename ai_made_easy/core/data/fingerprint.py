"""Data fingerprints: a short hash identifying the exact data a run trained on.

Files are hashed by content (large files: size + head + tail); folders by the
relative path, size and modification time of every file (fast enough to run at
every training start). Benchmarks, synthetic and remote datasets are
identified by their parameters. Hashes are cached by file signature.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from ai_made_easy.core.data.profile import FOLDER_SUFFIXES, resolve_path

_FULL_HASH_LIMIT = 256 * 2**20   # bytes hashed completely
_PARTIAL = 4 * 2**20             # head / tail hashed for bigger files
_cache: dict[tuple, str] = {}
_lock = threading.Lock()


def _short(h) -> str:
    return h.hexdigest()[:16]


def file_hash(path: Path) -> str:
    st = path.stat()
    key = ("file", str(path), st.st_size, st.st_mtime_ns)
    with _lock:
        if key in _cache:
            return _cache[key]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        if st.st_size <= _FULL_HASH_LIMIT:
            for chunk in iter(lambda: fh.read(2**20), b""):
                h.update(chunk)
        else:
            h.update(str(st.st_size).encode())
            h.update(fh.read(_PARTIAL))
            fh.seek(-_PARTIAL, 2)
            h.update(fh.read(_PARTIAL))
    digest = _short(h)
    with _lock:
        _cache[key] = digest
    return digest


def folder_hash(root: Path, suffixes: set[str] | None = None) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        if suffixes and f.suffix.lower() not in suffixes:
            continue
        st = f.stat()
        h.update(f"{f.relative_to(root).as_posix()}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return _short(h)


def fingerprint(type_id: str, params: dict, base=None) -> str:
    """``kind:hash`` for a dataset block, "" if its data is missing."""
    if type_id in FOLDER_SUFFIXES:
        root = resolve_path(params.get("root", ""), base)
        return f"folder:{folder_hash(root, FOLDER_SUFFIXES[type_id])}" if root.is_dir() else ""
    if "path" in params and type_id not in ("data.torchvision",):
        path = resolve_path(params.get("path", ""), base)
        return f"file:{file_hash(path)}" if path.is_file() else ""
    if type_id in ("data.torchvision", "data.sklearn"):
        return f"{type_id.split('.')[1]}:{params.get('dataset')}"
    if type_id == "data.huggingface":
        return f"hf:{params.get('repo_id')}@{params.get('split')}"
    blob = json.dumps({k: params[k] for k in sorted(params)}, default=str)
    return f"{type_id.split('.')[-1]}:{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


def graph_fingerprint(graph_dict: dict, base=None) -> str:
    """Fingerprint of the dataset block in a saved graph (``""`` when there is none)."""
    from ai_made_easy.core.graph import Graph

    try:
        graph = Graph.from_dict(graph_dict)
    except Exception:  # noqa: BLE001 — unknown blocks / malformed graphs
        return ""
    for node in graph.nodes.values():
        if node.type_id.startswith("data."):
            try:
                return fingerprint(node.type_id, dict(node.resolved_params()), base)
            except OSError:
                return ""
    return ""
