"""The feature vector a CSV dataset produces after the design's column preprocessing:
its width, and the positions / number of categories of ordinal-encoded columns.

Mirrors ``preprocess_tables`` in the training template: the target is removed, Feature
Columns select and order columns, Drop Columns removes some, Ordinal Encode keeps the
listed (or all) text columns in place as integer codes, and the remaining text columns
are one-hot encoded at the end (at most ``max_categories`` each).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    width: int
    categorical: tuple[int, ...]           # positions of ordinal-encoded columns
    cardinalities: tuple[int, ...]
    names: tuple[str, ...]                 # their column names
    unencoded: tuple[str, ...]             # text columns that are one-hot encoded


def _names(value) -> list[str]:  # noqa: ANN001
    return [c.strip() for c in str(value or "").split(",") if c.strip()]


@lru_cache(maxsize=8)
def _read(path: str, fmt: str, mtime: float):  # noqa: ANN202
    import pandas as pd

    if fmt == "tsv":
        return pd.read_csv(path, sep="\t")
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def layout_of(graph) -> Layout | None:  # noqa: ANN001
    """None when the design has no readable CSV / TSV / Parquet dataset."""
    data = next((n for n in graph.nodes.values()
                 if n.type_id in ("data.csv", "data.synthetic_table")), None)
    if data is None:
        return None
    p = data.resolved_params()
    if data.type_id == "data.synthetic_table":
        return _arrange(graph, _synthetic_frame(), "")
    path = Path(str(p["path"])).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file() or p.get("format") not in ("csv", "tsv", "parquet"):
        return None
    try:
        frame = _read(str(path), str(p["format"]), path.stat().st_mtime)
    except Exception:  # noqa: BLE001 — unreadable files are reported by the data profile
        return None
    target = str(p["target_column"])
    frame = frame.drop(columns=[target]) if target in frame.columns else frame
    return _arrange(graph, frame, str(p.get("feature_columns") or ""))


@lru_cache(maxsize=1)
def _synthetic_frame():  # noqa: ANN202
    """A stand-in with the Synthetic Table's columns and category counts (the generator
    always emits every category)."""
    import pandas as pd

    from ai_made_easy.core.tabular.synthetic import SCHEMA

    rows = max(cats or 1 for _name, cats in SCHEMA)
    return pd.DataFrame({name: [f"{name}{i % cats}" for i in range(rows)] if cats
                         else [0.0] * rows for name, cats in SCHEMA})


def _arrange(graph, frame, feature_columns: str) -> Layout:  # noqa: ANN001
    import pandas as pd

    wanted = _names(feature_columns)
    if wanted:
        frame = frame[[c for c in wanted if c in frame.columns]]

    def step(type_id: str):  # noqa: ANN202
        node = next((n for n in graph.nodes.values() if n.type_id == type_id), None)
        return node.resolved_params() if node is not None else None

    drop = step("prep.drop_columns")
    if drop:
        frame = frame.drop(columns=[c for c in _names(drop["columns"]) if c in frame.columns])
    text = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    ordinal_step = step("prep.ordinal_encode")
    ordinal = (_names(ordinal_step["columns"]) or text) if ordinal_step else []
    ordinal = [c for c in ordinal if c in frame.columns]
    one_hot_step = step("prep.one_hot")
    remaining = [c for c in text if c not in ordinal]
    one_hot = [c for c in (_names(one_hot_step["columns"]) if one_hot_step else []) or remaining
               if c in frame.columns]
    max_categories = int(one_hot_step["max_categories"]) if one_hot_step else 50
    kept = [c for c in frame.columns if c not in one_hot]
    positions = tuple(kept.index(c) for c in ordinal)
    cards = tuple(int(frame[c].dropna().astype(str).nunique()) for c in ordinal)
    width = len(kept) + sum(min(int(frame[c].dropna().astype(str).nunique()), max_categories)
                            for c in one_hot)
    return Layout(width, positions, cards, tuple(ordinal), tuple(one_hot))
