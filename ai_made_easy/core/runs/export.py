"""``aime runs export``: run records as a table for notebooks (CSV or Parquet).

One row per run: identity, status, timing, where it trained, then ``param.<node.key>``,
``final.<metric>`` and ``best.<metric>`` columns. ``epochs=True`` writes one row per
epoch instead (``run_id``, ``epoch``, then every metric).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_made_easy.core.runs.history import RunHistory

FORMATS = ("csv", "parquet")


def _cell(value):  # noqa: ANN001, ANN202
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return value


def run_rows(history: RunHistory, project: str | None = None) -> list[dict]:
    rows = []
    for rec in reversed(history.list(project=project)):
        row = {"run_id": rec.run_id, "name": rec.name, "project": rec.project,
               "kind": rec.kind, "framework": rec.framework, "status": rec.status,
               "created_at": rec.created_at, "finished_at": rec.finished_at,
               "duration_s": rec.duration, "epochs": rec.epochs_done,
               "backend": rec.backend or "local", "parent": rec.parent,
               "data_fingerprint": rec.data_fingerprint, "tags": ",".join(rec.tags)}
        row.update({f"param.{k}": _cell(v) for k, v in rec.params.items()})
        row.update({f"final.{k}": v for k, v in rec.final_metrics.items()})
        row.update({f"best.{k}": v for k, v in rec.best_metrics.items()})
        rows.append(row)
    return rows


def epoch_rows(history: RunHistory, project: str | None = None) -> list[dict]:
    rows = []
    for rec in reversed(history.list(project=project)):
        for event in history.epochs(rec.run_id):
            rows.append({"run_id": rec.run_id, "name": rec.name,
                         "epoch": event.get("epoch"),
                         **{k: v for k, v in (event.get("metrics") or {}).items()
                            if isinstance(v, (int, float))}})
    return rows


def export_runs(history: RunHistory, out: str | Path, *, project: str | None = None,
                fmt: str = "", epochs: bool = False) -> dict:
    out = Path(out).expanduser()
    fmt = (fmt or out.suffix.lstrip(".") or "csv").lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; one of {', '.join(FORMATS)}")
    rows = epoch_rows(history, project) if epochs else run_rows(history, project)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ValueError("Parquet needs pandas and pyarrow; write .csv instead") from exc
        frame = pd.DataFrame(rows, columns=columns)
        for col in frame.columns:  # mixed-type parameter columns become text
            if frame[col].dtype == object:
                frame[col] = frame[col].map(lambda v: None if v is None else str(v))
        try:
            frame.to_parquet(out, index=False)
        except ImportError as exc:
            raise ValueError("Parquet needs pyarrow; write .csv instead") from exc
    else:
        with open(out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return {"path": str(out), "format": fmt, "rows": len(rows), "columns": len(columns)}
