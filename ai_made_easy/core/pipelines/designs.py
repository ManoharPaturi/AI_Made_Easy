"""The designs a pipeline's stages train: resolving a stage's ``design`` value (a project
path, ``sample:<name>`` or ``embedded:<key>``) and comparing designs (architecture,
dataset, output) for the pipeline rules."""
from __future__ import annotations

import copy
import json
from pathlib import Path

DATA_PREFIXES = ("data.", "prep.")
CONFIG_PREFIXES = ("data.", "prep.", "train.", "eval.")


class DesignError(ValueError):
    pass


def resolve(value: str, pipeline, base: str | Path | None = None):  # noqa: ANN001, ANN201
    """The Graph a stage's design value names."""
    from ai_made_easy.core.graph import Graph

    value = str(value or "").strip()
    if not value:
        raise DesignError("choose a design (a project file, sample:<name> or embedded:<key>)")
    if value.startswith("embedded:"):
        key = value.split(":", 1)[1]
        designs = (getattr(pipeline, "meta", None) or {}).get("designs") or {}
        if key not in designs:
            raise DesignError(f"no embedded design {key!r} (have: {sorted(designs) or 'none'})")
        return Graph.from_dict(copy.deepcopy(designs[key]))
    if value.startswith("sample:"):
        from ai_made_easy.core import api

        try:
            return Graph.from_dict(api.read_sample(value.split(":", 1)[1]))
        except api.ApiError as exc:
            raise DesignError(str(exc)) from exc
    path = Path(value).expanduser()
    if not path.is_absolute():
        for root in (base, Path.cwd()):
            if root is not None and (Path(root) / path).is_file():
                path = Path(root) / path
                break
    if not path.is_file():
        raise DesignError(f"design file {value!r} not found")
    try:
        return Graph.from_dict(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError) as exc:
        raise DesignError(f"could not read {value}: {exc}") from exc


def embed(pipeline: dict, key: str, design: dict) -> dict:
    """The pipeline dict with ``design`` stored under ``embedded:<key>``."""
    out = copy.deepcopy(pipeline)
    out.setdefault("meta", {}).setdefault("designs", {})[key] = copy.deepcopy(design)
    return out


def canonical(graph) -> dict:  # noqa: ANN001
    """The design without layout (positions) for fingerprints."""
    data = graph.to_dict()
    return {"nodes": sorted(({"id": n["id"], "type": n["type"], "params": n.get("params", {})}
                             for n in data["nodes"]), key=lambda n: n["id"]),
            "edges": sorted(json.dumps(e, sort_keys=True) for e in data["edges"])}


def data_signature(graph) -> list:  # noqa: ANN001
    """Dataset and preprocessing blocks with their settings (same data, same split)."""
    rows = [(n.type_id, json.dumps(n.resolved_params(), sort_keys=True, default=str))
            for n in graph.nodes.values() if n.type_id.startswith(DATA_PREFIXES)]
    return sorted(rows)


def layers(graph) -> list[tuple[str, str]]:  # noqa: ANN001
    """The network's blocks in order (type, settings), without data / training blocks."""
    try:
        chain = graph.model_nodes()
    except Exception:  # noqa: BLE001 — an unreadable design has no layers
        return []
    return [(n.type_id, json.dumps(n.resolved_params(), sort_keys=True, default=str))
            for n in chain if not n.type_id.startswith(CONFIG_PREFIXES)
            and n.type_id not in ("core.input", "core.output")]


def output_shape(graph) -> list | None:  # noqa: ANN001
    from ai_made_easy.core.summary import summarize

    try:
        rows = summarize(graph).layers
    except Exception:  # noqa: BLE001
        return None
    return list(rows[-1].output_shape) if rows else None


def torch_supervised(graph) -> str:  # noqa: ANN001
    """'' when the design is a supervised PyTorch network, else why it is not."""
    from ai_made_easy.core.families import family_of
    from ai_made_easy.core.tasks import task_of

    family = family_of(graph)
    if family.id != "neural":
        return f"a {family.label} design"
    task = task_of(graph)
    if task is None:
        return "a design without a loss (no task)"
    if task.trainer_kind != "supervised":
        return f"a {task.label} design (it trains with its own loop)"
    return ""
