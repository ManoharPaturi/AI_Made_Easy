"""ProjectService: project identity, file IO, examples, dirty tracking —
all through ProjectStore so every view of the project name stays in sync.
"""
from __future__ import annotations

import json
from pathlib import Path

from ai_made_easy.core.graph import Graph

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"
DEMO_SEED = SAMPLES_DIR / "demo_seed.json"


def project_kind(data: dict) -> str:
    """Display label of the design's family ("Neural network", "Classic ML pipeline", ...)."""
    from ai_made_easy.core.families import family_of

    return family_of(data).label


class ProjectService:
    def __init__(self, store, graph_service, log, parent=None):
        self.store = store          # ProjectStore
        self.graph_service = graph_service
        self.log = log

    # ------------------------------------------------------------ project

    def new_project(self) -> None:
        """Empty design with an Input and an Output block."""
        graph = Graph.from_dict({"name": "untitled", "nodes": [
            {"id": "input", "type": "core.input", "params": {"shape": "784"},
             "position": [-300, 0]},
            {"id": "output", "type": "core.output", "params": {}, "position": [300, 0]},
        ], "edges": []})
        self.graph_service.load(graph)
        self.store.reset("untitled")

    def _reload_demo(self) -> None:
        self.graph_service.load(Graph.from_dict(json.loads(DEMO_SEED.read_text())))
        self.store.mark_clean()

    def snapshot(self) -> Graph:
        """IR snapshot stamped with the current project name."""
        ir = self.graph_service.snapshot()
        ir.name = self.store.name
        return ir

    def open_file(self, path) -> bool:
        try:
            graph = Graph.from_dict(json.loads(Path(path).read_text()))
        except Exception as exc:
            self.log.error(f"failed to open {path}: {exc}")
            return False
        self.graph_service.load(graph)
        self.store.set_path(Path(path))
        self.store.set_name(graph.name)
        self.store.mark_clean()
        self.log.info(f"opened {path}")
        return True

    def save(self) -> bool:
        if self.store.path is None:
            return False  # caller should use save_as with a chosen path
        return self._write(self.store.path)

    def save_as(self, path) -> bool:
        ok = self._write(Path(path))
        if ok:
            self.store.set_path(Path(path))
        return ok

    def _write(self, path: Path) -> bool:
        try:
            path.write_text(json.dumps(self.snapshot().to_dict(), indent=2))
        except Exception as exc:
            self.log.error(f"failed to save {path}: {exc}")
            return False
        self.store.mark_clean()
        self.log.info(f"saved {path}")
        return True

    def apply_graph_dict(self, data: dict) -> bool:
        """Assistant / external graph application with shared validation."""
        ok, message = self.graph_service.validate_dict(data)
        if not ok:
            self.log.error(f"proposed graph is invalid: {message}")
            return False
        try:
            graph = Graph.from_dict(data)
        except Exception as exc:
            self.log.error(f"invalid graph JSON: {exc}")
            return False
        self.graph_service.load(graph)
        if graph.name:
            self.store.set_name(graph.name)
        self.log.info("graph applied to the canvas")
        return True

    # ------------------------------------------------------------ samples

    @staticmethod
    def samples_dir() -> Path:
        return SAMPLES_DIR

    def list_samples(self) -> list[tuple[Path, str, str, str]]:
        entries = []
        for path in sorted(self.samples_dir().glob("*.json")):
            if path.name == "demo_seed.json":
                continue
            try:
                data = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 — skip unreadable files
                continue
            meta = data.get("meta", {})
            entries.append((path, meta.get("title", data.get("name", path.stem)),
                            meta.get("description", ""), project_kind(data)))
        return entries

    def open_sample(self, path: Path) -> bool:
        return self.open_file(path)
