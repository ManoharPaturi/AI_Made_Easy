"""Model registry: versioned, promotable copies of trained runs.

``$AIME_HOME/models/<name>/<version>/`` holds the run's training script,
weights, fitted preprocessing, ``run.json`` and ``model.json``; a version folder
is itself a valid run folder, so it can be packaged for deployment at any time
— even after the original run is deleted.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ai_made_easy.core.paths import subdir

STAGES = ("none", "staging", "production", "archived")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_COPY = ("run.json", "metrics.json", "classes.json", "inference_state.pkl", "inference.json",
         "epochs.jsonl", "predictions.json", "mistakes.json")
_COPY_GLOBS = ("*_train_*.py", "*_best.pt", "*_best.keras", "*_model.joblib")


class RegistryError(ValueError):
    pass


@dataclass
class ModelVersion:
    name: str
    version: int
    run_id: str
    framework: str
    task: str = ""
    stage: str = "none"
    metrics: dict = field(default_factory=dict)
    description: str = ""
    created_at: float = field(default_factory=time.time)
    stage_changed_at: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else subdir("models")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def path(self, name: str, version: int) -> Path:
        if not _NAME_RE.match(name or ""):
            raise RegistryError(f"invalid model name {name!r}")
        return self.root / name / str(int(version))

    def register(self, run_dir: Path | str, name: str, description: str = "") -> ModelVersion:
        run_dir = Path(run_dir)
        if not _NAME_RE.match(name or ""):
            raise RegistryError("model names use letters, digits, '.', '_' and '-' "
                                "(max 64 characters)")
        run_file = run_dir / "run.json"
        if not run_file.exists():
            raise RegistryError(f"{run_dir} is not a run folder")
        record = json.loads(run_file.read_text())
        if record.get("status") != "finished":
            raise RegistryError("only finished runs can be registered")
        files = [run_dir / f for f in _COPY if (run_dir / f).exists()]
        for pattern in _COPY_GLOBS:
            files.extend(sorted(run_dir.glob(pattern)))
        if not any(f.suffix in (".pt", ".keras", ".joblib") for f in files):
            raise RegistryError("the run has no trained model file")
        from ai_made_easy.core.deploy.package import _describe

        try:
            task = _describe(record.get("graph") or {})["task"]
        except Exception:  # noqa: BLE001 — informative only
            task = ""
        with self._lock:
            existing = [v.version for v in self.versions(name)]
            version = max(existing, default=0) + 1
            target = self.path(name, version)
            target.mkdir(parents=True)
            for file in files:
                shutil.copy2(file, target / file.name)
            mv = ModelVersion(name=name, version=version, run_id=record.get("run_id", ""),
                              framework=record.get("framework", ""), task=task,
                              metrics=record.get("final_metrics") or {},
                              description=description)
            self._save(mv)
            return mv

    def _save(self, mv: ModelVersion) -> None:
        (self.path(mv.name, mv.version) / "model.json").write_text(
            json.dumps(mv.to_dict(), indent=1, default=str))

    def get(self, name: str, version: int | str | None = None) -> ModelVersion:
        if version in (None, "", "latest"):
            versions = self.versions(name)
            if not versions:
                raise KeyError(f"no model named {name!r}")
            return versions[-1]
        if version == "production":
            prod = [v for v in self.versions(name) if v.stage == "production"]
            if not prod:
                raise KeyError(f"{name} has no production version")
            return prod[-1]
        file = self.path(name, int(version)) / "model.json"
        if not file.exists():
            raise KeyError(f"no version {version} of {name!r}")
        data = json.loads(file.read_text())
        return ModelVersion(**{k: v for k, v in data.items()
                               if k in ModelVersion.__dataclass_fields__})

    def versions(self, name: str) -> list[ModelVersion]:
        folder = self.root / name
        if not _NAME_RE.match(name or "") or not folder.is_dir():
            return []
        out = []
        for child in folder.iterdir():
            if child.name.isdigit() and (child / "model.json").exists():
                out.append(self.get(name, int(child.name)))
        return sorted(out, key=lambda v: v.version)

    def list(self) -> list[ModelVersion]:
        out = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            out.extend(self.versions(folder.name))
        return out

    def set_stage(self, name: str, version: int, stage: str) -> ModelVersion:
        if stage not in STAGES:
            raise RegistryError(f"stage must be one of {STAGES}")
        with self._lock:
            mv = self.get(name, version)
            if stage == "production":  # one production version per model
                for other in self.versions(name):
                    if other.version != mv.version and other.stage == "production":
                        other.stage, other.stage_changed_at = "archived", time.time()
                        self._save(other)
            mv.stage, mv.stage_changed_at = stage, time.time()
            self._save(mv)
            return mv

    def delete(self, name: str, version: int) -> None:
        with self._lock:
            path = self.path(name, version)
            if not (path / "model.json").exists():
                raise KeyError(f"no version {version} of {name!r}")
            shutil.rmtree(path)
            parent = path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
