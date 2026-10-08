"""Project archives (.aime): one portable file for a whole project.

zip layout:
    manifest.json        {format, version, app_version, created, name, entries}
    graph.json           the design (Graph.to_dict)
    model_card.md        model card (optional)
    thumbnail.png        canvas image (optional)
    dataset/<class>/…    image-folder samples (optional, size-capped)
    run/…                checkpoint, training script, metrics and predictions (optional)
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "aime"
BUNDLE_VERSION = 1
MAX_DATASET_FILES = 600
MAX_DATASET_BYTES = 40 * 1024 * 1024


def _manifest(name: str, entries: dict) -> dict:
    try:
        from importlib.metadata import version

        app_version = version("ai-made-easy")
    except Exception:
        app_version = "dev"
    return {
        "format": FORMAT,
        "version": BUNDLE_VERSION,
        "app": "AI Made Easy",
        "app_version": app_version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "entries": entries,
    }


def write_bundle(path: Path, graph_dict: dict, *, name: str = "project",
                 card_md: str | None = None, thumbnail_png: bytes | None = None,
                 dataset_dir: Path | None = None,
                 workdir: Path | None = None) -> Path:
    """Assemble the .aime zip. Missing pieces are simply omitted."""
    entries: dict[str, bool] = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        payload = {"graph.json": json.dumps(graph_dict, indent=1)}
        entries["graph"] = True
        if card_md:
            payload["model_card.md"] = card_md
            entries["model_card"] = True
        if thumbnail_png:
            payload["thumbnail.png"] = None  # binary, written separately
        for member, text in payload.items():
            if text is not None:
                zf.writestr(member, text)
        if thumbnail_png:
            zf.writestr("thumbnail.png", thumbnail_png)
            entries["thumbnail"] = True
        from ai_made_easy.core.block_packs import blocks_used

        custom = blocks_used(graph_dict)
        for slug, data in custom.items():
            zf.writestr(f"blocks/{slug}.json", json.dumps(data, indent=1))
        if custom:
            entries["custom_blocks"] = len(custom)
        if dataset_dir and Path(dataset_dir).exists():
            count = size = 0
            for f in sorted(Path(dataset_dir).rglob("*")):
                if not f.is_file() or f.suffix.lower() not in (
                        ".png", ".jpg", ".jpeg", ".bmp"):
                    continue
                if count >= MAX_DATASET_FILES or size > MAX_DATASET_BYTES:
                    break
                zf.write(f, f"dataset/{f.relative_to(dataset_dir)}")
                count += 1
                size += f.stat().st_size
            entries["dataset_files"] = count
        if workdir and Path(workdir).exists():
            for pattern, member in (("*_best.pt", "run/checkpoint.pt"),
                                    ("*_best.keras", "run/model.keras"),
                                    ("*_model.joblib", "run/model.joblib"),
                                    ("*_train_*.py", "run/train.py"),
                                    ("metrics.json", "run/metrics.json"),
                                    ("classes.json", "run/classes.json"),
                                    ("predictions.json", "run/predictions.json"),
                                    ("mistakes.json", "run/mistakes.json"),
                                    ("inference_state.pkl", "run/inference_state.pkl"),
                                    ("inference.json", "run/inference.json")):
                hits = sorted(Path(workdir).glob(pattern))
                if hits:
                    zf.write(hits[0], member)
                    entries[member.split("/")[0]] = True
        zf.writestr("manifest.json", json.dumps(
            _manifest(name, entries), indent=1))
    return Path(path)


def read_bundle(path: Path) -> dict:
    """Open + validate; returns manifest + members (dataset extracted to a
    temp dir the caller owns)."""
    import tempfile

    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names or "graph.json" not in names:
            raise ValueError(f"{path.name} is not an AI Made Easy project archive "
                             "(missing manifest or graph)")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError(f"unknown archive format "
                             f"{manifest.get('format')!r}")
        if int(manifest.get("version", 0)) > BUNDLE_VERSION:
            raise ValueError("this archive was created by a newer version of "
                             "AI Made Easy; please update")
        out = {
            "manifest": manifest,
            "graph": json.loads(zf.read("graph.json")),
            "card": (zf.read("model_card.md").decode()
                     if "model_card.md" in names else None),
            "has_dataset": any(n.startswith("dataset/") for n in names),
            "has_run": any(n in names for n in ("run/checkpoint.pt", "run/model.keras",
                                                "run/model.joblib")),
            "custom_blocks": {n[len("blocks/"):-len(".json")]: json.loads(zf.read(n))
                              for n in names if n.startswith("blocks/") and n.endswith(".json")},
            "run_dir": None,
            "dataset_dir": None,
        }
        if out["has_run"]:
            run_dir = Path(tempfile.mkdtemp(prefix="aime_bundle_run_"))
            for n in names:
                if n.startswith("run/"):
                    zf.extract(n, run_dir)
            out["run_dir"] = run_dir / "run"
        if out["has_dataset"]:
            ds_dir = Path(tempfile.mkdtemp(prefix="aime_bundle_ds_"))
            for n in names:
                if n.startswith("dataset/"):
                    zf.extract(n, ds_dir)
            out["dataset_dir"] = ds_dir / "dataset"
    return out


def restore_run(bundle: dict, history) -> str | None:  # noqa: ANN001
    """Add a bundle's trained run to the run history (so it can be analysed,
    registered and deployed). Returns the new run id, or None without a run."""
    import re
    import shutil

    run_dir = bundle.get("run_dir")
    if not run_dir or not (Path(run_dir) / "train.py").exists():
        return None
    run_dir = Path(run_dir)
    script = (run_dir / "train.py").read_text()
    framework = ("keras" if (run_dir / "model.keras").exists() else
                 "sklearn" if (run_dir / "model.joblib").exists() else "pytorch")
    match = re.search(r'^(?:CHECKPOINT|MODEL_FILE) = "([^"]+)"', script, re.M)
    record = history.create(bundle["graph"], framework=framework,
                            project=bundle["manifest"].get("name", ""), tags=["archive"])
    target = history.path(record.run_id)
    stem = (match.group(1).rsplit("_", 1)[0] if match else "model")
    (target / f"{stem}_train_{framework}.py").write_text(script)
    renames = {"checkpoint.pt": match.group(1) if match else f"{stem}_best.pt",
               "model.keras": match.group(1) if match else f"{stem}_best.keras",
               "model.joblib": match.group(1) if match else f"{stem}_model.joblib"}
    for f in run_dir.iterdir():
        if f.name == "train.py":
            continue
        shutil.copy2(f, target / renames.get(f.name, f.name))
    history.finalize(record.run_id, "finished", 0)
    return record.run_id
