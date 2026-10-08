"""Custom blocks on disk, and ``.aimeblocks`` packs for sharing them.

A custom block is a saved fragment (``$AIME_HOME/templates/<slug>.json``)
carrying a ``version`` that increases each time it is saved again under the
same name. Packs are zip files with a manifest; importing never silently
overwrites a different block of the same name.
"""
from __future__ import annotations

import json
import re
import time
import zipfile
from pathlib import Path

from ai_made_easy.core.paths import subdir

PACK_FORMAT = "aime-blocks"
PACK_VERSION = 1


class PackError(ValueError):
    pass


def templates_dir() -> Path:
    return subdir("templates")


def slugify(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    return slug or "template"


def _check(data: dict) -> None:
    from ai_made_easy.core.composites import fragment_from_dict

    if not isinstance(data, dict) or not data.get("nodes") or "entry" not in data:
        raise PackError("not a custom block definition")
    try:
        fragment_from_dict(data)
    except Exception as exc:  # noqa: BLE001
        raise PackError(f"invalid custom block: {exc}") from exc


def _content(data: dict) -> str:
    keep = {k: data[k] for k in ("nodes", "edges", "entry", "exit") if k in data}
    return json.dumps(keep, sort_keys=True)


def save_block(data: dict, name: str) -> Path:
    """Write a custom block, bumping its version when the name exists."""
    _check(data)
    path = templates_dir() / f"{slugify(name)}.json"
    version = 1
    if path.exists():
        old = json.loads(path.read_text())
        version = int(old.get("version", 1)) + (0 if _content(old) == _content(data) else 1)
    record = {**data, "name": name, "version": version, "updated_at": time.time()}
    path.write_text(json.dumps(record, indent=2))
    return path


def list_blocks() -> list[dict]:
    out = []
    for path in sorted(templates_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        out.append({"slug": path.stem, "name": data.get("name", path.stem),
                    "version": int(data.get("version", 1)),
                    "updated_at": data.get("updated_at"), "blocks": len(data.get("nodes", [])),
                    "type_id": f"custom.{path.stem}", "path": str(path)})
    return out


def export_pack(out_path: Path | str, slugs: list[str] | None = None) -> Path:
    out_path = Path(out_path)
    available = {b["slug"]: b for b in list_blocks()}
    chosen = slugs or list(available)
    missing = [s for s in chosen if s not in available]
    if missing:
        raise PackError(f"unknown custom blocks: {', '.join(missing)}")
    if not chosen:
        raise PackError("there are no custom blocks to export")
    manifest = {"format": PACK_FORMAT, "version": PACK_VERSION, "created_at": time.time(),
                "blocks": [{k: available[s][k] for k in ("slug", "name", "version")}
                           for s in chosen]}
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for slug in chosen:
            zf.write(available[slug]["path"], f"blocks/{slug}.json")
    return out_path


def import_pack(path: Path | str, overwrite: bool = False) -> dict:
    """Install a pack. Identical blocks are skipped; different blocks with a
    taken name are renamed (``name_2``) unless ``overwrite``."""
    result = {"imported": [], "skipped": [], "renamed": {}}
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise PackError(f"not a custom-block pack: {exc}") from exc
    with zf:
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except (KeyError, ValueError) as exc:
            raise PackError("the pack has no valid manifest") from exc
        if manifest.get("format") != PACK_FORMAT:
            raise PackError("not a custom-block pack")
        if int(manifest.get("version", 0)) > PACK_VERSION:
            raise PackError("the pack was made by a newer version of AI Made Easy")
        for entry in manifest.get("blocks", []):
            slug = slugify(entry["slug"])
            install(slug, json.loads(zf.read(f"blocks/{slug}.json")), result, overwrite)
    return result


def install(slug: str, data: dict, result: dict, overwrite: bool = False) -> None:
    """Write one block, recording into ``result`` what happened."""
    _check(data)
    slug = slugify(slug)
    target = templates_dir() / f"{slug}.json"
    if target.exists() and not overwrite:
        if _content(json.loads(target.read_text())) == _content(data):
            result["skipped"].append(slug)
            return
        n = 2
        while (templates_dir() / f"{slug}_{n}.json").exists():
            n += 1
        new = f"{slug}_{n}"
        data = {**data, "name": f"{data.get('name', slug)} {n}"}
        (templates_dir() / f"{new}.json").write_text(json.dumps(data, indent=2))
        result["renamed"][slug] = new
        return
    target.write_text(json.dumps(data, indent=2))
    result["imported"].append(slug)


def blocks_used(graph_dict: dict) -> dict[str, dict]:
    """Custom-block definitions a project uses (slug -> definition)."""
    out = {}
    for node in graph_dict.get("nodes", []):
        type_id = str(node.get("type", ""))
        if type_id.startswith("custom."):
            path = templates_dir() / f"{type_id.split('.', 1)[1]}.json"
            if path.exists():
                out[path.stem] = json.loads(path.read_text())
    return out


def install_for_project(graph_dict: dict, blocks: dict[str, dict]) -> dict:
    """Install a project's bundled custom blocks; a renamed block is re-pointed
    in ``graph_dict`` so the project keeps using the bundled definition."""
    result = {"imported": [], "skipped": [], "renamed": {}}
    for slug, data in blocks.items():
        install(slug, data, result)
    for node in graph_dict.get("nodes", []):
        type_id = str(node.get("type", ""))
        if type_id.startswith("custom."):
            slug = type_id.split(".", 1)[1]
            if slug in result["renamed"]:
                node["type"] = f"custom.{result['renamed'][slug]}"
    return result


def custom_block_definition(path: Path):  # noqa: ANN201 — BlockDefinition
    """The Custom-block definition for one saved block file."""
    from ai_made_easy.core.blocks._palette import family_color
    from ai_made_easy.core.composites import fragment_from_dict
    from ai_made_easy.core.spec import BlockDefinition, PortSpec

    def _builder(_params, p=path):  # noqa: ANN001, ANN202
        return fragment_from_dict(json.loads(p.read_text()))

    try:
        meta = json.loads(path.read_text())
    except ValueError:
        meta = {}
    version = int(meta.get("version", 1))
    return BlockDefinition(
        type_id=f"custom.{slugify(path.stem)}",
        display_name=str(meta.get("name") or path.stem.replace("_", " ").title())
        + (f" v{version}" if version > 1 else ""),
        category="Custom",
        color=family_color("custom"),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        builder=_builder,
    )


def register_custom_blocks() -> list:
    """Register saved custom blocks not yet in the registry; returns the new ones."""
    from ai_made_easy.core.registry import get_registry

    registry = get_registry()
    added = []
    for path in sorted(templates_dir().glob("*.json")):
        block = custom_block_definition(path)
        if not registry.has(block.type_id):
            registry.register(block)
            added.append(block)
    return added
