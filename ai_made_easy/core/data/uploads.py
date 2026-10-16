"""Data files dropped on the web UI (3.3): stored under ``$AIME_HOME/uploads`` so the
New Project wizard can read them like a local path.

A table (CSV, TSV, Parquet, Excel, JSON) or text file is kept as is; a ``.zip`` (image or
audio folders, one sub-folder per class) is unpacked after its paths and unpacked size are
checked. Uploads with the same content land in the same place.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from pathlib import Path

from ai_made_easy.core.paths import subdir

SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".jsonl",
            ".zip"}


class UploadError(ValueError):
    pass


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^\w.-]+", "_", Path(name).name).strip("._") or "data"
    return stem[:80]


def store(file: Path, filename: str, *, max_unpacked: int) -> Path:
    """Move ``file`` (already on disk) into the uploads folder; returns the path to
    detect: the file itself, or the folder a zip unpacked to."""
    name = _safe_name(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in SUFFIXES:
        raise UploadError(f"{filename}: drop a table (.csv, .parquet, .xlsx, .json), a text "
                          "file or a .zip of folders")
    digest = hashlib.sha256()
    with file.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    home = subdir("uploads") / f"{Path(name).stem}-{digest.hexdigest()[:10]}"
    if suffix != ".zip":
        home.mkdir(parents=True, exist_ok=True)
        target = home / name
        if not target.exists():
            shutil.move(str(file), target)
        return target
    if home.exists():
        return _root(home)
    try:
        archive = zipfile.ZipFile(file)
    except zipfile.BadZipFile as exc:
        raise UploadError(f"{filename} is not a valid zip file") from exc
    with archive:
        total = sum(i.file_size for i in archive.infolist())
        if total > max_unpacked:
            raise UploadError(f"{filename} unpacks to {total >> 20} MB, more than "
                              f"{max_unpacked >> 20} MB")
        tmp = home.with_name(home.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        root = tmp.resolve()
        for member in archive.namelist():
            if not (root / member).resolve().is_relative_to(root):
                shutil.rmtree(tmp, ignore_errors=True)
                raise UploadError(f"unsafe path in the zip: {member}")
        archive.extractall(tmp)
    for junk in tmp.rglob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)
    tmp.rename(home)
    return _root(home)


def _root(folder: Path) -> Path:
    """A zip of one folder unpacks to that folder."""
    entries = [p for p in folder.iterdir() if not p.name.startswith(".")]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else folder
