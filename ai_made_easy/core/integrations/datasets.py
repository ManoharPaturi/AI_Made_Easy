"""Datasets from Kaggle and the Hugging Face Hub, detected like a local folder.

A source names where the data lives:

* ``kaggle:owner/dataset`` (optionally ``/file``): a Kaggle dataset
* ``kaggle-competition:name`` (optionally ``/file``): a competition's files
* ``hf:owner/repo`` (optionally ``:split``): a Hub dataset, saved as a Parquet table,
  or as one folder per class for image datasets

``fetch`` downloads it once into ``$AIME_HOME/datasets`` and returns the path the
wizard, ``aime data`` and the dataset blocks use, so detection, profiling and remote
bundles treat it like any local file or folder. Sign-in is the tool's own:
``~/.kaggle/kaggle.json`` and ``hf auth login`` (public Hub datasets need none).
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from ai_made_easy.core.paths import subdir

KINDS = ("kaggle", "kaggle-competition", "hf")
_NAME = r"[A-Za-z0-9][\w.-]*"
TABLES = (".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx")
_TRAIN = re.compile(r"(^|[_-])train([_.-]|$)", re.I)
IMAGE_LIMIT = 20000   # images written per Hub image dataset


class SourceError(RuntimeError):
    pass


def is_source(value: str) -> bool:
    return isinstance(value, str) and value.split(":", 1)[0] in KINDS and ":" in value


def parse(source: str) -> tuple[str, str, str]:
    """``(kind, ref, file_or_split)``."""
    if not is_source(source):
        raise SourceError(f"{source!r} is not a dataset source; use kaggle:owner/name, "
                          "kaggle-competition:name or hf:owner/repo")
    kind, rest = source.split(":", 1)
    rest = rest.strip().strip("/")
    if kind == "hf":
        repo, _, split = rest.partition(":")
        if not re.fullmatch(rf"{_NAME}(/{_NAME})?", repo):
            raise SourceError(f"{repo!r} is not a Hub dataset id")
        return kind, repo, split or "train"
    parts = rest.split("/")
    if kind == "kaggle":
        if len(parts) < 2 or not all(re.fullmatch(_NAME, p) for p in parts[:2]):
            raise SourceError(f"{rest!r} is not a Kaggle dataset (owner/name)")
        if ".." in parts:
            raise SourceError(f"{rest!r} is not a Kaggle dataset file")
        return kind, "/".join(parts[:2]), "/".join(parts[2:])
    if not re.fullmatch(_NAME, parts[0]):
        raise SourceError(f"{rest!r} is not a Kaggle competition")
    if ".." in parts:
        raise SourceError(f"{rest!r} is not a Kaggle competition file")
    return kind, parts[0], "/".join(parts[1:])


def cache_dir(source: str) -> Path:
    kind, ref, extra = parse(source)
    slug = re.sub(r"[^\w.-]+", "__", f"{ref}__{extra}" if extra else ref)
    return subdir("datasets") / kind / slug


def _unzip_all(folder: Path) -> None:
    for archive in list(folder.rglob("*.zip")):
        target = archive.with_suffix("")
        root = target.resolve()
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise SourceError(f"unsafe path in {archive.name}: {member}")
            zf.extractall(target)
        archive.unlink()
        inner = [p for p in target.iterdir()]
        if len(inner) == 1 and inner[0].is_dir():  # name/name/... -> name/...
            for child in inner[0].iterdir():
                shutil.move(str(child), target / child.name)
            inner[0].rmdir()


def data_path(folder: Path) -> Path:
    """What detection should read: the training table when the download is tables
    (``train.csv`` before ``test.csv``), else the folder."""
    entries = [p for p in folder.rglob("*") if p.is_file()]
    tables = [p for p in entries if p.suffix.lower() in TABLES]
    others = [p for p in entries if p.suffix.lower() not in TABLES and p.name != ".aime-source"]
    if tables and len(others) <= len(tables):
        train = [p for p in tables if _TRAIN.search(p.stem)]
        pick = train or ([tables[0]] if len(tables) == 1 else
                         sorted(tables, key=lambda p: -p.stat().st_size))
        return pick[0]
    return folder


def _kaggle(kind: str, ref: str, file: str, dest: Path) -> None:
    from ai_made_easy.core.runner.backends import BackendError
    from ai_made_easy.core.runner.backends.kaggle import KaggleClient

    try:
        client = KaggleClient()
        client.download(ref, str(dest), competition=kind == "kaggle-competition", file=file)
    except BackendError as exc:
        raise SourceError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — the client's HTTP errors
        hint = (" (join the competition and accept its rules on kaggle.com first)"
                if kind == "kaggle-competition" else "")
        raise SourceError(f"Kaggle download of {ref} failed: {exc}{hint}") from exc
    _unzip_all(dest)


def _load_hf(repo: str, split: str):  # noqa: ANN202 — datasets.Dataset
    try:
        import datasets
    except ImportError as exc:
        raise SourceError("Hub datasets need the datasets package: "
                          "pip install 'ai-made-easy[hub]'") from exc
    try:
        return datasets.load_dataset(repo, split=split)
    except Exception as exc:  # noqa: BLE001
        raise SourceError(f"could not load {repo} ({split}) from the Hub: {exc}") from exc


def _hf(repo: str, split: str, dest: Path) -> None:
    ds = _load_hf(repo, split)
    features = getattr(ds, "features", {}) or {}
    image = next((k for k, v in features.items() if type(v).__name__ == "Image"), None)
    if image is None:
        ds.to_parquet(str(dest / f"{split}.parquet"))
        return
    label = next((k for k, v in features.items() if type(v).__name__ == "ClassLabel"), None)
    names = features[label].names if label else None
    root = dest / "images"
    for i, row in enumerate(ds):
        if i >= IMAGE_LIMIT:
            break
        cls = names[row[label]] if names else str(row.get(label, "unlabelled"))
        folder = root / re.sub(r"[^\w.-]+", "_", str(cls))
        folder.mkdir(parents=True, exist_ok=True)
        row[image].convert("RGB").save(folder / f"{i:06d}.png")


def fetch(source: str, *, force: bool = False) -> Path:
    """Download ``source`` once and return the file or folder to read."""
    kind, ref, extra = parse(source)
    dest = cache_dir(source)
    marker = dest / ".aime-source"
    if marker.exists() and not force:
        return data_path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    try:
        if kind == "hf":
            _hf(ref, extra, dest)
        else:
            _kaggle(kind, ref, extra, dest)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    marker.write_text(source + "\n", encoding="utf-8")
    return data_path(dest)


def submit(competition: str, file: str, message: str) -> dict:
    """Submit a predictions file to a Kaggle competition (callers confirm first)."""
    from ai_made_easy.core.runner.backends import BackendError
    from ai_made_easy.core.runner.backends.kaggle import KaggleClient

    path = Path(file).expanduser()
    if not path.is_file():
        raise SourceError(f"{file} does not exist")
    if not message.strip():
        raise SourceError("a submission needs a message")
    try:
        result = KaggleClient().submit(competition, str(path), message)
    except BackendError as exc:
        raise SourceError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise SourceError(f"Kaggle refused the submission: {exc}") from exc
    text = str(getattr(result, "message", "") or result or "")
    return {"competition": competition, "file": str(path), "message": message,
            "result": text,
            "url": f"https://www.kaggle.com/competitions/{competition}/submissions"}
