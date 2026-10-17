"""Run bundles: everything a remote machine needs to train one design.

A bundle is the run folder's training script, the worker, the launcher's metadata
(``bundle.json``), ``requirements.txt`` and every local data file the design points
at. Paths to local files are rewritten to ``data/…`` inside the bundle, so the script
finds them on the far side; nothing else from this computer travels.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ai_made_easy.core.graph import Graph

WORKER_NAME = "_aime_worker.py"
LAUNCHER_NAME = "aime_launch.py"
# bundles that travel inside one file (a Kaggle kernel, a Colab notebook) stay small;
# bigger data belongs in a Kaggle dataset or Google Drive
MAX_EMBEDDED_BYTES = 20_000_000


def worker_source() -> str:
    from ai_made_easy.core.runner.protocol import worker_script_path

    return worker_script_path().read_text()


def launcher_path() -> Path:
    import ai_made_easy.worker

    return Path(ai_made_easy.worker.__file__).parent / LAUNCHER_NAME


def _looks_like_path(value: str) -> bool:
    return bool(value) and (os.sep in value or "/" in value or value.startswith("~"))


def relocate_data(graph: Graph) -> tuple[Graph, dict[str, Path]]:
    """A copy of the design whose local file / folder params point into ``data/``,
    and the ``{bundle name: local path}`` files to ship."""
    copy = Graph.from_dict(graph.to_dict())
    files: dict[str, Path] = {}
    for node in copy.nodes.values():
        for key, value in list(node.params.items()):
            if not isinstance(value, str) or not _looks_like_path(value):
                continue
            local = Path(value).expanduser()
            if not local.exists():
                continue  # already a remote path (e.g. /kaggle/input/…) or a URL
            name = f"data/{node.instance_id}_{key}_{local.name}"
            files[name] = local.resolve()
            node.params[key] = name
    return copy, files


def training_requirements(script_text: str) -> list[str]:
    """Package names from the script's ``Needs:`` line (no pins: the remote keeps
    the versions it already has)."""
    match = re.search(r"^Needs:\s*(.+)$", script_text, re.M)
    names = [n.strip() for n in (match.group(1) if match else "").split(",") if n.strip()]
    return list(dict.fromkeys(names))


@dataclass
class Bundle:
    run_id: str
    run_dir: Path
    script: str                      # script file name inside the bundle
    files: dict[str, Path] = field(default_factory=dict)   # data files to ship
    requirements: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        return {"run_id": self.run_id, "script": self.script, "worker": WORKER_NAME,
                "requirements": self.requirements}

    def size(self) -> int:
        total = 0
        for path in self.files.values():
            if path.is_dir():
                total += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            else:
                total += path.stat().st_size
        return total + (self.run_dir / self.script).stat().st_size

    def tar_bytes(self) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            def add_text(name: str, text: str) -> None:
                data = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = int(time.time())  # not 1970: zip (the Colab results) refuses it
                tar.addfile(info, io.BytesIO(data))

            tar.add(self.run_dir / self.script, arcname=self.script)
            add_text(WORKER_NAME, worker_source())
            add_text("bundle.json", json.dumps(self.metadata(), indent=1))
            add_text("requirements.txt", "\n".join(self.requirements) + "\n")
            for name, path in sorted(self.files.items()):
                tar.add(path, arcname=name)
        return buf.getvalue()

    def launcher(self, embed: bool = False, args: list[str] | None = None) -> str:
        """The launcher's source; with ``embed`` the bundle travels inside it, and
        ``args`` become its arguments when it runs without any."""
        source = launcher_path().read_text()
        if args is not None:
            source = source.replace("DEFAULT_ARGS: list = []", f"DEFAULT_ARGS: list = {args!r}", 1)
        if not embed:
            return source
        data = self.tar_bytes()
        if len(data) > MAX_EMBEDDED_BYTES:
            raise ValueError(
                f"the run's data is {len(data) / 1e6:.0f} MB, too big to send inside one "
                f"file (limit {MAX_EMBEDDED_BYTES / 1e6:.0f} MB). Upload it as a Kaggle "
                "dataset or to Google Drive and point the dataset block at it there.")
        encoded = base64.b64encode(data).decode()
        return source.replace('BUNDLE = ""', f'BUNDLE = "{encoded}"', 1)


def prepare(graph: Graph, framework: str, run_id: str, run_dir: Path) -> Bundle:
    """Write the run's training script (with data paths relocated) and describe the
    bundle that carries it."""
    from ai_made_easy.core.codegen import export_training

    relocated, files = relocate_data(graph)
    script = export_training(relocated, framework, run_dir)
    return Bundle(run_id=run_id, run_dir=Path(run_dir), script=script.name, files=files,
                  requirements=training_requirements(script.read_text()))


def unpack_outputs(data: bytes, run_dir: Path) -> None:
    """Extract a ``.tar.gz`` of remote outputs into the local run folder (safely)."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(run_dir, filter="data")
            return
        root = os.path.realpath(run_dir)
        for member in tar.getmembers():
            target = os.path.realpath(os.path.join(run_dir, member.name))
            if not target.startswith(root) or member.issym() or member.islnk():
                raise ValueError(f"unsafe path in outputs: {member.name}")
        tar.extractall(run_dir)


def merge_folder(src: Path, run_dir: Path) -> None:
    """Move downloaded outputs into the run folder, keeping the local run.json."""
    for item in src.iterdir():
        if item.name in ("run.json", "epochs.jsonl"):
            continue
        target = run_dir / item.name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        shutil.move(str(item), str(target))
