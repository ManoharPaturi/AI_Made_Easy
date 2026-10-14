"""The Hugging Face Hub: push trained runs as model repos, deploy Spaces, pull back.

* ``push_model(run_dir, repo_id)``: a model repo with the trained files, the design,
  the run record and the model card AI Made Easy writes (``core.model_card``)
* ``pull_model(repo_id)``: download such a repo; the folder is a valid run folder, so
  it registers, packages and predicts like a local run
* ``deploy_space(package_dir, space_id)``: the FastAPI package as a Docker Space

Sign-in is the Hub's own (``hf auth login`` or ``HF_TOKEN``). Repos are private unless
you say otherwise.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from ai_made_easy.core.paths import subdir

EXTRA = "pip install 'ai-made-easy[hub]'"
MARKER = "aime.json"
_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")
_LIBRARY = {"pytorch": "pytorch", "keras": "keras", "sklearn": "sklearn"}


class HubError(RuntimeError):
    pass


def _api():  # noqa: ANN202 — huggingface_hub.HfApi (tests replace this)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise HubError(f"the Hugging Face Hub needs huggingface_hub: {EXTRA}") from exc
    return HfApi()


def _check(repo_id: str) -> None:
    if not _REPO.match(repo_id or ""):
        raise HubError(f"{repo_id!r} is not a Hub repo id (owner/name)")


def _signed_in(api) -> None:  # noqa: ANN001
    try:
        api.whoami()
    except Exception as exc:  # noqa: BLE001 — LocalTokenNotFoundError, HTTP 401
        raise HubError("not signed in to the Hugging Face Hub: run `hf auth login` "
                       "(or set HF_TOKEN)") from exc


def model_name(repo_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", repo_id.split("/")[-1]).strip("-.") or "model"
    return name[:64]


def model_card(run_dir: Path, record: dict, repo_id: str) -> str:
    """The run's model card with the Hub's front matter (library, tags)."""
    from ai_made_easy.core.model_card import build_card

    graph = record.get("graph") or {}
    nodes = graph.get("nodes") or []
    trainer = next((n.get("params") or {} for n in nodes
                    if str(n.get("type", "")).startswith("train.trainer")), {})
    data = next((n for n in nodes if str(n.get("type", "")).startswith("data.")), None)
    dataset = (f"{data['type']} ({', '.join(f'{k}={v}' for k, v in (data.get('params') or {}).items())})"
               if data else "see the design")
    task = ""
    try:
        from ai_made_easy.core.deploy.package import _describe

        task = _describe(graph)["task"]
    except Exception:  # noqa: BLE001 — informative only
        task = ""
    framework = record.get("framework", "pytorch")
    card = build_card(record.get("name") or model_name(repo_id), dataset, trainer, run_dir,
                      details={"framework": framework, "task": task or "n/a"})
    tags = ["ai-made-easy", framework] + ([task] if task else [])
    front = (f"library_name: {_LIBRARY.get(framework, framework)}\n"
             "tags:\n" + "".join(f"- {t}\n" for t in dict.fromkeys(tags)))
    usage = (f"\n## Use it\n\n```bash\naime hub pull {repo_id}      # registers it locally\n"
             f"aime deploy <name>/latest -o server   # a FastAPI package\n```\n")
    return card.replace("---\n", "---\n" + front, 1) + usage


def stage_model(run_dir: Path) -> tuple[Path, dict]:
    """A temporary folder with what a model repo holds."""
    from ai_made_easy.core.deploy.registry import _COPY, _COPY_GLOBS

    run_dir = Path(run_dir)
    run_file = run_dir / "run.json"
    if not run_file.exists():
        raise HubError(f"{run_dir} is not a run folder")
    record = json.loads(run_file.read_text())
    if record.get("status") != "finished":
        raise HubError("only finished runs can be pushed")
    files = [run_dir / f for f in _COPY if (run_dir / f).exists()]
    for pattern in _COPY_GLOBS:
        files.extend(sorted(run_dir.glob(pattern)))
    if not any(f.suffix in (".pt", ".keras", ".joblib") for f in files):
        raise HubError("the run has no trained model file")
    stage = Path(tempfile.mkdtemp(prefix="aime_hub_"))
    for file in files:
        shutil.copy2(file, stage / file.name)
    (stage / "design.json").write_text(json.dumps(record.get("graph") or {}, indent=1))
    (stage / MARKER).write_text(json.dumps({
        "format": "ai-made-easy-run", "version": 1, "run_id": record.get("run_id", ""),
        "name": record.get("name", ""), "framework": record.get("framework", "")}, indent=1))
    return stage, record


def push_model(run_dir: Path | str, repo_id: str, *, private: bool = True,
               message: str = "") -> dict:
    _check(repo_id)
    api = _api()
    _signed_in(api)
    stage, record = stage_model(Path(run_dir))
    try:
        (stage / "README.md").write_text(model_card(Path(run_dir), record, repo_id))
        url = api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="model",
                          commit_message=message or
                          f"AI Made Easy run {record.get('run_id', '')}")
    except HubError:
        raise
    except Exception as exc:  # noqa: BLE001 — HTTP errors
        raise HubError(f"push to {repo_id} failed: {exc}") from exc
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {"repo_id": repo_id, "url": str(url or f"https://huggingface.co/{repo_id}"),
            "private": private, "run_id": record.get("run_id", "")}


def pull_model(repo_id: str, *, revision: str | None = None) -> Path:
    """Download a model repo AI Made Easy pushed; returns a run folder."""
    _check(repo_id)
    api = _api()
    dest = subdir("hub") / repo_id.replace("/", "__")
    if dest.exists():
        shutil.rmtree(dest)
    try:
        folder = Path(api.snapshot_download(repo_id=repo_id, repo_type="model",
                                            revision=revision, local_dir=str(dest)))
    except Exception as exc:  # noqa: BLE001
        raise HubError(f"could not download {repo_id}: {exc}") from exc
    if not (folder / MARKER).exists() or not (folder / "run.json").exists():
        raise HubError(f"{repo_id} was not pushed by AI Made Easy (no {MARKER}); open its "
                       "model file with Import instead")
    return folder


def space_readme(package_dir: Path, space_id: str) -> str:
    readme = (package_dir / "README.md").read_text() if (package_dir / "README.md").exists() \
        else ""
    title = space_id.split("/")[-1]
    return (f"---\ntitle: {title}\nsdk: docker\napp_port: 8000\ntags:\n- ai-made-easy\n"
            f"---\n\n{readme}")


def deploy_space(package_dir: Path | str, space_id: str, *, private: bool = True) -> dict:
    """Upload a built package (``aime deploy``) as a Docker Space."""
    _check(space_id)
    package_dir = Path(package_dir)
    if not (package_dir / "Dockerfile").exists() or not (package_dir / "app.py").exists():
        raise HubError(f"{package_dir} is not a deployment package")
    api = _api()
    _signed_in(api)
    with tempfile.TemporaryDirectory(prefix="aime_space_") as tmp:
        stage = Path(tmp) / "space"
        shutil.copytree(package_dir, stage)
        (stage / "README.md").write_text(space_readme(package_dir, space_id))
        try:
            api.create_repo(space_id, repo_type="space", space_sdk="docker", private=private,
                            exist_ok=True)
            api.upload_folder(folder_path=str(stage), repo_id=space_id, repo_type="space",
                              commit_message="AI Made Easy model server")
        except Exception as exc:  # noqa: BLE001
            raise HubError(f"deploying the Space {space_id} failed: {exc}") from exc
    return {"space_id": space_id, "url": f"https://huggingface.co/spaces/{space_id}",
            "private": private}
