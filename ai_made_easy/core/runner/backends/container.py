"""Container backend: train in Docker on this computer, for a reproducible environment.

With no ``image`` option the backend builds one per set of requirements, like the
deploy package's Dockerfile (``python:3.11-slim`` + the run's packages, CPU PyTorch
unless ``gpus`` is on), and Docker caches it for later runs. For a container on another
machine, use an SSH backend with its ``docker_image`` option.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import threading
from typing import TYPE_CHECKING

from ai_made_easy.core.runner.backends import BackendError, RemoteBackend, split_command
from ai_made_easy.core.runner.backends.bundle import LAUNCHER_NAME
from ai_made_easy.core.runs.history import FINAL_STATES

if TYPE_CHECKING:
    from ai_made_easy.core.runner.manager import TrainingRun

BASE_IMAGE = "python:3.11-slim"
CPU_TORCH = "https://download.pytorch.org/whl/cpu"
BUILD_TIMEOUT = 3600  # seconds; a first build downloads PyTorch
_SKIP = ("bundle.tar.gz", LAUNCHER_NAME, "pid", "exit_code")


def docker_command() -> list[str]:
    """The docker client (``$AIME_DOCKER`` overrides it, e.g. for tests or podman)."""
    override = os.environ.get("AIME_DOCKER")
    if override:
        return split_command(override)
    found = shutil.which("docker") or shutil.which("podman")
    if not found:
        raise BackendError("no docker client found: install Docker Desktop (or podman)")
    return [found]


def dockerfile(requirements: list[str], gpus: bool) -> str:
    index = "" if gpus else f" --extra-index-url {CPU_TORCH}"
    lines = [f"FROM {BASE_IMAGE}", "WORKDIR /aime/run"]
    if requirements:
        packages = " ".join(shlex.quote(r) for r in requirements)
        lines.append(f"RUN pip install --no-cache-dir{index} {packages}")
    return "\n".join(lines) + "\n"


class ContainerBackend(RemoteBackend):
    kind = "container"
    options_help = {
        "image": "a ready image to train in (default: built from the run's requirements)",
        "gpus": "pass the GPUs through (--gpus all; needs the NVIDIA toolkit, default false)",
    }

    def test(self) -> dict:
        try:
            proc = subprocess.run([*docker_command(), "version", "--format",
                                   "{{.Server.Version}}"], capture_output=True, encoding="utf-8", errors="replace",
                                  timeout=30)
        except (BackendError, OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "message": str(exc), "accelerator": {}}
        if proc.returncode != 0:
            return {"ok": False, "accelerator": {},
                    "message": "Docker is not running: " + proc.stderr.strip()[-300:]}
        return {"ok": True, "message": f"Docker {proc.stdout.strip()}", "accelerator": {}}

    def image_for(self, run: TrainingRun) -> str:
        image = self.options.get("image")
        if image:
            return str(image)
        gpus = bool(self.options.get("gpus"))
        text = dockerfile(run.bundle.requirements, gpus)
        tag = "aime-train:" + hashlib.sha256(text.encode()).hexdigest()[:12]
        docker = docker_command()
        if subprocess.run([*docker, "image", "inspect", tag], capture_output=True,
                          timeout=60).returncode:
            self.note(run, f"building the training image {tag} (once per set of packages)")
            try:
                build = subprocess.run([*docker, "build", "-t", tag, "-"], input=text,
                                       capture_output=True, encoding="utf-8", errors="replace", timeout=BUILD_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                raise BackendError(f"the image did not build in {BUILD_TIMEOUT // 60} "
                                   "minutes") from exc
            if build.returncode:
                raise BackendError("the image did not build:\n" + build.stderr[-2000:])
        return tag

    def launch(self, run: TrainingRun, python: str) -> None:
        threading.Thread(target=self._launch, args=(run,), daemon=True).start()

    def _launch(self, run: TrainingRun) -> None:
        mount = run.workdir / "container"
        try:
            mount.mkdir(exist_ok=True)
            (mount / "bundle.tar.gz").write_bytes(run.bundle.tar_bytes())
            (mount / LAUNCHER_NAME).write_text(run.bundle.launcher(), encoding="utf-8")
            image = self.image_for(run)
            if run.state in FINAL_STATES:
                return
            argv = [*docker_command(), "run", "--rm", "--name", f"aime-{run.run_id}",
                    *(["--gpus", "all"] if self.options.get("gpus") else []),
                    "-v", f"{mount}:/aime/run", "-w", "/aime/run", image,
                    "python", LAUNCHER_NAME, "/aime/run", "--bundle", "bundle.tar.gz"]
            if self.options.get("image"):
                argv.append("--install")
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    encoding="utf-8", errors="replace", bufsize=1)
            run.process = proc
            run.accelerator = {"gpu": "passthrough" if self.options.get("gpus") else ""}
            if run.history is not None:
                run.history.update(run.run_id, remote=self.remote_state(run))
            self.note(run, f"training in {image}")
        except Exception as exc:  # noqa: BLE001
            run.record({"type": "error", "traceback": str(exc)})
            run.process_exited(1)
            return
        assert proc.stdout is not None and proc.stderr is not None
        stderr = threading.Thread(
            target=lambda: [run.record({"type": "log", "line": line.rstrip(),
                                        "stream": "stderr"}) for line in proc.stderr],
            daemon=True)
        stderr.start()
        for line in proc.stdout:
            self.feed(run, line)
        code = proc.wait()
        stderr.join(timeout=5)
        exit_file = mount / "exit_code"
        if exit_file.exists() and exit_file.read_text(encoding="utf-8").strip().lstrip("-").isdigit():
            code = int(exit_file.read_text(encoding="utf-8").strip())
        for item in mount.iterdir():
            if item.name not in _SKIP:
                target = run.workdir / item.name
                if target.exists() and target.is_dir():
                    shutil.rmtree(target)
                shutil.move(str(item), str(target))
        shutil.rmtree(mount, ignore_errors=True)
        self.finish(run, code)

    def remote_state(self, run: TrainingRun) -> dict:
        return {"container": f"aime-{run.run_id}", "accelerator": run.accelerator}

    def stop(self, run: TrainingRun) -> None:
        try:
            subprocess.run([*docker_command(), "stop", f"aime-{run.run_id}"],
                           capture_output=True, timeout=30)
        except (BackendError, OSError, subprocess.TimeoutExpired):
            pass
