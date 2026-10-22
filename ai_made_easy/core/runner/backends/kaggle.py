"""Kaggle backend: train as a private Kaggle notebook (a script kernel).

Sign-in is Kaggle's own: ``~/.kaggle/kaggle.json`` (or ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` in the environment), read by the ``kaggle`` package. The run travels
inside the kernel's one file; data too big for that belongs in a Kaggle dataset,
attached with the ``datasets`` option and read from ``/kaggle/input/<name>``.

Kaggle has no live log API, so the app polls the kernel's status and the curves
arrive when it finishes.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ai_made_easy.core.runner.backends import BackendError, RemoteBackend
from ai_made_easy.core.runner.backends.bundle import merge_folder
from ai_made_easy.core.runs.history import FINAL_STATES

if TYPE_CHECKING:
    from ai_made_easy.core.runner.manager import TrainingRun

# the accelerator option -> (kernel metadata, device profile)
ACCELERATORS = {
    "none": ({"enable_gpu": False, "enable_tpu": False}, "cpu_laptop"),
    "T4": ({"enable_gpu": True, "enable_tpu": False, "machine_shape": "NvidiaTeslaT4"}, "t4"),
    "P100": ({"enable_gpu": True, "enable_tpu": False, "machine_shape": "NvidiaTeslaP100"},
             "p100"),
    "TPU": ({"enable_gpu": False, "enable_tpu": True, "machine_shape": "Tpu1VmV38"}, ""),
}
RUN_DIR = "/kaggle/working/aime_run"
_FINAL = {"complete": 0, "error": 1, "cancelacknowledged": 130, "cancelrequested": 130}


class KaggleClient:
    """The few Kaggle API calls the backend needs (the ``kaggle`` package)."""

    def __init__(self) -> None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise BackendError("Kaggle needs the kaggle package: "
                               "pip install 'ai-made-easy[kaggle]'") from exc
        self.api = KaggleApi()
        try:
            self.api.authenticate()
        except (OSError, ValueError, SystemExit) as exc:
            raise BackendError("Kaggle is not signed in: create an API token on kaggle.com "
                               "(Settings → API) and save it as ~/.kaggle/kaggle.json") from exc

    def username(self) -> str:
        return str(self.api.get_config_value("username") or "")

    def push(self, folder: str) -> str:
        result = self.api.kernels_push(folder)
        error = getattr(result, "error", None) or (result.get("error")
                                                    if isinstance(result, dict) else None)
        if error:
            raise BackendError(f"Kaggle refused the kernel: {error}")
        return str(getattr(result, "url", "") or "")

    def status(self, slug: str) -> str:
        result = self.api.kernels_status(slug)
        raw = result.get("status") if isinstance(result, dict) else getattr(result, "status",
                                                                             result)
        return str(getattr(raw, "name", raw)).split(".")[-1].lower()

    def output(self, slug: str, dest: str) -> None:
        self.api.kernels_output(slug, dest)

    def download(self, ref: str, dest: str, *, competition: bool = False,
                 file: str = "") -> None:
        """A dataset (``owner/name``) or competition's files into ``dest``."""
        if competition:
            if file:
                self.api.competition_download_file(ref, file, path=dest, quiet=True)
            else:
                self.api.competition_download_files(ref, path=dest, quiet=True)
        elif file:
            self.api.dataset_download_file(ref, file, path=dest, quiet=True)
        else:
            self.api.dataset_download_files(ref, path=dest, quiet=True, unzip=True)

    def submit(self, competition: str, file: str, message: str):  # noqa: ANN201
        return self.api.competition_submit(file, message, competition, quiet=True)

    def cancel(self, slug: str) -> bool:
        cancel = getattr(self.api, "kernels_cancel", None)
        if cancel is None:
            return False
        cancel(slug)
        return True


class KaggleBackend(RemoteBackend):
    kind = "kaggle"
    live = False
    options_help = {
        "accelerator": "none, T4, P100 or TPU (default T4)",
        "datasets": "Kaggle datasets to attach, e.g. [\"owner/name\"] (read from /kaggle/input)",
        "internet": "let the kernel use the internet, needed to pip-install (default true)",
        "poll_seconds": "how often to check the kernel (default 30)",
    }
    client_factory = KaggleClient  # tests swap in a fake

    def __init__(self, name: str = "", **options) -> None:  # noqa: ANN003
        super().__init__(name, **options)
        self.accel = str(options.get("accelerator") or "T4")
        if self.accel not in ACCELERATORS:
            raise BackendError(f"Kaggle accelerator must be one of {', '.join(ACCELERATORS)}")
        self.poll = float(options.get("poll_seconds", 30))
        self._client: KaggleClient | None = None

    def client(self) -> KaggleClient:
        if self._client is None:
            self._client = self.client_factory()
        return self._client

    def accelerator(self) -> dict:
        device = ACCELERATORS[self.accel][1]
        return {"gpu": "" if self.accel == "none" else self.accel, "device": device}

    def test(self) -> dict:
        try:
            user = self.client().username()
        except BackendError as exc:
            return {"ok": False, "message": str(exc), "accelerator": self.accelerator()}
        return {"ok": True, "message": f"signed in to Kaggle as {user}",
                "accelerator": self.accelerator()}

    def metadata(self, run: TrainingRun, user: str) -> dict:
        title = f"aime-{run.run_id}".lower()
        meta = {"id": f"{user}/{title}", "title": title, "code_file": "aime_kernel.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_internet": bool(self.options.get("internet", True)),
                "dataset_sources": list(self.options.get("datasets") or []),
                "competition_sources": [], "kernel_sources": []}
        meta.update(ACCELERATORS[self.accel][0])
        return meta

    def launch(self, run: TrainingRun, python: str) -> None:
        threading.Thread(target=self._launch, args=(run,), daemon=True).start()

    def _launch(self, run: TrainingRun) -> None:
        try:
            client = self.client()
            meta = self.metadata(run, client.username())
            args = [RUN_DIR] + (["--install"] if meta["enable_internet"] else [])
            with tempfile.TemporaryDirectory() as folder:
                Path(folder, "aime_kernel.py").write_text(
                    run.bundle.launcher(embed=True, args=args), encoding="utf-8")
                Path(folder, "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
                url = client.push(folder)
            run.slug = meta["id"]
            run.accelerator = self.accelerator()
            self.check_budget(run)
            run.remote_url = url or f"https://www.kaggle.com/code/{meta['id']}"
            if run.history is not None:
                run.history.update(run.run_id, remote=self.remote_state(run))
            self.note(run, f"pushed to Kaggle ({self.accel}): {run.remote_url} — curves "
                           "arrive when the kernel finishes")
        except Exception as exc:  # noqa: BLE001
            run.record({"type": "error", "traceback": str(exc)})
            run.process_exited(1)
            return
        self._poll(run)

    def remote_state(self, run: TrainingRun) -> dict:
        return {"slug": run.slug, "url": run.remote_url, "accelerator": run.accelerator}

    def reattach(self, run: TrainingRun) -> None:
        threading.Thread(target=self._poll, args=(run,), daemon=True).start()

    def _poll(self, run: TrainingRun) -> None:
        client = self.client()
        last = ""
        failures = 0
        while run.state not in FINAL_STATES:
            try:
                status = client.status(run.slug)
                failures = 0
            except Exception as exc:  # noqa: BLE001 — a flaky poll is retried
                failures += 1
                if failures >= 10:
                    run.record({"type": "error", "traceback": f"Kaggle status failed: {exc}"})
                    run.process_exited(1)
                    return
                time.sleep(self.poll)
                continue
            if status != last:
                self.note(run, f"kernel {status}")
                last = status
            if status in _FINAL:
                self._collect(run, client, _FINAL[status])
                return
            time.sleep(self.poll)

    def _collect(self, run: TrainingRun, client: KaggleClient, code: int) -> None:
        with tempfile.TemporaryDirectory() as dest:
            try:
                client.output(run.slug, dest)
            except Exception as exc:  # noqa: BLE001
                self.note(run, f"could not download the kernel's outputs: {exc}")
            produced = Path(dest) / "aime_run"
            log_tail = _kernel_log(Path(dest))
            if produced.is_dir():
                merge_folder(produced, run.workdir)
        events = run.workdir / "events.jsonl"
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                self.feed(run, line)
        exit_file = run.workdir / "exit_code"
        if exit_file.exists() and exit_file.read_text(encoding="utf-8").strip().lstrip("-").isdigit():
            code = int(exit_file.read_text(encoding="utf-8").strip())
        if run.held_done is None and code and log_tail:
            run.record({"type": "error", "traceback": log_tail})
        self.finish(run, code)

    def stop(self, run: TrainingRun) -> None:
        slug = getattr(run, "slug", "")
        try:
            cancelled = bool(slug) and self.client().cancel(slug)
        except Exception:  # noqa: BLE001
            cancelled = False
        if not cancelled and slug:
            self.note(run, f"stop the kernel on Kaggle: {run.remote_url}")
        run.process_exited(130)


def _kernel_log(folder: Path) -> str:
    """The tail of a downloaded kernel log (a JSON list of stream chunks)."""
    for path in folder.glob("*.log"):
        try:
            chunks = json.loads(path.read_text(encoding="utf-8"))
            text = "".join(c.get("data", "") for c in chunks if isinstance(c, dict))
        except ValueError:
            text = path.read_text(errors="replace", encoding="utf-8")
        return text[-3000:]
    return ""
