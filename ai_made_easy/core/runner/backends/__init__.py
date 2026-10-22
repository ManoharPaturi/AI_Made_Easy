"""Where a run trains: this computer, an SSH host, Kaggle, Colab, Modal or a container.

Every backend speaks the same worker events (``epoch`` / ``log`` / ``done`` …) into the
:class:`~ai_made_easy.core.runner.manager.TrainingRun`, so live curves, stop, sweeps
and pipelines work the same everywhere.

**Credentials:** a backend never asks for, stores or logs a password, token or key. It
uses each tool's own sign-in: ``~/.ssh`` (config, keys, agent), ``~/.kaggle/kaggle.json``,
``modal token``, Google Drive for desktop. Saved backends (``$AIME_HOME/backends.json``)
hold only names and options such as a host alias or a GPU type.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_made_easy.core.paths import aime_home
from ai_made_easy.core.runner.protocol import parse_event, worker_script_path
from ai_made_easy.core.runs.history import FINAL_STATES

if TYPE_CHECKING:
    from ai_made_easy.core.runner.manager import TrainingRun

KINDS = ("local", "ssh", "kaggle", "colab", "modal", "container")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


class BackendError(RuntimeError):
    """A backend cannot be used (missing tool or package, bad option, no sign-in)."""


# GPU names (nvidia-smi, Kaggle, Modal) -> device profiles in devices.json
_GPU_DEVICES = (("h100", "h100"), ("a100", "a100_40"), ("a10g", "a10g"), ("l4", "l4"),
                ("t4", "t4"), ("p100", "p100"), ("4090", "rtx_4090"), ("3060", "rtx_3060"))


def device_for_gpu(name: str, memory_gb: float | None = None) -> str:
    """The device profile for a GPU name ("" when there is none)."""
    low = name.lower().replace(" ", "")
    for key, device in _GPU_DEVICES:
        if key in low:
            if device == "a100_40" and ((memory_gb or 0) > 50 or "80gb" in low):
                return "a100_80"
            return device
    return ""


class Backend:
    """Base class. ``launch`` must eventually finish the run (a ``done`` event or
    ``run.process_exited``); ``stop`` asks it to end early."""

    kind = "local"
    remote = False   # True: the run travels as a bundle (see bundle.py)
    live = True      # False: curves arrive per poll / at the end, not as they happen
    options_help: dict[str, str] = {}

    def __init__(self, name: str = "", **options: Any) -> None:
        self.name = name or self.kind
        self.options = options

    @property
    def spec(self) -> str:
        """How a run record refers to this backend (``--on`` syntax)."""
        return self.name

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "remote": self.remote,
                "live": self.live, "options": dict(self.options)}

    def accelerator(self) -> dict:
        """``{"gpu": name, "device": profile id}`` when known without connecting."""
        return {}

    def test(self) -> dict:
        """Check the backend is usable: ``{"ok", "message", "accelerator"}``."""
        return {"ok": True, "message": "ready", "accelerator": self.accelerator()}

    def launch(self, run: TrainingRun, python: str) -> None:
        raise NotImplementedError

    def stop(self, run: TrainingRun) -> None:
        raise NotImplementedError

    def reattach(self, run: TrainingRun) -> None:
        raise BackendError(f"{self.kind} runs cannot be followed again after the app exits")

    def remote_state(self, run: TrainingRun) -> dict:
        """What ``reattach`` needs after a restart (saved in the run record)."""
        return {}


class LocalBackend(Backend):
    """This computer: the worker runs as a child process (the 3.1 behaviour)."""

    def launch(self, run: TrainingRun, python: str) -> None:
        proc = subprocess.Popen(
            [python, str(worker_script_path()), str(run.script_path)],
            cwd=str(run.workdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        run.process = proc
        if run.history is not None:
            run.history.update(run.run_id, pid=proc.pid)

        def pump_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                run.record({"type": "log", "line": line.rstrip(), "stream": "stderr"})

        err_thread = threading.Thread(target=pump_stderr, daemon=True)

        def pump_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                event = parse_event(line)
                if event is not None:
                    run.record(event)
            code = proc.wait()
            err_thread.join(timeout=5)
            run.process_exited(code)

        err_thread.start()
        threading.Thread(target=pump_stdout, daemon=True).start()

    def stop(self, run: TrainingRun) -> None:
        proc = run.process
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def accelerator(self) -> dict:
        return {}


class RemoteBackend(Backend):
    """Shared plumbing for backends that train somewhere else from a bundle."""

    remote = True

    def feed(self, run: TrainingRun, line: str) -> None:
        """One events.jsonl line from the far side. The ``done`` event waits until
        the outputs are downloaded, so the run record finalises with its metrics."""
        run.remote_offset += 1
        event = parse_event(line)
        if event is None:
            return
        if event.get("type") == "done":
            run.held_done = event
            return
        run.record(event)

    def finish(self, run: TrainingRun, code: int) -> None:
        if run.held_done is not None and run.state not in FINAL_STATES:
            run.record(run.held_done)
        else:
            run.process_exited(code)

    def note(self, run: TrainingRun, text: str) -> None:
        run.record({"type": "log", "line": f"[{self.name}] {text}"})

    def check_budget(self, run: TrainingRun) -> None:
        """Once the accelerator is known, check the design's budget against it."""
        device = (run.accelerator or {}).get("device")
        if not device or run.graph is None:
            return
        for message in budget_warnings(run.graph, device):
            self.note(run, f"warning: {message}")


def budget_warnings(graph, device: str) -> list[str]:  # noqa: ANN001 — Graph
    """The design's budget checks on ``device`` (its memory, plus the project's limits)
    that the estimate goes over."""
    from ai_made_easy.core import budget
    from ai_made_easy.core.graph import Graph

    try:
        copy = Graph.from_dict(graph.to_dict())
        budget.set_budget(copy, device=device)
        label = budget.get_device(device).label
        checks = budget.check(copy)
    except Exception:  # noqa: BLE001 — an estimate must never stop a run
        return []
    names = {"train_memory": "training memory", "latency": "latency", "params": "parameters",
             "model_size": "model size"}
    return [f"estimated {names.get(c.kind, c.kind)} {c.used:.3g} {c.unit} is over "
            f"{c.limit:.3g} {c.unit} on {label}" for c in checks if c.over]


def _backend_classes() -> dict[str, type[Backend]]:
    from ai_made_easy.core.runner.backends.colab import ColabBackend
    from ai_made_easy.core.runner.backends.container import ContainerBackend
    from ai_made_easy.core.runner.backends.kaggle import KaggleBackend
    from ai_made_easy.core.runner.backends.modal_backend import ModalBackend
    from ai_made_easy.core.runner.backends.ssh import SSHBackend

    return {"local": LocalBackend, "ssh": SSHBackend, "kaggle": KaggleBackend,
            "colab": ColabBackend, "modal": ModalBackend, "container": ContainerBackend}


def create(kind: str, name: str = "", **options: Any) -> Backend:
    classes = _backend_classes()
    if kind not in classes:
        raise BackendError(f"unknown backend kind {kind!r}; one of {', '.join(KINDS)}")
    cls = classes[kind]
    unknown = sorted(set(options) - set(cls.options_help))
    if unknown:
        raise BackendError(f"{kind} backends have no option {', '.join(unknown)}; "
                           f"options: {', '.join(cls.options_help) or 'none'}")
    return cls(name, **options)


# ---------------------------------------------------------------- saved backends

def _store_path() -> Path:
    return aime_home() / "backends.json"


def saved() -> list[dict]:
    path = _store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []
    return [b for b in data if isinstance(b, dict) and "name" in b and "kind" in b]


def save(name: str, kind: str, **options: Any) -> dict:
    """Add or replace a saved backend (options only, never secrets)."""
    if not _NAME_RE.match(name) or name in KINDS:
        raise BackendError(f"backend names use letters, digits, . _ @ - and cannot be "
                           f"one of {', '.join(KINDS)}")
    create(kind, name, **options)  # validates kind and options
    entries = [b for b in saved() if b["name"] != name]
    entry = {"name": name, "kind": kind, **options}
    entries.append(entry)
    _store_path().parent.mkdir(parents=True, exist_ok=True)
    _store_path().write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return entry


def remove(name: str) -> None:
    entries = saved()
    if not any(b["name"] == name for b in entries):
        raise BackendError(f"no saved backend {name!r}")
    _store_path().write_text(json.dumps([b for b in entries if b["name"] != name], indent=1), encoding="utf-8")


def listing() -> list[dict]:
    """Every backend a run can use: this computer, the saved ones, and the
    built-in kinds with their defaults."""
    rows = [LocalBackend().describe()]
    for entry in saved():
        options = {k: v for k, v in entry.items() if k not in ("name", "kind")}
        try:
            rows.append(create(entry["kind"], entry["name"], **options).describe())
        except BackendError:
            continue
    names = {r["name"] for r in rows}
    for kind in ("kaggle", "colab", "modal", "container"):
        if kind not in names:
            rows.append(create(kind).describe())
    return rows


def resolve(on: str | None) -> Backend:
    """``None`` / ``local``, ``ssh:<host>``, a kind (``kaggle``, ``colab``, ``modal``,
    ``container``) or the name of a saved backend."""
    if not on or on == "local":
        return LocalBackend()
    for entry in saved():
        if entry["name"] == on:
            options = {k: v for k, v in entry.items() if k not in ("name", "kind")}
            return create(entry["kind"], entry["name"], **options)
    if on.startswith("ssh:"):
        host = on[4:]
        if not host:
            raise BackendError("ssh needs a host: --on ssh:<host from ~/.ssh/config>")
        return create("ssh", on, host=host)
    if on in KINDS:
        return create(on)
    raise BackendError(f"unknown backend {on!r}: use local, ssh:<host>, kaggle, colab, "
                       "modal, container or a saved backend's name")
