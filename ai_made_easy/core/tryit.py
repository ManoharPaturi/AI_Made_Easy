"""Try a finished run (3.5): its predictor is loaded once in a worker process, in the run's
own folder and interpreter, and answers predictions from both apps. Nothing leaves this
machine: no package is built and no server is started.

The worker imports the run's training script (``main()`` is not run), calls its
``load_predictor`` and then answers one JSON request per line with ``infer``. The two most
recently tried runs stay loaded.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

_MARK = "AIME-TRY "
_KEEP = 2

_WORKER = r'''
import importlib.util, json, sys
from pathlib import Path

run_dir, script = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(run_dir))
out = sys.stdout
sys.stdout = sys.stderr   # the script's own prints must not mix with the replies


def reply(obj):
    plain = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)  # noqa: E731
    out.write("AIME-TRY " + json.dumps(obj, default=plain) + "\n")
    out.flush()


try:
    spec = importlib.util.spec_from_file_location("model_def", script)
    model_def = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_def)
    model_def.load_predictor(run_dir, device="cpu")
except Exception as exc:  # noqa: BLE001
    reply({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
    sys.exit(1)
state = getattr(model_def, "INFERENCE_STATE", {}) or {}
reply({"ready": True, "signature": {
    "input_shape": list(getattr(model_def, "INPUT_SHAPE", ()) or ()),
    "classes": [str(c) for c in (getattr(model_def, "CLASS_NAMES", None) or [])] or None,
    "columns": ((state.get("table") or {}).get("columns")
                or getattr(model_def, "_COLUMNS", None)),
    "sample_rate": getattr(model_def, "SAMPLE_RATE", None),
}})
for line in sys.stdin:
    try:
        reply({"ok": True, "predictions": model_def.infer(json.loads(line))})
    except Exception as exc:  # noqa: BLE001 — reported to the app, the worker carries on
        reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
'''


class TryError(RuntimeError):
    pass


class _Worker:
    def __init__(self, run_dir: Path, script: Path, python: str, env: dict) -> None:
        self.proc = subprocess.Popen(
            [python, "-u", "-c", _WORKER, str(run_dir), str(script)], cwd=run_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env)
        self._replies: queue.Queue = queue.Queue()
        self._err: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        self.lock = threading.Lock()
        self.signature: dict = {}

    def _read(self) -> None:
        for line in self.proc.stdout:
            if line.startswith(_MARK):
                self._replies.put(json.loads(line[len(_MARK):]))
        self._replies.put(None)   # the worker exited

    def _drain(self) -> None:
        for line in self.proc.stderr:
            self._err = (self._err + [line])[-40:]

    def reply(self, timeout: float) -> dict:
        try:
            got = self._replies.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise TryError(f"the model did not answer within {timeout:.0f} s") from None
        if got is None:
            raise TryError("the model stopped:\n" + "".join(self._err)[-2000:])
        return got

    def ask(self, inputs: list, timeout: float) -> list:
        with self.lock:
            self.proc.stdin.write(json.dumps(inputs) + "\n")
            self.proc.stdin.flush()
            got = self.reply(timeout)
        if not got.get("ok"):
            raise TryError(got.get("error") or "the prediction failed")
        return got["predictions"]

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.alive():
            self.proc.kill()
        self.proc.wait(timeout=10)


class TryService:
    """Loaded predictors by run id (the most recent few)."""

    def __init__(self, keep: int = _KEEP) -> None:
        self._workers: OrderedDict[str, _Worker] = OrderedDict()
        self._keep = keep
        self._lock = threading.Lock()

    def worker(self, run_id: str, run_dir: Path, record: dict, *,
               load_timeout: float = 300) -> _Worker:
        from ai_made_easy.core.deploy.package import _artifacts
        from ai_made_easy.core.runner.manager import python_executable

        with self._lock:
            got = self._workers.get(run_id)
            if got is not None and got.alive():
                self._workers.move_to_end(run_id)
                return got
            if record.get("status") != "finished":
                raise TryError(f"the run is {record.get('status') or 'not finished'}; only "
                               "finished runs can be tried")
            framework = record.get("framework", "pytorch")
            try:
                script, _ = _artifacts(run_dir, framework)
            except Exception as exc:  # noqa: BLE001 — DeployError: what is missing
                raise TryError(str(exc)) from exc
            env = {**os.environ, "PYTHONWARNINGS": "ignore"}
            if framework == "keras":
                env["KERAS_BACKEND"] = (record.get("env") or {}).get("keras_backend") or "torch"
            worker = _Worker(run_dir, script, python_executable(), env)
            first = worker.reply(load_timeout)
            if not first.get("ready"):
                worker.close()
                raise TryError("the model could not be loaded: " + str(first.get("error")))
            worker.signature = first.get("signature") or {}
            self._workers[run_id] = worker
            while len(self._workers) > self._keep:
                _, old = self._workers.popitem(last=False)
                old.close()
            return worker

    def close(self) -> None:
        with self._lock:
            for worker in self._workers.values():
                worker.close()
            self._workers.clear()


def _target(service: TryService, history, run_id: str):  # noqa: ANN001, ANN202
    from ai_made_easy.core.deploy.package import _describe

    history.get(run_id)   # an unknown run raises here
    folder = history.path(run_id)
    record = json.loads((folder / "run.json").read_text())
    return service.worker(run_id, folder, record), _describe(record.get("graph") or {})


def describe(service: TryService, history, run_id: str) -> dict:  # noqa: ANN001
    """What a finished run takes: input kind and help, an example, its signature."""
    from ai_made_easy.core.deploy.package import _example

    worker, meta = _target(service, history, run_id)
    try:
        example = json.loads(_example(meta, worker.signature)).get("inputs")
    except ValueError:   # a placeholder such as "<base64 image>"
        example = None
    return {"run_id": run_id, "task": meta.get("task"), "modality": meta.get("modality"),
            "input_kind": meta["input_kind"], "input_help": meta.get("input_help", ""),
            "example": example, "signature": worker.signature}


def predict(service: TryService, history, run_id: str, inputs: list) -> dict:  # noqa: ANN001
    """The run's predictions for ``inputs`` (the same requests as its model server)."""
    import time

    if not isinstance(inputs, list) or not inputs:
        raise TryError("inputs must be a non-empty list")
    if len(inputs) > 64:
        raise TryError("try at most 64 inputs at a time")
    worker, meta = _target(service, history, run_id)
    kind = meta["input_kind"]
    if kind == "images":
        inputs = strip_data_urls(inputs)
    elif kind == "audio":
        inputs = wav_paths(inputs, temp_folder())
    started = time.perf_counter()
    predictions = worker.ask(inputs, timeout=120)
    return {"run_id": run_id, "input_kind": kind, "predictions": predictions,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def wav_paths(inputs: list, folder: Path) -> list:
    """Uploaded WAV files (base64, or a data: URL) become paths the audio predictor reads."""
    out = []
    for i, item in enumerate(inputs):
        if isinstance(item, str) and not Path(item).exists():
            data = item.split(",", 1)[1] if item.startswith("data:") else item
            path = folder / f"clip_{i}.wav"
            path.write_bytes(base64.b64decode(data))
            out.append(str(path))
        else:
            out.append(item)
    return out


def strip_data_urls(inputs: list) -> list:
    """Images from a browser come as data: URLs; the predictor takes the base64 part."""
    return [i.split(",", 1)[1] if isinstance(i, str) and i.startswith("data:") else i
            for i in inputs]


def temp_folder() -> Path:
    return Path(tempfile.mkdtemp(prefix="aime_try_"))
