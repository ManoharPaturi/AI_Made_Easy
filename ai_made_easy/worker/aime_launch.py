#!/usr/bin/env python3
"""Remote launcher: unpack a run bundle and train it, wherever this runs.

Usage: python aime_launch.py RUN_DIR [--bundle FILE] [--install] [--quiet]

Every remote backend (SSH, Kaggle, Colab, Modal, containers) runs this one file
on the far side. It needs only the Python standard library:

  1. unpacks the bundle (``--bundle``, or the copy embedded in this file) into RUN_DIR
  2. with ``--install``, pip-installs ``requirements.txt`` (names only, no pins, so
     packages the machine already has are kept)
  3. runs the training worker and writes its JSON events to ``RUN_DIR/events.jsonl``
     (stderr lines become log events); without ``--quiet`` it echoes them to stdout
  4. writes the exit code to ``RUN_DIR/exit_code`` (``pid`` holds the worker's pid
     while it runs, so a stop request can reach it)
"""
from __future__ import annotations

import base64
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading

PIP_TIMEOUT = 1800  # seconds
BUNDLE = ""  # base64 .tar.gz, filled in when the bundle travels inside this file
DEFAULT_ARGS: list = []  # used when the file runs without arguments (a Kaggle kernel)


def _extract(data: bytes, run_dir: str) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(run_dir, filter="data")
        else:  # Python < 3.12 without the backported filter
            root = os.path.realpath(run_dir)
            for member in tar.getmembers():
                target = os.path.realpath(os.path.join(run_dir, member.name))
                if not target.startswith(root) or member.issym() or member.islnk():
                    raise SystemExit(f"unsafe path in bundle: {member.name}")
            tar.extractall(run_dir)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = os.path.abspath(argv[0])
    os.makedirs(run_dir, exist_ok=True)
    quiet = "--quiet" in argv
    if "--bundle" in argv:
        with open(argv[argv.index("--bundle") + 1], "rb") as fh:
            _extract(fh.read(), run_dir)
    elif BUNDLE:
        _extract(base64.b64decode(BUNDLE), run_dir)
    with open(os.path.join(run_dir, "bundle.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    # a fresh start (a notebook run twice) begins a fresh event log
    for stale in ("exit_code", "pid"):
        if os.path.exists(os.path.join(run_dir, stale)):
            os.remove(os.path.join(run_dir, stale))
    events = open(os.path.join(run_dir, "events.jsonl"), "w", buffering=1, encoding="utf-8")
    lock = threading.Lock()

    def write(line: str) -> None:
        with lock:
            events.write(line + "\n")
            if not quiet:
                print(line, flush=True)

    reqs = os.path.join(run_dir, "requirements.txt")
    if "--install" in argv and os.path.exists(reqs) and os.path.getsize(reqs):
        write(json.dumps({"type": "log", "line": "installing requirements …"}))
        try:
            pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", reqs],
                                 capture_output=True, text=True, timeout=PIP_TIMEOUT)
            failed = pip.stderr[-2000:] if pip.returncode else ""
        except subprocess.TimeoutExpired:
            failed = f"pip install took longer than {PIP_TIMEOUT // 60} minutes; training anyway"
        if failed:
            write(json.dumps({"type": "log", "line": failed, "stream": "stderr"}))

    proc = subprocess.Popen([sys.executable, "-u", meta["worker"], meta["script"]],
                            cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    with open(os.path.join(run_dir, "pid"), "w", encoding="utf-8") as fh:
        fh.write(str(proc.pid))

    def stop(signum, frame):  # noqa: ANN001, ARG001
        proc.terminate()

    signal.signal(signal.SIGTERM, stop)

    def pump_stderr() -> None:
        for line in proc.stderr:
            write(json.dumps({"type": "log", "line": line.rstrip(), "stream": "stderr"}))

    err = threading.Thread(target=pump_stderr, daemon=True)
    err.start()
    for line in proc.stdout:
        if line.strip():
            write(line.rstrip())
    code = proc.wait()
    err.join(timeout=5)
    events.close()
    with open(os.path.join(run_dir, "exit_code"), "w", encoding="utf-8") as fh:
        fh.write(str(code))
    return code


def run(argv: list[str]) -> int:
    """``main`` that always leaves an exit code (and the reason) for the follower."""
    try:
        return main(argv)
    except BaseException as exc:  # noqa: BLE001 — report everything, then exit non-zero
        import traceback

        if not argv:
            raise
        run_dir = os.path.abspath(argv[0])
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "events.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "error", "traceback": traceback.format_exc()}) + "\n")
        with open(os.path.join(run_dir, "exit_code"), "w", encoding="utf-8") as fh:
            fh.write(str(exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int)
                         else 1))
        return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:] or DEFAULT_ARGS))
