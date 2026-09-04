#!/usr/bin/env python3
"""Training worker: run a generated AI Made Easy script and stream
line-delimited JSON events on stdout.

Usage: python train_worker.py <training_script.py>

Emitted events (one JSON object per line, flushed immediately):
  {"type": "log",   "line": "..."}
  {"type": "epoch", "epoch": 3, "total": 10, "metrics": {"train_loss": ...}}
  {"type": "error", "traceback": "..."}
  {"type": "done",  "returncode": 0}

The generated scripts stay clean and standalone — this wrapper owns all
IPC. Stderr passes through untouched (library warnings, tqdm, etc.).
"""
from __future__ import annotations

import io
import json
import re
import runpy
import sys
import traceback

_EPOCH_RE = re.compile(r"^epoch (\d+)/(\d+)\s+(.*)$")
_KV_RE = re.compile(r"([A-Za-z_]\w*)=(-?[\d.]+(?:[eE][+-]?\d+)?)")


def _emit(obj: dict, sink) -> None:
    print(json.dumps(obj), file=sink, flush=True)


class LineTap(io.TextIOBase):
    """Parses the training script's stdout line by line, re-emitting JSON."""

    def __init__(self, sink) -> None:
        self._sink = sink
        self._buf = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle(line)
        return len(s)

    def flush(self) -> None:  # tolerate flushes mid-line
        pass

    def _handle(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        m = _EPOCH_RE.match(line)
        if m:
            metrics = {k: float(v) for k, v in _KV_RE.findall(m.group(3))}
            _emit(
                {
                    "type": "epoch",
                    "epoch": int(m.group(1)),
                    "total": int(m.group(2)),
                    "metrics": metrics,
                },
                self._sink,
            )
        else:
            _emit({"type": "log", "line": line}, self._sink)


def main() -> int:
    if len(sys.argv) != 2:
        _emit({"type": "error", "traceback": "usage: train_worker.py <script>"}, sys.stdout)
        _emit({"type": "done", "returncode": 2}, sys.stdout)
        return 2
    script = sys.argv[1]
    real_stdout = sys.stdout
    sys.stdout = LineTap(real_stdout)
    try:
        runpy.run_path(script, run_name="__main__")
        _emit({"type": "done", "returncode": 0}, real_stdout)
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        _emit({"type": "done", "returncode": code}, real_stdout)
        return code
    except BaseException:
        _emit({"type": "error", "traceback": traceback.format_exc()}, real_stdout)
        _emit({"type": "done", "returncode": 1}, real_stdout)
        return 1
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    sys.exit(main())
