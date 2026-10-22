"""Observation and action spaces of an environment, read at design time.

Built-in environments are described in ``ENVS``. Anything else (a registered Gymnasium
id, or an environment defined in your own module) is created once in a subprocess, so a
broken or slow environment never touches the app, and its spaces are cached. The design
rules then check the network against them, as for the built-ins.
"""
from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

PROBE = r"""
import importlib, importlib.util, json, sys
from pathlib import Path
env_id, module = sys.argv[1], sys.argv[2]
try:
    if module:
        path = Path(module).expanduser()
        if path.suffix == ".py":
            spec = importlib.util.spec_from_file_location(path.stem, path.resolve())
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(path.resolve().parent))
            spec.loader.exec_module(mod)
        else:
            importlib.import_module(module)
    import gymnasium as gym
    try:
        import ale_py
        gym.register_envs(ale_py)
    except ImportError:
        pass
    env = gym.make(env_id)
    obs, act = env.observation_space, env.action_space
    out = {"observation": list(getattr(obs, "shape", ()) or ()),
           "dtype": str(getattr(obs, "dtype", "")), "space": type(obs).__name__,
           "action": "discrete" if hasattr(act, "n") else "continuous",
           "actions": int(act.n) if hasattr(act, "n") else
                      int(__import__("math").prod(act.shape or (1,)))}
    env.close()
except Exception as exc:
    out = {"error": f"{type(exc).__name__}: {exc}"}
print("SPACES " + json.dumps(out))
"""


def _stamp(module: str) -> float:
    path = Path(module).expanduser()
    return path.stat().st_mtime if module and path.is_file() else 0.0


@lru_cache(maxsize=64)
def _probe(env_id: str, module: str, stamp: float, cwd: str) -> dict:
    from ai_made_easy.core.runner.manager import python_executable  # not the app bundle

    env = {**os.environ, "PYTHONWARNINGS": "ignore",
           "PYTHONPATH": os.pathsep.join(filter(None, [cwd, os.environ.get("PYTHONPATH", "")]))}
    try:
        proc = subprocess.run([python_executable(), "-c", PROBE, env_id, module], cwd=cwd,
                              capture_output=True, encoding="utf-8", errors="replace", timeout=120, env=env)
    except subprocess.TimeoutExpired:
        return {"error": "creating the environment took longer than 2 minutes"}
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("SPACES ")), "")
    if not line:
        return {"error": (proc.stderr.strip().splitlines() or ["the environment crashed"])[-1]}
    return json.loads(line[len("SPACES "):])


def network_shape(spaces: dict) -> list[int]:
    """What the policy network receives: images channels-first (stable-baselines3
    transposes [H, W, C] pictures), everything else as the environment gives it."""
    shape = list(spaces.get("observation") or [])
    if len(shape) == 3 and spaces.get("dtype") == "uint8" and shape[-1] in (1, 3, 4) \
            and shape[0] > 4:
        return [shape[2], shape[0], shape[1]]
    return shape


def env_spaces(env_id: str, module: str = "", cwd: str | None = None) -> dict:
    """``{"observation", "dtype", "action", "actions", "input"}`` or ``{"error"}``."""
    cwd = cwd or os.getcwd()
    found = dict(_probe(env_id, module or "", _stamp(module or ""), cwd))
    if "error" not in found:
        found["input"] = network_shape(found)
    return found
