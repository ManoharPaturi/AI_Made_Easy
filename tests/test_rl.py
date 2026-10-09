"""Reinforcement learning: task, rules and Quick Fixes, generated stable-baselines3 scripts
that beat a random policy (discrete and continuous), serving and deploys."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_made_easy.core import api
from ai_made_easy.core.fixes import fix_for_issue
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.tasks import task_of

HAS_RL = all(importlib.util.find_spec(m) is not None
             for m in ("gymnasium", "stable_baselines3"))
needs_rl = pytest.mark.skipif(not HAS_RL, reason="needs gymnasium and stable-baselines3")
SAMPLES = ("rl_cartpole_ppo.json", "rl_pendulum_sac.json")


def n(nid: str, type_id: str, **params) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": [0, 0]}


def agent(obs: int, env: str, algo: str, params: dict | None = None,
          extra: list | None = None, last: str = "core.tanh") -> dict:
    return {"name": "agent", "nodes": [
        n("in", "core.input", shape=str(obs)), n("d", "core.dense", units=32), n("a", last),
        n("out", "core.output"), n("env", "rl.env", env_id=env, n_envs=4),
        n("algo", algo, **(params or {})), *(extra or [])],
        "edges": [{"from": "in/out", "to": "d/in"}, {"from": "d/out", "to": "a/in"},
                  {"from": "a/out", "to": "out/in"}]}


def messages(data: dict) -> list[str]:
    return [i.message for i in Graph.from_dict(data).validate()]


def test_task_and_samples():
    for sample in SAMPLES:
        graph = Graph.from_dict(api.read_sample(sample))
        assert task_of(graph).id == "reinforcement_learning"
        if HAS_RL:
            assert graph.validate() == []


def test_rules_and_fixes():
    wrong = agent(6, "CartPole-v1", "rl.ppo")
    g = Graph.from_dict(wrong)
    issue = next(i for i in g.validate() if "Set the Input shape to '4'" in i.message)
    assert fix_for_issue(g, issue)[2].nodes["in"].params["shape"] == "4"
    assert any("has continuous actions: use PPO, SAC" in m
               for m in messages(agent(3, "Pendulum-v1", "rl.dqn")))
    assert any("has 2 discrete actions: use PPO, A2C or DQN" in m
               for m in messages(agent(4, "CartPole-v1", "rl.sac")))
    big = agent(4, "CartPole-v1", "rl.ppo", {"n_steps": 16, "batch_size": 256})
    assert any("set batch_size to 64" in m for m in messages(big))
    soft = agent(4, "CartPole-v1", "rl.ppo", last="core.softmax")
    assert any("stable-baselines3 adds the action head" in m for m in messages(soft))
    ignored = agent(4, "CartPole-v1", "rl.a2c",
                    extra=[n("loss", "train.loss_mse"), n("data", "data.synthetic")])
    found = messages(ignored)
    assert any("defines its own loss" in m for m in found)
    assert any("learns from the environment" in m for m in found)
    custom = agent(4, "CartPole-v1", "rl.ppo")
    custom["nodes"][4]["params"].update(env_id="custom", custom_id="")
    assert any("set custom_id" in m for m in messages(custom))
    assert not any("no optimizer configured" in m
                   for m in messages(agent(4, "CartPole-v1", "rl.ppo",
                                           extra=[n("tr", "train.trainer")])))


def _run(data: dict, tmp_path: Path) -> tuple[Path, Path]:
    from ai_made_easy.core.codegen import export_training

    script = export_training(Graph.from_dict(data), "pytorch", tmp_path)
    proc = subprocess.run([sys.executable, script.name], cwd=tmp_path, capture_output=True,
                          text=True, timeout=1200)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    return tmp_path, script


def _infer(folder: Path, script: Path, items: list) -> list:
    code = (f"import sys, json; sys.path.insert(0, '.'); import {script.stem} as m; "
            f"m.load_predictor('.'); print('OUT ' + json.dumps(m.infer({items!r})))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True,
                          text=True, timeout=300, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert "OUT " in proc.stdout, proc.stderr[-3000:]
    return json.loads(proc.stdout.split("OUT ", 1)[1])


@needs_rl
@pytest.mark.parametrize("algo,params", [
    ("rl.ppo", {"total_timesteps": 30000, "n_steps": 256, "evaluations": 3}),
    ("rl.dqn", {"total_timesteps": 30000, "learning_rate": 0.0023, "train_freq": 256,
                "gradient_steps": 128, "target_update_interval": 10,
                "exploration_fraction": 0.16, "exploration_final_eps": 0.04,
                "evaluations": 3}),
], ids=["ppo", "dqn"])
def test_discrete_agents_beat_random(algo, params, tmp_path):
    run, script = _run(agent(4, "CartPole-v1", algo, params), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["mean_reward"] > 5 * metrics["random_reward"]          # random: ~20
    assert list((run / "samples").glob("epoch_*.png"))                   # live reward curve
    out = _infer(run, script, [[0.0, 0.0, 0.05, 0.0], {"observation": [0, 0, -0.05, 0]}])
    assert out[0]["action"] in (0, 1)
    key = "q_values" if algo == "rl.dqn" else "probabilities"
    assert len(out[1][key]) == 2


@needs_rl
def test_continuous_agent_swings_the_pendulum(tmp_path):
    data = api.read_sample(SAMPLES[1])
    for node in data["nodes"]:
        if node["type"] == "rl.sac":
            node["params"].update(total_timesteps=12000, evaluations=2)
    run, script = _run(data, tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["mean_reward"] > metrics["random_reward"] + 300        # random: ~-1100
    action = _infer(run, script, [[1.0, 0.0, 0.0]])[0]["action"]
    assert len(action) == 1 and -2.0 <= action[0] <= 2.0


@needs_rl
def test_agent_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    data = api.read_sample(SAMPLES[0])
    for node in data["nodes"]:
        if node["type"] == "rl.ppo":
            node["params"].update(total_timesteps=2048, evaluations=2)
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(data))
        status = mgr.wait(run_id, 900)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        assert mgr.history.epochs(run_id)                    # the reward curve
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["task"] == "reinforcement_learning" and meta["input_kind"] == "observations"
    finally:
        api.set_manager(None)
