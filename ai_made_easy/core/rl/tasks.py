"""The reinforcement-learning task and its resolver."""
from __future__ import annotations

from ai_made_easy.core.rl.blocks import ALGORITHMS
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

register_task(Task(
    "reinforcement_learning", "Reinforcement learning",
    "Learn a policy that maximises reward by acting in an environment.",
    target="reward", output_role="tensor", modalities=("environment",), trainer_kind="rl",
    serving="actions",
    meta={"loss_tasks": (), "losses": [], "metrics": [], "default_loss": None,
          "default_optimizer": None}))


def env_node(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id == "rl.env"), None)


def algorithm_node(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id in ALGORITHMS), None)


def atari_config(graph) -> dict | None:  # noqa: ANN001
    """Atari wrappers for an ALE game (defaults when the blocks are missing), else None."""
    from ai_made_easy.core.rl.blocks import ATARI

    env = env_node(graph)
    if env is None or env.resolved_params()["env_id"] not in ATARI:
        return None
    pre = next((n.resolved_params() for n in graph.nodes.values()
                if n.type_id == "rl.atari_preprocessing"), {})
    stack = next((n.resolved_params() for n in graph.nodes.values()
                  if n.type_id == "rl.frame_stack"), {})
    return {"frame_skip": int(pre.get("frame_skip", 4)),
            "screen_size": int(pre.get("screen_size", 84)),
            "noop_max": int(pre.get("noop_max", 30)),
            "terminal_on_life_loss": bool(pre.get("terminal_on_life_loss", False)),
            "stack": int(stack.get("frames", 4))}


def observation_info(graph) -> tuple | None:  # noqa: ANN001
    """(network input shape, action kind, actions, solved score or None, error) for the
    design's environment: built-in table, Atari wrappers, or a custom environment's
    spaces read in a subprocess."""
    from ai_made_easy.core.rl.blocks import ENVS

    env = env_node(graph)
    p = env.resolved_params() if env is not None else {"env_id": "CartPole-v1"}
    if p["env_id"] != "custom":
        info = ENVS.get(p["env_id"])
        if info is None:
            return None
        atari = atari_config(graph)
        shape = (atari["stack"], atari["screen_size"], atari["screen_size"]) if atari \
            else info[0]
        return tuple(shape), info[1], info[2], info[3], ""
    if not str(p.get("custom_id") or "").strip():
        return None
    from ai_made_easy.core.rl.spaces import env_spaces

    found = env_spaces(str(p["custom_id"]).strip(), str(p.get("module") or "").strip())
    if "error" in found:
        return None, "", 0, None, found["error"]
    return tuple(found["input"]), found["action"], found["actions"], None, ""


def rl_task(graph) -> str | None:  # noqa: ANN001
    if env_node(graph) is not None or algorithm_node(graph) is not None:
        return "reinforcement_learning"
    return None


register_task_resolver(rl_task)
