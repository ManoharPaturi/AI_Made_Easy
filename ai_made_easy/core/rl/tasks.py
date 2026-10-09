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


def rl_task(graph) -> str | None:  # noqa: ANN001
    if env_node(graph) is not None or algorithm_node(graph) is not None:
        return "reinforcement_learning"
    return None


register_task_resolver(rl_task)
