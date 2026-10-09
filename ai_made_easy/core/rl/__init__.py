"""Reinforcement learning: Gymnasium environments and stable-baselines3 algorithms (PPO,
A2C, DQN, SAC, TD3) whose policy feature extractor is the network on the canvas."""
from ai_made_easy.core.rl import blocks as _blocks

_blocks.register_all()

from ai_made_easy.core.rl import tasks  # noqa: E402, F401


def _register_trainer() -> None:
    from ai_made_easy.core.rl import template
    from ai_made_easy.core.training.generate import register_trainer

    register_trainer("rl", template.render, needs_spec=False)


_register_trainer()
from ai_made_easy.core.rl import rules as _rules  # noqa: E402, F401
