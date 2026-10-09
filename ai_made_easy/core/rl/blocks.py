"""Reinforcement-learning blocks (Gymnasium + stable-baselines3): the environment and the
algorithms. The network on the canvas is the policy's feature extractor: its Input is
the observation, its Output the features stable-baselines3's action / value heads read."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition

CATEGORY = "Reinforcement Learning"
REQUIRES = ("gymnasium", "stable_baselines3")
EXTRA = "rl"
# env id -> (observation size, action kind, number of actions or action size, solved score,
#            description)
ENVS = {
    "CartPole-v1": (4, "discrete", 2, 475.0, "balance a pole on a cart (2 actions)"),
    "Acrobot-v1": (6, "discrete", 3, -100.0, "swing a two-link arm above a line (3 actions)"),
    "MountainCar-v0": (2, "discrete", 3, -110.0, "drive an underpowered car up a hill"),
    "Pendulum-v1": (3, "continuous", 1, -200.0, "swing a pendulum up and hold it (torque)"),
    "MountainCarContinuous-v0": (2, "continuous", 1, 90.0, "the hill climb with throttle"),
}
TRAIN_FREQ = P("train_freq", "int", 1, lo=1, hi=100_000, help="Update every n steps")
GRADIENT_STEPS = P("gradient_steps", "int", -1, lo=-1, hi=10_000,
                   help="Gradient steps per update (-1: one per environment step collected)")
ALGORITHMS = ("rl.ppo", "rl.a2c", "rl.dqn", "rl.sac", "rl.td3")
DISCRETE_ONLY = ("rl.dqn",)
CONTINUOUS_ONLY = ("rl.sac", "rl.td3")

COMMON = (P("total_timesteps", "int", 50_000, lo=100, hi=1_000_000_000,
            help="Environment steps to train for"),
          P("gamma", "float", 0.99, lo=0.0, hi=1.0, help="Discount of future rewards"),
          P("head", "str", "64", help="Layers stable-baselines3 adds after the designed "
                                      "network (e.g. 64, 64; empty: none)"),
          P("evaluations", "int", 10, lo=1, hi=1000, help="Evaluation points (reward curve)"),
          P("eval_episodes", "int", 10, lo=1, hi=1000))


def _algo(type_id: str, name: str, params: tuple, desc: str) -> BlockDefinition:
    return BlockDefinition(
        type_id=type_id, display_name=name, category=CATEGORY, color=family_color("model"),
        params=(*params, *COMMON), library="stable-baselines3", requires=REQUIRES, extra=EXTRA,
        description=desc, meta={"rl": "algorithm"})


def _blocks() -> list[BlockDefinition]:
    lr = lambda v: P("learning_rate", "float", v, lo=1e-7, hi=1.0)  # noqa: E731
    return [
        BlockDefinition(
            type_id="rl.env", display_name="Gym Environment", category=CATEGORY,
            color=family_color("data"), library="Gymnasium", requires=REQUIRES, extra=EXTRA,
            params=(P("env_id", "enum", "CartPole-v1", options=(*ENVS, "custom")),
                    P("custom_id", "str", "", help="Any registered Gymnasium id (env_id = "
                                                   "custom)"),
                    P("n_envs", "int", 4, lo=1, hi=256, help="Parallel copies collecting "
                                                            "experience"),
                    P("seed", "int", 0, lo=0)),
            description="The world the agent acts in: classic-control tasks with known "
                        "observation and action spaces, or any registered Gymnasium id.",
            meta={"rl": "env"}),
        _algo("rl.ppo", "PPO",
              (lr(3e-4), P("n_steps", "int", 1024, lo=8, hi=100_000,
                           help="Steps per environment per update"),
               P("batch_size", "int", 64, lo=2, hi=100_000), P("n_epochs", "int", 10, lo=1),
               P("gae_lambda", "float", 0.95, lo=0.0, hi=1.0),
               P("clip_range", "float", 0.2, lo=0.01, hi=1.0),
               P("ent_coef", "float", 0.0, lo=0.0, hi=1.0)),
              "Proximal policy optimisation: stable on-policy actor-critic; discrete or "
              "continuous actions."),
        _algo("rl.a2c", "A2C",
              (lr(7e-4), P("n_steps", "int", 5, lo=1, hi=100_000),
               P("ent_coef", "float", 0.0, lo=0.0, hi=1.0)),
              "Advantage actor-critic: synchronous, simple and fast on-policy learning."),
        _algo("rl.dqn", "DQN",
              (lr(1e-3), P("buffer_size", "int", 100_000, lo=100, hi=100_000_000),
               P("learning_starts", "int", 1000, lo=0), P("batch_size", "int", 64, lo=2),
               P("train_freq", "int", 4, lo=1), P("gradient_steps", "int", 1, lo=-1, hi=10_000),
               P("target_update_interval", "int", 1000, lo=1),
               P("exploration_fraction", "float", 0.1, lo=0.0, hi=1.0),
               P("exploration_final_eps", "float", 0.05, lo=0.0, hi=1.0)),
              "Deep Q-network: learns the value of each discrete action from a replay "
              "buffer (discrete actions only)."),
        _algo("rl.sac", "SAC",
              (lr(1e-3), P("buffer_size", "int", 100_000, lo=100, hi=100_000_000),
               P("learning_starts", "int", 100, lo=0), P("batch_size", "int", 256, lo=2),
               P("tau", "float", 0.005, lo=0.0, hi=1.0), TRAIN_FREQ, GRADIENT_STEPS),
              "Soft actor-critic: off-policy with entropy bonus, sample-efficient for "
              "continuous actions."),
        _algo("rl.td3", "TD3",
              (lr(1e-3), P("buffer_size", "int", 100_000, lo=100, hi=100_000_000),
               P("learning_starts", "int", 100, lo=0), P("batch_size", "int", 256, lo=2),
               P("tau", "float", 0.005, lo=0.0, hi=1.0),
               P("action_noise", "float", 0.1, lo=0.0, hi=2.0,
                 help="Std of the Gaussian exploration noise"), TRAIN_FREQ, GRADIENT_STEPS),
              "Twin delayed DDPG: deterministic continuous control with clipped double "
              "Q-learning."),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in _blocks():
        reg.register(defn)
