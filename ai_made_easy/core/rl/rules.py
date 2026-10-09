"""Design rules for reinforcement learning ("Set the Input shape to '…'" and "set <param>
to <value>" are Quick Fixes)."""
from __future__ import annotations

from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule
from ai_made_easy.core.rl.blocks import CONTINUOUS_ONLY, DISCRETE_ONLY, ENVS
from ai_made_easy.core.rl.tasks import algorithm_node, env_node, rl_task
from ai_made_easy.core.training import catalog as cat


def rl_rules(ctx: LintContext) -> list:
    if rl_task(ctx.graph) is None:
        return []
    out = []
    env, algo = env_node(ctx.graph), algorithm_node(ctx.graph)
    if env is None:
        out.append(_issue("info", "no Gym Environment block: training on CartPole-v1",
                          algo.instance_id if algo else None))
    if algo is None:
        out.append(_issue("info", "no algorithm block: training with PPO",
                          env.instance_id if env else None))
    ep = env.resolved_params() if env is not None else {"env_id": "CartPole-v1",
                                                        "custom_id": ""}
    if ep["env_id"] == "custom" and not str(ep["custom_id"]).strip():
        out.append(_issue("error", "set custom_id to a registered Gymnasium environment id",
                          env.instance_id))
    info = ENVS.get(ep["env_id"])
    head = ctx.chain[0] if ctx.chain else None
    last = ctx.last_compute()
    if info is not None and head is not None and head.instance_id in ctx.shapes:
        if list(ctx.shapes[head.instance_id]) != [info[0]]:
            out.append(_issue("error", f"{ep['env_id']} observations have {info[0]} values: "
                                       f"Set the Input shape to '{info[0]}'", head.instance_id))
    if last is not None and last.instance_id in ctx.shapes and \
            len(ctx.shapes[last.instance_id]) != 1:
        out.append(_issue("error", "the policy network must output a feature vector [F]: "
                                   "add Flatten", last.instance_id))
    if info is not None and algo is not None:
        if algo.type_id in DISCRETE_ONLY and info[1] != "discrete":
            out.append(_issue("error", f"{_name(algo)} chooses among discrete actions but "
                                       f"{ep['env_id']} has continuous actions: use PPO, SAC "
                                       "or TD3", algo.instance_id))
        if algo.type_id in CONTINUOUS_ONLY and info[1] != "continuous":
            out.append(_issue("error", f"{_name(algo)} outputs continuous actions but "
                                       f"{ep['env_id']} has {info[2]} discrete actions: use "
                                       "PPO, A2C or DQN", algo.instance_id))
    if last is not None and last.type_id in ("core.softmax", "core.log_softmax"):
        out.append(_issue("warning", "the network is a feature extractor: stable-baselines3 adds "
                                     "the action head (and its softmax); remove this layer",
                          last.instance_id))
    if algo is not None:
        p = algo.resolved_params()
        n_envs = int(env.resolved_params()["n_envs"]) if env is not None else 4
        if algo.type_id == "rl.ppo":
            rollout = int(p["n_steps"]) * n_envs
            if int(p["batch_size"]) > rollout:
                out.append(_issue("error", f"each update collects {rollout} steps: set "
                                           f"batch_size to {rollout}", algo.instance_id))
            elif rollout % int(p["batch_size"]):
                out.append(_issue("warning", f"{rollout} steps per update do not split into "
                                             f"minibatches of {p['batch_size']}: the last one "
                                             "is truncated", algo.instance_id))
        if int(p["total_timesteps"]) < int(p["evaluations"]):
            out.append(_issue("warning", "fewer timesteps than evaluations: set evaluations to "
                                         "1", algo.instance_id))
    for node in ctx.graph.nodes.values():
        if node.type_id in (*cat.LOSS_IDS, *cat.OPTIMIZER_IDS):
            out.append(_issue("info", f"{_name(node)} is ignored: the algorithm defines its own "
                                      "loss and optimiser", node.instance_id))
        elif node.type_id.startswith("data."):
            out.append(_issue("warning", f"{_name(node)} is ignored: the agent learns from the "
                                         "environment", node.instance_id))
    return out


register_rule(rl_rules)
