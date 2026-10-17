"""Design rules for reinforcement learning ("Set the Input shape to '…'" and "set <param>
to <value>" are Quick Fixes)."""
from __future__ import annotations

import importlib.util

from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule
from ai_made_easy.core.rl.blocks import (
    ATARI,
    ATARI_GAMES,
    BOX2D,
    CONTINUOUS_ONLY,
    DISCRETE_ONLY,
)
from ai_made_easy.core.rl.tasks import (
    algorithm_node,
    atari_config,
    env_node,
    observation_info,
    rl_task,
)
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
    out.extend(_atari_issues(ctx, env, ep))
    info = observation_info(ctx.graph)
    if info is not None and info[4]:
        out.append(_issue("error", f"could not create {ep.get('custom_id')}: {info[4]}",
                          env.instance_id))
        info = None
    image_obs = info is not None and len(info[0]) == 3
    head = ctx.chain[0] if ctx.chain else None
    last = ctx.last_compute()
    if info is not None and head is not None and head.instance_id in ctx.shapes:
        want = list(info[0])
        if list(ctx.shapes[head.instance_id]) != want:
            what = (f"are {want[0]}-channel {want[1]}×{want[2]} images" if len(want) == 3
                    else f"have {want[0]} values")
            name = ep.get("custom_id") if ep["env_id"] == "custom" else ep["env_id"]
            out.append(_issue("error", f"{name} observations {what}: Set the Input "
                                       f"shape to '{', '.join(map(str, want))}'",
                              head.instance_id))
    if ep["env_id"] in BOX2D and importlib.util.find_spec("Box2D") is None:
        out.append(_issue("warning", f"{ep['env_id']} needs Box2D, which is not installed "
                                     "(pip install 'ai-made-easy[rl-box2d]')", env.instance_id))
    if image_obs:
        if head is not None and not any(n.type_id.startswith(("core.conv", "core.depthwise",
                                                              "core.separable"))
                                        for n in ctx.chain):
            out.append(_issue("warning", f"{ep['env_id']} observes pixels: add Conv2D layers "
                                         "(a CNN policy) before Flatten, or the network must "
                                         "learn from every pixel separately",
                              head.instance_id))
        if algo is not None and "buffer_size" in algo.resolved_params():
            frame = int(__import__("math").prod(info[0]))
            size = int(algo.resolved_params()["buffer_size"]) * frame * 2 / 2**30
            if size > 2:
                fit = max(1000, int(2 * 2**30 / (2 * frame)) // 1000 * 1000)
                out.append(_issue("warning", f"the replay buffer would hold about {size:.0f} GB "
                                             f"of images: set buffer_size to {fit}",
                                  algo.instance_id))
    if last is not None and last.instance_id in ctx.shapes and \
            len(ctx.shapes[last.instance_id]) != 1:
        out.append(_issue("error", "the policy network must output a feature vector [F]: "
                                   "add Flatten", last.instance_id))
    if info is not None and algo is not None:
        name = ep.get("custom_id") if ep["env_id"] == "custom" else ep["env_id"]
        if algo.type_id in DISCRETE_ONLY and info[1] != "discrete":
            out.append(_issue("error", f"{_name(algo)} chooses among discrete actions but "
                                       f"{name} has continuous actions: use PPO, SAC "
                                       "or TD3", algo.instance_id))
        if algo.type_id in CONTINUOUS_ONLY and info[1] != "continuous":
            out.append(_issue("error", f"{_name(algo)} outputs continuous actions but "
                                       f"{name} has {info[2]} discrete actions: use "
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


def _rom_missing(rom: str) -> bool | None:
    """True / False when ale-py can tell, None when it can't (an older ale-py)."""
    try:
        from ale_py import roms
    except ImportError:
        return None
    get = getattr(roms, "get_rom_path", None)
    if get is None:
        return None
    try:
        return get(rom) is None
    except Exception:  # noqa: BLE001
        return True


def _atari_issues(ctx: LintContext, env, ep: dict) -> list:  # noqa: ANN001
    out = []
    wrappers = [n for n in ctx.graph.nodes.values()
                if n.type_id in ("rl.atari_preprocessing", "rl.frame_stack")]
    if ep["env_id"] not in ATARI:
        for node in wrappers:
            if node.type_id == "rl.atari_preprocessing":
                out.append(_issue("warning", f"{_name(node)} only applies to Atari games: "
                                             "it is ignored", node.instance_id))
        return out
    if importlib.util.find_spec("ale_py") is None:
        out.append(_issue("warning", f"{ep['env_id']} needs ale-py, which is not installed: "
                                     "pip install 'ai-made-easy[atari]' (then add the game "
                                     "ROMs if your ale-py has none: see docs/RL.md)",
                          env.instance_id))
    else:
        rom = ATARI_GAMES[ep["env_id"]][0]
        if _rom_missing(rom):
            out.append(_issue("error", f"the {rom} ROM is missing: AI Made Easy never ships "
                                       "ROMs; install the ones you own (see docs/RL.md)",
                              env.instance_id))
    if not any(n.type_id == "rl.atari_preprocessing" for n in wrappers):
        out.append(_issue("info", "Atari games train with the standard preprocessing "
                                  "(frame skip 4, 84×84 grayscale, 4 stacked frames): add "
                                  "Atari Preprocessing / Frame Stack blocks to change it",
                          env.instance_id))
    cfg = atari_config(ctx.graph)
    if cfg and cfg["stack"] == 1:
        out.append(_issue("warning", "one frame shows no motion: the policy can't tell which "
                                     "way the ball moves; stack 4 frames",
                          next((n.instance_id for n in wrappers
                                if n.type_id == "rl.frame_stack"), env.instance_id)))
    return out


register_rule(rl_rules)
