# Reinforcement learning environments

The **Gym Environment** block chooses the world the agent acts in. The network on the
canvas is the policy's feature extractor: its Input is the observation, its Output the
features stable-baselines3's action and value heads read.

| Kind | Environments | Install |
| --- | --- | --- |
| Classic control | CartPole, Acrobot, MountainCar, Pendulum | `pip install 'ai-made-easy[rl]'` |
| Box2D | LunarLander, BipedalWalker, CarRacing | `pip install 'ai-made-easy[rl-box2d]'` |
| Pixels, built in | PixelChase (a starter for CNN policies) | the `rl` extra |
| Atari | Pong, Breakout, Space Invaders, Seaquest, Freeway | `pip install 'ai-made-easy[atari]'` |
| Your own | any Gymnasium environment | the `rl` extra |

## Atari

Atari games run through the Arcade Learning Environment (`ale-py`). Every game gets the
preprocessing from the DQN paper:

- **Atari Preprocessing:** frame skipping (the max of the last two frames), 84×84
  grayscale, and random no-op starts. Without the block, these defaults apply.
- **Frame Stack:** the last 4 frames become the channels, so the policy sees motion.

The observation is then `[frames, size, size]`; the default is `[4, 84, 84]`. The
**Atari PPO** recipe (`rl_atari_pong_ppo`) puts the Nature-DQN CNN in front of PPO. It needs
a GPU and a few million steps to win at Pong.

### ROMs

AI Made Easy never ships game ROMs. The design rule tells you when a game's ROM is
missing:

- **`ale-py` 0.9 and newer** include the ROMs of the supported games in their wheels,
  under the terms in ale-py's licence. Installing the `atari` extra is enough.
- **Older `ale-py`, or a build without ROMs:** install the ROMs you own with ale-py's
  import tool:

  ```bash
  ale-import-roms /path/to/your/roms
  ```

## Your own environments

Set **env_id** to `custom`, then:

- **custom_id:** the environment's registered id, for example `Corridor-v0`
- **module:** the Python file (`my_envs.py`) or module that registers it with
  `gymnasium.register(...)`

When you design, AI Made Easy creates the environment once in a separate process (so a
slow or broken environment never touches the app) and reads its spaces. The Problems
panel then checks the design against them, as for the built-in environments:

- the Input shape must match the observation (pictures arrive channels-first,
  `[C, H, W]`)
- DQN needs discrete actions; SAC and TD3 need continuous ones
- an environment that can't be created is reported with its error

Training scripts import the same module, so the run uses exactly the environment the
design was checked against. To train a custom environment remotely, give **module** as a
path (`./my_envs.py`) so it travels with the run.
