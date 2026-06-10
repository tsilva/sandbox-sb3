from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import gymnasium as gym
import numpy as np
import stable_retro as retro
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage

GAME = "SuperMarioBros-Nes-v0"
DEFAULT_STATE = "Level1-1"

# stable-retro button order for NES:
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
BUTTON_B = 0
BUTTON_LEFT = 6
BUTTON_RIGHT = 7
BUTTON_A = 8


def _button_mask(*buttons: int) -> np.ndarray:
    mask = np.zeros(9, dtype=np.int8)
    for button in buttons:
        mask[button] = 1
    return mask


MARIO_ACTIONS: tuple[np.ndarray, ...] = (
    _button_mask(),
    _button_mask(BUTTON_RIGHT),
    _button_mask(BUTTON_RIGHT, BUTTON_B),
    _button_mask(BUTTON_RIGHT, BUTTON_A),
    _button_mask(BUTTON_RIGHT, BUTTON_A, BUTTON_B),
    _button_mask(BUTTON_A),
    _button_mask(BUTTON_LEFT),
)

ACTION_NAMES = (
    "noop",
    "right",
    "right_b",
    "right_a",
    "right_a_b",
    "a",
    "left",
)


@dataclass(frozen=True)
class EnvConfig:
    game: str = GAME
    state: str = DEFAULT_STATE
    frame_skip: int = 4
    max_episode_steps: int = 4500
    observation_size: int = 84
    clip_rewards: bool = False
    progress_reward_scale: float = 0.0
    death_penalty: float = 0.0


class DiscreteMarioActions(gym.ActionWrapper):
    """Map a small discrete action set to stable-retro's NES MultiBinary controls."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(MARIO_ACTIONS))

    def action(self, action: int) -> np.ndarray:
        return MARIO_ACTIONS[int(action)].copy()


class FrameSkip(gym.Wrapper):
    """Repeat one action for several emulator frames and sum reward."""

    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        if skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.skip = skip

    def step(self, action: Any):
        total_reward = 0.0
        final_obs = None
        final_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for _ in range(self.skip):
            final_obs, reward, terminated, truncated, final_info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return final_obs, total_reward, terminated, truncated, final_info


class MarioProgressInfo(gym.Wrapper):
    """Add stable x-position metrics and optional shaping on top of Retro rewards."""

    def __init__(self, env: gym.Env, progress_reward_scale: float = 0.0, death_penalty: float = 0.0):
        super().__init__(env)
        self.progress_reward_scale = progress_reward_scale
        self.death_penalty = death_penalty
        self.prev_x_pos = 0
        self.max_x_pos = 0
        self.prev_lives: int | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x_pos = 0
        self.max_x_pos = 0
        self.prev_lives = None
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = int(info.get("xscrollHi", 0)) * 256 + int(info.get("xscrollLo", 0))
        lives = info.get("lives")
        progress_delta = max(0, x_pos - self.prev_x_pos)
        self.max_x_pos = max(self.max_x_pos, x_pos)
        self.prev_x_pos = x_pos

        shaped_reward = float(reward) + self.progress_reward_scale * progress_delta
        if self.prev_lives is not None and lives is not None and int(lives) < self.prev_lives:
            shaped_reward -= self.death_penalty
        if lives is not None:
            self.prev_lives = int(lives)

        info = dict(info)
        info["x_pos"] = x_pos
        info["max_x_pos"] = self.max_x_pos
        info["progress_delta"] = progress_delta
        return obs, shaped_reward, terminated, truncated, info


class MarioPreprocess(gym.ObservationWrapper):
    """Convert RGB frames to 84x84 grayscale channel-last observations."""

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 1),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        gray = np.dot(observation[..., :3], np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        y_idx = np.linspace(0, gray.shape[0] - 1, self.size).astype(np.int32)
        x_idx = np.linspace(0, gray.shape[1] - 1, self.size).astype(np.int32)
        resized = gray[np.ix_(y_idx, x_idx)]
        return resized[..., None]


def make_mario_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = config or EnvConfig()
    env = retro.make(config.game, state=config.state, render_mode="rgb_array")
    return wrap_mario_env(env, config=config, seed=seed)


def make_rendered_mario_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = config or EnvConfig()
    env = retro.make(config.game, state=config.state, render_mode="human")
    return wrap_mario_env(env, config=config, seed=seed)


def wrap_mario_env(env: gym.Env, config: EnvConfig, seed: int | None = None) -> gym.Env:
    env = DiscreteMarioActions(env)
    env = FrameSkip(env, config.frame_skip)
    env = MarioProgressInfo(
        env,
        progress_reward_scale=config.progress_reward_scale,
        death_penalty=config.death_penalty,
    )
    env = MarioPreprocess(env, config.observation_size)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    if config.clip_rewards:
        env = ClipRewardEnv(env)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_env_fn(rank: int, seed: int, config: EnvConfig) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = make_mario_env(config=config, seed=seed + rank)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_envs(config: EnvConfig, n_envs: int, seed: int, start_method: str = "fork"):
    env_fns = [make_env_fn(rank, seed, config) for rank in range(n_envs)]
    # stable-retro permits only one emulator instance per process. Keep every
    # VecEnv worker in its own process, including the n_envs=1 eval case.
    vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env = VecMonitor(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="last")
    return VecTransposeImage(vec_env)


def assert_rom_imported(game: str = GAME) -> str:
    try:
        return retro.data.get_romfile_path(game)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{game} is not imported. Run: uv run python scripts/import_roms.py ~/Desktop/roms",
        ) from exc


def default_run_dir(run_name: str) -> str:
    return os.path.join("runs", run_name)
