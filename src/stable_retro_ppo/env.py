from __future__ import annotations

# ruff: noqa: E402

import os
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import gymnasium as gym
import cv2
import numpy as np
import stable_retro as retro
from stable_retro import StableRetroNativeVecEnv
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecTransposeImage

from stable_retro_ppo.targets import GenericRetroTarget, target_for_game

GAME = os.environ.get("RETRO_GAME", "")
DEFAULT_STATE = os.environ.get("RETRO_STATE", "")
DEFAULT_OBS_RESIZE_ALGORITHM = "area"
DEFAULT_HUD_CROP_TOP = GenericRetroTarget.default_hud_crop_top
DEFAULT_COMPLETION_X_THRESHOLD = GenericRetroTarget.default_completion_x_threshold
FRAME_STACK_CHANNELS = {1, 3, 4}


def native_vec_env_supports_life_loss() -> bool:
    try:
        source = inspect.getsource(StableRetroNativeVecEnv.__init__)
    except (OSError, TypeError):
        return False
    return "terminate_on_life_loss" in source and "life_variable" in source


def action_names_for_set(action_set: str, game: str = GAME) -> tuple[str, ...]:
    return target_for_game(game).action_names_for_set(action_set)


@dataclass(frozen=True)
class EnvConfig:
    game: str = GAME
    state: str = DEFAULT_STATE
    states: tuple[str, ...] = ()
    frame_skip: int = 4
    max_pool_frames: bool = True
    max_episode_steps: int = 4500
    observation_size: int = 84
    hud_crop_top: int = -1
    obs_resize_algorithm: str = DEFAULT_OBS_RESIZE_ALGORITHM
    use_retro_reward: bool = False
    clip_rewards: bool = False
    reward_mode: str = "auto"
    progress_reward_cap: float = 30.0
    progress_reward_scale: float = 1.0
    terminal_reward: float = 50.0
    reward_scale: float = 10.0
    time_penalty: float = 0.0
    death_penalty: float = 25.0
    completion_reward: float = 0.0
    score_progress_clipped: bool = False
    no_progress_timeout_steps: int = 0
    no_progress_min_delta: int = 0
    completion_x_threshold: int = -1
    terminate_on_life_loss: bool | None = None
    terminate_on_level_change: bool = False
    terminate_on_completion: bool = False
    action_set: str = "auto"
    env_threads: int = 0


def resolve_env_config(config: EnvConfig) -> EnvConfig:
    if not config.game:
        raise ValueError("game is required; pass --game or set RETRO_GAME")
    target = target_for_game(config.game)
    updates: dict[str, Any] = {}
    if not config.state and target.default_state:
        updates["state"] = target.default_state
    if config.action_set == "auto":
        updates["action_set"] = target.default_action_set
    elif config.action_set not in target.action_sets and not target.action_sets:
        updates["action_set"] = target.default_action_set
    if config.reward_mode == "auto":
        updates["reward_mode"] = target.default_reward_mode
    if config.hud_crop_top < 0:
        updates["hud_crop_top"] = target.default_hud_crop_top
    if config.completion_x_threshold < 0:
        updates["completion_x_threshold"] = target.default_completion_x_threshold
    if config.terminate_on_life_loss is None:
        updates["terminate_on_life_loss"] = target.default_terminate_on_life_loss
    return replace(config, **updates) if updates else config


def retro_make_kwargs(config: EnvConfig) -> dict[str, Any]:
    return {"state": config.state} if config.state else {}


class DiscreteRetroActions(gym.ActionWrapper):
    """Map a target-specific discrete action set to stable-retro controls."""

    def __init__(self, env: gym.Env, config: EnvConfig):
        super().__init__(env)
        target = target_for_game(config.game)
        self.action_names = target.action_names_for_set(config.action_set)
        self.actions = target.action_masks_for_set(config.action_set)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action: int) -> np.ndarray:
        return self.actions[int(action)].copy()


class VecDiscreteRetroActions(VecEnvWrapper):
    """Map target-specific discrete SB3 actions to stable-retro controls."""

    def __init__(self, venv, config: EnvConfig):
        target = target_for_game(config.game)
        self.action_names = target.action_names_for_set(config.action_set)
        self.actions = np.stack(target.action_masks_for_set(config.action_set)).astype(np.int8)
        super().__init__(
            venv,
            observation_space=venv.observation_space,
            action_space=gym.spaces.Discrete(len(self.actions)),
        )

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        action_indices = np.asarray(actions, dtype=np.int64).reshape(-1)
        self.venv.step_async(self.actions[action_indices])

    def step_wait(self):
        return self.venv.step_wait()


class VecRetroProgressInfo(VecEnvWrapper):
    """Vectorized target reward shaping and progress metrics.

    Image preprocessing, frame skip, frame stacking, and max-pooling stay inside
    StableRetroNativeVecEnv. This wrapper only rewrites rewards and annotates info.
    """

    def __init__(self, venv, config: EnvConfig):
        super().__init__(venv)
        self.config = config
        target = target_for_game(config.game)
        self.trackers = [target.create_tracker(config) for _ in range(self.num_envs)]

    def reset(self):
        obs = self.venv.reset()
        self._reset_tracking(range(self.num_envs), getattr(self.venv, "reset_infos", None))
        return obs

    def _reset_tracking(self, indices, infos=None) -> None:
        infos = infos or [{} for _ in range(self.num_envs)]
        for index in indices:
            info = infos[index] if index < len(infos) else {}
            self.trackers[index].reset(info)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        infos = [dict(info) for info in infos]
        shaped_rewards = np.zeros(self.num_envs, dtype=np.float32)
        custom_dones = np.zeros(self.num_envs, dtype=bool)

        for index, info in enumerate(infos):
            progress = self.trackers[index].step(rewards[index], info, dones[index])
            shaped_rewards[index] = progress.reward
            if progress.done:
                custom_dones[index] = True
                if progress.terminal:
                    info["_custom_terminal"] = True
                if progress.truncated:
                    info["_custom_truncated"] = True
                    info["TimeLimit.truncated"] = True

        if self.config.clip_rewards:
            shaped_rewards = np.sign(shaped_rewards).astype(np.float32)

        dones = np.logical_or(dones, custom_dones)
        native_done_indices = [
            idx for idx, done in enumerate(dones) if done and not custom_dones[idx]
        ]
        if native_done_indices:
            self._reset_tracking(native_done_indices)

        if custom_dones.any():
            terminal_obs = np.asarray(obs).copy()
            for index, info in enumerate(infos):
                info.setdefault("terminal_observation", terminal_obs[index])
                if custom_dones[index]:
                    if info.pop("_custom_terminal", False):
                        info["TimeLimit.truncated"] = False
                    elif info.pop("_custom_truncated", False):
                        info["TimeLimit.truncated"] = True
                    else:
                        info.setdefault("TimeLimit.truncated", False)
                else:
                    # StableRetroNativeVecEnv does not expose per-env reset yet.
                    # If one Python-defined terminal condition fires, all slots
                    # must be reset together. Non-terminal slots are treated as
                    # time-limit-style truncations so SB3 bootstraps from the
                    # saved terminal_observation instead of cutting value flow.
                    info["global_reset"] = True
                    info["TimeLimit.truncated"] = True
            obs = self.venv.reset()
            dones[:] = True
            self._reset_tracking(range(self.num_envs), getattr(self.venv, "reset_infos", None))

        return obs, shaped_rewards, dones, infos


class FrameSkip(gym.Wrapper):
    """Repeat one action for several emulator frames and sum reward."""

    def __init__(self, env: gym.Env, skip: int, max_pool: bool = False):
        super().__init__(env)
        if skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.skip = skip
        self.max_pool = max_pool

    def step(self, action: Any):
        total_reward = 0.0
        final_obs = None
        pooled_obs: list[np.ndarray] = []
        final_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for step_idx in range(self.skip):
            final_obs, reward, terminated, truncated, final_info = self.env.step(action)
            total_reward += float(reward)
            if self.max_pool and step_idx >= self.skip - 2 and final_obs is not None:
                pooled_obs.append(final_obs)
            if terminated or truncated:
                break
        if self.max_pool and pooled_obs:
            final_obs = np.maximum.reduce(pooled_obs)
        return final_obs, total_reward, terminated, truncated, final_info


class RetroProgressInfo(gym.Wrapper):
    """Apply target-specific reward shaping and progress metrics."""

    def __init__(self, env: gym.Env, config: EnvConfig):
        super().__init__(env)
        if config.reward_mode not in {"baseline", "bounded", "additive", "score", "native"}:
            raise ValueError(
                "reward_mode must be 'baseline', 'bounded', 'additive', 'score', or 'native'"
            )
        if config.progress_reward_cap < 0:
            raise ValueError("progress_reward_cap must be >= 0")
        if config.terminal_reward < 0:
            raise ValueError("terminal_reward must be >= 0")
        if config.reward_scale < 0:
            raise ValueError("reward_scale must be >= 0")
        if config.no_progress_timeout_steps < 0:
            raise ValueError("no_progress_timeout_steps must be >= 0")
        if config.no_progress_min_delta < 0:
            raise ValueError("no_progress_min_delta must be >= 0")
        self.config = replace(config, max_episode_steps=0)
        self.tracker = target_for_game(config.game).create_tracker(self.config)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.tracker.reset(info)
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        progress = self.tracker.step(reward, info, terminated or truncated)
        terminated = terminated or progress.terminal
        truncated = truncated or progress.truncated
        return obs, progress.reward, terminated, truncated, info


class RetroPreprocess(gym.ObservationWrapper):
    """Crop optional HUD rows, then convert RGB frames to grayscale observations."""

    def __init__(self, env: gym.Env, size: int = 84, hud_crop_top: int = 0):
        super().__init__(env)
        if hud_crop_top < 0:
            raise ValueError("hud_crop_top must be >= 0")
        self.size = size
        self.hud_crop_top = hud_crop_top
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 1),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.hud_crop_top >= observation.shape[0]:
            raise ValueError(
                f"hud_crop_top={self.hud_crop_top} must be less than frame height {observation.shape[0]}",
            )
        frame = observation[self.hud_crop_top :, :, :]
        gray = np.dot(frame[..., :3], np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return resized[..., None]


def make_retro_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = resolve_env_config(config or EnvConfig())
    env = retro.make(config.game, render_mode="rgb_array", **retro_make_kwargs(config))
    return wrap_retro_env(env, config=config, seed=seed)


def make_fast_retro_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = resolve_env_config(config or EnvConfig())
    target = target_for_game(config.game)
    env = retro.make(
        config.game,
        render_mode="rgb_array",
        obs_resize=(config.observation_size, config.observation_size),
        obs_crop=(config.hud_crop_top, 0, 0, 0) if config.hud_crop_top else None,
        obs_grayscale=True,
        obs_resize_algorithm=config.obs_resize_algorithm,
        frame_skip=config.frame_skip,
        frame_stack=4,
        maxpool_last_two=config.max_pool_frames,
        **retro_make_kwargs(config),
    )
    if target.uses_discrete_actions(config.action_set):
        env = DiscreteRetroActions(env, config=config)
    env = RetroProgressInfo(env, config=config)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    if config.clip_rewards:
        env = ClipRewardEnv(env)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_rendered_retro_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = resolve_env_config(config or EnvConfig())
    env = retro.make(config.game, render_mode="human", **retro_make_kwargs(config))
    return wrap_retro_env(env, config=config, seed=seed)


def wrap_retro_env(env: gym.Env, config: EnvConfig, seed: int | None = None) -> gym.Env:
    config = resolve_env_config(config)
    target = target_for_game(config.game)
    if target.uses_discrete_actions(config.action_set):
        env = DiscreteRetroActions(env, config=config)
    env = FrameSkip(env, config.frame_skip, max_pool=config.max_pool_frames)
    env = RetroProgressInfo(env, config=config)
    env = RetroPreprocess(env, config.observation_size, hud_crop_top=config.hud_crop_top)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    if config.clip_rewards:
        env = ClipRewardEnv(env)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_env_fn(rank: int, seed: int, config: EnvConfig) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env_config = resolve_env_config(config)
        if config.states:
            env_config = replace(config, state=config.states[rank % len(config.states)])
        env = make_fast_retro_env(config=env_config, seed=seed + rank)
        env.reset(seed=seed + rank)
        return env

    return _init


def needs_vec_transpose_image(observation_space: gym.Space) -> bool:
    """Return whether SB3 needs VecTransposeImage to receive channel-first images."""

    shape = getattr(observation_space, "shape", None)
    if not isinstance(observation_space, gym.spaces.Box) or shape is None or len(shape) != 3:
        raise ValueError(
            "expected image observation_space with shape (H, W, C) or (C, H, W), "
            f"got {observation_space!r}",
        )

    channels_first = (
        int(shape[0]) in FRAME_STACK_CHANNELS and int(shape[-1]) not in FRAME_STACK_CHANNELS
    )
    channels_last = (
        int(shape[-1]) in FRAME_STACK_CHANNELS and int(shape[0]) not in FRAME_STACK_CHANNELS
    )
    if channels_first:
        return False
    if channels_last:
        return True
    raise ValueError(
        "could not infer observation channel order from shape "
        f"{tuple(int(dim) for dim in shape)}; expected channel count in first or last axis",
    )


def maybe_transpose_vec_image(vec_env):
    if needs_vec_transpose_image(vec_env.observation_space):
        return VecTransposeImage(vec_env)
    return vec_env


def make_vec_envs(config: EnvConfig, n_envs: int, seed: int, start_method: str = "fork"):
    os.environ.setdefault("STABLE_RETRO_DISABLE_AUDIO", "1")
    config = resolve_env_config(config)
    if config.states:
        raise ValueError(
            "StableRetroNativeVecEnv supports one homogeneous state per vector env; "
            "use --state instead of --states for native rollouts.",
        )
    target = target_for_game(config.game)
    num_threads = config.env_threads if config.env_threads > 0 else min(max(n_envs, 1), 16)
    native_life_variable = target.native_life_variable
    native_terminates_life_loss = bool(
        config.terminate_on_life_loss and native_life_variable
    )
    native_life_loss_supported = (
        native_terminates_life_loss and native_vec_env_supports_life_loss()
    )
    native_kwargs: dict[str, Any] = {
        "num_envs": n_envs,
        "state": config.state or None,
        "num_threads": num_threads,
        "render_mode": "rgb_array",
        "obs_resize": (config.observation_size, config.observation_size),
        "obs_crop": (config.hud_crop_top, 0, 0, 0) if config.hud_crop_top else None,
        "obs_grayscale": True,
        "obs_resize_algorithm": config.obs_resize_algorithm,
        "frame_skip": config.frame_skip,
        "frame_stack": 4,
        "maxpool_last_two": config.max_pool_frames,
        "copy_observations": False,
    }
    if native_life_loss_supported:
        native_kwargs["terminate_on_life_loss"] = True
        native_kwargs["life_variable"] = native_life_variable
    vec_env = StableRetroNativeVecEnv(config.game, **native_kwargs)
    vec_env.seed(seed)
    if target.uses_discrete_actions(config.action_set):
        vec_env = VecDiscreteRetroActions(vec_env, config=config)
    progress_config = (
        replace(config, terminate_on_life_loss=False)
        if native_life_loss_supported
        else config
    )
    vec_env = VecRetroProgressInfo(vec_env, config=progress_config)
    vec_env = VecMonitor(vec_env)
    return maybe_transpose_vec_image(vec_env)


def make_training_vec_env(config: EnvConfig, n_envs: int, seed: int, start_method: str = "fork"):
    return make_vec_envs(config=config, n_envs=n_envs, seed=seed, start_method=start_method)


def make_eval_vec_env(config: EnvConfig, n_envs: int, seed: int, start_method: str = "fork"):
    return make_vec_envs(config=config, n_envs=n_envs, seed=seed, start_method=start_method)


def make_rendered_replay_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    return make_retro_env(config=config, seed=seed)


def assert_rom_imported(game: str = GAME) -> str:
    if not game:
        raise ValueError("game is required; pass --game or set RETRO_GAME")
    try:
        return retro.data.get_romfile_path(game)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{game} is not imported. Run: uv run python scripts/import_roms.py ~/Desktop/roms",
        ) from exc




def default_run_dir(run_name: str, runs_dir: str = "runs") -> str:
    return os.path.join(runs_dir, run_name)
