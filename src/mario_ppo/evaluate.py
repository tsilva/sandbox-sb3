from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from stable_baselines3 import PPO

from mario_ppo.env import ACTION_NAMES, EnvConfig, assert_rom_imported, make_mario_env, make_vec_envs


def scripted_action(policy: str, step_idx: int) -> int:
    if policy == "random":
        raise ValueError("random policy is sampled from the env action space")
    if policy == "noop":
        return ACTION_NAMES.index("noop")
    if policy == "right":
        # Mostly sprint right, with periodic jumps to clear early obstacles.
        if step_idx % 55 in range(30, 42):
            return ACTION_NAMES.index("right_a_b")
        return ACTION_NAMES.index("right_b")
    raise ValueError(f"unknown policy: {policy}")


def run_scripted_episode(env, policy: str, max_steps: int):
    obs, _info = env.reset()
    total_reward = 0.0
    max_x = 0
    final_info = {}
    for step_idx in range(max_steps):
        if policy == "random":
            action = env.action_space.sample()
        else:
            action = scripted_action(policy, step_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        max_x = max(max_x, int(info.get("max_x_pos", 0)))
        final_info = info
        if terminated or truncated:
            break
    return {
        "reward": total_reward,
        "max_x_pos": max_x,
        "score": int(final_info.get("score", 0)),
        "lives": int(final_info.get("lives", 0)),
        "steps": step_idx + 1,
    }


def run_model_episode(env, model: PPO, max_steps: int):
    obs = env.reset()
    total_reward = 0.0
    max_x = 0
    final_info = {}
    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        info = dict(infos[0])
        total_reward += float(rewards[0])
        max_x = max(max_x, int(info.get("max_x_pos", 0)))
        final_info = info
        if bool(dones[0]):
            break
    return {
        "reward": total_reward,
        "max_x_pos": max_x,
        "score": int(final_info.get("score", 0)),
        "lives": int(final_info.get("lives", 0)),
        "steps": step_idx + 1,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Mario PPO or scripted baselines")
    parser.add_argument("--model", help="Path to PPO .zip model")
    parser.add_argument("--policy", choices=["random", "right", "noop"], default="random")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", help="Optional JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assert_rom_imported()
    config = EnvConfig(state=args.state, max_episode_steps=args.max_steps)
    model = PPO.load(args.model) if args.model else None

    if model is not None:
        env = make_vec_envs(config=config, n_envs=1, seed=args.seed)
        episodes = [run_model_episode(env, model=model, max_steps=args.max_steps) for _ in range(args.episodes)]
        env.close()
    else:
        env = make_mario_env(config=config, seed=args.seed)
        episodes = [
            run_scripted_episode(env, policy=args.policy, max_steps=args.max_steps)
            for _ in range(args.episodes)
        ]
        env.close()

    rewards = np.array([episode["reward"] for episode in episodes], dtype=np.float64)
    x_positions = np.array([episode["max_x_pos"] for episode in episodes], dtype=np.float64)
    summary = {
        "model": args.model,
        "policy": "ppo" if args.model else args.policy,
        "episodes": args.episodes,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "max_x_mean": float(x_positions.mean()),
        "max_x_max": int(x_positions.max()),
        "episode_results": episodes,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
