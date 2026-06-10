from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from mario_ppo.env import EnvConfig, assert_rom_imported, default_run_dir, make_vec_envs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on SuperMarioBros-Nes-v0")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-name", default="ppo_level1_1")
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=4500)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-rewards", action="store_true")
    parser.add_argument("--progress-reward-scale", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=0.0)
    parser.add_argument("--resume", help="Path to an existing PPO .zip checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--wandb-project", default="mario-ppo")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser


def init_wandb(args: argparse.Namespace, run_dir: str, config: EnvConfig):
    if not args.wandb:
        return None

    wandb_dir = os.path.abspath(run_dir)
    wandb_aux_dir = os.path.join(wandb_dir, "wandb")
    wandb_env_dirs = {
        "WANDB_DIR": wandb_dir,
        "WANDB_CACHE_DIR": os.path.join(wandb_aux_dir, "cache"),
        "WANDB_CONFIG_DIR": os.path.join(wandb_aux_dir, "config"),
        "WANDB_DATA_DIR": os.path.join(wandb_aux_dir, "data"),
        "WANDB_ARTIFACT_DIR": os.path.join(wandb_aux_dir, "artifacts"),
    }
    for env_name, path in wandb_env_dirs.items():
        os.environ.setdefault(env_name, path)
        os.makedirs(os.environ[env_name], exist_ok=True)

    import wandb

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    wandb_config: dict[str, Any] = {
        **vars(args),
        "game": config.game,
        "state": config.state,
        "frame_skip": config.frame_skip,
        "max_episode_steps": config.max_episode_steps,
        "observation_size": config.observation_size,
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.run_name,
        tags=tags,
        config=wandb_config,
        dir=wandb_dir,
        sync_tensorboard=True,
        save_code=True,
        mode=args.wandb_mode,
    )


def log_wandb_artifact(wandb_run, args: argparse.Namespace, model_path: str) -> None:
    if wandb_run is None:
        return

    import wandb

    artifact = wandb.Artifact(f"{args.run_name}-final-model", type="model")
    artifact.add_file(model_path)
    wandb_run.log_artifact(artifact)


def main() -> None:
    args = build_parser().parse_args()
    assert_rom_imported()
    set_random_seed(args.seed)

    run_dir = default_run_dir(args.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    best_dir = os.path.join(run_dir, "best")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    config = EnvConfig(
        state=args.state,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        clip_rewards=args.clip_rewards,
        progress_reward_scale=args.progress_reward_scale,
        death_penalty=args.death_penalty,
    )
    wandb_run = init_wandb(args, run_dir, config)

    env = make_vec_envs(config=config, n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_envs(config=config, n_envs=1, seed=args.seed + 10_000)

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=run_dir)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            tensorboard_log=run_dir,
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
            save_path=checkpoint_dir,
            name_prefix="ppo_mario",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=os.path.join(run_dir, "eval"),
            eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        ),
    ]

    final_model_path = os.path.join(run_dir, "final_model.zip")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
        model.save(os.path.join(run_dir, "final_model"))
        log_wandb_artifact(wandb_run, args, final_model_path)
    finally:
        env.close()
        eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()
    print(f"saved {final_model_path}")


if __name__ == "__main__":
    main()
