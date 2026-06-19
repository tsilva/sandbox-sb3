from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from stable_retro_ppo.artifacts import (
    apply_config_defaults,
    env_config_from_metadata,
    explicit_arg_dests,
    load_model_metadata,
    write_model_metadata,
)
from stable_retro_ppo.env import EnvConfig, resolve_env_config
from stable_retro_ppo.env_config import env_config_from_args
from stable_retro_ppo.wandb_artifacts import (
    artifact_download_dir,
    download_model_artifact,
    model_artifact_ref,
)
from stable_retro_ppo.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


def build_parser() -> argparse.ArgumentParser:
    defaults = EnvConfig()
    parser = argparse.ArgumentParser(
        description="Download a W&B model artifact and play it locally"
    )
    parser.add_argument(
        "run_name", nargs="?", help="Training run name, e.g. modal_gpu_short_improve"
    )
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT_PATH, help="W&B entity/project")
    parser.add_argument("--artifact", help="Full artifact ref, overriding run_name/kind/project")
    parser.add_argument("--kind", choices=["final", "best", "checkpoint"], default="final")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--root", default="runs/wandb_artifacts")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--game", default=defaults.game)
    parser.add_argument("--state", default=defaults.state)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sticky-action-prob",
        type=float,
        default=defaults.sticky_action_prob,
        help="Probability of replaying the previous high-level action; 0 disables sticky actions.",
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--observation-size", type=int, default=defaults.observation_size)
    parser.add_argument(
        "--hud-crop-top",
        type=int,
        default=defaults.hud_crop_top,
        help="Crop this many pixels from the top of raw frames before grayscale resize.",
    )
    parser.add_argument("--obs-resize-algorithm", default=defaults.obs_resize_algorithm)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-seeds", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=["auto", "baseline", "bounded", "additive", "score", "native"],
        default=defaults.reward_mode,
    )
    parser.add_argument("--use-retro-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-rewards", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-reward-cap", type=float, default=30.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--terminal-reward", type=float, default=50.0)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=25.0)
    parser.add_argument("--completion-reward", type=float, default=0.0)
    parser.add_argument("--score-progress-clipped", action="store_true")
    parser.add_argument("--no-progress-timeout-steps", type=int, default=0)
    parser.add_argument("--no-progress-min-delta", type=int, default=0)
    parser.add_argument("--completion-x-threshold", type=int, default=defaults.completion_x_threshold)
    parser.add_argument(
        "--terminate-on-life-loss",
        action=argparse.BooleanOptionalAction,
        default=defaults.terminate_on_life_loss,
    )
    parser.add_argument("--terminate-on-level-change", action="store_true")
    parser.add_argument("--terminate-on-completion", action="store_true")
    parser.add_argument("--action-set", default=defaults.action_set)
    parser.add_argument("--download-only", action="store_true")
    return parser


def artifact_ref(args: argparse.Namespace) -> str:
    if args.artifact:
        return args.artifact
    if not args.run_name:
        raise SystemExit("run_name is required unless --artifact is provided")
    return model_artifact_ref(
        project=args.project,
        run_name=args.run_name,
        kind=args.kind,
        version=args.version,
    )


def apply_artifact_run_config_defaults(
    args: argparse.Namespace,
    ref: str,
    parser_defaults: dict[str, object],
    explicit_dests: set[str],
) -> None:
    load_wandb_env()

    import wandb

    try:
        run = wandb.Api().artifact(ref, type="model").logged_by()
    except Exception as exc:
        print(f"warning: could not infer playback config from {ref}: {exc}", file=sys.stderr)
        return
    if run is None:
        return

    config = getattr(run, "config", {}) or {}
    apply_config_defaults(args, config, parser_defaults, explicit_dests)


def play_model(model_path: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "stable_retro_ppo.play",
        "--model",
        str(model_path),
        "--episodes",
        str(args.episodes),
        "--game",
        args.game,
        "--state",
        args.state,
        "--max-steps",
        str(args.max_steps),
        "--observation-size",
        str(args.observation_size),
        "--hud-crop-top",
        str(args.hud_crop_top),
        "--obs-resize-algorithm",
        args.obs_resize_algorithm,
        "--frame-skip",
        str(args.frame_skip),
        "--sticky-action-prob",
        str(args.sticky_action_prob),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--scale",
        str(args.scale),
        "--action-set",
        args.action_set,
        "--reward-mode",
        args.reward_mode,
        "--progress-reward-cap",
        str(args.progress_reward_cap),
        "--progress-reward-scale",
        str(args.progress_reward_scale),
        "--terminal-reward",
        str(args.terminal_reward),
        "--reward-scale",
        str(args.reward_scale),
        "--time-penalty",
        str(args.time_penalty),
        "--death-penalty",
        str(args.death_penalty),
        "--completion-reward",
        str(args.completion_reward),
        "--no-progress-timeout-steps",
        str(args.no_progress_timeout_steps),
        "--no-progress-min-delta",
        str(args.no_progress_min_delta),
        "--completion-x-threshold",
        str(args.completion_x_threshold),
    ]
    if args.score_progress_clipped:
        cmd.append("--score-progress-clipped")
    if args.use_retro_reward:
        cmd.append("--use-retro-reward")
    else:
        cmd.append("--no-use-retro-reward")
    if args.clip_rewards:
        cmd.append("--clip-rewards")
    else:
        cmd.append("--no-clip-rewards")
    if args.stochastic:
        cmd.append("--stochastic")
    if args.max_pool_frames:
        cmd.append("--max-pool-frames")
    else:
        cmd.append("--no-max-pool-frames")
    if args.random_seeds:
        cmd.append("--random-seeds")
    if args.terminate_on_life_loss is True:
        cmd.append("--terminate-on-life-loss")
    elif args.terminate_on_life_loss is False:
        cmd.append("--no-terminate-on-life-loss")
    if args.terminate_on_level_change:
        cmd.append("--terminate-on-level-change")
    if args.terminate_on_completion:
        cmd.append("--terminate-on-completion")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = build_parser()
    parser_defaults = vars(parser.parse_args([]))
    explicit_dests = explicit_arg_dests(parser, sys.argv[1:])
    args = parser.parse_args()
    ref = artifact_ref(args)
    download_root = artifact_download_dir(Path(args.root), ref)
    print(f"Downloading {ref} to {download_root}")
    model_path = download_model_artifact(ref, download_root)
    saved_config = env_config_from_metadata(load_model_metadata(model_path))
    if saved_config:
        apply_config_defaults(args, saved_config, parser_defaults, explicit_dests)
    else:
        apply_artifact_run_config_defaults(args, ref, parser_defaults, explicit_dests)
    config = resolve_env_config(env_config_from_args(args, max_episode_steps_attr="max_steps"))
    metadata_path = write_model_metadata(model_path, args, config, kind=args.kind)
    if metadata_path is not None:
        print(f"Wrote playback metadata: {metadata_path}")
    print(f"Downloaded model: {model_path}")
    if not args.download_only:
        play_model(model_path, args)


if __name__ == "__main__":
    main()
