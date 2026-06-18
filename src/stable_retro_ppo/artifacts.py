from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from stable_retro_ppo.env import EnvConfig
from stable_retro_ppo.wandb_utils import load_wandb_env


def init_wandb(args: argparse.Namespace, run_dir: str, config: EnvConfig):
    if not args.wandb:
        return None

    load_wandb_env()

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
        "states": list(config.states),
        "frame_skip": config.frame_skip,
        "max_pool_frames": config.max_pool_frames,
        "max_episode_steps": config.max_episode_steps,
        "observation_size": config.observation_size,
        "hud_crop_top": config.hud_crop_top,
        "obs_resize_algorithm": config.obs_resize_algorithm,
        "use_retro_reward": config.use_retro_reward,
        "reward_mode": config.reward_mode,
        "progress_reward_cap": config.progress_reward_cap,
        "progress_reward_scale": config.progress_reward_scale,
        "terminal_reward": config.terminal_reward,
        "reward_scale": config.reward_scale,
        "time_penalty": config.time_penalty,
        "death_penalty": config.death_penalty,
        "completion_reward": config.completion_reward,
        "score_progress_clipped": config.score_progress_clipped,
        "no_progress_timeout_steps": config.no_progress_timeout_steps,
        "no_progress_min_delta": config.no_progress_min_delta,
        "completion_x_threshold": config.completion_x_threshold,
        "terminate_on_life_loss": config.terminate_on_life_loss,
        "terminate_on_level_change": config.terminate_on_level_change,
        "terminate_on_completion": config.terminate_on_completion,
        "action_set": config.action_set,
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.run_name,
        notes=args.run_description or None,
        tags=tags,
        config=wandb_config,
        dir=wandb_dir,
        sync_tensorboard=True,
        save_code=True,
        mode=args.wandb_mode,
    )


def wandb_artifacts_enabled(wandb_run, args: argparse.Namespace) -> bool:
    return wandb_run is not None and not args.no_wandb_artifacts


def sanitize_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "stable-retro-ppo"


def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def format_wandb_run_path(run_path) -> str:
    if isinstance(run_path, (list, tuple)):
        return "/".join(str(part) for part in run_path)
    return str(run_path)


def wandb_artifact_storage_uri(args: argparse.Namespace) -> str:
    return (
        args.wandb_artifact_storage_uri.strip()
        or os.environ.get("WANDB_ARTIFACT_STORAGE_URI", "").strip()
        or os.environ.get("CHECKPOINT_BUCKET_URI", "").strip()
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/prefix URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def artifact_rom_prefix(game: str) -> str:
    return sanitize_artifact_name(game)


def artifact_storage_prefix(base_prefix: str, game: str) -> str:
    prefix = base_prefix.rstrip("/")
    rom_prefix = artifact_rom_prefix(game)
    if not prefix:
        return rom_prefix
    if prefix == rom_prefix or prefix.endswith(f"/{rom_prefix}"):
        return prefix
    return f"{prefix}/{rom_prefix}"


def build_s3_artifact_uri(base_uri: str, args: argparse.Namespace, model_path: Path, kind: str) -> str:
    bucket, prefix = parse_s3_uri(base_uri)
    prefix = artifact_storage_prefix(prefix, args.game)
    key_parts = [
        prefix,
        sanitize_artifact_name(args.run_name),
        kind,
        model_path.name,
    ]
    key = "/".join(part for part in key_parts if part)
    return f"s3://{bucket}/{key}"


def upload_s3_artifact(model_path: Path, destination_uri: str) -> None:
    bucket, key = parse_s3_uri(destination_uri)

    import boto3

    endpoint_url = os.environ.get("AWS_S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    s3_client = boto3.client("s3", endpoint_url=endpoint_url)
    s3_client.upload_file(
        str(model_path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/zip"},
    )


def log_wandb_model_artifact(
    wandb_run,
    args: argparse.Namespace,
    model_path: Path,
    kind: str,
    aliases: list[str] | None = None,
) -> None:
    if not wandb_artifacts_enabled(wandb_run, args):
        return
    if not model_path.is_file():
        return

    import wandb

    artifact_name = f"{sanitize_artifact_name(args.run_name)}-{kind}"
    step = checkpoint_step(model_path)
    metadata: dict[str, Any] = {
        "run_name": args.run_name,
        "run_description": args.run_description,
        "kind": kind,
        "filename": model_path.name,
        "checkpoint_step": step,
    }
    run_id = getattr(wandb_run, "id", None)
    if run_id:
        metadata["wandb_run_id"] = run_id
    run_path = getattr(wandb_run, "path", None)
    if run_path:
        metadata["wandb_run_path"] = format_wandb_run_path(run_path)

    storage_base_uri = wandb_artifact_storage_uri(args)
    reference_uri = None
    if storage_base_uri:
        reference_uri = build_s3_artifact_uri(storage_base_uri, args, model_path, kind)
        upload_s3_artifact(model_path, reference_uri)
        metadata["artifact_storage_uri"] = reference_uri

    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        metadata=metadata,
    )
    if reference_uri:
        artifact.add_reference(reference_uri, name=model_path.name)
    else:
        artifact.add_file(str(model_path), name=model_path.name)
    wandb_run.log_artifact(artifact, aliases=aliases)
    location = reference_uri or str(model_path)
    print(f"wandb artifact logged: {artifact_name} ({location})")


def write_wandb_url(wandb_run, run_dir: str) -> None:
    if wandb_run is None:
        return

    run_url = getattr(wandb_run, "url", None)
    if run_url:
        Path(run_dir, "wandb_url.txt").write_text(f"{run_url}\n", encoding="utf-8")
    run_id = getattr(wandb_run, "id", None)
    if run_id:
        Path(run_dir, "wandb_run_id.txt").write_text(f"{run_id}\n", encoding="utf-8")
    run_path = getattr(wandb_run, "path", None)
    if run_path:
        Path(run_dir, "wandb_run_path.txt").write_text(
            f"{format_wandb_run_path(run_path)}\n",
            encoding="utf-8",
        )


def write_run_description(args: argparse.Namespace, run_dir: str) -> None:
    description = args.run_description.strip()
    Path(run_dir, "run_description.txt").write_text(
        f"{description}\n" if description else "",
        encoding="utf-8",
    )
