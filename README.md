# Stable Retro PPO

PPO training scaffold for Stable Retro games using `stable-retro-turbo` and
Stable-Baselines3.

Runtime configuration is game-driven. Pass `--game <StableRetroGameId>` and,
when needed, `--state <StateName>`. Unknown games default to Stable Retro's
native action space and native emulator reward; registered targets can provide
custom action sets, reward shaping, HUD crop, and completion defaults.

## Setup

```bash
uv sync
uv run python scripts/import_roms.py ~/Desktop/roms
```

The supported macOS arm64 and Linux x86_64 environments are pinned to
`stable-retro-turbo==1.0.0.post12`; the runtime import remains `stable_retro`.
Stable Retro matches ROMs by SHA, not filename. The import must recognize
the game id you plan to pass with `--game`.

## Smoke Test

```bash
uv run python -m stable_retro_ppo.evaluate --game <GameId> --policy random --episodes 2 --max-steps 600
uv run python -m stable_retro_ppo.train --game <GameId> --preset smoke
uv run python -m stable_retro_ppo.evaluate --game <GameId> --model runs/smoke/final_model.zip --episodes 2 --max-steps 600
```

Watch the smoke model:

```bash
uv run python -m stable_retro_ppo.play --game <GameId> --model runs/smoke/final_model.zip --episodes 3 --max-steps 1200 --fps 30 --scale 4
```

## Training

Current defaults are intentionally close to the working upstream PPO recipe:

```text
n_envs=8
seed=123
n_steps=512
batch_size=256
n_epochs=10
learning_rate=1e-4
gamma=0.9
gae_lambda=1.0
ent_coef=0.01
vf_coef=1.0
clip_range=0.2
normalize_advantage=False
adam_eps=1e-8
action_set=auto
reward_mode=auto
```

Start with a bounded local run:

```bash
uv run python -m stable_retro_ppo.train \
  --game <GameId> \
  --timesteps 1000000 \
  --run-name ppo_retro_1m \
  --run-description "Local 1M Stable Retro PPO baseline check"
```

Use presets for common shapes:

```bash
uv run python -m stable_retro_ppo.train --game <GameId> --preset smoke
uv run python -m stable_retro_ppo.train --game <GameId> --preset modal-t4 --run-name modal_candidate --run-description "Candidate Modal T4 baseline"
```

By default, training-loop eval is disabled. Training should focus on throughput,
checkpointing, and artifact upload. Evaluate checkpoints out of process.

## Evaluation

Evaluate a local model:

```bash
uv run python -m stable_retro_ppo.evaluate --game <GameId> --model runs/ppo_retro_1m/final_model.zip --episodes 20
```

Model evaluation defaults to stochastic action sampling. Use `--no-stochastic`
for deterministic argmax evaluation.

Evaluate pending W&B checkpoint artifacts and promote the best checkpoint by
completion rate, then max x-position, then mean reward:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/eval_wandb_checkpoints.py <run-name> \
  --episodes 50 \
  --max-steps 2500 \
  --game <GameId> \
  --action-set native \
  --record-best-video
```

Local eval state is written under:

```text
runs/local_evals/<run-name>/checkpoint_eval_metrics.jsonl
runs/local_evals/<run-name>/videos/
```

## Playback

Download and play a W&B artifact:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/play_wandb_artifact.py <run-name> --game <GameId> --kind best --stochastic
```

Models trained with a non-native target action set must be played and evaluated
with the same `--game` and `--action-set`.

## Modal

Install and authenticate the Modal extra:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra modal
UV_CACHE_DIR=.uv-cache uv run modal setup
```

Upload ROMs once:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/stable_retro_ppo/modal_app.py::upload_roms --rom-dir ~/Desktop/roms
```

Run a remote smoke job:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/stable_retro_ppo/modal_app.py::train \
  --game <GameId> \
  --timesteps 512 \
  --n-envs 1 \
  --run-name modal_smoke \
  --run-description "Modal smoke test" \
  --max-episode-steps 600
```

For W&B online from Modal, create the secret once:

```bash
UV_CACHE_DIR=.uv-cache uv run modal secret create wandb-secret WANDB_API_KEY=...
```

Remote outputs are persisted to the Modal Volume `stable-retro-ppo-data` under
`/runs/<run-name>`.

## Experiment Notes

- Current experiment status and retained historical lessons live in `GOAL.md`.
- Current hardware and SkyPilot operating facts live in `INSTANCES.md`.
- Reusable SkyPilot experiment matrices live under `experiments/launches/`.
  `scripts/stable_retro_skypilot.py` can render, preflight, sparsely launch,
  clone W&B run configs into ROM-agnostic repro manifests, repair SkyPilot API
  selection, and write launch reports.
- Generated checkpoints, logs, videos, W&B files, caches, and ad hoc launch specs
  stay under ignored run/local directories.
- Root-level `sky_*.yaml` files are scratch launch specs and should not live in
  the active repo view.
- Training and model evaluation rollouts use `StableRetroNativeVecEnv`; videos
  replay selected action traces through the render-capable single-env path.
- Level completion and life-loss termination are target-specific. Unknown games
  default to no progress threshold and no life-loss termination.
