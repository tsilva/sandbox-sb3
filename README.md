# Mario PPO

PPO training scaffold for `SuperMarioBros-Nes-v0` using `stable-retro-apple-silicon` and Stable-Baselines3.

The goal is to train a CNN PPO policy that moves right through `Level1-1` and improves over random and simple scripted baselines.

## Setup

```bash
uv sync
uv run python scripts/import_roms.py ~/Desktop/roms
```

The ROM import must recognize `SuperMarioBros-Nes-v0`; stable-retro matches by ROM SHA, not by filename.

## Smoke Test

```bash
uv run python -m mario_ppo.evaluate --policy right --episodes 2 --max-steps 600
uv run python -m mario_ppo.train \
  --timesteps 512 \
  --n-envs 1 \
  --batch-size 128 \
  --eval-freq 256 \
  --eval-episodes 1 \
  --max-episode-steps 600 \
  --run-name smoke
uv run python -m mario_ppo.evaluate --model runs/smoke/final_model.zip --episodes 2 --max-steps 600
```

Watch the smoke model in a GUI window:

```bash
PYTHONPATH=src .venv/bin/python -m mario_ppo.play --model runs/smoke_doc/final_model.zip --episodes 3 --max-steps 1200 --fps 30 --scale 4
```

## Train

Start with a bounded run:

```bash
uv run python -m mario_ppo.train \
  --timesteps 1000000 \
  --n-envs 4 \
  --run-name ppo_level1_1_1m
```

Longer run:

```bash
uv run python -m mario_ppo.train \
  --timesteps 10000000 \
  --n-envs 4 \
  --run-name ppo_level1_1_10m
```

W&B online run:

```bash
wandb login
uv run python -m mario_ppo.train \
  --timesteps 10000000 \
  --n-envs 4 \
  --run-name ppo_level1_1_10m \
  --wandb \
  --wandb-project mario-ppo
```

W&B offline smoke run:

```bash
uv run python -m mario_ppo.train \
  --timesteps 512 \
  --n-envs 1 \
  --batch-size 128 \
  --eval-freq 256 \
  --eval-episodes 1 \
  --max-episode-steps 600 \
  --run-name wandb_smoke \
  --wandb \
  --wandb-mode offline
```

Evaluate:

```bash
uv run python -m mario_ppo.evaluate --model runs/ppo_level1_1_10m/final_model.zip --episodes 20
uv run python -m mario_ppo.evaluate --policy random --episodes 20
uv run python -m mario_ppo.evaluate --policy right --episodes 20
```

TensorBoard:

```bash
uv run tensorboard --logdir runs
```

## Notes

- The default stable-retro scenario rewards x-scroll progress.
- The wrapper adds frame skipping, discrete Mario actions, 84x84 grayscale observations, time limits, and progress metrics.
- Generated checkpoints and logs stay under `runs/`.
