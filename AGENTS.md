# Project Rules

## Upstream Emulator

Treat `../stable-retro-apple-silicon` as an external dependency checkout. Do not edit it from this repo unless the user explicitly asks for upstream package changes.

## Training Runs

Keep generated training artifacts out of source control. Use `runs/`, `logs/`, and `models/` for checkpoints, TensorBoard logs, and evaluation outputs.

## Dependencies

Use `uv` for dependency resolution and keep `uv.lock` committed. Preserve Python supply-chain hardening in `pyproject.toml`.
