# Goal: Sample-Efficient Level 1 Completion

Find the no-curriculum SuperMarioBros-NES PPO config that reaches reliable
`Level1-1` completion with the fewest aggregate policy timesteps.

## Target

Primary stop/score criterion:

```text
last 100 completed terminal training episodes are 100/100 completions
--stop-completion-episode-window 100
--stop-completion-rate-threshold 1.0
```

Use completed terminal episodes as the denominator. Do not use rollout-count
completion stops; they change meaning with `n_envs`, `n_steps`, and rollout
cadence.

Scope:

- Game/state: `SuperMarioBros-Nes-v0`, `Level1-1`
- No curriculum
- Native vector env training path
- W&B project: `SuperMarioBros-NES`
- Every run logs to W&B and has a specific `--run-description`

## Decision Metric

Primary metric: `time_to_100_of_100`, the first aggregate policy timestep where
the last 100 completed terminal training episodes are all completions.

If a run does not reach `100/100` before the active cap, record it as censored
at the cap with final completion rate, total completions, and total terminal
episodes.

Tie breakers:

1. Higher success count across confirmation seeds.
2. Lower median `time_to_100_of_100`.
3. Lower worst successful seed time.
4. Higher completion rate at fixed budgets: `2M`, `3M`, `4M`, and cap.
5. Higher external checkpoint eval completion rate, if available.

## Seeds And Cap

Use fixed paired seeds:

| Stage | Seeds | Purpose |
| --- | --- | --- |
| Screen | `23` | Cheap rejection of obvious losers |
| Confirm | `23,24,25` | Reproducibility check and baseline decision |

If a candidate is confirmed on `23,24,25`, compare only against an incumbent
evaluated on the same seeds. Same seed is not bitwise reproducibility under
default CUDA; treat it as a controlled randomization label.

Active cap:

```text
5,000,000 timesteps
```

Cap ratchet:

- If a confirmed baseline succeeds on all `23,24,25` seeds below the active cap,
  lower future caps to the slowest successful seed rounded up to the next `100k`.
- Do not increase the cap unless the goal changes or an infrastructure bug
  invalidates the baseline.

## Search Protocol

Screen with seed `23`. Reject obvious negatives. Advance candidates that reach
`100/100`, materially improve fixed-budget completion metrics, or look promising
enough to justify confirmation.

Confirm with seeds `23,24,25`. Promote only if success count is at least as good
as the incumbent, median `time_to_100_of_100` is better, there is no severe
reliability regression, and all runs either stop by `100/100` or reach the cap.

A baseline is the best confirmed config, not the best single run. Single lucky
runs generate hypotheses; they do not replace the baseline.

## Historical Context

The old `80/100` completed-episode target is retired. It was useful for finding
promising shapes, but it is not the current success criterion.

Most important retained facts from the retired experiment history:

- The best old single run was
  `b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906`, which reached the
  old `80/100` threshold at `2,558,256` timesteps with
  `learning_rate=2e-4 -> 1e-4 over 2M` and
  `ent_coef=0.01 -> 0.0003 over 2M`.
- That result was not a confirmed baseline. Same-seed/same-config repeats on
  later runs showed large PPO/CUDA/systems variance and did not reliably
  reproduce the old `80/100` stop.
- The upstream PPO reproduction proved that the simple-action, `gamma=0.9`,
  `gae_lambda=1.0`, fixed-entropy recipe can learn Level1-1, but checkpoint
  quality is non-monotonic. Do not evaluate a training run by final checkpoint
  alone.
- Use out-of-process stochastic checkpoint evaluation for promotion evidence.
  Rank checkpoints by completion rate first, then maximum x-position, then mean
  reward.
- Historical `BASELINE*` names are retired. In this repo, "baseline" now means a
  config confirmed under the current `100/100` protocol.
- Stable-retro runtime cautions and the current default runtime pin live in
  `AGENTS.md`; hardware and SkyPilot facts live in `INSTANCES.md`.

## RTX4090 Execution

Run SkyPilot jobs on:

```text
--infra k8s/rtx4090
```

Benchmark-backed scheduling decision for the current `n_envs=16`, `n_steps=512`
shape:

```text
concurrent child trainings: 5
env_threads per child: 4
expected aggregate wall fps: about 6.2k
```

Why: measured RTX4090/post10 sweeps showed `5` children with `env_threads=4`
was the aggregate-throughput winner. Lowering `env_threads` to `1` or `2`
reduced CPU pressure but underfed rollout collection; `env_threads=2,count=5`
was about `6%` slower than `env_threads=4,count=5`.

Use `3-4` children only when individual run latency matters or a long
confirmation batch should leave more CPU headroom. Refresh the safe limit when
changing `n_envs`, `n_steps`, model size, deterministic CUDA flags, runtime
package version, or target node CPU shape.

Use normal SkyPilot task files that start all child runs from the beginning.
Avoid ad hoc second trainers via `sky exec`; one such run previously collapsed
to about `140` fps and is not valid for comparison. Backfill finished slots only
while the next run still has decision value.

## Determinism

Default search should measure the real training distribution. Do not enable
deterministic CUDA flags by default because they can slow training and alter the
optimizer/kernel path.

Use deterministic CUDA only for debugging reproducibility:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
```

If used, mark deterministic CUDA as a separate runtime condition.

## Starting Config

Start from this near-best shape unless explicitly ablating one axis:

```text
stable-retro-turbo: 1.0.0.post12
state: Level1-1
n_envs: 16
env_threads: 4
n_steps: 512
batch_size: 512
n_epochs: 10
learning_rate: 2e-4 -> 1e-4 over 2M
ent_coef: 0.01 -> 0.0003 over 2M
gamma: 0.9
gae_lambda: 1.0
clip_range: 0.2
vf_coef: 1.0
normalize_advantage: false
adam_eps: 1e-8
reward_mode: score
terminal_reward: 50
reward_scale: 10
action_set: simple
frame_skip: 4
max_pool_frames: true
max_episode_steps: 4500
completion_x_threshold: 3160
terminate_on_completion: true
terminate_on_life_loss: true
eval_freq: 0
eval_episodes: 0
```

Always use:

```text
--stop-completion-episode-window 100
--stop-completion-rate-threshold 1.0
--timesteps <active cap>
```

## Batch Record

For each batch, record:

```text
hypothesis:
config_delta:
stage: screen|confirm
active_cap:
seeds:
wandb_group:
runs: seed, run_name, wandb_id, status, time_to_100_of_100, final_rate, totals
aggregate: success_count, median_time_to_100, worst_success_time, fixed-budget rates
decision: reject|confirm_more|promote
baseline_update: previous, new, old_cap, new_cap
```

## Current Status

The historical best single run reached the older `80/100` criterion at
`2,558,256` timesteps. That does not satisfy this `100/100` goal and was not
reliably reproduced as a same-seed `80/100` result.

```text
current_baseline_status: not confirmed
active_cap: 5,000,000
target_stop: 100/100 completed episodes
next_step: retry interrupted screen batch from scratch; prioritize ent0001 lead
```

## Interrupted Batch

On 2026-06-16, screen batch 1 was interrupted because the RTX4090 host had to
be shut down.

```text
task: sky_mario_100of100_screen_batch1_4090.yaml
cluster: sandbox-sb3-100of100-screen1
wandb_group: 100of100-screen-batch1-20260616_151115
runtime: stable-retro-turbo==1.0.0.post4
stage: screen
seed: 23
target_stop: 100/100
status: incomplete, no promotion
```

Last observed partial signal near `2.47M-2.48M` timesteps:

```text
ent0001: 147 total completions, recent rate 0.78 after briefly logging 0.80
control: 40 total completions, recent rate 0.22
lr125: 13 total completions, recent rate 0.02
fixedlr: 4 total completions, recent rate 0.01
clip015: 0 total completions
```

Retry guidance: relaunch from scratch after confirming the RTX4090 SkyPilot
state is clean. The local task now requests `cpus: 12+` and `memory: 48+`.
Either rerun the same five-way screen or bias the next batch toward `ent0001`
and nearby lower-entropy-floor schedules. Do not treat the interrupted
`ent0001` run as a confirmed recipe; it only reached the old `80/100` zone, not
the current `100/100` target.
