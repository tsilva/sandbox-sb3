# Super Mario PPO Baseline 3: Entropy-Decay Sample-Efficient Stop

Baseline 3 promotes the entropy-decay RTX 4090 run
`sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508`.

The important result is sample efficiency: it reached the completed-episode
early-stop criterion at `3,979,616` aggregate policy timesteps. The prior
completed-episode Baseline 2 follow-up stopped at `5,278,832`, so this run used
`24.6%` fewer samples, or about `1.33x` better sample efficiency. It is the new
default comparison target for future sample-efficiency work.

## Artifact

Recommended model:

[final model](runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/final_model.zip)

Local absolute path:

```text
/Users/tsilva/repos/tsilva/sandbox-sb3/runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/final_model.zip
```

SHA256:

```text
c6197172192df6d407aa098f128785c8bfcfd5185b3381b7cd366b414ba3a146
```

Latest periodic checkpoint before early stop:

[3.9M checkpoint](runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/checkpoints/ppo_mario_3900000_steps.zip)

SHA256:

```text
c9396fe16b3dd341f483db30b0e6ef3f26a4a1b7d826a6c2a66645a2d01d28d0
```

Artifacts are under `runs/`, which is intentionally ignored by source control.

## Result Summary

Stop criterion:

```text
stop when the last 100 terminal training episodes are >=80% complete
```

Stop summary:

| Field | Value |
| --- | ---: |
| Stop reason | completed-episode completion-rate threshold |
| Stop timestep | `3,979,616 / 10,000,000` |
| Episode window | `100` terminal episodes |
| Stop completion rate | `0.800` |
| Threshold | `0.800` |
| Total terminal episodes | `1,635` |
| Total completed episodes | `158` |

Comparison:

| Run | Stop timestep | Relative samples | Notes |
| --- | ---: | ---: | --- |
| Baseline 2 completed-episode stop | `5,278,832` | `1.00x` | Fixed `ent_coef=0.01` |
| Baseline 3 entropy decay | `3,979,616` | `0.754x` | `1.33x` better sample efficiency |
| LR-decay ablation | `6,956,400` | `1.318x` | Negative ablation |

The entropy-decay run delayed early completions until the entropy coefficient
approached its floor, then produced a much sharper reliability ramp. This is
the highest-ROI sample-efficiency improvement observed so far.

## Follow-up 5M Sample-Efficiency Ablations

On 2026-06-14, three Baseline 3 variants were run concurrently in SkyPilot job
`4` on cluster `sandbox-sb3-stop10-4090`, with W&B logging online under
`tsilva/SuperMarioBros-NES`. All used the same PPO geometry, seed `23`, `5M`
maximum timesteps, and the same completed-episode stop criterion.

| Variant | Key change | Result | W&B |
| --- | --- | ---: | --- |
| Fast entropy | `ent_coef 0.01 -> 0.0003` over `2M` | stopped at `3,774,448` | `pbfrcflj` |
| Clipped progress | `--score-progress-clipped --progress-reward-cap 5 --progress-reward-scale 0.05` | maxed at `5,005,312`, `0/100` recent completions | `oca6i52u` |
| No-progress truncation | Baseline 3 entropy plus `--no-progress-timeout-steps 800` | maxed at `5,005,312`, `37/100` recent completions | `clu7ef42` |

Fast entropy is a modest improvement over Baseline 3:

```text
3,979,616 -> 3,774,448 timesteps
205,168 fewer samples
5.2% fewer samples
1.05x better sample efficiency
```

Fast entropy stop marker:

```text
reason=training_completion_rate_threshold
timesteps=3774448
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1514
total_completed_episodes=131
```

Downloaded winning artifact:

```text
runs/b3_fast_entropy_5m_stop80ep100_seed23_20260614_120804/final_model.zip
sha256=e74f7708038ce2193ebdd8be182a78183e602aeb1c306ca8a7e387258a89d844
```

The clipped-progress ablation was strongly negative. The current vector
training reward already uses true wrapper-computed `progress_delta`; making it
very small and clipped removed too much useful dense signal. No-progress
truncation was also negative for sample efficiency under this threshold: it
produced nonzero clears by 5M but never approached the `80/100` stop criterion.

## Later Sample-Efficiency Findings

Subsequent 5M ablations found a better successor to Baseline 3:

```text
Baseline 3 entropy decay: 3,979,616 timesteps
Fast entropy, lr=1e-4: 3,774,448 timesteps
Fast entropy, lr=2e-4: 2,824,240 timesteps
Fast entropy, lr=2e-4 -> 1e-4 over 2M: 2,558,256 timesteps
```

The current best is:

[fast entropy lr=2e-4 -> 1e-4 final model](runs/b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/final_model.zip)

It reached the same `80/100` completed-episode stop criterion in `2,824,240`
timesteps in the fixed-LR version, then `2,558,256` timesteps with mild
learning-rate decay. The LR-decay successor is `35.7%` fewer samples than
Baseline 3 and `51.5%` fewer samples than the fixed-entropy completed-episode
Baseline 2 follow-up. It is still above the explicit `2M` target.

Learning rate is the highest-ROI axis found so far, but the optimum appears
near `2e-4` for this setup. A follow-up sweep showed:

| Variant | Result |
| --- | ---: |
| `learning_rate=2e-4` | stopped at `2,824,240` |
| `learning_rate=2.5e-4` | stopped at `3,066,816` |
| `learning_rate=3e-4` | stopped at `4,777,248` |
| `learning_rate=4e-4` | maxed at `5,005,312`, `33/100` |

Follow-up entropy/minibatch ablations at `learning_rate=2e-4` did not beat the
`2.824M` best:

| Variant | Result |
| --- | ---: |
| entropy `0.01 -> 0.0003` over `1.5M` | stopped at `3,593,696` |
| entropy `0.01 -> 0.0001` over `2M` | stopped at `3,800,496` |
| `batch_size=256` | stopped at `4,653,200` after an early-completion burst and collapse |
| entropy `0.01 -> 0.0005` over `2M` | maxed at `5,005,312`, `0/100` |

Other negative or weak axes: `normalize_advantage=True`, `gamma=0.95`,
`n_epochs=15`, clipped/down-weighted progress shaping, and composing
`learning_rate=2e-4` with `n_steps=256`.

The latest stability follow-ups showed that mild LR decay helps, while stronger
decay and KL-limiting do not:

| Variant | Result |
| --- | ---: |
| `learning_rate=2e-4 -> 1e-4` over `2M` | stopped at `2,558,256` |
| `target_kl=0.03` | stopped at `4,712,176` |
| `learning_rate=2e-4 -> 5e-5` over `2M` | maxed at `5,005,312`, `36/100` |
| `batch_size=256`, `learning_rate=2e-4 -> 1e-4` | maxed at `5,005,312`, `24/100` |

Current best artifact:

```text
runs/b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/final_model.zip
sha256=5e45554b09354b5c0ade678863a45df78633b3ac963d299103af66d9d3cb74d3
```

Current best checkpoint:

```text
runs/b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/checkpoints/ppo_mario_2500000_steps.zip
sha256=590c375c86038bf67e4edf8b6bd47ab6f996a8b0446d8a0aaae50623cf8f53b4
```

Post5 reproduction note:

```text
stable-retro-turbo==1.0.0.post5 reproduction:
  run: b10_post5_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260614_203813
  result: maxed at 5,005,312 timesteps, 0/100 completion rate, 0 total completions
  wall clock: 1,667s / 27m47s for 5M
  final fps: 3023
  W&B: d3dorh0d
```

The post5 build gave the expected systems speedup versus the post4 winner
(`~3023` final fps vs `~1038`), but it did not reproduce the learning result for
the same seed and hyperparameters. Follow-up audit isolated this to the
`StableRetroNativeVecEnv` path: post5 matched post4 in the single-env API, but
the post5 native vector env returned empty `info` dicts where post4 returned
RAM variables such as `xscrollHi`, `xscrollLo`, `score`, and `lives`. Our
reward wrapper computes progress, score deltas, deaths, and level completion
from those fields, so post5 zeroed the shaped training reward. Do not use
post5 for vector training until this regression is fixed; keep training on the
known-good post4 Linux wheel or a later build that passes the vector-info audit.

Post6 follow-up:

```text
stable-retro-turbo==1.0.0.post6 reproduction:
  run: b11_post6_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_073609
  result: maxed at 5,005,312 timesteps, 0/100 completion rate, 0 total completions
  wall clock: 1,615s / 26m55s for 5M
  final fps: 3120
  W&B: q0me90ft
```

Post6 fixed the specific post5 empty-`info` regression in the training path:
the deterministic audit reported `single_equal=True` and `vector_equal=True`
against post4, and native-vector `info` keys were populated again. However, the
full post6 reproduction still did not reproduce the post4 winner's learning
trajectory because post6 changed observation buffer ownership in a way that
breaks SB3 PPO with this project's current `copy_observations=False` setting.
The follow-up aliasing audit showed:

- post4 with `copy_observations=False`: reset/step observations used distinct
  buffers, and prior observation arrays did not mutate after later `step()`
  calls.
- post6 with `copy_observations=False`: reset and later step observations used
  the same pointer; each `step()` mutated all prior observation references.
- post6 with `copy_observations=True`: prior observation arrays did not mutate.
- The fully wrapped training env (`VecTransposeImage(VecMonitor(VecMario...))`)
  reproduced the post6 aliasing because `make_vec_envs()` currently passes
  `copy_observations=False`.

This explains why reward/info/observation hashes for the immediate returned
step could match while PPO learning failed: SB3 computes actions from
`_last_obs`, calls `env.step()`, then writes `_last_obs` into the rollout buffer.
If `env.step()` mutates `_last_obs` in place, the rollout buffer stores the
post-action observation for the pre-action action. Do not promote post6 as the
Baseline 3 runtime unless either stable-retro-turbo restores non-mutating
returned observations for `copy_observations=False` or the project switches to
`copy_observations=True` and reproduces learning.

Post7 macOS local follow-up:

```text
stable-retro-turbo==1.0.0.post7 local macOS/MPS reproduction:
  run: local_post7_repro_b9_lrd1e4_2m_5m_stop80ep100_seed23_20260615
  result: maxed at 5,005,312 timesteps, 40/100 completion rate, 101 total completions
  wall clock: 7,294s / 2h01m34s for 5M
  final fps: 686
  W&B: ft5i902b
```

Post7 is promising but not promoted yet. The local alias audit showed the
SB3-critical `_last_obs` lifetime is fixed compared with post6, and this local
training run learned enough to produce many Level1-1 clears. It still did not
reproduce the RTX/post4 best run's `80/100` completed-episode stop within 5M
timesteps. Treat this as evidence that the catastrophic regression is likely
fixed, not as a replacement for a Linux/RTX reproduction.

Post7 Linux/RTX follow-up:

```text
stable-retro-turbo==1.0.0.post7 Linux/RTX2060 reproduction:
  run: b12_post7_2060_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_170235
  result: stopped at 2,711,552 timesteps, 80/100 completion rate, 182 total completions
  wall clock: 2,055s / 34m15s
  final observed fps: about 1331 before early stop
  W&B: an80iif6
```

This confirms post7 fixes the training-breaking post5/post6 failure mode on a
Linux/RTX training path. The stop timestep is slower than the post4 RTX4090
winner (`2,558,256`) but close enough to count as a successful runtime
reproduction under changed hardware. Use post4 as the exact historical
comparison point for existing ablations, and treat post7 as a validated
candidate runtime for future Linux/RTX runs.

Post7 RTX4090 rerun:

```text
stable-retro-turbo==1.0.0.post7 Linux/RTX4090 reproduction:
  run: b13_post7_4090_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_190534
  result: maxed at 5,005,312 timesteps, 31/100 completion rate, 197 total completions
  wall clock: 1,635s / 27m15s for 5M
  final fps: 3093
  W&B: 4hepwv0x

stable-retro-turbo==1.0.0.post7 Linux/RTX4090 repeat:
  run: b14_post7_4090_repeat_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_194155
  result: stopped just after 4,227,072 logged timesteps, 80/100 completion criterion, 189 total completions
  wall clock: 1,403s / 23m23s
  final fps: 3045
  W&B: feqsvt6f
```

The second RTX4090 repeat did reach the `80/100` completed-episode threshold,
so post7 can reproduce the success criterion on the intended Linux/RTX4090
path. It did not reproduce exactly: the first same-seed RTX4090 post7 run
maxed at `5,005,312` with only `31/100`, while the repeat stayed at `0`
completions until after `3M` and then crossed `80/100` just after `4.227M`.
Treat post7 as training-validated but high-variance for sample-efficiency
claims; use repeated runs before promoting small timestep differences.

Post4 same-build repeatability check:

```text
stable-retro-turbo==1.0.0.post4 Linux/RTX4090 repeat A:
  run: b15_post4_4090_repeat_a_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_201454
  result: maxed at 5,005,312 timesteps, 0/100 final completion rate, 173 total completions
  wall clock: 3,352s / 55m52s for 5M while sharing the node with repeat B
  final fps: 1500
  W&B: pvsxz4u7

stable-retro-turbo==1.0.0.post4 Linux/RTX4090 repeat B:
  run: b15_post4_4090_repeat_b_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_201454
  result: maxed at 5,005,312 timesteps, 30/100 final completion rate, 111 total completions
  wall clock: 3,356s / 55m56s for 5M while sharing the node with repeat A
  final fps: 1498
  W&B: l1trgg71
```

These two post4 repeats used the same seed and hyperparameters as the current
best post4 run, but ran concurrently on one RTX4090 node. Neither reproduced
the original `2,558,256`-timestep `80/100` early stop. They did reproduce
learning and level clears, but reliability was unstable: repeat A peaked around
`60/100` recent completions before collapsing to `0/100` by the final log, and
repeat B peaked much lower and finished at `30/100`.

This means the post7 RTX4090 variability is not, by itself, proof of a post7
regression. The historical post4 build also shows large same-seed outcome
variance under a parallel-run setup. Keep post4 as the exact historical baseline
for old ablations, but treat the single original `2.558M` result as a lucky
high-end sample unless repeated isolated post4 runs reproduce it.

Post10 RTX4090 reproduction check:

```text
stable-retro-turbo==1.0.0.post10 Linux/RTX4090 reproduction:
  run: b16_post10_4090_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260616_063000
  result: maxed at 5M, no early stop, final observed 47/100 completion rate, 253 total completions
  wall clock: 1,719s / 28m39s for 5M
  final observed fps: about 2939

stable-retro-turbo==1.0.0.post10 Linux/RTX4090 clean repeat:
  run: b16_post10_4090_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260616_070550
  result: manually stopped before conclusion; last checked at about 2.78M with 4 total completions
  wall clock before manual stop: 1,307s / 21m47s
```

Post10 does not show the catastrophic post5/post6 failure mode: the isolated
run learned, cleared Level1-1 many times, and reached a transient recent
completion window around `66/100` before regressing. It still did not reproduce
the `80/100` stop criterion within 5M in the completed run. The clean repeat
started clearing late and was stopped manually, so it is only partial evidence.
Treat post10 as promising/runtime-validating, not as a promoted baseline.

## GUI Playback

GUI command:

```bash
uv --cache-dir .uv-cache run python -m mario_ppo.play \
  --model runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/final_model.zip \
  --episodes 0 \
  --max-steps 4500 \
  --fps 30 \
  --scale 4 \
  --reward-mode score \
  --terminal-reward 50 \
  --reward-scale 10 \
  --action-set simple \
  --completion-x-threshold 3160 \
  --terminate-on-completion \
  --device auto \
  --stochastic
```

Observed GUI sample:

```text
episode=1 seed=7 reward=319.35 max_x=3127 steps=513 status=terminated died=False complete=True
```

## Training Run

SkyPilot job:

| Field | Value |
| --- | --- |
| SkyPilot cluster | `sandbox-sb3-stop10-4090` |
| SkyPilot parent job id | `3` |
| Task name | `mario-sampleeff-ablate-parallel-4090` |
| Infra | `k8s/rtx4090` |
| GPU | NVIDIA GeForce RTX 4090 |
| Run name | `sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508` |
| Parent log path | `~/sky_logs/sky-2026-06-14-11-24-56-120668` |
| Per-run log path | `~/sky_workdir/logs/parallel_ablation/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508.log` |
| Status | `SUCCEEDED` |
| Parent job duration | `1h 5m 30s` |

The parent SkyPilot task ran this entropy-decay ablation and the LR-decay
ablation concurrently inside one GPU pod. The parent duration is therefore not
the wall time of this child run alone.

Package/runtime facts observed in the run:

- Linux package: `stable-retro-turbo==1.0.0.post4`.
- Python: `3.14`.
- Torch on the SkyPilot node: `2.12.0+cu130`.
- CUDA was available.
- ROM import succeeded for `SuperMarioBros-Nes-v0`.

## Training Command

The SkyPilot task was launched from
`sky_mario_sampleeff_ablate_parallel_4090.yaml`. The entropy-decay child
process ran:

```bash
uv run --python 3.14 --no-dev python -m mario_ppo.train \
  --timesteps 10000000 \
  --n-envs 16 \
  --run-name "${ENT_RUN}" \
  --seed 23 \
  --n-steps 512 \
  --batch-size 512 \
  --n-epochs 10 \
  --learning-rate 0.0001 \
  --gamma 0.9 \
  --gae-lambda 1.0 \
  --ent-coef 0.01 \
  --ent-coef-final 0.001 \
  --ent-coef-schedule-timesteps 3000000 \
  --clip-range 0.2 \
  --reward-mode score \
  --terminal-reward 50 \
  --reward-scale 10 \
  --action-set simple \
  --frame-skip 4 \
  --max-pool-frames \
  --max-episode-steps 4500 \
  --completion-x-threshold 3160 \
  --terminate-on-completion \
  --checkpoint-freq 100000 \
  --stop-completion-episode-window 100 \
  --stop-completion-rate-threshold 0.8 \
  --eval-freq 0 \
  --eval-episodes 0 \
  --device cuda \
  --env-threads 8 \
  --torch-num-threads 1 \
  --wandb \
  --wandb-project mario-ppo \
  --wandb-mode offline
```

The command did not pass `--no-terminate-on-life-loss`, so life loss was
terminal during training.

## Environment And Observation Path

Environment:

- Game: `SuperMarioBros-Nes-v0`.
- State: `Level1-1`.
- stable-retro provider: `stable-retro-turbo`.
- Native vector path: `StableRetroNativeVecEnv`.
- Parallel envs: `16`.
- Native env threads: `8`.
- Action set: `simple`.

Observation preprocessing:

- HUD crop top: `32` pixels.
- Resize: `84x84`.
- Grayscale.
- Frame skip: `4`.
- Max-pool over the last two raw frames inside each frame skip.
- Frame stack: `4`.
- Tensor layout entering SB3 policy: channel-first image stack compatible with
  SB3 `CnnPolicy`.

Action set:

```text
noop
right
right_b
right_a
right_a_b
a
left
```

## Reward And Termination Semantics

This run used `--reward-mode score`.

For the vector wrapper, score mode computes:

```text
reward = progress_delta + 0.01 * score_delta
if completion_event: reward += completion_reward
if died: reward -= death_penalty
reward -= time_penalty
```

For this run:

| Parameter | Value |
| --- | ---: |
| `progress_reward_scale` | `1.0` |
| `completion_reward` | `0.0` |
| `death_penalty` | `25.0` |
| `time_penalty` | `0.0` |
| `use_retro_reward` | `False` |
| `completion_x_threshold` | `3160` |
| `terminate_on_life_loss` | `True` |
| `terminate_on_completion` | `True` |
| `terminate_on_level_change` | `False` |

Completion was counted if either stable-retro reported a level change or the
level-local x-position reached the threshold.

## PPO Configuration

| Parameter | Value |
| --- | ---: |
| Algorithm | SB3 PPO |
| Policy | `CnnPolicy` |
| Device | `cuda` |
| Seed | `23` |
| Parallel envs | `16` |
| Rollout length per env | `512` |
| Aggregate rollout size | `8192` |
| Batch size | `512` |
| Minibatches per epoch | `16` |
| PPO epochs per rollout | `10` |
| Learning rate | `1e-4` |
| Gamma | `0.9` |
| GAE lambda | `1.0` |
| Entropy coefficient | `0.01 -> 0.001` |
| Entropy schedule | linear over first `3,000,000` timesteps |
| Value coefficient | `1.0` |
| Clip range | `0.2` |
| Advantage normalization | `False` |
| Adam epsilon | `1e-8` |
| Checkpoint frequency | `100,000` aggregate policy steps |
| Training-loop eval | disabled |

Timestep accounting:

```text
16 envs * 512 steps = 8192 aggregate policy decisions per PPO rollout
```

Each policy decision repeats the selected action for `4` emulator frames.

## Interpretation

Baseline 3 should replace the fixed-entropy completed-episode run as the
current sample-efficiency target. It did not achieve the desired 2x improvement,
but it moved the stop point from `5.28M` to `3.98M` aggregate timesteps without
changing env count, rollout geometry, reward mode, or completion semantics.

The main lesson is that fixed `ent_coef=0.01` is useful for discovery but likely
too exploratory once the policy has found level-clearing behavior. Decaying the
entropy coefficient allowed reliability to consolidate earlier.

The next sample-efficiency runs should tune entropy scheduling before changing
other axes. Promising variants:

- Delay decay until `1M` or `2M`, then decay to `0.001`.
- Use a higher floor such as `0.003`.
- Decay over `4M` instead of `3M`.

Continue to use the completed-episode stop as the training budget guardrail,
and continue using out-of-process stochastic checkpoint evals before promoting
policies for robustness claims.
