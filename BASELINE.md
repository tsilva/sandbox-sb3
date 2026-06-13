# Super Mario PPO Baseline

This baseline documents the Modal reproduction of
`vietnh1009/Super-mario-bros-PPO-pytorch` for `SuperMarioBros-1-1-v0`.
The goal was to check whether the upstream implementation actually works and
estimate how many timesteps it needs to solve Level 1-1.

## Result Summary

The upstream PPO implementation works: it learns policies that can clear
Level 1-1, and it is much stronger than our local SB3 attempts so far. It did
not reach the robust solve threshold in one 5M-step seed-123 run.

Key results from `viet_level1_repro_5m_seed123`:

| Step | Eval clears | Completion rate | Notes |
| ---: | ---: | ---: | --- |
| 401,408 | 1/20 | 5% | First nonzero clear observed. |
| 802,816 | 6/20 | 30% | First strong early checkpoint. |
| 2,101,248 | 11/20 | 55% | First large jump. |
| 2,953,216 | 12/20 | 60% | Roughly 3M aggregate env decisions. |
| 3,403,776 | 13/20 | 65% | Best before 4M. |
| 4,603,904 | 14/20 | 70% | Best checkpoint in the 5M run. |
| 4,751,360 | 14/20 | 70% | Same clear count, lower reward/position. |
| 5,001,216 | 12/20 | 60% | Final checkpoint, below best. |

Best checkpoint:

```text
/vol/runs/viet_level1_repro_5m_seed123/checkpoints/ppo_super_mario_bros_1_1_4603904_steps.pt
```

Best checkpoint metrics:

- `step`: `4,603,904`
- `completion_count`: `14/20`
- `completion_rate`: `0.70`
- `reward_mean`: `276.39`
- `max_x_mean`: `2828.3`
- `max_x_max`: `3161`

Solve threshold used in the reproduction:

- 20-episode stochastic checkpoint scan.
- A checkpoint would trigger 100-episode confirmation at `>=16/20` clears
  (`>=80%`).
- No checkpoint reached that threshold, so no 100-episode confirmation ran.

## Timestep Accounting

The logged timesteps are aggregate policy-decision timesteps across all
parallel envs, not per-env timesteps.

The baseline used `8` parallel env processes:

```text
2,953,216 aggregate decisions / 8 envs = 369,152 decisions per env
```

Each selected action is repeated by the env wrapper for `4` emulator frames.
So `2,953,216` policy decisions corresponds to about `11,812,864` raw emulator
frames across all envs.

One PPO rollout/update uses:

```text
num_processes * num_local_steps = 8 * 512 = 4096 policy decisions
```

## Upstream PPO Hyperparameters

These are the upstream defaults used by `train.py` and preserved in the Modal
reproduction unless noted.

| Parameter | Value |
| --- | ---: |
| World | `1` |
| Stage | `1` |
| Env id | `SuperMarioBros-1-1-v0` |
| Action set | `simple` (`SIMPLE_MOVEMENT`) |
| Parallel envs | `8` |
| Learning rate | `1e-4` |
| Optimizer | `Adam` |
| Discount factor `gamma` | `0.9` |
| GAE parameter `tau` | `1.0` |
| Entropy coefficient `beta` | `0.01` |
| PPO clip epsilon | `0.2` |
| PPO epochs per rollout | `10` |
| Rollout length per env | `512` policy decisions |
| Aggregate rollout size | `4096` policy decisions |
| Global training target | `5e6` upstream default; `5,000,000` in Modal run |
| Minibatch split count | `16` |
| Minibatch size | `256` samples |
| Gradient clipping | global norm `0.5` |
| Value loss | Smooth L1 loss |
| Actor loss | clipped PPO surrogate |
| Entropy schedule | none; fixed `beta=0.01` |
| Seed | `123` in upstream and Modal reproduction |
| CPU threading | `OMP_NUM_THREADS=1` |
| Upstream `save_interval` | `50` episodes, but save calls are commented out in `train.py` |
| Upstream `max_actions` | `200`, used by upstream test/eval process |
| Upstream `log_path` | `tensorboard/ppo_super_mario_bros` |
| Upstream `saved_path` | `trained_models` |

The upstream `batch_size` name is slightly misleading: it is the number of
minibatches per epoch, not the number of samples per minibatch. With `4096`
rollout samples and `batch_size=16`, each minibatch has `256` samples.

Modal reproduction/evaluation knobs:

| Parameter | Value |
| --- | ---: |
| Checkpoint interval | `50,000` aggregate policy decisions |
| Eval episodes per checkpoint | `20` |
| Confirmation episodes | `100` |
| Confirmation trigger | `completion_rate >= 0.8` |
| Max eval steps per episode | `2500` policy decisions |
| Eval action selection | stochastic `Categorical(policy)` |
| Run seed | `123` |

## Environment And Preprocessing

Environment construction:

1. `gym_super_mario_bros.make("SuperMarioBros-1-1-v0")`
2. `JoypadSpace(env, SIMPLE_MOVEMENT)`
3. `CustomReward`
4. `CustomSkipFrame(skip=4)`

Observation preprocessing:

- Convert RGB frame to grayscale with OpenCV.
- Resize to `84x84`.
- Scale pixels by `/ 255.0`.
- Maintain a 4-frame stack.
- For each repeated action, keep the last two processed frames and max-pool
  them into the newest frame-stack slot.
- Observation shape entering the policy: `(4, 84, 84)`.

Frame/action behavior:

- The policy samples one discrete action.
- `CustomSkipFrame` repeats that action for `4` env steps.
- If the env ends during the repeat, the wrapper resets internally and returns
  the current frame stack with `done=True`.

Reward shaping:

```text
reward = env_reward
reward += (score - previous_score) / 40
if done and flag_get: reward += 50
if done and not flag_get: reward -= 50
reward = reward / 10
```

Special extra termination penalties exist upstream for Worlds 7-4 and 4-4,
but they do not affect Level 1-1.

## Model Architecture

The upstream `PPO` module is a shared CNN torso with separate actor and critic
heads.

Input:

```text
(batch, 4, 84, 84)
```

Network:

| Layer | Details |
| --- | --- |
| Conv1 | `in=4`, `out=32`, kernel `3`, stride `2`, padding `1`, ReLU |
| Conv2 | `in=32`, `out=32`, kernel `3`, stride `2`, padding `1`, ReLU |
| Conv3 | `in=32`, `out=32`, kernel `3`, stride `2`, padding `1`, ReLU |
| Conv4 | `in=32`, `out=32`, kernel `3`, stride `2`, padding `1`, ReLU |
| Linear | `32 * 6 * 6 -> 512` |
| Actor head | `512 -> num_actions` |
| Critic head | `512 -> 1` |

Initialization:

- Orthogonal weights with ReLU gain for conv and linear layers.
- Biases initialized to `0`.

Action sampling:

- Actor logits are converted with `softmax`.
- Training samples actions from `Categorical(policy)`.
- The Modal evaluation scans also used stochastic `Categorical(policy)`
  actions to match training-time sampling.

## Modal Reproduction Additions

The reproduction kept upstream PPO/env/model behavior but added operational
infrastructure so we could run and measure it on Modal:

- Pinned old dependency stack for `gym-super-mario-bros`/`nes-py`/`gym`.
- Headless training and evaluation.
- Real `num_global_steps` stopping at `5,000,000`.
- Checkpoints approximately every `50,000` aggregate policy decisions.
- `metrics.jsonl` with train/eval rows.
- 20-episode stochastic checkpoint scans.
- 100-episode confirmation only if a checkpoint scan reached `>=80%`.
- Worker cleanup for multiprocessing envs.
- Deployed Modal app launcher so the run survived local process exit.

Modal run identifiers:

- App name: `viet-mario-ppo`
- App id: `ap-f17d64qgIJxeqYkHcnbQKl`
- Function call: `fc-01KTYS8P75TVHHRYC3G0NKTXYJ`
- Run name: `viet_level1_repro_5m_seed123`
- Volume: `viet-mario-ppo-data`

Useful artifact commands:

```bash
/tmp/modal-cli-venv/bin/modal volume get --force viet-mario-ppo-data /runs/viet_level1_repro_5m_seed123/metrics.jsonl /tmp/viet_level1_metrics.jsonl
/tmp/modal-cli-venv/bin/modal volume ls viet-mario-ppo-data /runs/viet_level1_repro_5m_seed123/checkpoints
/tmp/modal-cli-venv/bin/modal app logs ap-f17d64qgIJxeqYkHcnbQKl
```

## Interpretation

The baseline answers two separate questions:

1. Does the upstream repo work?

Yes. It learns a policy that clears Level 1-1 frequently under stochastic
evaluation.

2. How many timesteps does it need to solve Level 1-1?

In this seed-123 reproduction, it did not robustly solve Level 1-1 within 5M
aggregate policy-decision timesteps. It reached a useful but not solved
`12/20` around 3M and a best `14/20` around 4.6M.

The most important practical lesson is that checkpoint quality is highly
non-monotonic. The final checkpoint was worse than the best checkpoint, so
future comparisons should use systematic checkpoint scans and confirmation
rollouts rather than final policy quality alone.

Likely next baseline extensions:

- Re-evaluate the best checkpoint deterministically or with lower-temperature
  action selection.
- Run multiple seeds to estimate variance.
- Test an entropy coefficient schedule against the fixed `beta=0.01` upstream
  baseline.
- Keep the same eval protocol so improvements are attributable to the change,
  not to a looser measurement.
