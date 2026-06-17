# Super Mario PPO Sample-Efficiency Ablation Log

Goal: find a hyperparameter configuration that reaches the Level 1 completed-episode
early-stop criterion with maximum sample efficiency. The concrete success target is
`>=80%` completion over the last `100` terminal training episodes by `2,000,000`
aggregate policy timesteps.

Unless noted otherwise, runs use:

- Environment: `SuperMarioBros-Nes-v0`, `Level1-1`.
- Hardware: SkyPilot `k8s/rtx4090`, NVIDIA RTX 4090.
- Seed: `23`.
- PPO shape: `n_envs=16`, `n_steps=512`, `batch_size=512`, `n_epochs=10`.
- Reward: `--reward-mode score`, `terminal_reward=50`, `reward_scale=10`.
- Termination: life loss terminal, completion threshold `x >= 3160`, terminate on completion.
- Stop: `--stop-completion-episode-window 100 --stop-completion-rate-threshold 0.8`.
- W&B project: `tsilva/SuperMarioBros-NES`.

## Current Ranking

| Rank | Run | Stop / max timestep | Result | Last checkpoint | Final model | W&B |
| ---: | --- | ---: | --- | --- | --- | --- |
| 1 | `b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906` | `2,558,256` | stopped at `80/100` | [2.5M](runs/b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/checkpoints/ppo_mario_2500000_steps.zip) | [final](runs/b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/final_model.zip) | [5jvhkenl](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/5jvhkenl) |
| 2 | `b6_fastent_lr2e4_5m_stop80ep100_seed23_20260614_152409` | `2,824,240` | stopped at `80/100` | [2.8M](runs/b6_fastent_lr2e4_5m_stop80ep100_seed23_20260614_152409/checkpoints/ppo_mario_2800000_steps.zip) | [final](runs/b6_fastent_lr2e4_5m_stop80ep100_seed23_20260614_152409/final_model.zip) | [9i1wphn6](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/9i1wphn6) |
| 3 | `b7_fastent_lr25e5_5m_stop80ep100_seed23_20260614_162644` | `3,066,816` | stopped at `80/100` | [3.0M](runs/b7_fastent_lr25e5_5m_stop80ep100_seed23_20260614_162644/checkpoints/ppo_mario_3000000_steps.zip) | [final](runs/b7_fastent_lr25e5_5m_stop80ep100_seed23_20260614_162644/final_model.zip) | [c3sp7hhe](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/c3sp7hhe) |
| 4 | `b8_lr2e4_ent0003_1500k_5m_stop80ep100_seed23_20260614_174804` | `3,593,696` | stopped at `80/100` | [3.5M](runs/b8_lr2e4_ent0003_1500k_5m_stop80ep100_seed23_20260614_174804/checkpoints/ppo_mario_3500000_steps.zip) | [final](runs/b8_lr2e4_ent0003_1500k_5m_stop80ep100_seed23_20260614_174804/final_model.zip) | [mmr1pp6m](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/mmr1pp6m) |
| 5 | `b3_fast_entropy_5m_stop80ep100_seed23_20260614_120804` | `3,774,448` | stopped at `80/100` | [3.7M](runs/b3_fast_entropy_5m_stop80ep100_seed23_20260614_120804/checkpoints/ppo_mario_3700000_steps.zip) | [final](runs/b3_fast_entropy_5m_stop80ep100_seed23_20260614_120804/final_model.zip) | [pbfrcflj](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pbfrcflj) |
| 6 | `b8_lr2e4_ent0001_2000k_5m_stop80ep100_seed23_20260614_174804` | `3,800,496` | stopped at `80/100` | [3.8M](runs/b8_lr2e4_ent0001_2000k_5m_stop80ep100_seed23_20260614_174804/checkpoints/ppo_mario_3800000_steps.zip) | [final](runs/b8_lr2e4_ent0001_2000k_5m_stop80ep100_seed23_20260614_174804/final_model.zip) | [9lhkxaj1](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/9lhkxaj1) |
| 7 | `sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508` | `3,979,616` | stopped at `80/100` | [3.9M](runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/checkpoints/ppo_mario_3900000_steps.zip) | [final](runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/final_model.zip) | offline |
| 8 | `b6_fastent_nsteps256_5m_stop80ep100_seed23_20260614_152409` | `4,047,184` | stopped at `80/100` | [4.0M](runs/b6_fastent_nsteps256_5m_stop80ep100_seed23_20260614_152409/checkpoints/ppo_mario_4000000_steps.zip) | [final](runs/b6_fastent_nsteps256_5m_stop80ep100_seed23_20260614_152409/final_model.zip) | [h5lrzqqz](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/h5lrzqqz) |
| 9 | `b4_ent0001_1500k_5m_stop80ep100_seed23_20260614_131847` | `4,292,304` | stopped at `80/100` | [4.2M](runs/b4_ent0001_1500k_5m_stop80ep100_seed23_20260614_131847/checkpoints/ppo_mario_4200000_steps.zip) | [final](runs/b4_ent0001_1500k_5m_stop80ep100_seed23_20260614_131847/final_model.zip) | [sgqwua3g](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/sgqwua3g) |
| 10 | `b8_lr2e4_batch256_5m_stop80ep100_seed23_20260614_174804` | `4,653,200` | stopped at `80/100` | [4.6M](runs/b8_lr2e4_batch256_5m_stop80ep100_seed23_20260614_174804/checkpoints/ppo_mario_4600000_steps.zip) | [final](runs/b8_lr2e4_batch256_5m_stop80ep100_seed23_20260614_174804/final_model.zip) | [3xmixuxf](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/3xmixuxf) |
| 11 | `b9_lr2e4_targetkl003_5m_stop80ep100_seed23_20260614_190906` | `4,712,176` | stopped at `80/100` | [4.7M](runs/b9_lr2e4_targetkl003_5m_stop80ep100_seed23_20260614_190906/checkpoints/ppo_mario_4700000_steps.zip) | [final](runs/b9_lr2e4_targetkl003_5m_stop80ep100_seed23_20260614_190906/final_model.zip) | [47esf8e7](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/47esf8e7) |
| 12 | `b7_fastent_lr3e4_5m_stop80ep100_seed23_20260614_162644` | `4,777,248` | stopped at `80/100` | [4.7M](runs/b7_fastent_lr3e4_5m_stop80ep100_seed23_20260614_162644/checkpoints/ppo_mario_4700000_steps.zip) | [final](runs/b7_fastent_lr3e4_5m_stop80ep100_seed23_20260614_162644/final_model.zip) | [eolol9qn](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/eolol9qn) |
| 13 | `b7_fastent_lr2e4_nsteps256_5m_stop80ep100_seed23_20260614_162644` | `5,001,216` | maxed at `9/100` | [5.0M](runs/b7_fastent_lr2e4_nsteps256_5m_stop80ep100_seed23_20260614_162644/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b7_fastent_lr2e4_nsteps256_5m_stop80ep100_seed23_20260614_162644/final_model.zip) | [2f10ec1p](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/2f10ec1p) |
| 14 | `b7_fastent_lr4e4_5m_stop80ep100_seed23_20260614_162644` | `5,005,312` | maxed at `33/100` | [5.0M](runs/b7_fastent_lr4e4_5m_stop80ep100_seed23_20260614_162644/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b7_fastent_lr4e4_5m_stop80ep100_seed23_20260614_162644/final_model.zip) | [ygkh3bsq](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ygkh3bsq) |
| 15 | `b9_lr2e4_lrd5e5_2m_5m_stop80ep100_seed23_20260614_190906` | `5,005,312` | maxed at `36/100` | [5.0M](runs/b9_lr2e4_lrd5e5_2m_5m_stop80ep100_seed23_20260614_190906/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b9_lr2e4_lrd5e5_2m_5m_stop80ep100_seed23_20260614_190906/final_model.zip) | [sdoyp170](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/sdoyp170) |
| 16 | `b9_lr2e4_batch256_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906` | `5,005,312` | maxed at `24/100` | [5.0M](runs/b9_lr2e4_batch256_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b9_lr2e4_batch256_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/final_model.zip) | [vo4t330k](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/vo4t330k) |
| 17 | `b8_lr2e4_ent0005_2000k_5m_stop80ep100_seed23_20260614_174804` | `5,005,312` | maxed at `0/100`, `106` total completions | [5.0M](runs/b8_lr2e4_ent0005_2000k_5m_stop80ep100_seed23_20260614_174804/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b8_lr2e4_ent0005_2000k_5m_stop80ep100_seed23_20260614_174804/final_model.zip) | [3ievym54](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/3ievym54) |
| 18 | `sky_score_style_simple_maxpool_10m_stop80ep100_seed23_20260614_091939` | `5,278,832` | stopped at `80/100` | [5.2M](runs/sky_score_style_simple_maxpool_10m_stop80ep100_seed23_20260614_091939/checkpoints/ppo_mario_5200000_steps.zip) | [final](runs/sky_score_style_simple_maxpool_10m_stop80ep100_seed23_20260614_091939/final_model.zip) | offline |
| 19 | `sky_score_style_simple_maxpool_10m_stop80ep100_lrdecay_seed23_20260614_102508` | `6,956,400` | stopped at `80/100` | [6.9M](runs/sky_score_style_simple_maxpool_10m_stop80ep100_lrdecay_seed23_20260614_102508/checkpoints/ppo_mario_6900000_steps.zip) | [final](runs/sky_score_style_simple_maxpool_10m_stop80ep100_lrdecay_seed23_20260614_102508/final_model.zip) | offline |
| 20 | `b3_no_progress_trunc_5m_stop80ep100_seed23_20260614_120804` | `5,005,312` | maxed at `37/100` | [5.0M](runs/b3_no_progress_trunc_5m_stop80ep100_seed23_20260614_120804/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b3_no_progress_trunc_5m_stop80ep100_seed23_20260614_120804/final_model.zip) | [clu7ef42](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/clu7ef42) |
| 21 | `b4_ent0003_1500k_5m_stop80ep100_seed23_20260614_131847` | `5,005,312` | maxed at `36/100` | [5.0M](runs/b4_ent0003_1500k_5m_stop80ep100_seed23_20260614_131847/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b4_ent0003_1500k_5m_stop80ep100_seed23_20260614_131847/final_model.zip) | [u2tz14ug](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/u2tz14ug) |
| 22 | `b5_fastent_normadv_5m_stop80ep100_seed23_20260614_144059` | `5,005,312` | maxed at `7/100` | [5.0M](runs/b5_fastent_normadv_5m_stop80ep100_seed23_20260614_144059/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b5_fastent_normadv_5m_stop80ep100_seed23_20260614_144059/final_model.zip) | [z38m7mkp](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/z38m7mkp) |
| 23 | `b4_ent0005_1500k_5m_stop80ep100_seed23_20260614_131847` | `5,005,312` | maxed at `6/100` | [5.0M](runs/b4_ent0005_1500k_5m_stop80ep100_seed23_20260614_131847/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b4_ent0005_1500k_5m_stop80ep100_seed23_20260614_131847/final_model.zip) | [ui5ejfgs](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ui5ejfgs) |
| 24 | `b6_fastent_gamma095_5m_stop80ep100_seed23_20260614_152409` | `5,005,312` | maxed at `0/100`, `6` total completions | [5.0M](runs/b6_fastent_gamma095_5m_stop80ep100_seed23_20260614_152409/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b6_fastent_gamma095_5m_stop80ep100_seed23_20260614_152409/final_model.zip) | [w12leawr](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/w12leawr) |
| 25 | `b10_post5_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260614_203813` | `5,005,312` | maxed at `0/100`, `0` total completions | [5.0M](runs/b10_post5_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260614_203813/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b10_post5_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260614_203813/final_model.zip) | [d3dorh0d](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/d3dorh0d) |
| 26 | `b11_post6_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_073609` | `5,005,312` | maxed at `0/100`, `0` total completions | [5.0M](runs/b11_post6_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_073609/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b11_post6_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_073609/final_model.zip) | [q0me90ft](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/q0me90ft) |
| 27 | `b3_clipped_progress_5m_stop80ep100_seed23_20260614_120804` | `5,005,312` | maxed at `0/100` | [5.0M](runs/b3_clipped_progress_5m_stop80ep100_seed23_20260614_120804/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b3_clipped_progress_5m_stop80ep100_seed23_20260614_120804/final_model.zip) | [oca6i52u](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/oca6i52u) |

## Completed Runs

### `b6_fastent_lr2e4_5m_stop80ep100_seed23_20260614_152409`

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `38`.
- Key change: current-best entropy schedule plus `learning_rate=2e-4`.
- Result: early-stopped at `2,824,240` timesteps.
- Stop marker:

```text
reason=training_completion_rate_threshold
timesteps=2824240
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1752
total_completed_episodes=158
```

- Last checkpoint: [ppo_mario_2800000_steps.zip](runs/b6_fastent_lr2e4_5m_stop80ep100_seed23_20260614_152409/checkpoints/ppo_mario_2800000_steps.zip)
  - SHA256: `979672b703c23a9a6e0adc1a885676724c47e44406d0545f57dd6156465d648b`
- Final model: [final_model.zip](runs/b6_fastent_lr2e4_5m_stop80ep100_seed23_20260614_152409/final_model.zip)
  - SHA256: `9d57a72c7cd708776ba20145fa31a198993781c82bb0c046fc7f959c92d5d502`
- W&B: [9i1wphn6](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/9i1wphn6)
- Interpretation: current best. Doubling PPO learning rate gave the first
  large sample-efficiency jump after Baseline 3, improving the previous best by
  `950,208` timesteps (`25.2%`).

### `b3_fast_entropy_5m_stop80ep100_seed23_20260614_120804`

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `4`.
- Key change: `ent_coef 0.01 -> 0.0003` over `2,000,000` timesteps.
- Result: early-stopped at `3,774,448` timesteps.
- Stop marker:

```text
reason=training_completion_rate_threshold
timesteps=3774448
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1514
total_completed_episodes=131
```

- Last checkpoint: [ppo_mario_3700000_steps.zip](runs/b3_fast_entropy_5m_stop80ep100_seed23_20260614_120804/checkpoints/ppo_mario_3700000_steps.zip)
  - SHA256: `70e02e2388b94d369baa91fa59dee8b1c111e87f23ade15ff2e86d4f05663cb6`
- Final model: [final_model.zip](runs/b3_fast_entropy_5m_stop80ep100_seed23_20260614_120804/final_model.zip)
  - SHA256: `e74f7708038ce2193ebdd8be182a78183e602aeb1c306ca8a7e387258a89d844`
- W&B: [pbfrcflj](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pbfrcflj)
- Interpretation: current best. It improves Baseline 3 by `205,168` timesteps,
  or `5.2%`, but remains far from the `2M` target.

### `b3_clipped_progress_5m_stop80ep100_seed23_20260614_120804`

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `4`.
- Key change: `--score-progress-clipped --progress-reward-cap 5 --progress-reward-scale 0.05`.
- Result: maxed at `5,005,312` timesteps with `0/100` recent completions.
- Last checkpoint: [ppo_mario_5000000_steps.zip](runs/b3_clipped_progress_5m_stop80ep100_seed23_20260614_120804/checkpoints/ppo_mario_5000000_steps.zip)
  - SHA256: `f29ff9a09d3757fa023aae84de0e0a3356bef07cb32d57117537b5c92b213ffd`
- Final model: [final_model.zip](runs/b3_clipped_progress_5m_stop80ep100_seed23_20260614_120804/final_model.zip)
  - SHA256: `dfea56976e5c4814b16cce7595fa30cfdc6ed044f2ac33c5cff339d38a2ebaec`
- W&B: [oca6i52u](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/oca6i52u)
- Interpretation: hard negative. The current vector training reward already uses
  wrapper-computed true `progress_delta`; clipping and down-weighting it removed
  too much useful dense signal.

### `b3_no_progress_trunc_5m_stop80ep100_seed23_20260614_120804`

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `4`.
- Key change: Baseline 3 entropy schedule plus `--no-progress-timeout-steps 800`.
- Result: maxed at `5,005,312` timesteps with `37/100` recent completions.
- Last checkpoint: [ppo_mario_5000000_steps.zip](runs/b3_no_progress_trunc_5m_stop80ep100_seed23_20260614_120804/checkpoints/ppo_mario_5000000_steps.zip)
  - SHA256: `76af7e28d7ac73fdf9e7171761fc9b79dfddee99ce39b8977f1d2dc6a5f5f100`
- Final model: [final_model.zip](runs/b3_no_progress_trunc_5m_stop80ep100_seed23_20260614_120804/final_model.zip)
  - SHA256: `e07d38f5386e78a901095cff16d9cfae7a656da30a23b5d4f16e957d0f13ff95`
- W&B: [clu7ef42](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/clu7ef42)
- Interpretation: negative for sample efficiency. It produced clears, but did not
  approach the stop threshold within `5M`.

### Batch 2 entropy schedule follow-ups

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `20`.
- Result summary: none reached the `2M` target. Only the `0.0001` floor run
  eventually reached the stop criterion, but it was slower than the current best.

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b4_ent0001_1500k_5m_stop80ep100_seed23_20260614_131847` | `ent_coef 0.01 -> 0.0001` over `1.5M` | stopped at `4,292,304`, `80/100` | [4.2M](runs/b4_ent0001_1500k_5m_stop80ep100_seed23_20260614_131847/checkpoints/ppo_mario_4200000_steps.zip) | [final](runs/b4_ent0001_1500k_5m_stop80ep100_seed23_20260614_131847/final_model.zip) | [sgqwua3g](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/sgqwua3g) |
| `b4_ent0003_1500k_5m_stop80ep100_seed23_20260614_131847` | `ent_coef 0.01 -> 0.0003` over `1.5M` | maxed at `5,005,312`, `36/100` | [5.0M](runs/b4_ent0003_1500k_5m_stop80ep100_seed23_20260614_131847/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b4_ent0003_1500k_5m_stop80ep100_seed23_20260614_131847/final_model.zip) | [u2tz14ug](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/u2tz14ug) |
| `b4_ent0005_1500k_5m_stop80ep100_seed23_20260614_131847` | `ent_coef 0.01 -> 0.0005` over `1.5M` | maxed at `5,005,312`, `6/100` | [5.0M](runs/b4_ent0005_1500k_5m_stop80ep100_seed23_20260614_131847/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b4_ent0005_1500k_5m_stop80ep100_seed23_20260614_131847/final_model.zip) | [ui5ejfgs](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ui5ejfgs) |
| `b4_fastent_epochs15_5m_stop80ep100_seed23_20260614_131847` | best fast entropy plus `n_epochs=15` | maxed at `5,005,312`, `0/100` recent, `73` total completions | [5.0M](runs/b4_fastent_epochs15_5m_stop80ep100_seed23_20260614_131847/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b4_fastent_epochs15_5m_stop80ep100_seed23_20260614_131847/final_model.zip) | [ino0h27p](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ino0h27p) |

Stop marker for `b4_ent0001_1500k...`:

```text
reason=training_completion_rate_threshold
timesteps=4292304
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1699
total_completed_episodes=137
```

Hashes:

- `b4_ent0001_1500k...` final: `0f7d0c1ccbb337187005a57a2b3e0d996383f472bc21abd1732bce23b6aafeaa`
- `b4_ent0001_1500k...` checkpoint: `a78ea074fed683d341a5c095d5d42183669b21e7373bfad122954315a3d36e75`
- `b4_ent0003_1500k...` final: `7d3134c073e46c2d6f8dc5d97750aca482509698610d678b0b8dfc1b99a0587f`
- `b4_ent0003_1500k...` checkpoint: `e6efd0e28fb3e7882aeff402e56098ef8bc4bba924cfbb8b947a15a54c87b112`
- `b4_ent0005_1500k...` final: `b42ff7dfef9b91af4640930a1bac938fc68b9b8d3902f1468e467702f5133c00`
- `b4_ent0005_1500k...` checkpoint: `07af39313ae49e77ad0745d8f353913f4bf6b7f4d9dcade20078269df4d11bd7`
- `b4_fastent_epochs15...` final: `32ee149d6d0a8aaa13a3df13175e412cbe32c9fae79b11b007b4496178e9d12a`
- `b4_fastent_epochs15...` checkpoint: `555150808f05a317b1c8b22de894add891e8e80880154c48616011910d812b83`

Interpretation: shortening the entropy decay to `1.5M` hurt sample efficiency
for the `0.0003` and `0.0005` floors. The `0.0001` floor produced a sharp late
phase transition and eventually stopped, but slower than the `0.0003 over 2M`
winner. Increasing PPO epochs to `15` caused an unstable burst of clears followed
by collapse, so more update reuse is not a clean sample-efficiency win here.

### `b5_fastent_normadv_5m_stop80ep100_seed23_20260614_144059`

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `36`.
- Key change: current-best entropy schedule plus `--normalize-advantage`.
- Result: maxed at `5,005,312` timesteps with `7/100` recent completions.
- Last checkpoint: [ppo_mario_5000000_steps.zip](runs/b5_fastent_normadv_5m_stop80ep100_seed23_20260614_144059/checkpoints/ppo_mario_5000000_steps.zip)
  - SHA256: `e7b4e6233837eac28432da361d9f21275e15087eec588ed7c9dc045feb1a4617`
- Final model: [final_model.zip](runs/b5_fastent_normadv_5m_stop80ep100_seed23_20260614_144059/final_model.zip)
  - SHA256: `e8e23e05cfe75bd235e3e011eaae2af01b4372d6c3335c71c8817559951e6131`
- W&B: [z38m7mkp](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/z38m7mkp)
- Interpretation: hard negative. It stayed at zero completions past `4.8M`
  and only reached sparse clears at the very end.

### Batch 3 learning-rate and update-frequency follow-ups

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `38`.
- Result summary: `learning_rate=2e-4` is the new best; `n_steps=256`
  eventually stopped but was slower; `gamma=0.95` was negative.

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b6_fastent_lr2e4_5m_stop80ep100_seed23_20260614_152409` | `learning_rate=2e-4` | stopped at `2,824,240`, `80/100` | [2.8M](runs/b6_fastent_lr2e4_5m_stop80ep100_seed23_20260614_152409/checkpoints/ppo_mario_2800000_steps.zip) | [final](runs/b6_fastent_lr2e4_5m_stop80ep100_seed23_20260614_152409/final_model.zip) | [9i1wphn6](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/9i1wphn6) |
| `b6_fastent_nsteps256_5m_stop80ep100_seed23_20260614_152409` | `n_steps=256` | stopped at `4,047,184`, `80/100` | [4.0M](runs/b6_fastent_nsteps256_5m_stop80ep100_seed23_20260614_152409/checkpoints/ppo_mario_4000000_steps.zip) | [final](runs/b6_fastent_nsteps256_5m_stop80ep100_seed23_20260614_152409/final_model.zip) | [h5lrzqqz](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/h5lrzqqz) |
| `b6_fastent_gamma095_5m_stop80ep100_seed23_20260614_152409` | `gamma=0.95` | maxed at `5,005,312`, `0/100`, `6` total completions | [5.0M](runs/b6_fastent_gamma095_5m_stop80ep100_seed23_20260614_152409/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b6_fastent_gamma095_5m_stop80ep100_seed23_20260614_152409/final_model.zip) | [w12leawr](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/w12leawr) |

Hashes:

- `b6_fastent_lr2e4...` final: `9d57a72c7cd708776ba20145fa31a198993781c82bb0c046fc7f959c92d5d502`
- `b6_fastent_lr2e4...` checkpoint: `979672b703c23a9a6e0adc1a885676724c47e44406d0545f57dd6156465d648b`
- `b6_fastent_nsteps256...` final: `61746b92192115b4483ef10811386df4546dd7ab4dce83b04e36fd2a015447ba`
- `b6_fastent_nsteps256...` checkpoint: `cc7109ac54e544177860e3166d80ff2d8ec4fb2c76c48dc01f62a4589c66915b`
- `b6_fastent_gamma095...` final: `28a1a389830f64e7d3b2e67848131021f216daa5b32e0f370de79309a1bec5a7`
- `b6_fastent_gamma095...` checkpoint: `0217c92b1ca58b1d97eff2409220302bf0e4b7ebf05c6c69f6960e9bbf57cfb5`

Interpretation: learning rate is now the highest-ROI axis. `2e-4` produced a
`25.2%` sample-efficiency improvement over the previous best. Shorter rollouts
alone eventually worked but were slower, and higher `gamma` was harmful.

### Batch 4 learning-rate sweep follow-ups

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `39`.
- Result summary: pushing learning rate above `2e-4` did not improve sample
  efficiency. `2.5e-4` was the closest runner-up; `3e-4` crossed the threshold
  only late; `4e-4` did not stop by `5M`; combining `2e-4` with `n_steps=256`
  was unstable and much worse than either change alone.

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b7_fastent_lr25e5_5m_stop80ep100_seed23_20260614_162644` | `learning_rate=2.5e-4` | stopped at `3,066,816`, `80/100` | [3.0M](runs/b7_fastent_lr25e5_5m_stop80ep100_seed23_20260614_162644/checkpoints/ppo_mario_3000000_steps.zip) | [final](runs/b7_fastent_lr25e5_5m_stop80ep100_seed23_20260614_162644/final_model.zip) | [c3sp7hhe](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/c3sp7hhe) |
| `b7_fastent_lr3e4_5m_stop80ep100_seed23_20260614_162644` | `learning_rate=3e-4` | stopped at `4,777,248`, `80/100` | [4.7M](runs/b7_fastent_lr3e4_5m_stop80ep100_seed23_20260614_162644/checkpoints/ppo_mario_4700000_steps.zip) | [final](runs/b7_fastent_lr3e4_5m_stop80ep100_seed23_20260614_162644/final_model.zip) | [eolol9qn](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/eolol9qn) |
| `b7_fastent_lr4e4_5m_stop80ep100_seed23_20260614_162644` | `learning_rate=4e-4` | maxed at `5,005,312`, `33/100` | [5.0M](runs/b7_fastent_lr4e4_5m_stop80ep100_seed23_20260614_162644/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b7_fastent_lr4e4_5m_stop80ep100_seed23_20260614_162644/final_model.zip) | [ygkh3bsq](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ygkh3bsq) |
| `b7_fastent_lr2e4_nsteps256_5m_stop80ep100_seed23_20260614_162644` | `learning_rate=2e-4`, `n_steps=256` | maxed at `5,001,216`, `9/100` | [5.0M](runs/b7_fastent_lr2e4_nsteps256_5m_stop80ep100_seed23_20260614_162644/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b7_fastent_lr2e4_nsteps256_5m_stop80ep100_seed23_20260614_162644/final_model.zip) | [2f10ec1p](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/2f10ec1p) |

Stop marker for `b7_fastent_lr25e5...`:

```text
reason=training_completion_rate_threshold
timesteps=3066816
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1282
total_completed_episodes=133
```

Stop marker for `b7_fastent_lr3e4...`:

```text
reason=training_completion_rate_threshold
timesteps=4777248
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1878
total_completed_episodes=198
```

Hashes:

- `b7_fastent_lr25e5...` final: `5fc4ad102b07dc791bff4a514502ab7184324b410148e26f3d41b6e7175d3372`
- `b7_fastent_lr25e5...` checkpoint: `644b2a931c36330ac5ce0678a2e67fc99ce0c4e1f52ce32a3fdec89e57a90ad4`
- `b7_fastent_lr3e4...` final: `269dc90d77b7974ebf55d0d48fe4152578df6d656cb94ea37ff2ea692f0bb891`
- `b7_fastent_lr3e4...` checkpoint: `5a1b721ff8b634bde7c9215288912ac5ad1cf8b8971f978f52cb259bcfcd8c98`
- `b7_fastent_lr4e4...` final: `24baa62b1c573ec7795e2320b06a871e0f7477fd956c5682967b5fbfe0b5e2ff`
- `b7_fastent_lr4e4...` checkpoint: `c03039147923745726d17787af92f145295911492f71e2b127097cc0f935aea4`
- `b7_fastent_lr2e4_nsteps256...` final: `219f4c859f696a862d30f74ad1b840aa8a6813c5c2d066cee17f60555f9db51b`
- `b7_fastent_lr2e4_nsteps256...` checkpoint: `823f89dc3a3d36dfa93fa5a715c287345ee83c8f95c1ae778cf079cb3c174358`

Interpretation: the learning-rate optimum for this setup appears close to
`2e-4`. Increasing it to `2.5e-4` remains viable but slower, while `3e-4` and
`4e-4` look too unstable for the `2M` target. The `n_steps=256` result also
shows that the previously decent shorter-rollout ablation does not compose
cleanly with the higher learning rate.

### Batch 5 entropy and minibatch follow-ups

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `40`.
- Run stamp: `20260614_174804`.
- Shared base: current-best `learning_rate=2e-4`, fast entropy schedule
  (`0.01 -> 0.0003` over `2M`) unless noted otherwise.
- Result summary: none reached the `2M` target and none beat the current best
  `2.824M` run. The `1.5M` entropy decay stopped fastest in this batch but was
  still slower than the existing winner. `batch_size=256` produced early clears
  before `2M`, then collapsed and recovered late.

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b8_lr2e4_ent0003_1500k_5m_stop80ep100_seed23_20260614_174804` | entropy decay to `0.0003` over `1.5M` | stopped at `3,593,696`, `80/100` | [3.5M](runs/b8_lr2e4_ent0003_1500k_5m_stop80ep100_seed23_20260614_174804/checkpoints/ppo_mario_3500000_steps.zip) | [final](runs/b8_lr2e4_ent0003_1500k_5m_stop80ep100_seed23_20260614_174804/final_model.zip) | [mmr1pp6m](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/mmr1pp6m) |
| `b8_lr2e4_ent0001_2000k_5m_stop80ep100_seed23_20260614_174804` | entropy floor `0.0001` over `2M` | stopped at `3,800,496`, `80/100` | [3.8M](runs/b8_lr2e4_ent0001_2000k_5m_stop80ep100_seed23_20260614_174804/checkpoints/ppo_mario_3800000_steps.zip) | [final](runs/b8_lr2e4_ent0001_2000k_5m_stop80ep100_seed23_20260614_174804/final_model.zip) | [9lhkxaj1](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/9lhkxaj1) |
| `b8_lr2e4_batch256_5m_stop80ep100_seed23_20260614_174804` | `batch_size=256` | stopped at `4,653,200`, `80/100` | [4.6M](runs/b8_lr2e4_batch256_5m_stop80ep100_seed23_20260614_174804/checkpoints/ppo_mario_4600000_steps.zip) | [final](runs/b8_lr2e4_batch256_5m_stop80ep100_seed23_20260614_174804/final_model.zip) | [3xmixuxf](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/3xmixuxf) |
| `b8_lr2e4_ent0005_2000k_5m_stop80ep100_seed23_20260614_174804` | entropy floor `0.0005` over `2M` | maxed at `5,005,312`, `0/100`, `106` total completions | [5.0M](runs/b8_lr2e4_ent0005_2000k_5m_stop80ep100_seed23_20260614_174804/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b8_lr2e4_ent0005_2000k_5m_stop80ep100_seed23_20260614_174804/final_model.zip) | [3ievym54](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/3ievym54) |

Stop marker for `b8_lr2e4_ent0003_1500k...`:

```text
reason=training_completion_rate_threshold
timesteps=3593696
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=2617
total_completed_episodes=154
```

Stop marker for `b8_lr2e4_ent0001_2000k...`:

```text
reason=training_completion_rate_threshold
timesteps=3800496
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1544
total_completed_episodes=175
```

Stop marker for `b8_lr2e4_batch256...`:

```text
reason=training_completion_rate_threshold
timesteps=4653200
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1839
total_completed_episodes=365
```

Hashes:

- `b8_lr2e4_ent0003_1500k...` final: `4ec4d783171c4491c277fa6a3f0a8ff400e72a373034913d33b5272552c15dbd`
- `b8_lr2e4_ent0003_1500k...` checkpoint: `dca0ff9f008f3745ea90c0a3d3c706f20be62b7367cf5b3db0700558c681ea16`
- `b8_lr2e4_ent0001_2000k...` final: `a8b987f5fd7ae259d8988711d3e75837ad910aa7c2d7553ab895b3222780a2e3`
- `b8_lr2e4_ent0001_2000k...` checkpoint: `ba4162e693e2e34bbb41e79769904d7b1df69309cdd9a98d389d9b33857fc10b`
- `b8_lr2e4_batch256...` final: `5e6609a113ecac7ccd3de5023bdf0f9ecf840021c50dd13aa969fc4c483374db`
- `b8_lr2e4_batch256...` checkpoint: `ee924e3413dc957c502703dec6b214d72dfce73214954177e3919a59c22c4cfb`
- `b8_lr2e4_ent0005_2000k...` final: `d3d82f4830de2478922f048206b2b7db7cd35083815d2ae861b2cb323ed65958`
- `b8_lr2e4_ent0005_2000k...` checkpoint: `ae0dd7ca31b097e73b56ce4ed95010d76f70e1f67da10513d8ed3db7b002a329`

Interpretation: Batch 5 did not improve sample efficiency. Faster decay to the
same `0.0003` entropy floor was viable but slower than `2M` decay with `lr=2e-4`.
The lower `0.0001` floor also stopped but later. The higher `0.0005` floor was
unstable and finished at `0/100`. `batch_size=256` is interesting only as a
failure mode: it produced early completions before `2M`, collapsed to `0/100`
around `2.5M`, and recovered only near `4.65M`.

### Batch 6 stability follow-ups

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `41`.
- Run stamp: `20260614_190906`.
- Shared base: current-best `learning_rate=2e-4`, fast entropy schedule
  (`0.01 -> 0.0003` over `2M`) unless noted otherwise.
- Result summary: LR decay to `1e-4` over the first `2M` is the new best at
  `2,558,256` timesteps. The current runs were allowed to finish and no
  replacement runs were launched.

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906` | LR `2e-4 -> 1e-4` over `2M` | stopped at `2,558,256`, `80/100` | [2.5M](runs/b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/checkpoints/ppo_mario_2500000_steps.zip) | [final](runs/b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/final_model.zip) | [5jvhkenl](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/5jvhkenl) |
| `b9_lr2e4_targetkl003_5m_stop80ep100_seed23_20260614_190906` | `target_kl=0.03` | stopped at `4,712,176`, `80/100` | [4.7M](runs/b9_lr2e4_targetkl003_5m_stop80ep100_seed23_20260614_190906/checkpoints/ppo_mario_4700000_steps.zip) | [final](runs/b9_lr2e4_targetkl003_5m_stop80ep100_seed23_20260614_190906/final_model.zip) | [47esf8e7](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/47esf8e7) |
| `b9_lr2e4_lrd5e5_2m_5m_stop80ep100_seed23_20260614_190906` | LR `2e-4 -> 5e-5` over `2M` | maxed at `5,005,312`, `36/100`, `172` total completions | [5.0M](runs/b9_lr2e4_lrd5e5_2m_5m_stop80ep100_seed23_20260614_190906/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b9_lr2e4_lrd5e5_2m_5m_stop80ep100_seed23_20260614_190906/final_model.zip) | [sdoyp170](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/sdoyp170) |
| `b9_lr2e4_batch256_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906` | `batch_size=256`, LR `2e-4 -> 1e-4` over `2M` | maxed at `5,005,312`, `24/100`, `157` total completions | [5.0M](runs/b9_lr2e4_batch256_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b9_lr2e4_batch256_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906/final_model.zip) | [vo4t330k](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/vo4t330k) |

Stop marker for `b9_lr2e4_lrd1e4_2m...`:

```text
reason=training_completion_rate_threshold
timesteps=2558256
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1276
total_completed_episodes=120
```

Stop marker for `b9_lr2e4_targetkl003...`:

```text
reason=training_completion_rate_threshold
timesteps=4712176
episode_window=100
completion_rate=0.800000
threshold=0.800000
total_terminal_episodes=1725
total_completed_episodes=368
```

Hashes:

- `b9_lr2e4_lrd1e4_2m...` final: `5e45554b09354b5c0ade678863a45df78633b3ac963d299103af66d9d3cb74d3`
- `b9_lr2e4_lrd1e4_2m...` checkpoint: `590c375c86038bf67e4edf8b6bd47ab6f996a8b0446d8a0aaae50623cf8f53b4`
- `b9_lr2e4_targetkl003...` final: `0614a92002f1f98518d9510d1d4115ff3065b0b41a2d11cb6fed05681e8d8434`
- `b9_lr2e4_targetkl003...` checkpoint: `40d558827859a401cfc47d2b78ac25a20a56d73db64729df16707d5e8f8c5825`
- `b9_lr2e4_lrd5e5_2m...` final: `c48c170a32c394a58e9f8e5e8761282407a3cafdf8d5932aa3cce12b84a97cad`
- `b9_lr2e4_lrd5e5_2m...` checkpoint: `f8a1d38aa31d9d07558faab83c9f34812e5b1c9c9d420c4fddb3f0c97e120d2a`
- `b9_lr2e4_batch256_lrd1e4_2m...` final: `bf1051b06ef218a9ca41bba59e1d999d10f81a02c6b0449a70603b83af227386`
- `b9_lr2e4_batch256_lrd1e4_2m...` checkpoint: `f344ae9c6b887ec23b992f8a871570b54c31753c7aa970a3af847a0b77c3fb17`

Interpretation: the useful axis was a mild post-discovery learning-rate decay,
not a heavy one. Decaying `2e-4 -> 1e-4` over `2M` improved the prior best
from `2,824,240` to `2,558,256`, a `265,984` timestep reduction (`9.4%`
fewer samples). Decaying to `5e-5` looked too conservative and failed to reach
reliable completion by `5M`. `target_kl=0.03` did stop, but much later, so
limiting updates via KL was not competitive. The `batch_size=256` combination
again failed to preserve reliable completion.

### Post5 reproduction of current best

- Date: 2026-06-14.
- SkyPilot job: `sandbox-sb3-stop10-4090`, job `42`.
- Run stamp: `20260614_203813`.
- Package change: `stable-retro-turbo==1.0.0.post5` on Linux x86_64.
- Intended test: reproduce the current best `b9_lr2e4_lrd1e4_2m...` run with
  the same seed and hyperparameters, changing only the stable-retro-turbo build.
- Result: maxed at `5,005,312` timesteps with `0/100` completion rate and
  `0` total completions. This did not reproduce the current best learning
  result.
- Wall clock: `1,667s` (`27m47s`) for the full `5M` budget.
- Throughput: final SB3 `time/fps=3023`; progress bar reported about `3094 it/s`.
- Previous post4 winner comparison: stopped at `2,558,256` in about `41m03s`
  with final `time/fps=1038`. Post5 delivered about `2.9x` higher reported fps,
  but did not preserve the learning trajectory.

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b10_post5_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260614_203813` | post5 build, otherwise current best config | maxed at `5,005,312`, `0/100`, `0` total completions | [5.0M](runs/b10_post5_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260614_203813/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b10_post5_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260614_203813/final_model.zip) | [d3dorh0d](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/d3dorh0d) |

Hashes:

- final: `6feb8079b8c92c2d419f76640e05cf0117259de5cdd39b17939c989c0190f08e`
- checkpoint: `5d0c6a033242a7eb8c0e4482ee5495f66e3fc109ec89f315a024d1f79c5bfe52`
- log: `5e570425b1ef03429db901b9349e0f97c2da8c05d78152876448645d2ba47106`
- walltime: `a6734dc03c6d451634284e3117704c72b6a5a90dc822cb10e51e632ede109576`

Follow-up regression audit:

- Audit SkyPilot job: `46`.
- Audit files:
  - [post4_behavior_seed23.json](logs/retro_version_audit/post4_behavior_seed23.json)
    sha256=`45c4e5b780fb845960cb78f9e017b6e971e0fca5494e6b06cb392a91a6b8dab8`
  - [post5_behavior_seed23.json](logs/retro_version_audit/post5_behavior_seed23.json)
    sha256=`01f78b8fa8572eceaf099fda1e4dc011205c22a0c333b6b357023737d931f0a2`
  - [comparison_seed23.json](logs/retro_version_audit/comparison_seed23.json)
    sha256=`d9c6d872f324e50380d777cdc5e70064bcdd216d1d3f1e0b4bdfe9e6bfc9bd29`
- Single-env API result: post4 and post5 matched on deterministic action
  traces, including native reward totals, done timing, deaths, and x position.
- Raw `StableRetroNativeVecEnv` result: post4 returned native RAM variables in
  `info` (`xscrollHi`, `xscrollLo`, `score`, `lives`, level bytes, etc.); post5
  returned empty `info` dicts while still producing nonzero native rewards.
- Wrapped training-env result: post4 produced nonzero project rewards and
  progress/completion info; post5 produced all-zero project rewards and default
  wrapper fields (`x_pos=0`, `progress_delta=0`, `score_delta=0`) because the
  source RAM variables were missing.

Interpretation: this is a post5 native-vector info regression, not a normal
RL seed miss or hyperparameter-transfer problem. The project reward wrapper
depends on native `info` fields, so post5 makes the training signal zero while
still stepping observations. Do not use `stable-retro-turbo==1.0.0.post5` for
`StableRetroNativeVecEnv` training until a fixed build passes this audit. Keep
training on Linux `1.0.0.post4` or a later verified build.

### Post6 fixed-info reproduction attempt

- Date: 2026-06-15.
- SkyPilot cluster/job: `sandbox-sb3-post6-audit-4090`, audit job `2`, training
  job `4`.
- Package change: installed `stable-retro-turbo==1.0.0.post6` into the locked
  training venv with `pip install --no-deps --force-reinstall`, so the intended
  runtime change was stable-retro-turbo only. A first attempted training job was
  canceled after plain `pip install` upgraded `gymnasium` to `1.3.0`; the clean
  run restored locked `gymnasium==1.2.3` before installing post6 with no deps.
- Audit result: post6 fixed the post5 empty-info regression for the training
  path. Single-env traces matched post4 (`single_equal=True`), wrapped vector
  traces matched post4 (`vector_equal=True`), and raw native-vector `info` keys
  were populated again. `raw_vector_equal=False` only because one raw random
  trace's native reward total differed; the wrapped reward/progress path used
  for training matched on the deterministic audit.
- Training run:
  `b11_post6_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_073609`.
- Result: maxed at `5,005,312` timesteps with `0/100` completion rate and
  `0` total completions. It did not reproduce the post4 winner.
- Wall clock: `1,615s` (`26m55s`) for the full `5M` budget.
- Throughput: final SB3 `time/fps=3120`; progress bar reported about
  `3093 it/s`.
- W&B: [q0me90ft](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/q0me90ft).

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b11_post6_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_073609` | post6 build, otherwise current best config | maxed at `5,005,312`, `0/100`, `0` total completions | [5.0M](runs/b11_post6_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_073609/checkpoints/ppo_mario_5000000_steps.zip) | [final](runs/b11_post6_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_073609/final_model.zip) | [q0me90ft](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/q0me90ft) |

Hashes:

- final:
  `5e41d76e23a59979cb4d816321c1f15189c26ad0e93f71164b019ebc47d025d1`
- checkpoint:
  `8ad6c6ce365999572347db1138374d5fb199c7305a67726fd7856d464a4c16d2`
- log:
  `bace47ec7b88207279ab334f5322abe09681fa6e908b14f06d74df22b51f28d3`
- walltime:
  `f311f4a688bee65632b6770749d3ef9c9162bd2cd1abd44a921f2300b3517043`
- post6 audit:
  `a385276b5c2d11282b9489943540f6a60c4e972c701df026301c480c29032caa`

Follow-up aliasing audit:

- Script: [audit_retro_observation_aliasing.py](../../scripts/audits/audit_retro_observation_aliasing.py).
- Artifacts:
  - [post4_aliasing_seed23.json](logs/retro_version_audit/post4_aliasing_seed23.json)
    SHA256 `872a1f589d6b9ca2010322bf3298f738513b68733d5fa10d4f7fd9d11a7eb2c9`
  - [post6_aliasing_seed23.json](logs/retro_version_audit/post6_aliasing_seed23.json)
    SHA256 `0bf1e62af4add1ae5f4896aec8daaf136912b3b650501e6c24b366b19f9d571f`
  - [comparison_post4_post6_aliasing_seed23.json](logs/retro_version_audit/comparison_post4_post6_aliasing_seed23.json)
    SHA256 `ecffae6fe632157e40968e408869e69f8bbf7facfdd5bb2cd2bbcd2e4b531cad`
- Result: post6 fixed the `info` regression but changed observation buffer
  ownership for `copy_observations=False`. Post4 returned distinct observation
  buffers and did not mutate prior observation arrays. Post6 returned the same
  pointer for reset/step observations and each `step()` mutated all prior
  observation references. Post6 with `copy_observations=True` did not mutate
  prior observations.
- Training-path confirmation: the fully wrapped SB3 training env reproduced the
  post6 aliasing because `make_vec_envs()` passes `copy_observations=False`.

Interpretation: post6 does not reproduce the post4 learning outcome because the
PPO rollout buffer can receive corrupted observation/action pairs. SB3 computes
an action from `_last_obs`, calls `env.step()`, then writes `_last_obs` into the
rollout buffer; with post6 `copy_observations=False`, `env.step()` has already
mutated that `_last_obs` reference. This explains why deterministic reward/info
audits matched while learning failed. Do not promote post6 as the new
sample-efficiency baseline unless a later stable-retro-turbo build restores
non-mutating returned observations for `copy_observations=False`, or the project
switches to `copy_observations=True` and reproduces the post4 learning curve.

### Post7 Linux/RTX2060 reproduction attempt

- Date: 2026-06-15.
- SkyPilot cluster/job: `sandbox-sb3-post7-2060`, training job `1`.
- Package change: installed `stable-retro-turbo==1.0.0.post7` into the locked
  training venv with `pip install --no-deps --force-reinstall`.
- Hardware: SSH SkyPilot target `ssh/beast2`, GPU `NVIDIA GeForce RTX 2060`.
- Training run:
  `b12_post7_2060_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_170235`.
- Result: reproduced the completed-episode stop criterion at `2,711,552`
  timesteps with `80/100` completion rate and `182` total completions.
- Wall clock: `2,055s` (`34m15s`) from training command start to finish.
- Throughput: last logged SB3 `time/fps` before early stop was about `1331`.
- W&B: [an80iif6](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/an80iif6).

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b12_post7_2060_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_170235` | post7 build on RTX2060, otherwise current best config | stopped at `2,711,552`, `80/100`, `182` total completions | `runs/.../checkpoints/ppo_mario_2700000_steps.zip` | `runs/.../final_model.zip` | [an80iif6](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/an80iif6) |

Interpretation: post7 fixes the training-breaking post5/post6 regressions on a
Linux/RTX training path. The result is slightly slower in sample count than the
post4 RTX4090 winner (`2,558,256`), but it reproduces the same success
criterion under changed hardware. Treat post7 as a validated candidate runtime
for future Linux/RTX experiments; keep post4 as the exact historical baseline
for prior sample-efficiency comparisons unless the dependency pins are
deliberately updated.

### Post7 Linux/RTX4090 reproduction attempt

- Date: 2026-06-15.
- SkyPilot cluster/job: `sandbox-sb3-post7-4090`, training job `1`.
- Package change: installed `stable-retro-turbo==1.0.0.post7` into the locked
  training venv with `pip install --no-deps --force-reinstall`; setup printed
  `stable-retro-turbo 1.0.0.post7` on `NVIDIA GeForce RTX 4090`.
- Training run:
  `b13_post7_4090_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_190534`.
- Result: did not reproduce the completed-episode stop criterion. It maxed at
  `5,005,312` timesteps with `31/100` recent completions and `197` total
  completions.
- Wall clock: `1,635s` (`27m15s`) for the full `5M` budget.
- Throughput: final SB3 `time/fps=3093`.
- W&B: [4hepwv0x](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/4hepwv0x).

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b13_post7_4090_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_190534` | post7 build on RTX4090, otherwise current best config | maxed at `5,005,312`, `31/100`, `197` total completions | `runs/.../checkpoints/ppo_mario_5000000_steps.zip` | `runs/.../final_model.zip` | [4hepwv0x](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/4hepwv0x) |
| `b14_post7_4090_repeat_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_194155` | post7 build on RTX4090, same seed/config repeat | stopped just after `4,227,072` logged timesteps, `80/100`, `189` total completions | `runs/.../checkpoints/ppo_mario_4200000_steps.zip` | `runs/.../final_model.zip` | [feqsvt6f](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/feqsvt6f) |

Interpretation: post7 can reproduce the completed-episode success criterion on
Linux/RTX4090, but the result is not exactly reproducible even with the same
seed and config. The first RTX4090 post7 run maxed at `31/100`; the immediate
repeat stayed at `0` completions past `2.9M`, climbed late, and stopped after
the last logged update at `4,227,072` timesteps. This points to normal PPO/RL
and systems nondeterminism being large enough to change sample-efficiency
conclusions from one run. Treat post7 as training-validated, but compare future
sample-efficiency changes with repeat seeds/runs rather than a single run.

### Post4 Linux/RTX4090 same-build repeatability check

- Date: 2026-06-15.
- SkyPilot cluster/job: `sandbox-sb3-post4-4090-repeat2`, training job `1`.
- Package: locked Linux `stable-retro-turbo==1.0.0.post4`, verified in setup
  output on `NVIDIA GeForce RTX 4090`.
- Test: two concurrent child processes on the same SkyPilot RTX4090 node, both
  using the same seed/config as the historical best run
  `b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906`.

| Run | Key change | Result | Last checkpoint | Final model | W&B |
| --- | --- | --- | --- | --- | --- |
| `b15_post4_4090_repeat_a_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_201454` | post4 build, same seed/config as current best, concurrent repeat A | maxed at `5,005,312`, final `0/100`, `173` total completions | `runs/.../checkpoints/ppo_mario_5000000_steps.zip` | `runs/.../final_model.zip` | [pvsxz4u7](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pvsxz4u7) |
| `b15_post4_4090_repeat_b_best_lrd1e4_2m_5m_stop80ep100_seed23_20260615_201454` | post4 build, same seed/config as current best, concurrent repeat B | maxed at `5,005,312`, final `30/100`, `111` total completions | `runs/.../checkpoints/ppo_mario_5000000_steps.zip` | `runs/.../final_model.zip` | [l1trgg71](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/l1trgg71) |

Wall clock: repeat A `3,352s` (`55m52s`), repeat B `3,356s` (`55m56s`).
Throughput was about `1500` SB3 fps per child while the two trainers shared the
same node.

Interpretation: this did not reproduce the original post4 winner's `80/100`
early stop at `2,558,256` timesteps. It did reproduce nonzero learning and
level clears, but not reliability: repeat A peaked near `60/100` before
collapsing to `0/100`, and repeat B finished at `30/100`. The old post4 build
therefore also exhibits large same-seed outcome variance under this parallel
setup. The post7 RTX4090 split result is not sufficient evidence of a unique
post7 regression; deciding whether post7 is worse requires repeated runs or a
matched isolated post4/post7 distribution comparison.

### Post10 Linux/RTX4090 reproduction attempt

- Date: 2026-06-16.
- SkyPilot cluster/job: `sandbox-sb3-post10-4090-repro`, training jobs `1` and
  `20`; job `9` was an invalid ad hoc `sky exec` parallel attempt and should be
  ignored for throughput/learning comparisons.
- Package change: installed `stable-retro-turbo==1.0.0.post10` into the locked
  training venv with `pip install --no-deps --force-reinstall`; setup printed
  `stable-retro-turbo 1.0.0.post10` on `NVIDIA GeForce RTX 4090`.
- Test: same seed/config as the current best run
  `b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906`.

| Run | Key change | Result | Notes |
| --- | --- | --- | --- |
| `b16_post10_4090_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260616_063000` | post10 build, same seed/config | maxed at 5M, final observed `47/100`, `253` total completions | Wall clock `1,719s` (`28m39s`), final observed fps about `2939`; transiently reached about `66/100` near `3.2M`, then regressed and partially recovered. |
| `b16_post10_4090_repro_best_lrd1e4_2m_5m_stop80ep100_seed23_20260616_070550` | clean post10 repeat, same seed/config | manually stopped before conclusion | Last checked around `2.78M` with `4` total completions; first completions appeared around `2.70M`; stopped at status `143` after `1,307s` because the user requested all runs stop. |

Interpretation: post10 is not exhibiting the catastrophic post5/post6 failure
mode: it trains at expected RTX4090 throughput and learns real level clears.
It did not reproduce the `80/100` success criterion in the one completed
isolated run, and the clean repeat was stopped too early to classify. Keep
post10 as a candidate runtime, but do not claim sample-efficiency improvement
without completed matched repeat distributions.
