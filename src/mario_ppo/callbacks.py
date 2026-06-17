from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from mario_ppo.artifacts import checkpoint_step, log_wandb_model_artifact, wandb_artifacts_enabled


class WandbCheckpointArtifactCallback(BaseCallback):
    def __init__(
        self,
        wandb_run,
        args: argparse.Namespace,
        checkpoint_dir: str,
        scan_freq: int,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.args = args
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scan_freq = scan_freq
        self.logged_paths: set[Path] = set()

    def _on_step(self) -> bool:
        if self.scan_freq <= 1 or self.n_calls % self.scan_freq == 0:
            self.log_new_checkpoints()
        return True

    def log_new_checkpoints(self) -> None:
        if not wandb_artifacts_enabled(self.wandb_run, self.args):
            return

        for checkpoint_path in sorted(self.checkpoint_dir.glob("*.zip")):
            resolved_path = checkpoint_path.resolve()
            if resolved_path in self.logged_paths:
                continue
            step = checkpoint_step(checkpoint_path)
            aliases = ["latest"]
            if step is not None:
                aliases.append(f"step-{step}")
            log_wandb_model_artifact(
                self.wandb_run,
                self.args,
                checkpoint_path,
                kind="checkpoint",
                aliases=aliases,
            )
            self.logged_paths.add(resolved_path)


class RollingCompletionStopCallback(BaseCallback):
    def __init__(
        self,
        rolling_window: int,
        threshold: float,
        run_dir: str,
        wandb_run=None,
    ):
        super().__init__()
        self.rolling_window = rolling_window
        self.threshold = threshold
        self.run_dir = Path(run_dir)
        self.wandb_run = wandb_run
        self.rollout_completion_count = 0
        self.total_completion_count = 0
        self.rollout_counts: deque[int] = deque(maxlen=rolling_window)
        self.rolling_mean = 0.0
        self.stop_requested = False

    def _on_step(self) -> bool:
        if self.stop_requested:
            return False

        infos = self.locals.get("infos", [])
        for info in infos:
            if bool(info.get("completion_event", info.get("level_complete", False))):
                self.rollout_completion_count += 1
                self.total_completion_count += 1

        return True

    def _on_rollout_end(self) -> None:
        self.rollout_counts.append(self.rollout_completion_count)
        self.rolling_mean = sum(self.rollout_counts) / len(self.rollout_counts)
        window_full = len(self.rollout_counts) >= self.rolling_window

        self.logger.record("train/completion_events_rollout", self.rollout_completion_count)
        self.logger.record("train/completion_events_rolling_mean", self.rolling_mean)
        self.logger.record("train/completion_events_total", self.total_completion_count)

        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "train/completion_events_rollout": self.rollout_completion_count,
                    "train/completion_events_rolling_mean": self.rolling_mean,
                    "train/completion_events_total": self.total_completion_count,
                    "global_step": self.num_timesteps,
                },
                step=self.num_timesteps,
            )

        print(
            "completion rolling: "
            f"rollout={self.rollout_completion_count} "
            f"mean={self.rolling_mean:.3f}/{self.threshold:g} "
            f"window={len(self.rollout_counts)}/{self.rolling_window} "
            f"total={self.total_completion_count}",
            flush=True,
        )

        if window_full and self.rolling_mean >= self.threshold:
            self.stop_requested = True
            stop_path = self.run_dir / "early_stop.txt"
            stop_path.write_text(
                "\n".join(
                    [
                        "reason=rolling_completion_threshold",
                        f"timesteps={self.num_timesteps}",
                        f"rolling_window={self.rolling_window}",
                        f"rolling_mean={self.rolling_mean:.6f}",
                        f"threshold={self.threshold:.6f}",
                        f"total_completion_count={self.total_completion_count}",
                    ],
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                "early stop requested: "
                f"rolling completion mean {self.rolling_mean:.3f} >= {self.threshold:g}",
                flush=True,
            )

        self.rollout_completion_count = 0


class TrainingCompletionRateStopCallback(BaseCallback):
    def __init__(
        self,
        episode_window: int,
        rate_threshold: float,
        run_dir: str,
        wandb_run=None,
    ):
        super().__init__()
        if not 0.0 < rate_threshold <= 1.0:
            raise ValueError("rate_threshold must be in (0, 1]")
        self.episode_window = episode_window
        self.rate_threshold = rate_threshold
        self.run_dir = Path(run_dir)
        self.wandb_run = wandb_run
        self.completed_episode_outcomes: deque[int] = deque(maxlen=episode_window)
        self.total_terminal_episodes = 0
        self.total_completed_episodes = 0
        self.stop_requested = False

    def _on_step(self) -> bool:
        if self.stop_requested:
            return False

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos, strict=False):
            if not bool(done) or bool(info.get("global_reset", False)):
                continue

            completed = bool(info.get("completion_event", info.get("level_complete", False)))
            self.completed_episode_outcomes.append(int(completed))
            self.total_terminal_episodes += 1
            if completed:
                self.total_completed_episodes += 1

            completion_rate = self.completion_rate
            self.logger.record("train/completion_episode_rate", completion_rate)
            self.logger.record(
                "train/completion_episode_window_size", len(self.completed_episode_outcomes)
            )
            self.logger.record("train/completion_episodes_total", self.total_completed_episodes)
            self.logger.record("train/terminal_episodes_total", self.total_terminal_episodes)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "train/completion_episode_rate": completion_rate,
                        "train/completion_episode_window_size": len(
                            self.completed_episode_outcomes,
                        ),
                        "train/completion_episodes_total": self.total_completed_episodes,
                        "train/terminal_episodes_total": self.total_terminal_episodes,
                        "global_step": self.num_timesteps,
                    },
                    step=self.num_timesteps,
                )

            if (
                len(self.completed_episode_outcomes) >= self.episode_window
                and completion_rate >= self.rate_threshold
            ):
                self.request_stop(completion_rate)
                return False

        return True

    @property
    def completion_rate(self) -> float:
        if not self.completed_episode_outcomes:
            return 0.0
        return sum(self.completed_episode_outcomes) / len(self.completed_episode_outcomes)

    def request_stop(self, completion_rate: float) -> None:
        self.stop_requested = True
        stop_path = self.run_dir / "early_stop.txt"
        stop_path.write_text(
            "\n".join(
                [
                    "reason=training_completion_rate_threshold",
                    f"timesteps={self.num_timesteps}",
                    f"episode_window={self.episode_window}",
                    f"completion_rate={completion_rate:.6f}",
                    f"threshold={self.rate_threshold:.6f}",
                    f"total_terminal_episodes={self.total_terminal_episodes}",
                    f"total_completed_episodes={self.total_completed_episodes}",
                ],
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "early stop requested: "
            f"training completion rate {completion_rate:.3f} >= {self.rate_threshold:g} "
            f"over last {self.episode_window} completed episodes",
            flush=True,
        )
