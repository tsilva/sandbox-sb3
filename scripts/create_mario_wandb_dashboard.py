#!/usr/bin/env python3
"""Create the focused W&B workspace for Mario PPO sweep diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

try:
    from wandb_workspaces import workspaces as ws
    from wandb_workspaces.reports import v2 as wr
except ImportError as exc:  # pragma: no cover - operator-facing dependency hint
    raise SystemExit(
        "Missing W&B workspace support. Install it with: "
        "uv --cache-dir .uv-cache pip install --python .venv/bin/python "
        "'wandb[workspaces]==0.22.3'"
    ) from exc


DEFAULT_QUERY = r"^b2[0-9]_"
DEFAULT_PROJECT = "SuperMarioBros-NES"
DEFAULT_ENTITY = "tsilva"


def line(
    title: str,
    metric: str,
    *,
    x: int,
    y: int,
    w: int = 12,
    h: int = 6,
    ymin: float | None = None,
    ymax: float | None = None,
) -> wr.LinePlot:
    return wr.LinePlot(
        title=title,
        x="global_step",
        y=[metric],
        range_y=(ymin, ymax),
        smoothing_type="none",
        max_runs_to_show=30,
        legend_position="east",
        legend_template="${run:displayName}",
        layout=wr.Layout(x=x, y=y, w=w, h=h),
    )


def markdown(text: str, *, x: int, y: int, w: int = 24, h: int = 3) -> wr.MarkdownPanel:
    return wr.MarkdownPanel(markdown=text.strip(), layout=wr.Layout(x=x, y=y, w=w, h=h))


def build_workspace(entity: str, project: str, name: str, query: str) -> ws.Workspace:
    return ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        settings=ws.WorkspaceSettings(
            x_axis="global_step",
            smoothing_type="none",
            smoothing_weight=0,
            ignore_outliers=False,
            max_runs=30,
            tooltip_number_of_runs="all_runs",
            tooltip_color_run_names=True,
            group_by_prefix="last",
        ),
        runset_settings=ws.RunsetSettings(
            query=query,
            regex_query=True,
            order=[ws.Ordering(ws.Summary("train/completion_episode_rate"), ascending=False)],
            pinned_runs=["6hvqs5et", "5ktcw6dm", "lugd5cth", "iab5gq4b", "j0q58wg4", "cyyfs6s5"],
            pinned_columns=[
                "Name",
                "State",
                "group",
                "summary.train/completion_episode_rate",
                "summary.train/completion_episodes_total",
                "summary.global_step",
                "config.learning_rate",
                "config.ent_coef_final",
                "config.ent_coef_schedule_timesteps",
                "config.clip_range",
                "config.target_kl",
                "config.n_steps",
            ],
        ),
        sections=[
            ws.Section(
                name="Read This First",
                is_open=True,
                panels=[
                    markdown(
                        """
                        Primary question: which runs become reliable at Level1-1, not merely high reward?

                        Read `train/completion_episode_rate` first. It is already the rolling success rate over the last 100 completed terminal episodes, so the dashboard uses no extra smoothing. Then use `completion_episodes_total` to distinguish early discovery from reliable exploitation, and use PPO internals to diagnose whether promising policies were destroyed by later updates.
                        """,
                        x=0,
                        y=0,
                    ),
                ],
            ),
            ws.Section(
                name="Outcome: sample efficiency and reliability",
                is_open=True,
                panels=[
                    line(
                        "North star: Level1-1 completion rate over last 100 terminal episodes",
                        "train/completion_episode_rate",
                        x=0,
                        y=0,
                        w=24,
                        h=8,
                        ymin=0,
                        ymax=1.05,
                    ),
                    line(
                        "Discovery volume: cumulative completed episodes",
                        "train/completion_episodes_total",
                        x=0,
                        y=8,
                    ),
                    line(
                        "Denominator: cumulative terminal episodes",
                        "train/terminal_episodes_total",
                        x=12,
                        y=8,
                    ),
                    wr.BarPlot(
                        title="Final summary: completion rate",
                        metrics=["train/completion_episode_rate"],
                        orientation="h",
                        max_runs_to_show=30,
                        layout=wr.Layout(x=0, y=14, w=12, h=6),
                    ),
                    wr.ScatterPlot(
                        title="Final reliability vs total completions",
                        x="train/completion_episodes_total",
                        y="train/completion_episode_rate",
                        z="global_step",
                        range_y=(0, 1.05),
                        legend_template="${run:displayName}",
                        layout=wr.Layout(x=12, y=14, w=12, h=6),
                    ),
                ],
            ),
            ws.Section(
                name="Behavior: reward can lie",
                is_open=True,
                panels=[
                    line("Episode reward mean", "rollout/ep_rew_mean", x=0, y=0),
                    line("Episode length mean", "rollout/ep_len_mean", x=12, y=0),
                    line("Training fps", "time/fps", x=0, y=6),
                ],
            ),
            ws.Section(
                name="Mechanism: PPO update health",
                is_open=True,
                panels=[
                    line("Approx KL: did updates get too large?", "train/approx_kl", x=0, y=0),
                    line("Clip fraction: how constrained were updates?", "train/clip_fraction", x=12, y=0, ymin=0),
                    line("Policy gradient loss", "train/policy_gradient_loss", x=0, y=6),
                    line("Value loss", "train/value_loss", x=12, y=6, ymin=0),
                    line("Explained variance", "train/explained_variance", x=0, y=12, ymin=-0.1, ymax=1.05),
                ],
            ),
            ws.Section(
                name="Mechanism: exploration and step size",
                is_open=True,
                panels=[
                    line("Learning rate schedule", "train/learning_rate", x=0, y=0),
                    line("Entropy loss", "train/entropy_loss", x=12, y=0),
                    wr.ParameterImportancePlot(
                        with_respect_to="train/completion_episode_rate",
                        layout=wr.Layout(x=0, y=6, w=24, h=8),
                    ),
                ],
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="Mario 100of100 Sweep Diagnosis")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    workspace = build_workspace(args.entity, args.project, args.name, args.query).save()
    print(workspace.url)


if __name__ == "__main__":
    main()
