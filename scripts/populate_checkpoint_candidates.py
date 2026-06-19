from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from stable_retro_ppo.wandb_artifacts import safe_artifact_stem
from stable_retro_ppo.wandb_utils import DEFAULT_WANDB_ENTITY, load_wandb_env


DEFAULT_PROJECT = f"{DEFAULT_WANDB_ENTITY}/SuperMarioBros-NES"
COMPLETION_RATE_KEY = "train/completion_episode_rate"


SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoint_candidates (
  id BIGSERIAL PRIMARY KEY,
  artifact_ref TEXT NOT NULL UNIQUE,
  run_name TEXT,
  run_path TEXT,
  checkpoint_step INTEGER,
  priority INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending',
  note TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


@dataclass(frozen=True)
class Candidate:
    artifact_ref: str
    run_name: str
    run_path: str
    checkpoint_step: int
    priority: int
    note: str


def artifact_aliases(artifact: Any) -> list[str]:
    aliases = []
    for alias in getattr(artifact, "aliases", []) or []:
        aliases.append(str(getattr(alias, "alias", alias)))
    return aliases


def checkpoint_step_from_artifact(artifact: Any) -> int | None:
    metadata = getattr(artifact, "metadata", {}) or {}
    step = metadata.get("checkpoint_step")
    if step is not None:
        return int(step)
    for alias in artifact_aliases(artifact):
        if alias.startswith("step-"):
            return int(alias.removeprefix("step-"))
    return None


def artifact_ref(artifact: Any) -> str:
    value = getattr(artifact, "qualified_name", None)
    if value:
        return str(value)
    return str(getattr(artifact, "name"))


def format_run_path(run: Any, project: str) -> str:
    path = getattr(run, "path", None)
    if isinstance(path, list | tuple):
        return "/".join(str(part) for part in path)
    if path:
        return str(path)
    return f"{project}/runs/{run.id}"


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def database_url(use_direct: bool) -> str:
    load_dotenv()
    if use_direct:
        value = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    else:
        value = os.environ.get("DATABASE_URL") or os.environ.get("DIRECT_DATABASE_URL")
    if not value:
        raise SystemExit("DATABASE_URL or DIRECT_DATABASE_URL must be set in .env")
    return value


def connect(url: str):
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def apply_schema(conn) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)


def completion_history(run: Any, key: str, *, samples: int) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for row in run.history(keys=[key, "global_step"], samples=samples, pandas=False):
        value = row.get(key)
        step = row.get("global_step", row.get("_step"))
        if value is None or step is None:
            continue
        try:
            rows.append((int(step), float(value)))
        except (TypeError, ValueError):
            continue
    return rows


def peak_steps(
    history: list[tuple[int, float]],
    *,
    min_peak_rate: float,
    top_peaks_per_run: int,
) -> list[tuple[int, float]]:
    seen_steps: set[int] = set()
    peaks: list[tuple[int, float]] = []
    for step, rate in sorted(history, key=lambda item: item[1], reverse=True):
        if rate < min_peak_rate:
            break
        if step in seen_steps:
            continue
        seen_steps.add(step)
        peaks.append((step, rate))
        if len(peaks) >= top_peaks_per_run:
            break
    return peaks


def nearest_checkpoint_steps(
    checkpoint_steps: list[int],
    target_step: int,
    *,
    neighbor_window: int,
) -> list[int]:
    if not checkpoint_steps:
        return []
    ordered = sorted(set(checkpoint_steps))
    nearest_index = min(
        range(len(ordered)),
        key=lambda index: (abs(ordered[index] - target_step), ordered[index]),
    )
    start = max(0, nearest_index - neighbor_window)
    end = min(len(ordered), nearest_index + neighbor_window + 1)
    return ordered[start:end]


def checkpoint_artifacts_for_run(api: Any, project: str, run_name: str) -> dict[int, Any]:
    collection = f"{project}/{safe_artifact_stem(run_name)}-checkpoint"
    artifacts = list(api.artifacts("model", collection))
    by_step: dict[int, Any] = {}
    for artifact in artifacts:
        step = checkpoint_step_from_artifact(artifact)
        if step is not None:
            by_step[step] = artifact
    return by_step


def candidates_for_run(
    api: Any,
    run: Any,
    project: str,
    *,
    min_peak_rate: float,
    top_peaks_per_run: int,
    neighbor_window: int,
    history_samples: int,
) -> list[Candidate]:
    history = completion_history(run, COMPLETION_RATE_KEY, samples=history_samples)
    peaks = peak_steps(
        history,
        min_peak_rate=min_peak_rate,
        top_peaks_per_run=top_peaks_per_run,
    )
    if not peaks:
        return []

    try:
        artifacts_by_step = checkpoint_artifacts_for_run(api, project, run.name)
    except Exception as exc:
        print(f"skip {run.name}: checkpoint artifacts unavailable: {exc}", flush=True)
        return []

    selected: dict[int, tuple[float, int]] = {}
    for peak_step, peak_rate in peaks:
        for step in nearest_checkpoint_steps(
            list(artifacts_by_step),
            peak_step,
            neighbor_window=neighbor_window,
        ):
            current = selected.get(step)
            if current is None or peak_rate > current[0]:
                selected[step] = (peak_rate, peak_step)

    run_path = format_run_path(run, project)
    candidates: list[Candidate] = []
    for step, (peak_rate, peak_step) in sorted(selected.items()):
        artifact = artifacts_by_step[step]
        priority = int(round(peak_rate * 1000))
        note = (
            f"training_peak_rate={peak_rate:.3f}; "
            f"training_peak_step={peak_step}; "
            f"selected_step={step}; "
            f"source=wandb_population"
        )
        candidates.append(
            Candidate(
                artifact_ref=artifact_ref(artifact),
                run_name=str(run.name),
                run_path=run_path,
                checkpoint_step=step,
                priority=priority,
                note=note,
            )
        )
    return candidates


def insert_candidates(conn, candidates: list[Candidate]) -> int:
    if not candidates:
        return 0
    rows = [candidate.__dict__ for candidate in candidates]
    with conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO checkpoint_candidates (
                  artifact_ref, run_name, run_path, checkpoint_step, priority, note
                )
                VALUES (
                  %(artifact_ref)s, %(run_name)s, %(run_path)s,
                  %(checkpoint_step)s, %(priority)s, %(note)s
                )
                ON CONFLICT (artifact_ref) DO UPDATE SET
                  run_name = EXCLUDED.run_name,
                  run_path = EXCLUDED.run_path,
                  checkpoint_step = EXCLUDED.checkpoint_step,
                  priority = GREATEST(checkpoint_candidates.priority, EXCLUDED.priority),
                  note = EXCLUDED.note
                """,
                rows,
                page_size=100,
            )
            return cur.rowcount


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Populate the Neon checkpoint eval inbox from worthwhile W&B checkpoints."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--min-peak-rate", type=float, default=0.8)
    parser.add_argument("--top-peaks-per-run", type=int, default=2)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--history-samples", type=int, default=10000)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_wandb_env()

    import wandb

    api = wandb.Api()
    conn = None if args.dry_run else connect(database_url(args.direct))
    total_candidates = 0
    changed_rows = 0
    inspected_runs = 0

    try:
        if conn is not None:
            apply_schema(conn)
        runs = api.runs(args.project)
        for run in runs:
            if args.max_runs and inspected_runs >= args.max_runs:
                break
            inspected_runs += 1
            candidates = candidates_for_run(
                api,
                run,
                args.project,
                min_peak_rate=args.min_peak_rate,
                top_peaks_per_run=args.top_peaks_per_run,
                neighbor_window=args.neighbor_window,
                history_samples=args.history_samples,
            )
            total_candidates += len(candidates)
            if candidates:
                print(f"{run.name}: {len(candidates)} candidates", flush=True)
            if not args.dry_run:
                changed_rows += insert_candidates(conn, candidates)
    finally:
        if conn is not None:
            conn.close()

    print(
        "done "
        "db=neon "
        f"runs_inspected={inspected_runs} "
        f"candidates_seen={total_candidates} "
        f"rows_changed={changed_rows} "
        f"dry_run={args.dry_run}",
        flush=True,
    )


if __name__ == "__main__":
    main()
