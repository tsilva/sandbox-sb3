# GPU Instances

Last updated: 2026-06-17

Use this file as the repo-local source of truth for known GPU instances, launch targets, benchmark-backed concurrency, and operational gotchas. Re-check live availability before launching, but do not rediscover these basics from scratch unless the facts here fail.

## Quick Choice

| Use case | Target | Default shape |
| --- | --- | --- |
| Highest-throughput Mario PPO screening | `k8s/rtx4090` | 5 concurrent children, `env_threads=4` |
| Lower-contention RTX4090 confirmation batch | `k8s/rtx4090` | 3-4 concurrent children, `env_threads=4` |
| Small-GPU batch screening | `ssh/beast2` | 4 concurrent children, `env_threads=2` |
| Faster individual turnaround on RTX2060 | `ssh/beast2` | 2 concurrent children, `env_threads=4` |

Refresh these defaults when changing `n_envs`, `n_steps`, model size, deterministic CUDA flags, runtime package version, W&B/artifact behavior, or target node CPU shape.

## W&B Artifact Storage: Cloudflare R2

Use Cloudflare R2 as the default byte store for W&B model artifacts so W&B keeps
metrics, config, aliases, lineage, and reference metadata instead of storing every
checkpoint zip directly.

Current repo-local `.env` is expected to define:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_S3_ENDPOINT_URL
AWS_REGION
CHECKPOINT_BUCKET_URI
WANDB_API_KEY
```

The active smoke-tested storage target is the bucket referenced by
`CHECKPOINT_BUCKET_URI`. As of 2026-06-17, the R2 smoke test succeeded: bucket
exists, object upload/download worked, and the smoke object was deleted.

Training support:

- `mario_ppo.train` reads `--wandb-artifact-storage-uri`, or falls back to
  `WANDB_ARTIFACT_STORAGE_URI`, then `CHECKPOINT_BUCKET_URI`.
- When that URI is set, checkpoint/final/best model zips upload to R2/S3 and W&B
  logs reference artifacts with the existing aliases such as `latest`, `final`,
  `best`, and `step-<N>`.
- Without that URI, W&B artifact behavior remains direct file upload.

Preferred SkyPilot launch pattern from this repo:

```bash
(
  set -a
  . ./.env
  set +a

  sky launch -c <cluster-name> -y <task.yaml> \
    --env AWS_REGION \
    --env AWS_S3_ENDPOINT_URL \
    --env CHECKPOINT_BUCKET_URI \
    --secret AWS_ACCESS_KEY_ID \
    --secret AWS_SECRET_ACCESS_KEY \
    --secret WANDB_API_KEY
)
```

This passes only the bucket config and secrets needed by the task. Older task
YAMLs often mount `.env` to `~/.env` and source it in `run`; that works, but
prefer `--env`/`--secret` for new tasks and remove the `.env` file mount when
editing those YAMLs.

Local R2 smoke test:

```bash
(
  set -a
  . ./.env
  set +a

  uv --cache-dir .uv-cache run python scripts/setup_r2_bucket.py \
    "$(python - <<'PY'
import os
from urllib.parse import urlparse
print(urlparse(os.environ["CHECKPOINT_BUCKET_URI"]).netloc)
PY
)" \
    --prefix "$(python - <<'PY'
import os
from urllib.parse import urlparse
print(urlparse(os.environ["CHECKPOINT_BUCKET_URI"]).path.strip("/") or "mario-ppo")
PY
)"
)
```

## RTX4090: beast-3

### Access

- SkyPilot infra: `k8s/rtx4090`
- GPU host SSH: `ssh tsilva@beast-3`
- SkyPilot dashboard/API server: `http://192.168.0.151:46580`
- GPU: NVIDIA GeForce RTX 4090, 24 GB VRAM
- Observed driver: `595.71.05`, reporting CUDA support up to `13.2`
- Default SkyPilot image: `docker:us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:latest`
- Task Python bootstrap path inside the standard image: `/home/sky/skypilot-runtime/bin/python`

### Mario PPO Scheduling

Benchmarked for the current near-best Mario PPO shape: `n_envs=16`, `n_steps=512`, `batch_size=512`, `n_epochs=10`, `stable-retro-turbo==1.0.0.post10`, Torch `2.12.0+cu130`, 262,144 timesteps per child, W&B logging enabled, model artifact uploads disabled.

| Shape | Aggregate wall fps | Avg GPU util | Max VRAM | Notes |
| --- | ---: | ---: | ---: | --- |
| 1 child, `env_threads=4` | 2,881 | 16.7% | 990 MiB | Single trainer severely underuses the GPU. |
| 3 children, `env_threads=4` | 5,578 | 44.3% | 2,604 MiB | Good lower-contention confirmation shape. |
| 4 children, `env_threads=4` | 5,825-5,858 | ~52% | ~3,412 MiB | Good confirmation/screening compromise. |
| 5 children, `env_threads=4` | 6,242-6,271 | 57-58% | 4,219 MiB | Best measured aggregate throughput. |
| 5 children, `env_threads=2` | 5,878 | 51.9% | not recorded here | Lower contention but about 6% slower. |
| 5 children, `env_threads=1` | 4,615 | 37.0% | not recorded here | Underfeeds rollout collection. |

Default to 5 children with `env_threads=4` for screening queues. Use 3-4 children with `env_threads=4` when individual run latency, lower CPU contention, or easier debugging matters more than total aggregate throughput.

### Operational Notes

- If a single RTX4090 is already held by an `UP` SkyPilot cluster with no active managed jobs, launching a second Kubernetes cluster may fail with `Insufficient nvidia.com/gpu`. Prefer submitting to the warm cluster with `sky launch -c <existing-cluster> -y <task.yaml>` when reuse is intended.
- Use normal `sky launch -c <warm-cluster> -y <task.yaml>` for valid repeat runs. Starting a second long trainer via ad hoc `sky exec` inside an already-running task produced pathological throughput around 140 fps.
- `sky cancel` against `k8s/rtx4090` jobs can fail with `PermissionError: [Errno 13] Permission denied` while trying to `os.killpg`. If that happens, identify the training process group with `sky exec <cluster> 'ps -eo pid,ppid,pgid,stat,cmd | grep <run-name>'`, terminate the process group with `kill -TERM -<pgid>`, verify no trainer remains, then run `sky down -y <cluster>`.
- The local SkyPilot CLI may not expose a useful general `sky cp`. For small artifact retrieval from Kubernetes-backed clusters, identify the pod on `beast-3` and stream files from `/home/sky/sky_workdir` with `kubectl exec ... -- cat <remote-file> > <local-file>`.
- The default interactive kube context on the server may not have the `rtx4090` alias. For manual Kubernetes inspection, explicitly use `KUBECONFIG=/home/tsilva/.kube/config`.

Manual inspection examples:

```bash
ssh tsilva@beast-3 'KUBECONFIG=/home/tsilva/.kube/config kubectl get pods -n default -o wide'
ssh tsilva@beast-3 'KUBECONFIG=/home/tsilva/.kube/config kubectl logs -n default <pod> -c ray-node --tail=200'
ssh tsilva@beast-3 'KUBECONFIG=/home/tsilva/.kube/config kubectl exec -n default <pod> -c ray-node -- bash -lc "<command>"'
```

## RTX2060: beast-2

### Access

- SkyPilot infra: `ssh/beast2`
- GPU host IP observed from the current Mac/Codex network: `192.168.133.26`
- SSH command: `ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26`
- GPU: NVIDIA GeForce RTX 2060, 6 GB VRAM
- Observed driver: `595.71.05`
- SkyPilot server venv on host: `/home/tsilva/skypilot-server/.venv`
- Host `uv`: `/home/tsilva/.local/bin/uv`
- Host-side SkyPilot API server: localhost-only at `http://127.0.0.1:46580`; do not bind to `0.0.0.0` without explicit security approval.

`beast-2`, `beast2`, `beast-2.local`, and `beast2.local` may not resolve from the current Mac/Codex network. Use the IP plus `HostKeyAlias=beast-2` unless hostname resolution has been re-established. `ssh-keyscan -T 5 192.168.133.26` previously matched the existing `beast-2` host keys in `~/.ssh/known_hosts`.

When running SkyPilot from the host-side CLI on `beast-2`, set `KUBECONFIG="$HOME/.kube/config"` first. Without it, `sky check ssh` may try `/etc/rancher/k3s/k3s.yaml` and fail with permission denied.

Example host-side launch shape:

```bash
ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26 \
  'cd /home/tsilva/sandbox-sb3 && export KUBECONFIG="$HOME/.kube/config" && PATH="$HOME/.local/bin:$PATH" /home/tsilva/skypilot-server/.venv/bin/sky launch --infra ssh/beast2 -c <cluster-name> -y <task.yaml>'
```

### Mario PPO Scheduling

Benchmarked on 2026-06-17 with `stable-retro-turbo==1.0.0.post7`, Torch `2.12.0+cu130`, 131,072 timesteps per child, and W&B disabled to avoid upload-time contamination.

| Shape | Aggregate wall fps | Per-child wall fps | Avg GPU util | Max VRAM | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 child, `env_threads=4` | 1,248 | 1,248 | 32% | 695 MiB | Fastest single-job setting. |
| 2 children, `env_threads=4` | 1,971 | 986-993 | 56% | 1,175 MiB | Best fast-turnaround setting. |
| 3 children, `env_threads=4` | 2,473 | 824-830 | 70% | 1,655 MiB | Good compromise. |
| 4 children, `env_threads=2` | 2,661 | 665-669 | 77% | 2,135 MiB | Best measured aggregate throughput. |
| 4 children, `env_threads=4` | 2,661 | 665-672 | 75% | 2,135 MiB | Same aggregate as `env_threads=2`, no clear advantage. |

Default to 4 children with `env_threads=2` for aggregate screening throughput. Use 2 children with `env_threads=4` when individual result latency matters.

### Operational Notes

- `sky check ssh --verbose` is the authoritative confirmation that `ssh/beast2` is enabled; `sky ssh up` alone is not enough.
- Persistent setup to reuse: SkyPilot server venv, localhost API server, `ssh/beast2`, node-pool enablement, host tools, and GPU visibility.
- Disposable per-job setup: cluster/pod creation, workdir sync, YAML `setup`, package installs, and dataset download unless cached.
- The RTX2060 training-validated Mario path includes `stable-retro-turbo==1.0.0.post7`; it reproduced the completed-episode stop criterion at 2,711,552 timesteps on 2026-06-15.

## Cleanup

After one-off experiments, clean up clusters unless the user explicitly wants a warm cluster left running:

```bash
sky down -y <cluster-name>
```

For `beast-2` host-side commands, run cleanup through the host SkyPilot venv and explicit kubeconfig:

```bash
ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26 \
  'cd /home/tsilva/sandbox-sb3 && export KUBECONFIG="$HOME/.kube/config" && PATH="$HOME/.local/bin:$PATH" /home/tsilva/skypilot-server/.venv/bin/sky down -y <cluster-name>'
```

## Related Repo Files

- `experiments/history/EXPERIMENT_MEMORY.md`: benchmark history and interpretation.
- `GOAL.md`: current RTX4090 scheduling decision for the active screening goal.
- root-level `sky_*.yaml`: ignored local launch files; promote reusable shapes under
  `experiments/launches/`.
