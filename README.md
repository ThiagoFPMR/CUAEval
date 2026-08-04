# CUAEval

A small harness for **lining up several computer-use-agent models and testing
each on a benchmark in sequence**. OSWorld is the only benchmark wired up so far.

For every model in a plan, CUAEval brings up an OpenAI-compatible inference
endpoint, runs OSWorld against it, then tears the endpoint down before moving to
the next model — so **only one model is loaded at a time**, and a plan can mix
models served in different places from weights stored in different places.

```
plan.yaml ──► for each job, in sequence:
                1. serve the model behind http://localhost:PORT/v1
                     • local   → Docker container on THIS host
                     • remote  → process on a vast.ai instance (over SSH)
                2. wait until the endpoint serves that model
                3. run OSWorld's own run_multienv_<runner>.py against it
                4. unload the model (docker rm -f  /  kill the tmux session)
              then print a per-model success-rate summary
```

## What you supply

- **A fully configured OSWorld checkout** (`osworld_repo` in the plan). CUAEval
  shells out to its `scripts/python/run_multienv_<runner>.py`; it does not
  reimplement OSWorld. The endpoint is wired in via `OPENAI_BASE_URL` /
  `OPENAI_API_KEY`, exactly like `run_osworld_holo3_here.sh`.
- **Where each model is served** — `local` (Docker here) or `remote` (a vast.ai
  instance; give its `ssh_host`).
- **Where each model's weights live** — an `s3://` URI or a filesystem path.

## Install

```bash
cd CUAEval
uv sync                      # creates .venv from pyproject.toml / uv.lock
uv run cuaeval check plans/example.yaml
```

## Use

```bash
uv run cuaeval check plans/example.yaml            # validate + show resolved jobs
uv run cuaeval run   plans/example.yaml --dry-run  # print every ssh/docker/runner cmd
uv run cuaeval run   plans/example.yaml            # for real
uv run cuaeval run   plans/example.yaml --only holo3-gimp-local
uv run cuaeval run   plans/example.yaml --keep-going   # don't stop on a failed job
```

See [`plans/example.yaml`](plans/example.yaml) for the full schema. Results nest
under each job's `label`, and the OSWorld runner is resumable, so re-running a
plan continues where it left off and models never overwrite each other.

## Serving model, by location

| | **local** | **remote (vast.ai)** |
|---|---|---|
| mechanism | Docker container (this host has a daemon) | **process** inside the instance (a vast instance is itself a container — no nested Docker) |
| image / server | stock `sglang`/`vllm` image, pinned & overridable (`serve.image`) | the instance's own base image already provides the server |
| deploy | `docker run` | rsync a tiny bundle (`deploy.sh` + `s3.py`), run it in a **tmux** session |
| switch model | `docker rm -f` | `tmux kill-session` |
| endpoint | `localhost:port` | `localhost:tunnel_port` via a CUAEval-owned SSH tunnel |

Both frameworks (`sglang`, `vllm`) are selectable per job via `serve.framework`.

### Weights

- **S3** — downloaded (layout-preserving, resumable, size-skip) by
  [`cuaeval/s3.py`](cuaeval/s3.py). Locally it stages into `~/.cuaeval/models/`
  and mounts that into the container. On vast it runs on the instance (the bundle
  ships `s3.py`; `boto3` is `pip install`ed if the image lacks it), staging into
  `remote_models_dir`. AWS creds are read from this host's `AWS_*` env vars and,
  for remote jobs, written to a `0600 aws.env` on the instance (kept off the
  command line / `ps` / logs).
- **Filesystem path** — used in place. For a `local` job that's a path on this
  host; for a `remote` job it's a path already staged on the instance.

Model-specific code (HF `trust_remote_code` files, a pruned checkpoint's patched
`config.json`) rides **inside the weights directory** — nothing model-specific
is baked into CUAEval or assumed to pre-exist on the remote.

## Layout

```
cuaeval/
  config.py        plan schema (dataclasses) + YAML loader with defaults-merge
  models.py        weight-source classification + sglang/vllm launch-arg builder
  s3.py            layout-preserving, resumable S3 model download (also run on vast)
  serving.py       endpoint health polling (wait_serving / wait_down)
  backends/
    base.py        ServerBackend contract (start / wait_ready / stop; ctx manager)
    local.py       LocalDockerServer   (docker run / rm -f on this host)
    remote.py      RemoteProcessServer (vast.ai: rsync + tmux process + SSH tunnel)
  remote/
    deploy.sh      runs ON the instance: stage weights, exec the server (foreground)
  osworld.py       build/run run_multienv_<runner>.py; read result.txt scores
  orchestrator.py  sequential job loop + summary
  cli.py           `cuaeval run|check`
plans/
  example.yaml     three-model sample (two remote S3 + one local)
```

## Notes / limits

- **Remote = vast.ai only** by design (process-based). A generic Docker host
  would want a `docker`-mechanism remote backend; not built (YAGNI for now).
- `--dry-run` prints every `ssh`/`docker`/`rsync`/runner command without
  executing — use it to eyeball a plan before spending GPU time.
- One model in VRAM at a time; staged weights persist on disk for warm reruns.
