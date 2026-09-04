# CUAEval

Run **computer-use-agent models on OSWorld**, one at a time, with the **model
served on a vast.ai GPU box** and the **OSWorld task VMs on AWS** — all driven
from a single EC2 host you clone this repo onto.

CUAEval owns three things OSWorld itself doesn't:

1. **Bootstrap** — clone a pristine upstream OSWorld at a pinned commit and copy
   your custom harness *adapters* (e.g. the Holo3 agent + runner) into place.
   The adapters live here, version-controlled, not loose inside an OSWorld checkout.
2. **Serving** — bring each model up behind an OpenAI-compatible endpoint (a
   vast.ai box it can **rent and destroy**, or local Docker), tunnel it to
   `localhost`, run OSWorld against it, then tear it down before the next model.
3. **Orchestration** — a `plan.yaml` lines up several models; results nest per
   model and the OSWorld runner is resumable, so reruns continue where they left off.

```
plan.yaml ──► cuaeval bootstrap   → clone OSWorld@ref + copy adapters/<name> into it
          ──► cuaeval run, per job in sequence:
                1. (vast) rent a GPU box, or reuse yours
                2. serve the model behind http://localhost:TUNNEL_PORT/v1
                     • remote → server process on the vast box, over SSH
                     • local  → Docker container on THIS host
                3. run OSWorld's run_multienv_<runner>.py against it
                     • provider_name=aws → OSWorld spawns worker EC2 VMs
                4. unload the model; (vast) destroy the box
              then print a per-model success-rate summary
```

## Quick start (fresh EC2 host)

```bash
git clone <this repo> && cd CUAEval
./setup_ec2.sh plans/ec2_aws_holo3.yaml     # installs uv/vastai/awscli, syncs venv,
                                            #   clones OSWorld, copies adapters

# credentials (host-side)
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
export AWS_SECURITY_GROUP_ID=sg-...  AWS_SUBNET_ID=subnet-...
export AWS_DEFAULT_REGION=us-east-1
export VAST_API_KEY=...                      # or: vastai set api-key <key>

uv run cuaeval run plans/ec2_aws_holo3.yaml --dry-run   # eyeball every command
uv run cuaeval run plans/ec2_aws_holo3.yaml             # for real
```

Local dev (adapters already vendored, no EC2):

```bash
uv sync
uv run cuaeval check plans/example.yaml
```

## Commands

```bash
uv run cuaeval bootstrap plan.yaml            # clone OSWorld@ref + copy adapters
uv run cuaeval bootstrap plan.yaml --force    #   overwrite adapter files if present
uv run cuaeval check     plan.yaml            # validate + show resolved jobs
uv run cuaeval run       plan.yaml --dry-run  # print every git/ssh/docker/vast cmd
uv run cuaeval run       plan.yaml            # for real
uv run cuaeval run       plan.yaml --only holo3-gimp-pruned-half
uv run cuaeval run       plan.yaml --keep-going   # don't stop on a failed job
```

## Bootstrap: OSWorld + adapters

`cuaeval bootstrap` reads the plan's `osworld:` block:

```yaml
osworld_repo: ./OSWorld                        # where the checkout goes
osworld:
  repo_url: https://github.com/xlang-ai/OSWorld
  ref: main                                    # pin a commit SHA / tag for reproducibility
  adapters: [holo3]                            # sets under adapters/<name>/ to copy on top
```

It clones upstream at `ref`, builds `OSWorld/.venv`, `pip install -r requirements.txt`,
then copies each adapter's files into the checkout per its manifest and verifies the
runner script resolves. It's idempotent — an existing checkout is left alone; use
`--force` to re-copy adapter files over it.

### Adapters

An adapter set is a folder under [`adapters/`](adapters/) with a `manifest.yaml`
mapping vendored files to their destination inside an OSWorld checkout:

```
adapters/holo3/
  manifest.yaml                       # name, runner, files: [{src, dest}, ...]
  mm_agents/holo3_agent.py            # the Holo3 OpenAI-compatible CUA agent
  mm_agents/holo3_format.py
  scripts/python/run_multienv_holo3.py   # aws/docker/vmware-capable multi-env runner
  scripts/python/holo3_grounding_smoketest.py
```

A new harness = a sibling `adapters/<name>/` dir + its manifest, then
`adapters: [<name>]` in a plan. Nothing model-specific is baked into CUAEval's code.

## Serving a model, by location

| | **remote (vast.ai)** | **local** |
|---|---|---|
| mechanism | server **process** inside the instance (a vast instance is itself a container — no nested Docker) | Docker container (this host has a daemon) |
| image / server | the instance's base image provides the server (pinned via `serve.vast.image`) | stock `sglang`/`vllm` image (`serve.image`) |
| deploy | rsync a tiny bundle (`deploy.sh` + `s3.py`), run it in a **tmux** session | `docker run` |
| switch model | `tmux kill-session` | `docker rm -f` |
| endpoint | `localhost:tunnel_port` via a CUAEval-owned SSH tunnel | `localhost:port` |

Both frameworks (`sglang`, `vllm`) are selectable per job via `serve.framework`.

### vast.ai instance lifecycle

A `remote` job can rent its own box and/or destroy one on teardown. `provision`
(create it?) and `destroy` (kill it?) are **independent** — you can destroy a box
you supplied by `ssh_host`, and keep a box CUAEval rented.

```yaml
serve:
  location: remote
  ssh_host: my-box            # optional if provision: true (derived after create)
  vast:
    provision: true           # rent a fresh instance for this job
    destroy: true             # default = provision's value; set independently to override
    keep_on_error: false      # if the job errored, leave the box up for debugging
    instance_id: 1234567      # reuse a box (provision:false) or the id to destroy
    gpu_name: RTX_4090        # offer selection (or set a raw `query`)
    num_gpus: 1
    disk: 80                  # GB
```

- **provision + destroy** (the EC2 plan): rent → serve → run → destroy, per job.
- **reuse + destroy**: give `ssh_host` (and `instance_id` if the SSH→id lookup
  can't find it); CUAEval serves on it, then destroys it at the end.
- **reuse only** (no `vast:` block): exactly the old behavior — connect to a box
  you manage via its `~/.ssh/config` alias.

Destroy runs in a `finally` after the serve/tunnel teardown, so a rented box is
never leaked even if a job crashes. For provisioned boxes CUAEval writes an SSH
alias into `~/.cuaeval/ssh_config` (Included from `~/.ssh/config`) so `ssh`,
`rsync`, and the tunnel all reach the instance the same way.

### Weights

- **S3** (`s3://...`) — downloaded layout-preserving/resumable by
  [`cuaeval/s3.py`](cuaeval/s3.py): locally into `~/.cuaeval/models/` (mounted into
  the container); on vast into `remote_models_dir`. AWS creds come from this host's
  `AWS_*` env and, for remote jobs, are written to a `0600 aws.env` on the instance
  (kept off the command line / `ps` / logs).
- **Filesystem path** — used in place (a path on this host for `local`, or one
  already staged on the instance for `remote`).

Model-specific code (`trust_remote_code` files, a pruned checkpoint's patched
`config.json`) rides **inside the weights directory** — nothing is assumed to
pre-exist on the box.

## OSWorld VM provider

Set per job via `provider_name` (threaded straight to the runner):

- **`aws`** — OSWorld's host/client mode: this EC2 host spawns worker EC2 client
  VMs for the tasks (README AWS path). CUAEval preflights that
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SECURITY_GROUP_ID`,
  `AWS_SUBNET_ID` are set and passes `--region`. See OSWorld's `SETUP_GUIDELINE.md` §3
  for the security-group rules and subnet setup.
- **`docker`** — task VMs as Docker containers on a single KVM-capable box.
- **`vmware`** — local desktop/laptop only (won't run on EC2).

## Layout

```
adapters/
  holo3/           vendored Holo3 harness + manifest.yaml
cuaeval/
  config.py        plan schema (osworld / serve / vast / job) + YAML loader
  bootstrap.py     clone OSWorld@ref, build venv, copy adapters per manifest
  vast.py          vast.ai CLI wrapper (search/create/wait/destroy) + ssh-alias writer
  models.py        weight-source classification + sglang/vllm launch-arg builder
  s3.py            layout-preserving, resumable S3 model download (also run on vast)
  serving.py       endpoint health polling (wait_serving / wait_down)
  osworld.py       build/run run_multienv_<runner>.py; read result.txt scores
  orchestrator.py  sequential job loop + aws preflight + summary
  backends/
    base.py        ServerBackend contract (start / wait_ready / stop; ctx manager)
    local.py       LocalDockerServer     (docker run / rm -f on this host)
    remote.py      RemoteProcessServer   (vast: rsync + tmux + SSH tunnel)
                   VastRemoteServer      (+ rent/destroy the instance)
  remote/
    deploy.sh      runs ON the instance: stage weights, exec the server (foreground)
  cli.py           `cuaeval bootstrap|run|check`
plans/
  ec2_aws_holo3.yaml   EC2 + aws provider + vast-provisioned serving
  example.yaml         local/BYO-box sample
setup_ec2.sh           one-shot host provisioning + bootstrap
```

## Notes / limits

- **Remote = vast.ai only** by design (process-based). A generic Docker-host
  remote backend isn't built (YAGNI).
- `--dry-run` prints every `git`/`ssh`/`docker`/`rsync`/`vastai`/runner command
  without executing — use it before spending GPU time (or a rented box).
- One model in VRAM at a time; staged weights persist on disk for warm reruns.
