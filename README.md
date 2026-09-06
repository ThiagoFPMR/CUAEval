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
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...   # OSWorld task VMs only
export AWS_SECURITY_GROUP_ID=sg-...  AWS_SUBNET_ID=subnet-...
export AWS_DEFAULT_REGION=us-east-1
export B2_ACCESS_KEY_ID=...  B2_SECRET_ACCESS_KEY=...     # weights + results
export B2_S3_ENDPOINT=https://s3.us-west-004.backblazeb2.com
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
uv run cuaeval run       plan.yaml --results-b2 b2://bucket/prefix   # override where results go
uv run cuaeval run       plan.yaml --no-videos    # save everything except recording.mp4
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
vendors our custom evaluation metas (see below), then copies each adapter's files into
the checkout per its manifest and verifies the runner script resolves. It's idempotent —
an existing checkout is left alone; use `--force` to re-copy adapter/meta files over it.

### Evaluation metas

A plan's `meta:` selects which OSWorld tasks a job runs. Upstream ships some
(`test_all.json`, ...) but the ones we author don't come with a clone, so they live
version-controlled in [`evaluation_examples/`](evaluation_examples/) and bootstrap
copies them into the checkout's `evaluation_examples/`. Only custom metas live there,
so vendoring never clobbers an upstream file. The default,
`evaluation_examples/test_nogdrive.json`, is `test_all.json` minus the 8 Google-Drive
`multi_apps` tasks. Add a subset by dropping a `{domain: [ids]}` JSON there and pointing
a job's `meta:` at it.

To restrict a single run to some of a meta's domains without a new file, set the job's
`domains:` list (e.g. `domains: [chrome, vlc]`; empty/unset = every domain in the meta).

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
| deploy | rsync a tiny bundle (`deploy.sh` + `b2.py`), run it in a **tmux** session | `docker run` |
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

- **Backblaze B2** (`b2://...`) — downloaded layout-preserving/resumable by
  [`cuaeval/b2.py`](cuaeval/b2.py): locally into `~/.cuaeval/models/` (mounted into
  the container); on vast into `remote_models_dir`. Credentials come from this
  host's `B2_*` env (the same ones the results mirror uses) and, for remote jobs,
  are written to a `0600 b2.env` on the instance (kept off the command line /
  `ps` / logs):

  ```bash
  export B2_ACCESS_KEY_ID=...       # Backblaze application keyID
  export B2_SECRET_ACCESS_KEY=...   # Backblaze applicationKey
  export B2_S3_ENDPOINT=https://s3.us-west-004.backblazeb2.com
  ```

  `cuaeval run` preflights these whenever a job has `b2://` weights, so a missing
  key fails before a GPU box is rented rather than after.
- **Filesystem path** — used in place (a path on this host for `local`, or one
  already staged on the instance for `remote`).

Model-specific code (`trust_remote_code` files, a pruned checkpoint's patched
`config.json`) rides **inside the weights directory** — nothing is assumed to
pre-exist on the box.

## Saving results

The OSWorld runner writes each finished example (`result.txt`, `traj.jsonl`,
screenshots, `recording.mp4`) into the results tree **on the host's disk** — one
un-backed-up EBS volume. A terminated or spot-reclaimed host loses the campaign:
every rented GPU hour and every task VM it drove. `results_sync` mirrors that tree
to Backblaze B2 while the run is still going:

```yaml
results_sync:
  b2_uri: b2://bucket/cuaeval-results
  endpoint_url: https://s3.us-west-004.backblazeb2.com   # your Backblaze S3 endpoint
  interval: 600            # sync every 10 min DURING a job (<=0 = only at job end)
  include_videos: true     # false => skip recording.mp4 (the bulk of the bytes)
  exclude: []              # extra globs, matched on path or basename
  run_id: null             # optional extra prefix to keep campaigns from merging
```

Backblaze is reached through its S3-compatible API, so set these in the
environment (never in the plan) — the same variables that read the weights:

```bash
export B2_ACCESS_KEY_ID=...       # Backblaze application keyID
export B2_SECRET_ACCESS_KEY=...   # Backblaze applicationKey
export B2_S3_ENDPOINT=https://s3.us-west-004.backblazeb2.com   # or set endpoint_url above
```

- Syncs on `interval` while the benchmark runs, **plus once when the job ends** —
  including when it ends by raising, so a crashed job still saves partial results.
- The final sync runs *after* the vast box is destroyed: stop paying for the GPU
  first, then spend time on Backblaze.
- Uploads are size-compared and additive. The runner writes each example directory
  once and never rewrites it, so a tick with nothing new costs one `LIST`.
- **A failed sync never fails the job** — the results still exist locally. It's
  logged loudly and the summary ends with `RESULTS NOT UPLOADED for: ...`.
- Not keyed by job label: the tree already nests as
  `<action_space>/<obs>/<label>/<domain>/<example>/`, so one mirror keeps jobs
  separate, and resumed runs merge into the same place.

Per-run overrides: `--results-b2 b2://...`, `--no-videos`, `--no-results-sync`.
With no `results_sync` block at all, `cuaeval run` warns on every start.

### Credentials

Two independent credential sets, because two unrelated things are being rented:

**Backblaze B2** — object storage, and the only storage CUAEval uses. One
application key covers both halves; note that it lands on a rented third-party
GPU box (remote jobs stage their own weights), so scope it to the bucket:

| Prefix | Capabilities |
|---|---|
| `models/*` | `listBuckets`, `listFiles`, `readFiles` |
| `cuaeval-results/*` | `listFiles`, `writeFiles` |

No `deleteFiles` — the sync should never be able to erase results. Backblaze
scopes an application key to one bucket, optionally with a single name prefix; to
keep weights read-only while results stay writable, use **two keys on two
buckets** (or two prefix-scoped keys) rather than one key over the whole bucket.

**AWS** — only for OSWorld's `aws` VM provider (`provider_name: aws`), which
spawns the task VMs. Nothing in CUAEval reads or writes S3 any more, so this IAM
user needs EC2 only: `RunInstances`, `TerminateInstances`, `StartInstances`,
`DescribeInstances`, `DescribeInstanceStatus`, `DescribeImages`, `DescribeSubnets`,
`DescribeSecurityGroups`, `CreateTags`. OSWorld also tries to register an
EventBridge TTL schedule per task VM and will attempt `iam:CreateRole` to do it;
set `ENABLE_TTL=false`, or pre-create the role and set `AWS_SCHEDULER_ROLE_ARN`
(+ `scheduler:CreateSchedule` and `iam:PassRole` on it). Skip AWS entirely if you
run the task VMs on `vmware` or `docker`.

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
evaluation_examples/
  test_nogdrive.json   custom OSWorld task metas, vendored into the checkout on bootstrap
cuaeval/
  config.py        plan schema (osworld / serve / vast / job) + YAML loader
  bootstrap.py     clone OSWorld@ref, build venv, vendor metas, copy adapters per manifest
  vast.py          vast.ai CLI wrapper (search/create/wait/destroy) + ssh-alias writer
  models.py        weight-source classification + sglang/vllm launch-arg builder
  b2.py            layout-preserving, resumable B2 download (weights) + upload (results)
  results.py       periodic + end-of-job mirror of the results tree to B2
  serving.py       endpoint health polling (wait_serving / wait_down)
  osworld.py       build/run run_multienv_<runner>.py; read result.txt scores
  orchestrator.py  sequential job loop + credential preflights + summary
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
