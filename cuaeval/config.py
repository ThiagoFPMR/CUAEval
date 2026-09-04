"""Plan schema + loader.

A *plan* is a YAML file describing which OSWorld checkout to use, some shared
defaults, and a list of model *jobs* to run in sequence. Per-job values override
the plan-level `defaults`. See plans/example.yaml for a worked example.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Pinned, Holo3 (qwen3_5_moe VLM-MoE) capable images for LOCAL docker serving.
# Overridable per job via serve.image. Remote (vast.ai) does not use these — there
# the instance's own base image provides the server.
DEFAULT_IMAGES = {
    "sglang": "lmsysorg/sglang:v0.4.6.post1-cu124",
    "vllm": "vllm/vllm-openai:v0.8.5",
}


@dataclass
class VastConfig:
    """Optional vast.ai instance lifecycle for a `remote` job.

    `provision` (do I create the box?) and `destroy` (do I kill it on teardown?)
    are independent: you can destroy a box you handed CUAEval by ssh_host, and you
    can keep a box CUAEval created. `destroy` defaults to mirroring `provision`.
    """
    provision: bool = False           # create a fresh instance for this job
    destroy: bool | None = None       # kill on teardown; None => same as `provision`
    keep_on_error: bool = False       # if the job errored, skip destroy (leave it for debugging)
    instance_id: int | None = None    # reuse this instance (provision=False) or the id to destroy

    # create-time offer selection (used only when provision=True)
    query: str | None = None          # raw `vastai search offers` query; overrides gpu_name/num_gpus
    gpu_name: str | None = None        # e.g. "RTX_4090" (used to build the query if `query` unset)
    num_gpus: int = 1
    image: str | None = None          # instance base image (default: DEFAULT_IMAGES[framework])
    disk: int = 60                    # GB
    label: str | None = None          # vast instance label (default: cuaeval-<job label>)
    onstart_cmd: str | None = None    # optional onstart script contents
    create_args: list[str] = field(default_factory=list)  # extra argv to `vastai create instance`

    # how CUAEval reaches the box over SSH (an alias is written to ~/.cuaeval/ssh_config)
    ssh_alias: str | None = None      # default: cuaeval-vast-<instance_id>
    ssh_user: str = "root"            # vast default
    identity_file: str | None = None  # SSH key for the alias (default: your agent / ~/.ssh keys)

    wait_timeout: int = 1200          # secs to wait for the instance to reach 'running' + sshd up

    def resolved_destroy(self) -> bool:
        return self.provision if self.destroy is None else self.destroy


@dataclass
class ServeConfig:
    # where + how to serve
    location: str = "remote"          # "local" (docker here) | "remote" (vast.ai)
    framework: str = "sglang"         # "sglang" | "vllm"

    # server params (both frameworks)
    port: int = 60000                 # port the server listens on (on its own host)
    tp: int = 1                       # tensor-parallel size = GPU count
    context_length: int = 32768
    mem_fraction: float = 0.9
    trust_remote_code: bool = True
    extra_args: list[str] = field(default_factory=list)  # appended to the launch cmd
    command: list[str] | None = None  # full launch argv override (advanced escape hatch)

    # local-docker only
    image: str | None = None          # defaults to DEFAULT_IMAGES[framework]
    gpus: str = "all"                 # docker --gpus value
    container_name: str = "cuaeval-serve"

    # remote (vast.ai) only
    ssh_host: str = ""                # ssh alias/target of the instance (derived if vast.provision)
    vast: VastConfig = field(default_factory=VastConfig)  # optional instance lifecycle
    tunnel_port: int | None = None    # local port forwarded to the remote `port` (default: port)
    tmux_session: str = "cuaeval-serve"
    remote_models_dir: str = "/models"      # where weights are staged on the instance
    remote_workdir: str = "~/.cuaeval"      # where the deploy bundle is rsynced
    remote_python: str = "python3"          # interpreter on the instance
    aws_env: list[str] = field(default_factory=lambda: [
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
    ])  # env var names forwarded to the instance for the S3 download

    # timeouts (seconds)
    up_timeout: int = 2400            # includes any S3 download before the server is ready
    down_timeout: int = 180
    poll: int = 10

    def resolved_image(self) -> str:
        return self.image or DEFAULT_IMAGES.get(self.framework, "")

    def resolved_tunnel_port(self) -> int:
        return self.tunnel_port or self.port


@dataclass
class JobConfig:
    label: str                        # --model label + result subdir; must be unique
    weights: str                      # local path, remote path, or s3:// URI
    serve: ServeConfig = field(default_factory=ServeConfig)

    # OSWorld runner selection + knobs
    runner: str = "holo3"             # -> scripts/python/run_multienv_<runner>.py
    meta: str = "evaluation_examples/test_nogdrive.json"
    result_dir: str = "./results"
    max_steps: int = 15
    num_envs: int = 4
    domain: str | None = None         # None => all domains
    provider_name: str = "vmware"     # OSWorld VM provider: aws | docker | vmware | ...
    region: str = "us-east-1"         # AWS region (only used when provider_name=aws)
    runner_args: list[str] = field(default_factory=list)  # extra argv to the runner
    api_key: str = "EMPTY"            # OPENAI_API_KEY sent to the runner


@dataclass
class OSWorldSource:
    """How `cuaeval bootstrap` provisions the OSWorld checkout at `osworld_repo`.

    A pristine upstream OSWorld is cloned at a pinned `ref`, its venv is built,
    then each named adapter set (CUAEval/adapters/<name>/) is copied on top. This
    keeps OSWorld upgradable (bump `ref`) while the custom harness lives, version
    controlled, in CUAEval.
    """
    repo_url: str = "https://github.com/xlang-ai/OSWorld"
    ref: str | None = None            # commit SHA or tag to pin; None => default branch (warns)
    adapters: list[str] = field(default_factory=list)  # adapter set names under CUAEval/adapters/
    python: str = "python3"           # interpreter used to build the repo's .venv
    pip_install: bool = True          # pip install -r requirements.txt into that venv


@dataclass
class Plan:
    osworld_repo: Path
    osworld_python: str               # interpreter used to run the OSWorld runner
    jobs: list[JobConfig]
    osworld: OSWorldSource = field(default_factory=OSWorldSource)


def _filter_known(cls, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that are real dataclass fields; error loudly on typos."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown key(s) {sorted(unknown)}; "
            f"known keys are {sorted(known)}"
        )
    return {k: v for k, v in data.items() if k in known}


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively overlay `over` on `base` (used for defaults <- per-job)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_plan(path: str | Path) -> Plan:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "osworld_repo" not in raw:
        raise ValueError("plan is missing required top-level key: osworld_repo")
    repo = Path(raw["osworld_repo"]).expanduser()

    # Default the OSWorld interpreter to the repo's own venv if present.
    venv_py = repo / ".venv" / "bin" / "python"
    osworld_python = raw.get("osworld_python") or (
        str(venv_py) if venv_py.exists() else "python3"
    )

    osworld_src = OSWorldSource(**_filter_known(OSWorldSource, raw.get("osworld", {}) or {}))

    defaults = raw.get("defaults", {}) or {}
    jobs_raw = raw.get("jobs") or []
    if not jobs_raw:
        raise ValueError("plan has no jobs")

    jobs: list[JobConfig] = []
    seen_labels: set[str] = set()
    for i, jr in enumerate(jobs_raw):
        merged = _deep_merge(defaults, jr)
        serve_raw = merged.pop("serve", {}) or {}
        vast_raw = serve_raw.pop("vast", {}) or {}
        job_kwargs = _filter_known(JobConfig, merged)
        serve_kwargs = _filter_known(ServeConfig, serve_raw)
        serve_kwargs["vast"] = VastConfig(**_filter_known(VastConfig, vast_raw))
        job_kwargs["serve"] = ServeConfig(**serve_kwargs)
        if "label" not in job_kwargs or "weights" not in job_kwargs:
            raise ValueError(f"job #{i} is missing required key 'label' and/or 'weights'")
        job = JobConfig(**job_kwargs)

        if job.label in seen_labels:
            raise ValueError(f"duplicate job label {job.label!r} — labels must be unique "
                             "(they become result subdirs)")
        seen_labels.add(job.label)
        _validate_job(job)
        jobs.append(job)

    return Plan(osworld_repo=repo, osworld_python=osworld_python, jobs=jobs,
                osworld=osworld_src)


def _validate_job(job: JobConfig) -> None:
    s = job.serve
    if s.location not in ("local", "remote"):
        raise ValueError(f"{job.label}: serve.location must be 'local' or 'remote'")
    if s.framework not in ("sglang", "vllm"):
        raise ValueError(f"{job.label}: serve.framework must be 'sglang' or 'vllm'")
    if s.location == "remote" and not s.ssh_host and not s.vast.provision:
        raise ValueError(f"{job.label}: remote serving requires serve.ssh_host "
                         "(or serve.vast.provision: true to create a box)")
    if s.location == "remote" and s.vast.resolved_destroy() and not s.vast.provision \
            and s.vast.instance_id is None and not s.ssh_host:
        raise ValueError(f"{job.label}: serve.vast.destroy needs a way to identify the "
                         "box — set serve.vast.instance_id or serve.ssh_host")
    if s.location == "local" and not s.resolved_image():
        raise ValueError(f"{job.label}: no docker image for framework {s.framework!r}; "
                         "set serve.image")
    # remote_workdir is injected raw into remote shell commands (so ~/$HOME expand),
    # so it must be a plain path token — reject anything that could break/inject.
    if s.location == "remote" and not re.fullmatch(r"~?[\w./+-]+", s.remote_workdir):
        raise ValueError(f"{job.label}: serve.remote_workdir must be a plain path "
                         f"(no spaces/quotes/shell metachars): {s.remote_workdir!r}")
