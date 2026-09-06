"""Weight-source classification + server launch-command construction.

A job's `weights` string is one of:
  * a b2:// URI             -> must be staged (downloaded from Backblaze B2)
                               before serving
  * a filesystem path       -> used in place (local job: path on this host;
                               remote job: path already on the vast instance)

`build_launch_args` produces the sglang/vllm argv given the *path the server
should load from* — the same for local (a docker-mounted dir) and remote (a dir
on the instance), so the two backends share one command builder.
"""
from __future__ import annotations

from .config import ServeConfig


def is_b2(weights: str) -> bool:
    return weights.startswith("b2://")


def local_stage_dir(weights: str, models_root: str) -> str:
    """Where a b2:// checkpoint gets staged: <models_root>/<basename-of-uri>."""
    from pathlib import PurePosixPath
    base = PurePosixPath(weights.rstrip("/")).name
    return f"{models_root.rstrip('/')}/{base}"


def build_launch_args(serve: ServeConfig, model_path: str, label: str) -> list[str]:
    """Build the server launch argv (without the python interpreter prefix).

    Returned as e.g. ["-m", "sglang.launch_server", "--model-path", ...] so both
    backends can prefix it with the right interpreter (docker image entrypoint or
    the instance's python). A full override is honoured via serve.command.
    """
    if serve.command:
        return list(serve.command)

    if serve.framework == "sglang":
        args = [
            "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--served-model-name", label,
            "--tp", str(serve.tp),
            "--mem-fraction-static", str(serve.mem_fraction),
            "--context-length", str(serve.context_length),
            "--host", "0.0.0.0",
            "--port", str(serve.port),
        ]
        if serve.trust_remote_code:
            args.append("--trust-remote-code")
    elif serve.framework == "vllm":
        args = [
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--served-model-name", label,
            "--tensor-parallel-size", str(serve.tp),
            "--gpu-memory-utilization", str(serve.mem_fraction),
            "--max-model-len", str(serve.context_length),
            "--host", "0.0.0.0",
            "--port", str(serve.port),
        ]
        if serve.trust_remote_code:
            args.append("--trust-remote-code")
    else:
        raise ValueError(f"unknown framework {serve.framework!r}")

    args.extend(serve.extra_args)
    return args
