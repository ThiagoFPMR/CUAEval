"""Small shared helpers: logging, and a single subprocess entry point that every
backend/runner funnels through so that `--dry-run` can print commands instead of
executing them, uniformly, in one place.
"""
from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from typing import Mapping, Sequence

log = logging.getLogger("cuaeval")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="\x1b[1;36m[cuaeval %(asctime)s]\x1b[0m %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def fmt_cmd(cmd: Sequence[str]) -> str:
    """Render an argv list as a copy-pasteable shell string (for logs/dry-run)."""
    return " ".join(shlex.quote(str(c)) for c in cmd)


def run(
    cmd: Sequence[str],
    *,
    dry_run: bool = False,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    capture: bool = False,
    background: bool = False,
) -> subprocess.CompletedProcess | subprocess.Popen | None:
    """Run (or, when dry_run, just print) a command.

    - background=True returns a live Popen (used for the long-lived SSH tunnel and
      for streaming eval runs); the caller owns its lifetime.
    - capture=True returns CompletedProcess with captured stdout/stderr text.
    - dry_run prints the command and returns None (background) or a dummy
      CompletedProcess(returncode=0) so callers can proceed without special cases.
    """
    printable = fmt_cmd(cmd)
    if dry_run:
        log.info("[dry-run] %s", printable)
        if background:
            return None
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    log.debug("exec: %s", printable)
    if background:
        return subprocess.Popen(cmd, env=_merged_env(env), cwd=cwd)
    return subprocess.run(
        cmd,
        env=_merged_env(env),
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture,
    )


def _merged_env(env: Mapping[str, str] | None) -> dict[str, str] | None:
    """Overlay `env` on top of the current process environment (never replace it —
    SSH/docker need PATH, HOME, etc.). None means 'inherit unchanged'."""
    if env is None:
        return None
    import os

    merged = dict(os.environ)
    merged.update(env)
    return merged
