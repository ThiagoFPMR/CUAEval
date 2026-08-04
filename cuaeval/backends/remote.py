"""Remote serving on a vast.ai instance = a server PROCESS inside the instance.

A vast.ai instance is itself a container with no usable docker daemon, so we do
NOT `docker run` there. Instead the instance is rented from a stock sglang/vllm
image (its base image already provides the server), and CUAEval:

  1. rsyncs a tiny deploy bundle (deploy.sh + s3.py) to the instance;
  2. runs deploy.sh inside a detached tmux session — it stages the weights
     (S3 download via s3.py, or a pre-staged path) then launches the server as a
     foreground process the tmux session holds;
  3. opens an SSH tunnel so the endpoint is reachable at http://localhost:PORT/v1.

Switching models = killing the tmux session (kills the process, frees VRAM);
staged weights stay on the instance disk for warm reruns. This mirrors the vast
path of OSWorld/run_two_models_holo3.sh, generalised and self-managing.

AWS credentials for the S3 download are written to a 0600 aws.env on the instance
(sourced by deploy.sh) rather than placed on the command line, so they never
appear in `ps` output or in CUAEval's logs.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from ..config import JobConfig
from ..models import build_launch_args, is_s3
from ..util import fmt_cmd, log, run
from .base import ServerBackend

_BUNDLE_DIR = Path(__file__).resolve().parent.parent / "remote"
_S3_PY = Path(__file__).resolve().parent.parent / "s3.py"


class RemoteProcessServer(ServerBackend):
    def __init__(self, job: JobConfig, *, dry_run: bool = False) -> None:
        super().__init__(job, dry_run=dry_run)
        self._tunnel: subprocess.Popen | None = None
        self._launched = False

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.serve.resolved_tunnel_port()}/v1"

    # --- helpers -------------------------------------------------------------
    def _ssh(self, remote_cmd: str, *, check: bool = True):
        return run(["ssh", self.serve.ssh_host, remote_cmd],
                   dry_run=self.dry_run, check=check)

    def _remote_model_dir(self) -> str:
        """Path the server loads from on the instance."""
        if is_s3(self.job.weights):
            base = self.job.weights.rstrip("/").rsplit("/", 1)[-1]
            return f"{self.serve.remote_models_dir.rstrip('/')}/{base}"
        return self.job.weights  # pre-staged path on the instance

    # --- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._preflight()
        self._sync_bundle()
        self._push_aws_env()
        self._launch()
        self._open_tunnel()
        self._launched = True

    def _preflight(self) -> None:
        log.info("preflight: ssh %s ...", self.serve.ssh_host)
        run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             self.serve.ssh_host, "true"], dry_run=self.dry_run)

    def _sync_bundle(self) -> None:
        # remote_workdir is a trusted, validated plain path (config._validate_job).
        # It is injected RAW so the remote shell expands ~ / $HOME — quoting it
        # would create a directory literally named '~'. rsync/scp likewise expand
        # ~ in their remote-path argv, so the two agree.
        workdir = self.serve.remote_workdir
        log.info("rsyncing deploy bundle -> %s:%s", self.serve.ssh_host, workdir)
        self._ssh(f"mkdir -p {workdir}")
        run(["rsync", "-az",
             f"{_BUNDLE_DIR}/", str(_S3_PY),
             f"{self.serve.ssh_host}:{workdir}/"], dry_run=self.dry_run)

    def _push_aws_env(self) -> None:
        """Write AWS creds (from this host's env) to a 0600 aws.env on the instance,
        keeping them off the command line. Skipped for non-S3 (pre-staged) weights."""
        if not is_s3(self.job.weights):
            return
        lines = [f"export {name}={shlex.quote(os.environ[name])}"
                 for name in self.serve.aws_env if os.environ.get(name)]
        remote_path = f"{self.serve.remote_workdir}/aws.env"
        if not lines:
            log.warning("no AWS_* env vars set locally; S3 download on the instance "
                        "will rely on the instance's own credentials/role.")
            return
        if self.dry_run:
            log.info("[dry-run] would write %d AWS var(s) to %s:%s (values hidden)",
                     len(lines), self.serve.ssh_host, remote_path)
            return
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tf:
            tf.write("\n".join(lines) + "\n")
            tmp = tf.name
        try:
            os.chmod(tmp, 0o600)
            run(["scp", "-p", tmp, f"{self.serve.ssh_host}:{remote_path}"], dry_run=False)
            self._ssh(f"chmod 600 {remote_path}")  # raw path: let remote shell expand ~
        finally:
            os.unlink(tmp)

    def _launch(self) -> None:
        s = self.serve
        model_dir = self._remote_model_dir()
        launch = build_launch_args(s, model_dir, self.job.label)
        # env consumed by deploy.sh (non-secret; safe to appear in ps/logs)
        env = {
            "CUAEVAL_PYTHON": s.remote_python,
            "MODEL_SRC": self.job.weights,
            "MODEL_DIR": model_dir,
            "IS_S3": "1" if is_s3(self.job.weights) else "0",
            "S3_WORKERS": "4",
            "SERVE_ARGS": " ".join(shlex.quote(a) for a in launch),
        }
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        # remote_workdir raw so tmux's `sh -c` expands ~ (see _sync_bundle note).
        inner = (f"cd {s.remote_workdir} && "
                 f"{env_prefix} bash deploy.sh 2>&1 | tee serve.log")
        remote_cmd = (
            f"tmux kill-session -t {shlex.quote(s.tmux_session)} 2>/dev/null || true; "
            f"tmux new-session -d -s {shlex.quote(s.tmux_session)} {shlex.quote(inner)}"
        )
        log.info("launching %r on %s (tmux '%s', framework=%s)",
                 self.job.label, s.ssh_host, s.tmux_session, s.framework)
        self._ssh(remote_cmd)

    def _open_tunnel(self) -> None:
        s = self.serve
        local_port = s.resolved_tunnel_port()
        cmd = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes",
               "-o", "ServerAliveInterval=30",
               "-L", f"{local_port}:localhost:{s.port}", s.ssh_host]
        log.info("opening SSH tunnel localhost:%s -> %s:%s", local_port, s.ssh_host, s.port)
        if self.dry_run:
            log.info("[dry-run] %s", fmt_cmd(cmd))
            return
        self._tunnel = subprocess.Popen(cmd)

    def stop(self) -> None:
        s = self.serve
        if self._launched or self.dry_run:
            log.info("killing tmux session %r on %s (unloads model)",
                     s.tmux_session, s.ssh_host)
            self._ssh(f"tmux kill-session -t {shlex.quote(s.tmux_session)} 2>/dev/null || true",
                      check=False)
            # Best-effort wait for the endpoint to drop, then close the tunnel.
            if not self.dry_run:
                from ..serving import wait_down
                wait_down(self.endpoint, s.down_timeout, s.poll)
        if self._tunnel is not None:
            log.info("closing SSH tunnel")
            self._tunnel.terminate()
            try:
                self._tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._tunnel.kill()
            self._tunnel = None
        self._launched = False
