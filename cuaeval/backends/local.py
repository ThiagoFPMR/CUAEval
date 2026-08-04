"""Local serving = a Docker container of a stock sglang/vllm image on THIS host
(the same machine that runs the OSWorld VMs — it has a real docker daemon).

Flow: stage s3:// weights locally (skip if already present) -> `docker run -d`
the image with the weights dir mounted and the server port published -> endpoint
is http://localhost:PORT/v1. stop() = `docker rm -f`, which frees VRAM; staged
weights stay on disk for warm reruns.
"""
from __future__ import annotations

from pathlib import Path

from ..config import JobConfig
from ..models import build_launch_args, is_s3, local_stage_dir
from ..util import log, run
from .base import ServerBackend

CONTAINER_MODEL_DIR = "/model"  # where the weights dir is mounted inside the container
LOCAL_MODELS_ROOT = "~/.cuaeval/models"  # where s3:// checkpoints are staged on this host


class LocalDockerServer(ServerBackend):
    def __init__(self, job: JobConfig, *, dry_run: bool = False) -> None:
        super().__init__(job, dry_run=dry_run)
        self._started = False

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.serve.port}/v1"

    def _stage_weights(self) -> str:
        """Return the host path to mount. Download from S3 if needed."""
        if not is_s3(self.job.weights):
            path = str(Path(self.job.weights).expanduser())
            log.info("using local weights at %s", path)
            return path

        root = str(Path(LOCAL_MODELS_ROOT).expanduser())
        dest = local_stage_dir(self.job.weights, root)
        log.info("staging %s -> %s", self.job.weights, dest)
        if self.dry_run:
            log.info("[dry-run] would download %s to %s", self.job.weights, dest)
            return dest
        from ..s3 import download_model
        download_model(self.job.weights, dest)
        return dest

    def start(self) -> None:
        host_model_dir = self._stage_weights()
        # Ensure no stale container from a previous (crashed) run holds the name/port.
        run(["docker", "rm", "-f", self.serve.container_name],
            dry_run=self.dry_run, check=False)

        launch = build_launch_args(self.serve, CONTAINER_MODEL_DIR, self.job.label)
        cmd = [
            "docker", "run", "-d",
            "--name", self.serve.container_name,
            "--gpus", self.serve.gpus,
            "--shm-size", "16g",
            "-v", f"{host_model_dir}:{CONTAINER_MODEL_DIR}:ro",
            "-p", f"{self.serve.port}:{self.serve.port}",
            self.serve.resolved_image(),
            "python3", *launch,
        ]
        log.info("starting local container %r (%s)",
                 self.serve.container_name, self.serve.framework)
        run(cmd, dry_run=self.dry_run)
        self._started = True

    def stop(self) -> None:
        if not self._started and not self.dry_run:
            return
        log.info("stopping local container %r", self.serve.container_name)
        run(["docker", "rm", "-f", self.serve.container_name],
            dry_run=self.dry_run, check=False)
        self._started = False
