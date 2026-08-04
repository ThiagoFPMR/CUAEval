"""Backend contract.

A backend owns one model's serving lifecycle for one job:

    with make_backend(job) as backend:
        backend.start()          # stage weights + launch server + (remote) tunnel
        backend.wait_ready()     # block until the endpoint serves this model
        ...run the benchmark against backend.endpoint...
    # __exit__ -> backend.stop(): unload the model, free VRAM, drop the tunnel

`endpoint` is always a *local* OpenAI base URL (http://localhost:PORT/v1): for
local jobs that's the docker-published port, for remote jobs the tunnelled port.
So the OSWorld runner is pointed at localhost regardless of where the model runs.
"""
from __future__ import annotations

import abc

from ..config import JobConfig
from ..serving import wait_serving
from ..util import log


class ServerBackend(abc.ABC):
    def __init__(self, job: JobConfig, *, dry_run: bool = False) -> None:
        self.job = job
        self.serve = job.serve
        self.dry_run = dry_run

    # --- lifecycle -----------------------------------------------------------
    @abc.abstractmethod
    def start(self) -> None:
        """Stage weights and launch the server (returns once launched, not ready)."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Unload the model and release resources. Must be idempotent + safe to
        call in a finally/except even if start() only partly succeeded."""

    # --- endpoint ------------------------------------------------------------
    @property
    @abc.abstractmethod
    def endpoint(self) -> str:
        """Local OpenAI base URL, e.g. http://localhost:60000/v1."""

    # --- readiness -----------------------------------------------------------
    def wait_ready(self) -> None:
        if self.dry_run:
            log.info("[dry-run] would wait for %r at %s", self.job.label, self.endpoint)
            return
        wait_serving(
            self.endpoint,
            want=self.job.label,
            up_timeout=self.serve.up_timeout,
            poll=self.serve.poll,
            what=f"{self.serve.location}/{self.serve.framework}",
        )

    # --- context manager -----------------------------------------------------
    def __enter__(self) -> "ServerBackend":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.stop()
        except Exception:  # noqa: BLE001 — teardown must never mask the real error
            log.exception("error during backend teardown for %r", self.job.label)
