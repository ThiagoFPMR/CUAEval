"""Serving backends: bring a model up behind an OpenAI-compatible endpoint and
tear it down. `make_backend` picks the implementation from the job's serve config.
"""
from __future__ import annotations

from ..config import JobConfig
from .base import ServerBackend
from .local import LocalDockerServer
from .remote import RemoteProcessServer, VastRemoteServer


def make_backend(job: JobConfig, *, dry_run: bool = False) -> ServerBackend:
    if job.serve.location == "local":
        return LocalDockerServer(job, dry_run=dry_run)
    v = job.serve.vast
    if v.provision or v.resolved_destroy():
        return VastRemoteServer(job, dry_run=dry_run)
    return RemoteProcessServer(job, dry_run=dry_run)


__all__ = ["ServerBackend", "LocalDockerServer", "RemoteProcessServer",
           "VastRemoteServer", "make_backend"]
