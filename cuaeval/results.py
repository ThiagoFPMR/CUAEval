"""Get benchmark results off the host's disk while the run is still going.

The OSWorld runner writes each finished example's result.txt / traj.jsonl /
screenshots / recording.mp4 into the results tree on THIS host's EBS volume, and
nothing else ever copies them anywhere. A terminated or reclaimed host therefore
loses a whole campaign — hours of rented GPU plus every task VM it drove.

ResultsSyncer mirrors that tree to Backblaze B2 on an interval while the
benchmark runs, and once more when the job ends (including when it ends by
raising), so the worst case is losing the examples finished since the last tick
rather than all of them. Uploads are size-compared and additive — the runner writes each example
directory once and never rewrites it, so a tick after a quiet period costs one
LIST and nothing else.

A failed sync never fails the job: the results still exist locally, and killing
a finished run because an upload 500'd would be a worse outcome than the thing
we're defending against. Failures are logged loudly and reported in the summary.
"""
from __future__ import annotations

import threading
from pathlib import Path

from .config import JobConfig, Plan, ResultsSyncConfig
from .b2 import upload_dir
from .util import log

# Videos dominate the byte count; everything else in a results tree is small.
VIDEO_PATTERNS = ("*.mp4",)


class ResultsSyncer:
    """Periodic + final mirror of one job's results tree to Backblaze B2.

    Used as a context manager: the final sync runs on __exit__, so results are
    saved whether the job returned or raised.
    """

    def __init__(self, cfg: ResultsSyncConfig, local_dir: Path, dest: str,
                 *, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.local_dir = local_dir
        self.dest = dest
        self.dry_run = dry_run
        self.failed = False           # read by the orchestrator for the summary
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- one pass ------------------------------------------------------------
    def sync_once(self, *, final: bool = False) -> None:
        what = "final" if final else "periodic"
        if self.dry_run:
            log.info("[dry-run] would %s sync %s -> %s (exclude=%s)",
                     what, self.local_dir, self.dest, list(self._exclude()) or "none")
            return
        if not self.local_dir.is_dir():
            # Nothing written yet — normal for an early tick or a job that died
            # before its first example finished.
            log.debug("results sync: %s does not exist yet, skipping", self.local_dir)
            return
        try:
            uploaded, skipped = upload_dir(
                self.local_dir, self.dest,
                workers=self.cfg.workers, exclude=self._exclude(), quiet=True,
                endpoint_url=self.cfg.endpoint_url or None,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed = True
            log.error("results %s sync FAILED (%s -> %s): %s", what,
                      self.local_dir, self.dest, exc)
            log.error("results are still on local disk at %s — copy them off "
                      "before terminating this host.", self.local_dir)
            return
        self.failed = False          # a later success clears an earlier failure
        if uploaded or final:
            log.info("results %s sync: %d uploaded, %d unchanged -> %s",
                     what, uploaded, skipped, self.dest)

    def _exclude(self) -> tuple[str, ...]:
        pats = tuple(self.cfg.exclude)
        if not self.cfg.include_videos:
            pats += VIDEO_PATTERNS
        return pats

    # --- background loop -----------------------------------------------------
    def start(self) -> None:
        if self.cfg.interval <= 0 or self.dry_run:
            return
        self._thread = threading.Thread(target=self._loop, name="results-sync",
                                        daemon=True)
        self._thread.start()
        log.info("results sync every %ds -> %s", self.cfg.interval, self.dest)

    def _loop(self) -> None:
        # wait() rather than sleep() so stop() interrupts a long interval at once.
        while not self._stop.wait(self.cfg.interval):
            self.sync_once()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.join_timeout)
            self._thread = None

    # --- context manager -----------------------------------------------------
    def __enter__(self) -> "ResultsSyncer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
        self.sync_once(final=True)


class _NullSyncer:
    """Stand-in when no results_sync is configured, so callers stay branch-free."""

    failed = False

    def sync_once(self, *, final: bool = False) -> None:
        pass

    def __enter__(self) -> "_NullSyncer":
        return self

    def __exit__(self, *exc) -> None:
        pass


def destination(cfg: ResultsSyncConfig) -> str:
    """Where the results tree is mirrored.

    Deliberately NOT keyed by job label: the runner already nests results as
    <action_space>/<observation_type>/<label>/<domain>/<example>/, so one mirror
    of the whole tree keeps every job separate without duplicating the others'
    files under each job's prefix. Reruns and resumed runs merge into the same
    place, matching the runner's own resumability. Set `run_id` to namespace
    separate campaigns that must not merge.
    """
    base = cfg.b2_uri.rstrip("/")
    return f"{base}/{cfg.run_id}" if cfg.run_id else base


def make_syncer(plan: Plan, job: JobConfig, *, dry_run: bool = False):
    """A live ResultsSyncer, or a no-op stand-in when results_sync is unset."""
    cfg = plan.results_sync
    if cfg is None or not cfg.enabled:
        return _NullSyncer()
    # The runner is invoked with cwd=osworld_repo, so its --result_dir is relative
    # to the checkout (see osworld.run_benchmark).
    local = (plan.osworld_repo / job.result_dir).resolve()
    return ResultsSyncer(cfg, local, destination(cfg), dry_run=dry_run)
