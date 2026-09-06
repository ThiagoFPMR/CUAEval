"""Run the jobs of a plan in sequence: for each, bring the model up, benchmark it,
then tear it down before moving to the next. One model is loaded at a time."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .b2 import B2_APP_KEY_ENV, B2_ENDPOINT_ENV, B2_KEY_ID_ENV
from .backends import make_backend
from .config import JobConfig, Plan
from .models import is_b2
from .osworld import read_score, run_benchmark, runner_script
from .results import make_syncer
from .util import log

# Env vars OSWorld's `aws` provider needs to launch client VMs (see SETUP_GUIDELINE §3).
_AWS_REQUIRED_ENV = (
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "AWS_SECURITY_GROUP_ID", "AWS_SUBNET_ID",
)


@dataclass
class JobResult:
    label: str
    status: str                       # "ok" | "failed" | "skipped"
    detail: str = ""
    num_examples: int | None = None
    score: float | None = None
    results_synced: bool = True       # False => the Backblaze mirror of this job's results failed


def _preflight(plan: Plan, *, dry_run: bool = False) -> None:
    """Fail fast on config that can't possibly work, before spending a run."""
    if not plan.osworld_repo.exists():
        raise FileNotFoundError(
            f"osworld_repo does not exist: {plan.osworld_repo}\n"
            f"  run `cuaeval bootstrap <plan>` first to clone OSWorld + install adapters."
        )
    for job in plan.jobs:
        runner_script(plan.osworld_repo, job.runner)  # raises if missing
    _preflight_results_sync(plan)
    if not dry_run:
        _preflight_aws(plan)
        _preflight_b2_weights(plan)


def _preflight_aws(plan: Plan) -> None:
    """If any job uses the aws VM provider, the client-launch env must be set."""
    if not any(j.provider_name == "aws" for j in plan.jobs):
        return
    missing = [v for v in _AWS_REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            "provider_name=aws needs these env vars set (see SETUP_GUIDELINE §3): "
            + ", ".join(missing)
        )


def _preflight_b2_weights(plan: Plan) -> None:
    """A b2:// checkpoint is unreachable without credentials + an endpoint. Say so
    now rather than after renting a GPU box and waiting for the stage to fail."""
    if not any(is_b2(j.weights) for j in plan.jobs):
        return
    missing = [v for v in (B2_KEY_ID_ENV, B2_APP_KEY_ENV, B2_ENDPOINT_ENV)
               if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            "b2:// weights need these env vars set (see the README's Weights section): "
            + ", ".join(missing)
        )


def _preflight_results_sync(plan: Plan) -> None:
    """Results live on one un-backed-up EBS volume unless they're mirrored. That's
    a whole campaign's worth of GPU time riding on one instance not dying, so say
    so up front rather than at the end."""
    cfg = plan.results_sync
    if cfg is None or not cfg.enabled:
        log.warning("no results_sync configured — benchmark results will exist ONLY on "
                    "this host's disk. If it is terminated or reclaimed, the run is lost. "
                    "Add a results_sync block to the plan (or pass --results-b2).")
        return
    log.info("results sync: %s (every %ds, videos=%s)",
             cfg.b2_uri, cfg.interval, "yes" if cfg.include_videos else "no")


def run_plan(plan: Plan, *, dry_run: bool = False, only: set[str] | None = None,
             keep_going: bool = False) -> list[JobResult]:
    _preflight(plan, dry_run=dry_run)
    results: list[JobResult] = []

    jobs = [j for j in plan.jobs if not only or j.label in only]
    log.info("plan: %d job(s) to run (of %d) against OSWorld at %s",
             len(jobs), len(plan.jobs), plan.osworld_repo)

    for i, job in enumerate(jobs, 1):
        log.info("=" * 72)
        log.info("[%d/%d] JOB %r  serve=%s/%s  weights=%s",
                 i, len(jobs), job.label, job.serve.location, job.serve.framework,
                 job.weights)
        syncer = make_syncer(plan, job, dry_run=dry_run)
        try:
            _run_one(plan, job, syncer, dry_run=dry_run)
            score = read_score(plan, job)
            res = JobResult(job.label, "ok",
                            num_examples=score[0] if score else None,
                            score=score[1] if score else None,
                            results_synced=not syncer.failed)
        except Exception as exc:  # noqa: BLE001
            log.exception("job %r failed", job.label)
            # The syncer's final pass has already run (its __exit__ fires on the
            # way out of _run_one), so partial results are saved even here.
            res = JobResult(job.label, "failed", detail=str(exc),
                            results_synced=not syncer.failed)
            if not keep_going and not dry_run:
                results.append(res)
                _summary(results)
                raise
        results.append(res)

    _summary(results)
    return results


def _run_one(plan: Plan, job: JobConfig, syncer, *, dry_run: bool) -> None:
    # Syncer OUTSIDE the backend so its final upload happens after the GPU box is
    # torn down — stop paying for the instance first, then spend time on Backblaze.
    with syncer:
        with make_backend(job, dry_run=dry_run) as backend:
            backend.start()
            backend.wait_ready()
            run_benchmark(plan, job, backend.endpoint, dry_run=dry_run)
        # backend.__exit__ -> stop(): unload the model before the next job.


def _summary(results: list[JobResult]) -> None:
    log.info("=" * 72)
    log.info("SUMMARY")
    for r in results:
        if r.score is not None:
            log.info("  %-28s %-8s  %5.1f%%  (n=%d)",
                     r.label, r.status, r.score * 100, r.num_examples)
        else:
            extra = f"  ({r.detail})" if r.detail else ""
            log.info("  %-28s %-8s%s", r.label, r.status, extra)
    unsaved = [r.label for r in results if not r.results_synced]
    if unsaved:
        log.error("RESULTS NOT UPLOADED for: %s — they exist only on this host's "
                  "disk. Copy them off before terminating it.", ", ".join(unsaved))
