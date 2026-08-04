"""Run the jobs of a plan in sequence: for each, bring the model up, benchmark it,
then tear it down before moving to the next. One model is loaded at a time."""
from __future__ import annotations

from dataclasses import dataclass

from .backends import make_backend
from .config import JobConfig, Plan
from .osworld import read_score, run_benchmark, runner_script
from .util import log


@dataclass
class JobResult:
    label: str
    status: str                       # "ok" | "failed" | "skipped"
    detail: str = ""
    num_examples: int | None = None
    score: float | None = None


def _preflight(plan: Plan) -> None:
    """Fail fast on config that can't possibly work, before spending a run."""
    if not plan.osworld_repo.exists():
        raise FileNotFoundError(f"osworld_repo does not exist: {plan.osworld_repo}")
    for job in plan.jobs:
        runner_script(plan.osworld_repo, job.runner)  # raises if missing


def run_plan(plan: Plan, *, dry_run: bool = False, only: set[str] | None = None,
             keep_going: bool = False) -> list[JobResult]:
    _preflight(plan)
    results: list[JobResult] = []

    jobs = [j for j in plan.jobs if not only or j.label in only]
    log.info("plan: %d job(s) to run (of %d) against OSWorld at %s",
             len(jobs), len(plan.jobs), plan.osworld_repo)

    for i, job in enumerate(jobs, 1):
        log.info("=" * 72)
        log.info("[%d/%d] JOB %r  serve=%s/%s  weights=%s",
                 i, len(jobs), job.label, job.serve.location, job.serve.framework,
                 job.weights)
        try:
            _run_one(plan, job, dry_run=dry_run)
            score = read_score(plan, job)
            res = JobResult(job.label, "ok",
                            num_examples=score[0] if score else None,
                            score=score[1] if score else None)
        except Exception as exc:  # noqa: BLE001
            log.exception("job %r failed", job.label)
            res = JobResult(job.label, "failed", detail=str(exc))
            if not keep_going and not dry_run:
                results.append(res)
                _summary(results)
                raise
        results.append(res)

    _summary(results)
    return results


def _run_one(plan: Plan, job: JobConfig, *, dry_run: bool) -> None:
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
