"""CLI entry point.

    cuaeval bootstrap plans/example.yaml [--dry-run] [--force]  # clone OSWorld + adapters
    cuaeval run       plans/example.yaml [--dry-run] [--only LABEL ...] [--keep-going]
    cuaeval check     plans/example.yaml        # validate + print the resolved plan
"""
from __future__ import annotations

import argparse
import sys

from .bootstrap import bootstrap
from .config import ResultsSyncConfig, load_plan
from .orchestrator import run_plan
from .results import destination
from .util import log, setup_logging


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    bootstrap(plan, dry_run=args.dry_run, force=args.force)
    return 0


def _apply_sync_overrides(plan, args: argparse.Namespace) -> None:
    """CLI overrides for the plan's results_sync block (--results-b2/--no-videos/
    --no-results-sync), so a one-off run can change where results land without
    editing the plan."""
    if args.no_results_sync:
        if plan.results_sync is not None:
            plan.results_sync.enabled = False
        return
    if args.results_b2:
        if plan.results_sync is None:
            plan.results_sync = ResultsSyncConfig(b2_uri=args.results_b2)
        else:
            plan.results_sync.b2_uri = args.results_b2
            plan.results_sync.enabled = True
    if args.no_videos:
        if plan.results_sync is None:
            log.warning("--no-videos has no effect without results_sync configured")
        else:
            plan.results_sync.include_videos = False


def _cmd_run(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    _apply_sync_overrides(plan, args)
    run_plan(
        plan,
        dry_run=args.dry_run,
        only=set(args.only) if args.only else None,
        keep_going=args.keep_going,
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    log.info("plan OK: %d job(s), OSWorld=%s, python=%s",
             len(plan.jobs), plan.osworld_repo, plan.osworld_python)
    for job in plan.jobs:
        s = job.serve
        where = (f"local docker ({s.resolved_image()})" if s.location == "local"
                 else f"remote {s.ssh_host} (tunnel :{s.resolved_tunnel_port()})")
        if job.domains:
            scope = "domains=" + ",".join(job.domains)
        elif job.domain and job.domain != "all":
            scope = "domain=" + job.domain
        else:
            scope = "all-domains"
        log.info("  - %-24s %-6s %-7s %s  weights=%s  %s",
                 job.label, s.framework, f"tp{s.tp}", where, job.weights, scope)
    cfg = plan.results_sync
    if cfg is not None and cfg.enabled:
        log.info("results sync: %s  every %ds  videos=%s",
                 destination(cfg), cfg.interval,
                 "included" if cfg.include_videos else "EXCLUDED")
    else:
        log.warning("results sync: DISABLED — results will exist only on local disk")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cuaeval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a plan's jobs in sequence")
    r.add_argument("plan")
    r.add_argument("--dry-run", action="store_true",
                   help="print every ssh/docker/runner command without executing")
    r.add_argument("--only", nargs="+", metavar="LABEL",
                   help="run only these job label(s)")
    r.add_argument("--keep-going", action="store_true",
                   help="continue to the next job if one fails")
    r.add_argument("--results-b2", metavar="B2_URI",
                   help="mirror results here (overrides/sets the plan's results_sync.b2_uri)")
    r.add_argument("--no-videos", action="store_true",
                   help="exclude recording.mp4 from the results upload (they dominate "
                        "the byte count); everything else is still saved")
    r.add_argument("--no-results-sync", action="store_true",
                   help="disable the Backblaze results mirror entirely (results stay on local disk)")
    r.set_defaults(func=_cmd_run)

    b = sub.add_parser("bootstrap",
                       help="clone OSWorld at the pinned ref + copy adapters into place")
    b.add_argument("plan")
    b.add_argument("--dry-run", action="store_true",
                   help="print every git/venv/copy step without executing")
    b.add_argument("--force", action="store_true",
                   help="overwrite adapter/meta files that already exist in the checkout")
    b.set_defaults(func=_cmd_bootstrap)

    c = sub.add_parser("check", help="validate a plan and print the resolved jobs")
    c.add_argument("plan")
    c.set_defaults(func=_cmd_check)

    args = p.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
