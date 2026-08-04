"""CLI entry point.

    cuaeval run   plans/example.yaml [--dry-run] [--only LABEL ...] [--keep-going]
    cuaeval check plans/example.yaml            # validate + print the resolved plan
"""
from __future__ import annotations

import argparse
import sys

from .config import load_plan
from .orchestrator import run_plan
from .util import log, setup_logging


def _cmd_run(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
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
        log.info("  - %-24s %-6s %-7s %s  weights=%s",
                 job.label, s.framework, f"tp{s.tp}", where, job.weights)
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
    r.set_defaults(func=_cmd_run)

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
