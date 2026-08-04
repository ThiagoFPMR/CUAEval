"""Drive a *separately configured* OSWorld checkout.

We don't reimplement OSWorld — we shell out to its own multi-env runner
(scripts/python/run_multienv_<runner>.py) with the endpoint wired in via the
OPENAI_BASE_URL/OPENAI_API_KEY env vars those runners already read, exactly like
OSWorld/run_osworld_holo3_here.sh does. Results nest under the job label and are
resumable (the runner's get_unfinished()), so lined-up jobs never collide.
"""
from __future__ import annotations

from pathlib import Path

from .config import JobConfig, Plan
from .util import log, run


def runner_script(repo: Path, runner: str) -> Path:
    path = repo / "scripts" / "python" / f"run_multienv_{runner}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"OSWorld runner not found: {path}\n"
            f"  (job runner={runner!r}; expected scripts/python/run_multienv_{runner}.py "
            f"in {repo})"
        )
    return path


def build_command(plan: Plan, job: JobConfig) -> list[str]:
    script = runner_script(plan.osworld_repo, job.runner)
    cmd = [
        plan.osworld_python, str(script),
        "--model", job.label,
        "--result_dir", job.result_dir,
        "--test_all_meta_path", job.meta,
        "--max_steps", str(job.max_steps),
        "--num_envs", str(job.num_envs),
        "--provider_name", job.provider_name,
    ]
    if job.domain and job.domain != "all":
        cmd += ["--domain", job.domain]
    cmd += job.runner_args
    return cmd


def run_benchmark(plan: Plan, job: JobConfig, endpoint: str, *, dry_run: bool = False) -> None:
    cmd = build_command(plan, job)
    env = {"OPENAI_BASE_URL": endpoint, "OPENAI_API_KEY": job.api_key}
    log.info("running OSWorld: model=%s runner=%s endpoint=%s", job.label, job.runner, endpoint)
    # Stream the runner's output straight through (no capture) so long runs are
    # observable live; cwd is the repo so its relative paths (evaluation_examples,
    # results, .env) resolve.
    run(cmd, env=env, cwd=str(plan.osworld_repo), dry_run=dry_run, check=True)


def read_score(plan: Plan, job: JobConfig) -> tuple[int, float] | None:
    """Average of all result.txt values under the job's label dir. Returns
    (num_examples, mean_success) or None if nothing has been scored yet."""
    result_root = (plan.osworld_repo / job.result_dir).resolve()
    if not result_root.exists():
        return None
    scores: list[float] = []
    # The runner nests .../<action_space>/<obs>/<label>/<domain>/<example>/result.txt;
    # find the label dir wherever it sits and average every result.txt beneath it.
    for label_dir in result_root.rglob(job.label):
        if not label_dir.is_dir():
            continue
        for rt in label_dir.rglob("result.txt"):
            try:
                scores.append(float(rt.read_text().strip()))
            except (ValueError, OSError):
                scores.append(0.0)
    if not scores:
        return None
    return len(scores), sum(scores) / len(scores)
