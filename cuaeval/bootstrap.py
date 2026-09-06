"""Provision the OSWorld checkout a plan runs against.

This is the "clone CUAEval into a fresh EC2 box and it sets everything up" piece:

    1. git clone a PRISTINE upstream OSWorld into `osworld_repo`, pinned at `ref`;
    2. build its .venv and pip install -r requirements.txt;
    3. vendor our custom evaluation metas (CUAEval/evaluation_examples/) into the
       checkout — these task lists (e.g. test_nogdrive.json) are ours, not shipped
       by upstream OSWorld, and plans reference them via `meta:`;
    4. copy each named adapter set (CUAEval/adapters/<name>/) on top per its
       manifest, so the custom harness (e.g. Holo3) lands in the right places;
    5. verify each adapter's runner script now resolves.

Idempotent: an existing checkout is left in place (use --force to re-copy
adapters over it; a clean re-clone means removing the dir yourself). Nothing
model-specific is baked in here — adapters are data under adapters/.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import Plan
from .util import log, run

# CUAEval/adapters/  (this file is CUAEval/cuaeval/bootstrap.py)
ADAPTERS_ROOT = Path(__file__).resolve().parent.parent / "adapters"
# CUAEval/evaluation_examples/ — custom OSWorld task metas we keep version-controlled
# here and copy into the checkout's evaluation_examples/ (upstream doesn't ship them).
META_ROOT = Path(__file__).resolve().parent.parent / "evaluation_examples"


@dataclass
class AdapterManifest:
    name: str
    runner: str | None
    files: list[tuple[str, str]]   # (src-relative-to-adapter-dir, dest-relative-to-repo)
    root: Path

    @classmethod
    def load(cls, name: str) -> "AdapterManifest":
        adir = ADAPTERS_ROOT / name
        mpath = adir / "manifest.yaml"
        if not mpath.exists():
            available = sorted(p.name for p in ADAPTERS_ROOT.iterdir() if p.is_dir()) \
                if ADAPTERS_ROOT.exists() else []
            raise FileNotFoundError(
                f"adapter set {name!r} not found: {mpath} is missing "
                f"(available: {available or 'none'})"
            )
        data = yaml.safe_load(mpath.read_text()) or {}
        files = [(f["src"], f["dest"]) for f in (data.get("files") or [])]
        if not files:
            raise ValueError(f"adapter {name!r}: manifest lists no files")
        return cls(name=data.get("name", name), runner=data.get("runner"),
                   files=files, root=adir)


def bootstrap(plan: Plan, *, dry_run: bool = False, force: bool = False) -> None:
    src = plan.osworld
    repo = plan.osworld_repo

    _clone(src.repo_url, src.ref, repo, dry_run=dry_run)
    _build_venv(repo, src.python, pip_install=src.pip_install, dry_run=dry_run)
    _vendor_metas(repo, dry_run=dry_run, force=force)

    for name in src.adapters:
        _apply_adapter(AdapterManifest.load(name), repo, dry_run=dry_run, force=force)

    _verify_runners(plan, src.adapters, dry_run=dry_run)
    log.info("bootstrap complete: OSWorld ready at %s", repo)


def _clone(repo_url: str, ref: str | None, repo: Path, *, dry_run: bool) -> None:
    if repo.exists() and any(repo.iterdir()):
        log.info("OSWorld checkout already present at %s — leaving it in place "
                 "(remove it for a clean re-clone)", repo)
        return
    log.info("cloning %s -> %s", repo_url, repo)
    run(["git", "clone", repo_url, str(repo)], dry_run=dry_run)
    if ref:
        log.info("pinning OSWorld at ref %s", ref)
        run(["git", "-C", str(repo), "checkout", "--detach", ref], dry_run=dry_run)
    else:
        log.warning("no osworld.ref pinned — using the repo's default branch HEAD. "
                    "Pin a commit/tag for reproducible runs.")


def _build_venv(repo: Path, python: str, *, pip_install: bool, dry_run: bool) -> None:
    venv = repo / ".venv"
    if venv.exists():
        log.info("venv already present at %s — skipping create", venv)
    else:
        log.info("creating venv at %s (%s)", venv, python)
        run([python, "-m", "venv", str(venv)], dry_run=dry_run)
    if not pip_install:
        return
    pip = venv / "bin" / "pip"
    req = repo / "requirements.txt"
    if not dry_run and not req.exists():
        log.warning("no requirements.txt at %s — skipping pip install", req)
        return
    log.info("installing OSWorld requirements (this can take a while)")
    run([str(pip), "install", "-r", str(req)], cwd=str(repo), dry_run=dry_run)


def _vendor_metas(repo: Path, *, dry_run: bool, force: bool) -> None:
    """Copy CUAEval/evaluation_examples/* into <repo>/evaluation_examples/.

    These are our custom task lists (e.g. test_nogdrive.json) that upstream OSWorld
    doesn't ship; a plan's `meta:` points at one of them. We keep only the CUSTOM
    metas here, so this never touches upstream files like test_all.json. Idempotent:
    an existing file is left in place unless `force`.
    """
    if not META_ROOT.exists():
        return
    metas = sorted(p for p in META_ROOT.rglob("*") if p.is_file() and p.name != "README.md")
    if not metas:
        return
    dest_root = repo / "evaluation_examples"
    log.info("vendoring %d custom eval meta(s) -> %s", len(metas), dest_root)
    for src in metas:
        rel = src.relative_to(META_ROOT)
        dest = dest_root / rel
        if dest.exists() and not force:
            log.warning("  skip (exists, use --force to overwrite): evaluation_examples/%s", rel)
            continue
        log.info("  copy evaluation_examples/%s", rel)
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _apply_adapter(m: AdapterManifest, repo: Path, *, dry_run: bool, force: bool) -> None:
    log.info("applying adapter %r (%d file(s)) -> %s", m.name, len(m.files), repo)
    for src_rel, dest_rel in m.files:
        src = m.root / src_rel
        dest = repo / dest_rel
        if not dry_run and not src.exists():
            raise FileNotFoundError(f"adapter {m.name!r}: source file missing: {src}")
        if dest.exists() and not force:
            log.warning("  skip (exists, use --force to overwrite): %s", dest_rel)
            continue
        log.info("  copy %s -> %s", src_rel, dest_rel)
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _verify_runners(plan: Plan, adapters: list[str], *, dry_run: bool) -> None:
    if dry_run:
        return
    for name in adapters:
        runner = AdapterManifest.load(name).runner
        if not runner:
            continue
        script = plan.osworld_repo / "scripts" / "python" / f"run_multienv_{runner}.py"
        if not script.exists():
            raise FileNotFoundError(
                f"adapter {name!r} declares runner {runner!r} but {script} is "
                f"missing after copy — check the manifest's `dest` paths."
            )
        log.info("verified runner for %r: %s", name, script.relative_to(plan.osworld_repo))
