"""vast.ai instance lifecycle via the `vastai` CLI.

Provision (rent) a GPU box, wait until it is running with sshd reachable, and
destroy it on teardown — so a run can stand up its own serving box and never
leak it. `provision` and `destroy` are independent knobs (see VastConfig): you
can destroy a box you rented by hand just as well as one CUAEval created.

CUAEval reaches the box over plain SSH. Rather than thread host/port/user through
every ssh/rsync/scp call, we write a Host alias into ~/.cuaeval/ssh_config (and
Include that from ~/.ssh/config once), so `ssh <alias>` Just Works everywhere the
existing RemoteProcessServer already uses `serve.ssh_host`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import VastConfig
from .util import fmt_cmd, log, run

_SSH_DIR = Path.home() / ".ssh"
_CUAEVAL_SSH_CONFIG = Path.home() / ".cuaeval" / "ssh_config"


class VastError(RuntimeError):
    pass


class VastManager:
    """Thin wrapper over `vastai <cmd> --raw`, parsing JSON output."""

    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def _json(self, args: list[str], *, what: str) -> object:
        cmd = ["vastai", *args, "--raw"]
        if self.dry_run:
            log.info("[dry-run] %s", fmt_cmd(cmd))
            return {}
        cp = run(cmd, capture=True, check=False)
        if cp.returncode != 0:
            raise VastError(f"{what} failed: {fmt_cmd(cmd)}\n{cp.stderr or cp.stdout}")
        text = (cp.stdout or "").strip()
        try:
            return json.loads(text) if text else {}
        except json.JSONDecodeError as e:
            raise VastError(f"{what}: could not parse vastai JSON: {e}\noutput: {text[:500]}")

    # --- offers / create -----------------------------------------------------
    def search_offer(self, cfg: VastConfig) -> int:
        query = cfg.query or self._build_query(cfg)
        log.info("searching vast offers: %s", query)
        offers = self._json(["search", "offers", query, "-o", "dph_total"],
                            what="search offers")
        if self.dry_run:
            return 0
        if not isinstance(offers, list) or not offers:
            raise VastError(f"no vast offers matched query: {query!r}")
        offer = offers[0]  # cheapest (ordered by dph_total ascending)
        log.info("selected offer %s: %s x%s @ $%.3f/hr",
                 offer.get("id"), offer.get("gpu_name"), offer.get("num_gpus"),
                 offer.get("dph_total", 0.0))
        return int(offer["id"])

    @staticmethod
    def _build_query(cfg: VastConfig) -> str:
        parts = [f"num_gpus={cfg.num_gpus}", "rentable=true"]
        if cfg.gpu_name:
            parts.insert(0, f"gpu_name={cfg.gpu_name}")
        if cfg.disk:
            parts.append(f"disk_space>={cfg.disk}")
        return " ".join(parts)

    def create(self, offer_id: int, cfg: VastConfig, image: str, label: str) -> int:
        args = ["create", "instance", str(offer_id),
                "--image", image, "--disk", str(cfg.disk),
                "--ssh", "--direct", "--label", label]
        if cfg.onstart_cmd:
            args += ["--onstart-cmd", cfg.onstart_cmd]
        args += cfg.create_args
        log.info("creating vast instance from offer %s (image=%s, disk=%sGB)",
                 offer_id, image, cfg.disk)
        res = self._json(args, what="create instance")
        if self.dry_run:
            return 0
        if not (isinstance(res, dict) and res.get("success") and res.get("new_contract")):
            raise VastError(f"create instance did not return a contract id: {res}")
        instance_id = int(res["new_contract"])
        log.info("created vast instance %s", instance_id)
        return instance_id

    # --- inspect / wait / destroy -------------------------------------------
    def info(self, instance_id: int) -> dict:
        res = self._json(["show", "instance", str(instance_id)], what="show instance")
        # vastai may return the instance dict directly or wrapped in a list.
        if isinstance(res, list):
            res = res[0] if res else {}
        return res if isinstance(res, dict) else {}

    def wait_running(self, instance_id: int, cfg: VastConfig) -> tuple[str, int]:
        """Block until the instance is running with ssh_host/ssh_port populated.
        Returns (ssh_host, ssh_port)."""
        if self.dry_run:
            log.info("[dry-run] would wait for vast instance %s to be running", instance_id)
            return ("dry-run-host", 22)
        deadline = time.time() + cfg.wait_timeout
        last = ""
        while time.time() < deadline:
            info = self.info(instance_id)
            status = info.get("actual_status")
            host, port = info.get("ssh_host"), info.get("ssh_port")
            msg = f"status={status} ssh={host}:{port}"
            if msg != last:
                log.info("vast instance %s: %s", instance_id, msg)
                last = msg
            if status == "running" and host and port:
                return str(host), int(port)
            if status in ("exited", "error"):
                raise VastError(f"vast instance {instance_id} entered status {status!r}: "
                                f"{info.get('status_msg')}")
            time.sleep(10)
        raise VastError(f"vast instance {instance_id} not running after {cfg.wait_timeout}s")

    def find_by_ssh(self, ssh_host: str) -> int | None:
        """Best-effort: map a user-provided ssh_host (a vast ssh_host, or an alias
        whose HostName is one) to an instance id, for destroy of a box we didn't create."""
        if self.dry_run:
            return None
        res = self._json(["show", "instances"], what="show instances")
        if not isinstance(res, list):
            return None
        hostname = _alias_hostname(ssh_host) or ssh_host
        for inst in res:
            if str(inst.get("ssh_host")) == hostname or str(inst.get("public_ipaddr")) == hostname:
                return int(inst["id"])
        return None

    def destroy(self, instance_id: int) -> None:
        log.info("destroying vast instance %s", instance_id)
        cmd = ["vastai", "destroy", "instance", str(instance_id)]
        if self.dry_run:
            log.info("[dry-run] %s", fmt_cmd(cmd))
            return
        cp = run(cmd, capture=True, check=False)
        if cp.returncode != 0:
            log.error("failed to destroy vast instance %s: %s",
                      instance_id, cp.stderr or cp.stdout)
        else:
            log.info("vast instance %s destroyed", instance_id)


# --- ssh alias management ----------------------------------------------------
def write_ssh_alias(alias: str, host: str, port: int, cfg: VastConfig,
                    *, dry_run: bool = False) -> None:
    """Upsert a Host block for `alias` into ~/.cuaeval/ssh_config, and make sure
    ~/.ssh/config Includes it. After this, `ssh <alias>` (and rsync/scp) work."""
    block = _render_alias_block(alias, host, port, cfg)
    if dry_run:
        log.info("[dry-run] would write ssh alias %r -> %s:%s into %s",
                 alias, host, port, _CUAEVAL_SSH_CONFIG)
        return
    _CUAEVAL_SSH_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    existing = _CUAEVAL_SSH_CONFIG.read_text() if _CUAEVAL_SSH_CONFIG.exists() else ""
    merged = _replace_host_block(existing, alias, block)
    _CUAEVAL_SSH_CONFIG.write_text(merged)
    _CUAEVAL_SSH_CONFIG.chmod(0o600)
    _ensure_ssh_include()
    log.info("wrote ssh alias %r -> %s:%s (%s)", alias, host, port, _CUAEVAL_SSH_CONFIG)


def _render_alias_block(alias: str, host: str, port: int, cfg: VastConfig) -> str:
    lines = [
        f"Host {alias}",
        f"    HostName {host}",
        f"    Port {port}",
        f"    User {cfg.ssh_user}",
        "    StrictHostKeyChecking accept-new",
        "    UserKnownHostsFile /dev/null",
        "    ServerAliveInterval 30",
    ]
    if cfg.identity_file:
        lines.append(f"    IdentityFile {cfg.identity_file}")
    return "\n".join(lines) + "\n"


def _replace_host_block(existing: str, alias: str, block: str) -> str:
    """Remove any prior `Host <alias>` stanza, then append the new one."""
    out, skipping = [], False
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped.startswith("Host "):
            skipping = stripped.split()[1:] == [alias]
        if not skipping:
            out.append(line)
    body = "\n".join(out).rstrip()
    return (body + "\n\n" if body else "") + block


def _ensure_ssh_include() -> None:
    _SSH_DIR.mkdir(mode=0o700, exist_ok=True)
    cfg = _SSH_DIR / "config"
    include = f"Include {_CUAEVAL_SSH_CONFIG}"
    text = cfg.read_text() if cfg.exists() else ""
    if include in text:
        return
    # Include must precede host-specific blocks to take effect; prepend it.
    cfg.write_text(include + "\n\n" + text)
    cfg.chmod(0o600)


def _alias_hostname(alias: str) -> str | None:
    """If `alias` is defined in ~/.cuaeval/ssh_config, return its HostName."""
    if not _CUAEVAL_SSH_CONFIG.exists():
        return None
    in_block = False
    for line in _CUAEVAL_SSH_CONFIG.read_text().splitlines():
        s = line.strip()
        if s.startswith("Host "):
            in_block = s.split()[1:] == [alias]
        elif in_block and s.lower().startswith("hostname "):
            return s.split(None, 1)[1]
    return None
