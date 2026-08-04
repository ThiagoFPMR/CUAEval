"""Endpoint health helpers — the Python equivalent of the curl/grep polling in
OSWorld/run_two_models_holo3.sh. Everything talks to the *local* end of the
endpoint (for remote jobs that's the tunnelled port), so these are host-agnostic.
"""
from __future__ import annotations

import time

import requests

from .util import log


def endpoint_up(base_url: str, timeout: float = 8.0) -> bool:
    """Is the OpenAI-compatible server answering /models at all?"""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        return r.ok
    except requests.RequestException:
        return False


def serving(base_url: str, want: str, timeout: float = 8.0) -> bool:
    """Is the server up AND is `want` (checkpoint basename/label) in its model list?

    Mirrors the `grep -q -- "$want"` check in the bash orchestrator: the served
    model id is whatever the launch command labelled it, which we set to the job
    label, so this confirms we're talking to the model we think we are.
    """
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        if not r.ok:
            return False
        return want in r.text
    except requests.RequestException:
        return False


def wait_serving(base_url: str, want: str, up_timeout: int, poll: int, what: str = "") -> None:
    """Block until the endpoint serves `want`, or raise TimeoutError."""
    log.info("waiting for endpoint to serve %r (%s, timeout %ss)...", want, what, up_timeout)
    waited = 0
    while waited < up_timeout:
        if serving(base_url, want):
            log.info("endpoint is serving %r after %ss.", want, waited)
            return
        time.sleep(poll)
        waited += poll
    raise TimeoutError(f"{want!r} not serving after {up_timeout}s (endpoint={base_url})")


def wait_down(base_url: str, down_timeout: int, poll: int) -> None:
    """Block until the endpoint stops answering (best-effort; warns and returns on
    timeout rather than raising — a stuck old server shouldn't abort the plan)."""
    log.info("waiting for old server to go down (timeout %ss)...", down_timeout)
    waited = 0
    while waited < down_timeout:
        if not endpoint_up(base_url):
            log.info("server is down after %ss.", waited)
            return
        time.sleep(poll)
        waited += poll
    log.warning("server still answering after %ss; proceeding anyway.", down_timeout)
