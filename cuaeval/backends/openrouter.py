"""OpenRouter "backend": the model is already hosted, so there is nothing to serve.

The runner is pointed straight at OpenRouter's OpenAI-compatible API. The key is
read from the env var named by serve.api_key_env and handed to the runner as
OPENAI_API_KEY; serve.api_model travels as CUAEVAL_API_MODEL so the agents send
the OpenRouter id while --model (the job label) still names the results dir.
"""
from __future__ import annotations

import os

from ..serving import serving
from ..util import log
from .base import ServerBackend


class OpenRouterBackend(ServerBackend):
    @property
    def endpoint(self) -> str:
        return self.serve.api_base_url

    def start(self) -> None:
        log.info("openrouter: using hosted model %r at %s (nothing to serve)",
                 self.serve.api_model, self.endpoint)

    def stop(self) -> None:
        pass

    def wait_ready(self) -> None:
        if self.dry_run:
            log.info("[dry-run] would check %r is listed at %s", self.serve.api_model, self.endpoint)
            return
        # /models is public on OpenRouter and lists every id, so a typo fails here
        # instead of on every step of the benchmark.
        if not serving(self.endpoint, self.serve.api_model, timeout=30):
            raise RuntimeError(f"{self.job.label}: model {self.serve.api_model!r} not found "
                               f"at {self.endpoint}/models")

    def runner_env(self) -> dict[str, str]:
        env = {"CUAEVAL_API_MODEL": self.serve.api_model}
        key = os.environ.get(self.serve.api_key_env)
        if key:
            env["OPENAI_API_KEY"] = key
        return env
