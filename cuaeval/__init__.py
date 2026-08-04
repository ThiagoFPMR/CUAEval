"""CUAEval — a small harness that lines up several computer-use-agent models and
tests each on a benchmark (OSWorld for now) in sequence.

Each model is served behind an OpenAI-compatible endpoint that CUAEval brings up
and tears down between jobs — either as a local Docker container on this host, or
as a process on a remote vast.ai instance reached over SSH. The benchmark itself
is run by shelling out to a *separately configured* OSWorld checkout.
"""

__version__ = "0.1.0"
