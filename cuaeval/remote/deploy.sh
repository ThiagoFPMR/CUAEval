#!/usr/bin/env bash
# Runs ON a vast.ai instance (rsynced here by CUAEval's RemoteProcessServer).
# The instance's base image already provides sglang/vllm; this script only:
#   1. sources Backblaze creds (b2.env, if present) — kept out of argv/ps/logs;
#   2. stages the model weights (B2 download via b2.py, or a pre-staged path);
#   3. execs the inference server in the FOREGROUND so the enclosing tmux session
#      owns the process (killing the session unloads the model / frees VRAM).
#
# All inputs arrive as environment variables set by the caller (see remote.py):
#   CUAEVAL_PYTHON  python interpreter on the instance          (e.g. python3)
#   MODEL_SRC       b2:// URI or a filesystem path on the instance
#   MODEL_DIR       dir the server loads from (== MODEL_SRC when not B2)
#   IS_B2           "1" if MODEL_SRC is a b2:// URI, else "0"
#   B2_WORKERS      parallel B2 downloads                        (default 4)
#   SERVE_ARGS      the server launch argv, already shell-quoted, WITHOUT the
#                   python prefix (e.g. -m sglang.launch_server --model-path ...)
set -euo pipefail

PY="${CUAEVAL_PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. credentials (optional) --------------------------------------------------
if [[ -f "${HERE}/b2.env" ]]; then
    # shellcheck disable=SC1091
    source "${HERE}/b2.env"
fi

# 2. stage weights -----------------------------------------------------------
if [[ "${IS_B2:-0}" == "1" ]]; then
    echo "=== [deploy] staging ${MODEL_SRC} -> ${MODEL_DIR} ==="
    # boto3 (B2 speaks the S3 API) may or may not be in the stock image.
    if ! "${PY}" -c 'import boto3' 2>/dev/null; then
        echo "=== [deploy] installing boto3 ==="
        "${PY}" -m pip install --quiet --disable-pip-version-check boto3
    fi
    "${PY}" "${HERE}/b2.py" --b2-src "${MODEL_SRC}" --model-dir "${MODEL_DIR}" \
        --workers "${B2_WORKERS:-4}"
else
    echo "=== [deploy] using pre-staged weights at ${MODEL_DIR} ==="
    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "ERROR: MODEL_DIR '${MODEL_DIR}' not found on the instance." >&2
        exit 1
    fi
fi

# 3. serve (foreground) ------------------------------------------------------
echo "=== [deploy] launching server: ${PY} ${SERVE_ARGS} ==="
# SERVE_ARGS is a shell-quoted argv; eval so its quoting is honoured.
eval "exec ${PY} ${SERVE_ARGS}"
