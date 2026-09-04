#!/usr/bin/env bash
# One-shot provisioning for a FRESH Ubuntu EC2 host.
#
#   git clone <this repo> && cd CUAEval
#   ./setup_ec2.sh [plan.yaml]        # default: plans/ec2_aws_holo3.yaml
#
# Installs the host-side toolchain (uv, awscli, the vast.ai CLI), syncs CUAEval's
# own venv, then runs `cuaeval bootstrap` — which clones OSWorld at the plan's
# pinned ref, builds its venv, and copies the adapters into place. After this you
# only need to export your AWS + vast credentials and `cuaeval run` the plan.
set -euo pipefail
cd "$(dirname "$0")"

PLAN="${1:-plans/ec2_aws_holo3.yaml}"
[[ -f "$PLAN" ]] || { echo "plan not found: $PLAN" >&2; exit 1; }

log() { echo -e "\033[1;36m[setup_ec2]\033[0m $*"; }

# --- system packages ---------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
    log "installing system packages (git, python venv, rsync, tmux, curl)"
    sudo apt-get update -qq
    sudo apt-get install -y -qq git python3-venv python3-pip rsync tmux curl openssh-client
fi

# --- uv (drives CUAEval's own venv) ------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- CUAEval venv ------------------------------------------------------------
log "syncing CUAEval venv (uv sync)"
uv sync

# --- vast.ai CLI -------------------------------------------------------------
if ! uv run vastai --version >/dev/null 2>&1 && ! command -v vastai >/dev/null 2>&1; then
    log "installing vastai CLI"
    uv pip install vastai
fi
if [[ -n "${VAST_API_KEY:-}" ]]; then
    log "configuring vast.ai api key from \$VAST_API_KEY"
    uv run vastai set api-key "$VAST_API_KEY" >/dev/null 2>&1 || \
        vastai set api-key "$VAST_API_KEY" >/dev/null 2>&1 || \
        log "WARN: could not set vast api-key; run 'vastai set api-key <key>' yourself"
fi

# --- awscli (optional convenience; boto3 in OSWorld's venv does the real work) -
if ! command -v aws >/dev/null 2>&1; then
    log "installing awscli"
    uv pip install awscli || true
fi

# --- bootstrap OSWorld + adapters --------------------------------------------
log "bootstrapping OSWorld from plan: $PLAN"
uv run cuaeval bootstrap "$PLAN"

cat <<EOF

$(log "done.")
Next steps:
  1. Export the AWS host/client env (see SETUP_GUIDELINE §3):
       export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
       export AWS_SECURITY_GROUP_ID=sg-...  AWS_SUBNET_ID=subnet-...
       export AWS_DEFAULT_REGION=us-east-1
  2. Make sure the vast.ai api key is set (export VAST_API_KEY=... or 'vastai set api-key').
  3. Preview, then run:
       uv run cuaeval run $PLAN --dry-run
       uv run cuaeval run $PLAN
EOF
