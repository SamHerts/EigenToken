#!/usr/bin/env bash
# Copies this checkout to the Pi, installs it with CPU-only torch, and (re)starts
# the service. Run from your workstation:  deploy/pi/install.sh [SSH_HOST]
#
# The public address, https://eigentoken-api.samuelherts.com, is a Cloudflare
# tunnel managed in the dashboard (Zero Trust → Networks → Tunnels → eigentoken,
# public hostname → HTTP 127.0.0.1:8077). On the Pi it is the cloudflared service,
# installed once with `sudo cloudflared service install <TOKEN>`; this script
# doesn't touch it.
set -euo pipefail

host=${1:-ras1}
root=$(cd "$(dirname "$0")/../.." && pwd)

rsync -a --delete --exclude .venv --exclude .env --exclude .git \
  --exclude __pycache__ --exclude NOTES.md "$root/" "$host:eigentoken/"
ssh "$host" 'set -e
  command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  cd ~/eigentoken
  mkdir -p ~/eigentoken-models
  [ -d .venv ] || ~/.local/bin/uv venv --python 3.11 .venv
  # The PyPI torch for ARM pulls in CUDA; the CPU index has the small build.
  ~/.local/bin/uv pip install -q --python .venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cpu
  ~/.local/bin/uv pip install -q --python .venv/bin/python --no-sources ".[web]"
  sudo cp deploy/pi/eigentoken.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable -q eigentoken
  sudo systemctl restart eigentoken
  systemctl --no-pager --lines=0 status eigentoken'
