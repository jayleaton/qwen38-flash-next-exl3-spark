#!/usr/bin/env bash
# Convenience launcher: docker compose up + optional Tailscale exposure.
#
#   ./scripts/start.sh              # start (loopback only)
#   SERVE=1 ./scripts/start.sh      # also publish over Tailscale (https, tailnet only)
#
# The model must be downloaded (scripts/download-model.sh) and MODEL_DIR set in .env.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "missing .env (cp .env.example .env, set MODEL_DIR)"; exit 2; }
[ -f api_tokens.yml ] || { echo "missing api_tokens.yml (cp api_tokens.yml.example api_tokens.yml)"; exit 2; }

docker compose up -d
echo "waiting for /v1/models ..."
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:5000/v1/models || true)
  [ "$code" = "200" ] && { echo "ready on http://127.0.0.1:5000"; break; }
  sleep 5
done

if [ "${SERVE:-0}" = "1" ]; then
  sudo tailscale serve --bg 5000
  tailscale serve status
fi
