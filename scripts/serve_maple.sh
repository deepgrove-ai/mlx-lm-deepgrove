#!/usr/bin/env bash
# Start an OpenAI-compatible HTTP endpoint for Maple-Preview.
#
# Serves the local ./maple-2bit-mlx checkpoint under two aliases:
#   maple-preview        exact lm_head (default)
#   maple-preview-flash  approximate FlashHead lm_head (~20% faster decode)
#
# Point MAPLE_MODEL at the Hugging Face repo id to auto-download instead:
#   MAPLE_MODEL=deepgrove/maple-preview-2bit-mlx ./scripts/serve_maple.sh
#
# Usage: ./scripts/serve_maple.sh [PORT] [HOST]
set -euo pipefail

PORT="${1:-8080}"
HOST="${2:-127.0.0.1}"
MODEL="${MAPLE_MODEL:-./maple-2bit-mlx}"

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "No .venv found — run ./setup.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

exec python -m mlx_lm.server \
  --model "$MODEL" \
  --model-registry maple_models.json \
  --trust-remote-code \
  --host "$HOST" \
  --port "$PORT" \
  --max-tokens 4096 \
  --prefill-step-size 2048
