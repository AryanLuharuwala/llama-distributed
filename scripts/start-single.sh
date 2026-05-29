#!/usr/bin/env bash
# start-single.sh — run llama-distributed on a single machine (no GPU cluster needed)
# Usage:  ./scripts/start-single.sh <model.gguf>
#         ./scripts/start-single.sh <model.gguf> --cuda
set -euo pipefail

MODEL="${1:-}"
shift || true

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model.gguf> [--cuda]"
  echo ""
  echo "This script starts a coordinator + one node on this machine, then opens"
  echo "the dashboard in your browser. Great for testing without a GPU cluster."
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Error: model file not found: $MODEL"
  exit 1
fi

# Build if needed
if [[ ! -f "./build/gpunet-coordinator" || ! -f "./build/gpunet-node" || ! -f "./build/gpunet-client" ]]; then
  echo "==> Binaries not found, building first..."
  ./scripts/build.sh "$@"
fi

COORD_HOST="127.0.0.1"
COORD_PORT=7700
API_PORT=7702
DASH_PORT=7780
NODE_DATA_PORT=7701

echo ""
echo "=== llama-distributed single-node mode ==="
echo "  Model:     $MODEL"
echo "  Dashboard: http://${COORD_HOST}:${DASH_PORT}"
echo "  API:       ${COORD_HOST}:${API_PORT}  (OpenAI-compatible)"
echo ""
echo "  Press Ctrl+C to stop all processes."
echo ""

# Start coordinator
./build/gpunet-coordinator \
  --model      "$MODEL" \
  --model-name "local" \
  --control-port "${COORD_PORT}" \
  --api-port     "${API_PORT}" \
  --dashboard-port "${DASH_PORT}" \
  --public-host  "${COORD_HOST}" \
  --min-nodes    1 &
COORD_PID=$!

# Give coordinator a moment to bind ports
sleep 1

# Start a single node
./build/gpunet-node \
  --server       "${COORD_HOST}" \
  --control-port "${COORD_PORT}" \
  --data-port    "${NODE_DATA_PORT}" \
  --n-gpu-layers 999 &
NODE_PID=$!

# Open dashboard in browser
sleep 1
DASH_URL="http://${COORD_HOST}:${DASH_PORT}"
echo "==> Dashboard: $DASH_URL"
if command -v xdg-open &>/dev/null;   then xdg-open "$DASH_URL" &>/dev/null &
elif command -v open &>/dev/null;     then open "$DASH_URL" &>/dev/null &
fi

echo ""
echo "To send a prompt:"
echo "  ./build/gpunet-client --server ${COORD_HOST} --model local --prompt \"Hello!\""
echo ""

cleanup(){
  echo ""
  echo "==> Stopping…"
  kill "$COORD_PID" "$NODE_PID" 2>/dev/null || true
  wait "$COORD_PID" "$NODE_PID" 2>/dev/null || true
  echo "==> Done."
}
trap cleanup EXIT INT TERM
wait
