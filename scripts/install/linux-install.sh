#!/usr/bin/env bash
# linux-install.sh — installs the llama-distributed agent binaries into
# /usr/local/bin and registers a user systemd service that auto-starts on
# login and re-pairs with the coordinator.
#
# Usage:
#   sudo ./linux-install.sh install --token <pool-token> [--server host:port]
#   sudo ./linux-install.sh uninstall
#
# The token is a short-lived pool-join token generated from the website.
# When missing, the agent is installed but not auto-started.

set -euo pipefail

PREFIX="${PREFIX:-/usr/local}"
BIN_DIR="$PREFIX/bin"
ETC_DIR="/etc/llama-distributed"
SERVICE_USER_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="llama-distributed-agent.service"
SERVICE_FILE="$SERVICE_USER_DIR/$SERVICE_NAME"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"   # the extracted tarball root

cmd="${1:-}"; shift || true

pair=""
# Legacy flags (kept so older bootstrappers still work): if --token and
# --server are supplied we synthesise a distpool://pair URL from them.
token=""
server=""
with_comfyui=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pair)         pair="$2";   shift 2 ;;
    --token)        token="$2";  shift 2 ;;
    --server)       server="$2"; shift 2 ;;
    --with-comfyui) with_comfyui=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

if [ -z "$pair" ] && [ -n "$token" ]; then
  srv="${server:-wss://pool.llamadist.dev/ws/agent}"
  pair="distpool://pair?token=${token}&server=${srv}"
fi

install_bins() {
  echo "[install] copying binaries into $BIN_DIR"
  install -d "$BIN_DIR"
  for b in dist-node dist-join dist-client dist-coordinator; do
    if [ -x "$PKG_DIR/bin/$b" ]; then
      install -m 0755 "$PKG_DIR/bin/$b" "$BIN_DIR/$b"
    fi
  done
}

write_config() {
  install -d "$ETC_DIR"
  cat > "$ETC_DIR/agent.env" <<EOF
# llama-distributed agent configuration — edit and restart the service.
# DIST_PAIR is a distpool://pair?token=…&server=ws(s)://… URL minted by
# the dashboard.  The agent consumes it once on start, re-registers using
# the persistent WebSocket, and stays running.
DIST_PAIR=$pair
DIST_GPU_LAYERS=999
${with_comfyui:+DIST_WITH_COMFYUI=1}
${with_comfyui:+DIST_COMFY_URL=http://127.0.0.1:8188}
EOF
  chmod 0600 "$ETC_DIR/agent.env"
}

install_comfyui() {
  [ -z "$with_comfyui" ] && return 0
  comfy_dir="$HOME/.local/share/llama-distributed/ComfyUI"
  if [ -d "$comfy_dir/.git" ]; then
    echo "[install] ComfyUI already cloned at $comfy_dir — pulling latest"
    git -C "$comfy_dir" pull --ff-only || true
  else
    echo "[install] cloning ComfyUI into $comfy_dir"
    install -d "$(dirname "$comfy_dir")"
    git clone --depth=1 https://github.com/comfyanonymous/ComfyUI "$comfy_dir"
  fi
  if [ ! -d "$comfy_dir/.venv" ]; then
    echo "[install] creating ComfyUI venv"
    python3 -m venv "$comfy_dir/.venv"
  fi
  # shellcheck disable=SC1091
  . "$comfy_dir/.venv/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet -U -r "$comfy_dir/requirements.txt"
  deactivate
  # Optional companion systemd user unit — only enabled if user opts in
  # later, but write the file so it's a single command away.
  comfy_unit="$SERVICE_USER_DIR/llama-distributed-comfyui.service"
  cat > "$comfy_unit" <<EOF
[Unit]
Description=ComfyUI for llama-distributed agent
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$comfy_dir
ExecStart=$comfy_dir/.venv/bin/python $comfy_dir/main.py --listen 127.0.0.1 --port 8188
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload 2>/dev/null || true
  echo "[install] ComfyUI ready at $comfy_dir"
  echo "          start it with:  systemctl --user enable --now llama-distributed-comfyui.service"
}

write_service() {
  install -d "$SERVICE_USER_DIR"
  cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=llama-distributed pool agent
After=network-online.target

[Service]
Type=simple
EnvironmentFile=$ETC_DIR/agent.env
ExecStart=$BIN_DIR/dist-node --pair \${DIST_PAIR} -g \${DIST_GPU_LAYERS}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
}

start_service() {
  if [ -z "$token" ]; then
    echo "[install] no --token supplied — service installed but not enabled"
    echo "          edit $ETC_DIR/agent.env and run:"
    echo "            systemctl --user enable --now $SERVICE_NAME"
    return
  fi
  systemctl --user enable --now "$SERVICE_NAME"
  echo "[install] agent started — view logs with:"
  echo "            journalctl --user -u $SERVICE_NAME -f"
}

case "$cmd" in
  install)
    install_bins
    write_config
    write_service
    install_comfyui
    start_service
    ;;
  uninstall)
    systemctl --user disable --now "$SERVICE_NAME" 2>/dev/null || true
    rm -f "$SERVICE_FILE"
    systemctl --user daemon-reload 2>/dev/null || true
    for b in dist-node dist-join dist-client dist-coordinator; do
      rm -f "$BIN_DIR/$b"
    done
    rm -rf "$ETC_DIR"
    echo "[uninstall] done"
    ;;
  *)
    echo "usage: $0 install|uninstall [--token <t>] [--server <host:port>]"
    exit 2
    ;;
esac
