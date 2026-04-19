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
while [ $# -gt 0 ]; do
  case "$1" in
    --pair)   pair="$2";   shift 2 ;;
    --token)  token="$2";  shift 2 ;;
    --server) server="$2"; shift 2 ;;
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
EOF
  chmod 0600 "$ETC_DIR/agent.env"
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
