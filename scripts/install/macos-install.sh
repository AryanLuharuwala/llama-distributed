#!/usr/bin/env bash
# macos-install.sh — installs the agent under /usr/local/bin and registers a
# per-user LaunchAgent so it starts on login.
#
# Usage:
#   ./macos-install.sh install --token <pool-token> [--server host:port]
#   ./macos-install.sh uninstall

set -euo pipefail

PREFIX="${PREFIX:-/usr/local}"
BIN_DIR="$PREFIX/bin"
LABEL="dev.llamadist.agent"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/llama-distributed"
ETC_DIR="$HOME/.config/llama-distributed"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cmd="${1:-}"; shift || true
pair=""; token=""; server=""
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
  echo "[install] copying binaries into $BIN_DIR (may prompt for sudo)"
  sudo install -d "$BIN_DIR"
  for b in dist-node dist-join dist-client dist-coordinator; do
    if [ -x "$PKG_DIR/bin/$b" ]; then
      sudo install -m 0755 "$PKG_DIR/bin/$b" "$BIN_DIR/$b"
    fi
  done
}

write_config() {
  mkdir -p "$ETC_DIR" "$LOG_DIR"
  cat > "$ETC_DIR/agent.env" <<EOF
# llama-distributed agent configuration — edit and reload the LaunchAgent.
# DIST_PAIR is a distpool://pair?token=…&server=ws(s)://… URL minted by
# the dashboard.  The agent consumes it once on start and stays running.
DIST_PAIR=$pair
DIST_GPU_LAYERS=999
EOF
  chmod 0600 "$ETC_DIR/agent.env"
}

write_plist() {
  mkdir -p "$(dirname "$PLIST")"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string><string>-c</string>
    <string>. $ETC_DIR/agent.env; exec $BIN_DIR/dist-node --pair "\$DIST_PAIR" -g \$DIST_GPU_LAYERS</string>
  </array>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>$LOG_DIR/agent.log</string>
  <key>StandardErrorPath</key> <string>$LOG_DIR/agent.err</string>
</dict>
</plist>
EOF
}

start_service() {
  if [ -z "$pair" ]; then
    echo "[install] no --pair URL — service installed but not loaded"
    echo "         edit $ETC_DIR/agent.env then run:"
    echo "           launchctl load $PLIST"
    return
  fi
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  echo "[install] agent started — logs at $LOG_DIR/"
}

case "$cmd" in
  install)
    install_bins
    write_config
    write_plist
    start_service
    ;;
  uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    for b in dist-node dist-join dist-client dist-coordinator; do
      sudo rm -f "$BIN_DIR/$b"
    done
    rm -rf "$ETC_DIR"
    echo "[uninstall] done"
    ;;
  *)
    echo "usage: $0 install|uninstall [--pair <url>] [--token <t>] [--server <ws-url>]"
    exit 2
    ;;
esac
