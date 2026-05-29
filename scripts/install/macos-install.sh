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
COMFY_LABEL="dev.llamadist.comfyui"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
COMFY_PLIST="$HOME/Library/LaunchAgents/$COMFY_LABEL.plist"
LOG_DIR="$HOME/Library/Logs/llama-distributed"
ETC_DIR="$HOME/.config/llama-distributed"
COMFY_DIR="$HOME/Library/Application Support/llama-distributed/ComfyUI"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cmd="${1:-}"; shift || true
pair=""; token=""; server=""; with_comfyui=""
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
  echo "[install] copying binaries into $BIN_DIR (may prompt for sudo)"
  sudo install -d "$BIN_DIR"
  for b in gpunet-node gpunet-join gpunet-client gpunet-coordinator; do
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
${with_comfyui:+DIST_WITH_COMFYUI=1}
${with_comfyui:+DIST_COMFY_URL=http://127.0.0.1:8188}
EOF
  chmod 0600 "$ETC_DIR/agent.env"
}

install_comfyui() {
  [ -z "$with_comfyui" ] && return 0
  if ! command -v git >/dev/null 2>&1; then
    echo "[install] git not found — install Xcode CLT (xcode-select --install) then re-run with --with-comfyui"
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[install] python3 not found — install Python 3.10+ then re-run with --with-comfyui"
    return 0
  fi
  mkdir -p "$(dirname "$COMFY_DIR")"
  if [ -d "$COMFY_DIR/.git" ]; then
    echo "[install] ComfyUI already cloned — pulling latest"
    git -C "$COMFY_DIR" pull --ff-only || true
  else
    echo "[install] cloning ComfyUI into $COMFY_DIR"
    git clone --depth=1 https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
  fi
  if [ ! -d "$COMFY_DIR/.venv" ]; then
    python3 -m venv "$COMFY_DIR/.venv"
  fi
  # shellcheck disable=SC1091
  . "$COMFY_DIR/.venv/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet -U -r "$COMFY_DIR/requirements.txt"
  deactivate

  cat > "$COMFY_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>$COMFY_LABEL</string>
  <key>WorkingDirectory</key>  <string>$COMFY_DIR</string>
  <key>ProgramArguments</key>
  <array>
    <string>$COMFY_DIR/.venv/bin/python</string>
    <string>$COMFY_DIR/main.py</string>
    <string>--listen</string><string>127.0.0.1</string>
    <string>--port</string><string>8188</string>
  </array>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>$LOG_DIR/comfyui.log</string>
  <key>StandardErrorPath</key> <string>$LOG_DIR/comfyui.err</string>
</dict>
</plist>
EOF
  launchctl unload "$COMFY_PLIST" 2>/dev/null || true
  launchctl load "$COMFY_PLIST"
  echo "[install] ComfyUI ready at $COMFY_DIR — logs in $LOG_DIR/comfyui.log"
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
    <string>. $ETC_DIR/agent.env; exec $BIN_DIR/gpunet-node --pair "\$DIST_PAIR" -g \$DIST_GPU_LAYERS</string>
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
    install_comfyui
    start_service
    ;;
  uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl unload "$COMFY_PLIST" 2>/dev/null || true
    rm -f "$PLIST" "$COMFY_PLIST"
    for b in gpunet-node gpunet-join gpunet-client gpunet-coordinator; do
      sudo rm -f "$BIN_DIR/$b"
    done
    rm -rf "$ETC_DIR"
    echo "[uninstall] done (ComfyUI dir at $COMFY_DIR left in place)"
    ;;
  *)
    echo "usage: $0 install|uninstall [--pair <url>] [--token <t>] [--server <ws-url>] [--with-comfyui]"
    exit 2
    ;;
esac
