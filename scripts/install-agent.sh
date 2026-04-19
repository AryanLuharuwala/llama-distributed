#!/usr/bin/env bash
# install-agent.sh
#
# Registers the dist-node binary as the handler for distpool:// URLs on Linux.
# No sudo required — everything goes into the user's ~/.local tree.
#
# Usage:
#   ./install-agent.sh [path/to/dist-node]
#
#   If the path argument is omitted, the script uses the dist-node it finds
#   next to itself (build-tree layout) or on $PATH.

set -euo pipefail

BIN_PATH="${1:-}"
if [[ -z "$BIN_PATH" ]]; then
    # Try sibling directory (if we're under scripts/ in a source tree)
    for candidate in \
        "$(dirname "$0")/../build/dist-node" \
        "$(dirname "$0")/../build/bin/dist-node" \
        "$(command -v dist-node || true)"; do
        if [[ -x "$candidate" ]]; then
            BIN_PATH="$(readlink -f "$candidate")"
            break
        fi
    done
fi

if [[ -z "$BIN_PATH" || ! -x "$BIN_PATH" ]]; then
    echo "error: could not locate dist-node binary." >&2
    echo "       pass the path as the first argument." >&2
    exit 1
fi

echo "→ using binary: $BIN_PATH"

PREFIX="$HOME/.local"
APP_DIR="$PREFIX/share/applications"
BIN_DIR="$PREFIX/bin"

mkdir -p "$APP_DIR" "$BIN_DIR"

# Copy (or re-link) the binary to ~/.local/bin/dist-node so it is on PATH.
if [[ "$BIN_PATH" != "$BIN_DIR/dist-node" ]]; then
    ln -sf "$BIN_PATH" "$BIN_DIR/dist-node"
    echo "→ symlinked $BIN_DIR/dist-node -> $BIN_PATH"
fi

# A tiny launcher script that parses the distpool:// URL and re-invokes
# dist-node with the right flags.  xdg-open passes the URL as the last
# argument after %u expansion.
LAUNCH="$BIN_DIR/distpool-handler"
cat > "$LAUNCH" <<'EOF'
#!/usr/bin/env bash
# Wrapper that converts a distpool:// URL into dist-node invocation.
set -euo pipefail
URL="${1:-}"
if [[ -z "$URL" ]]; then
    echo "distpool-handler: no URL given" >&2
    exit 1
fi
# Log for debugging
LOG="$HOME/.local/share/distpool/handler.log"
mkdir -p "$(dirname "$LOG")"
echo "[$(date -Is)] invoked with: $URL" >> "$LOG"

# Hand off.  dist-node parses the URL itself (see --pair flag).
exec dist-node --pair "$URL" >> "$LOG" 2>&1
EOF
chmod +x "$LAUNCH"
echo "→ wrote launcher $LAUNCH"

# .desktop entry that advertises the distpool/x-scheme-handler.
DESKTOP="$APP_DIR/distpool-handler.desktop"
cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=distpool agent
Exec=$LAUNCH %u
StartupNotify=false
MimeType=x-scheme-handler/distpool;
Terminal=false
NoDisplay=true
Categories=Network;
EOF
echo "→ wrote $DESKTOP"

# Tell the desktop environment to re-scan.
update-desktop-database "$APP_DIR" 2>/dev/null || true
xdg-mime default distpool-handler.desktop x-scheme-handler/distpool

# Verify.
handler="$(xdg-mime query default x-scheme-handler/distpool 2>/dev/null || echo '?')"
echo
echo "done."
echo "  distpool:// handler: $handler"
echo
echo "Make sure \$HOME/.local/bin is on your PATH, then click the"
echo "\"Pair this machine\" button in the web UI — your browser will"
echo "launch dist-node automatically."
