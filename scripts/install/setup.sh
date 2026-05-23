#!/usr/bin/env sh
# setup.sh — minimal one-line installer for the SURD CLI.
#
#   curl -fsSL https://<host>/setup.sh | sh
#
# This is the modern, lightweight installer.  Downloads the `surd` binary
# (Phase 1 CLI) and places it on PATH.  The user then runs:
#
#   surd login         # device-code flow, opens browser
#   surd connect       # bring the rig online
#
# Legacy installers (with --pair etc.) still live at /install.sh.

set -eu

# DIST_SERVER is injected by the server when serving this script — points
# back at the dashboard so the rig can fetch /releases/* without going to
# GitHub.  Empty when run from a static checkout.
DIST_SERVER="${DIST_SERVER:-}"
PREFIX="${PREFIX:-$HOME/.local/bin}"

# Forwarded args (pool, invite, …) come from the install-page configurator
# as query-string-style assignments.
POOL=""
INVITE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pool)   POOL="$2"; shift 2 ;;
    --invite) INVITE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --server) DIST_SERVER="$2"; shift 2 ;;
    *) echo "[setup] unknown arg: $1 (ignored)" >&2; shift ;;
  esac
done

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
case "$os"   in linux) tos="linux";; darwin) tos="darwin";; *) echo "unsupported OS: $os" >&2; exit 1;; esac
case "$arch" in x86_64|amd64) tarch="amd64";; arm64|aarch64) tarch="arm64";; *) echo "unsupported arch: $arch" >&2; exit 1;; esac

if [ -z "$DIST_SERVER" ]; then
  echo "[setup] DIST_SERVER not set — re-run via the dashboard's /setup.sh URL." >&2
  exit 2
fi

mkdir -p "$PREFIX"
asset="surd-${tos}-${tarch}"
echo "[setup] downloading ${asset} from ${DIST_SERVER}"
if ! curl -fSL "${DIST_SERVER%/}/releases/${asset}" -o "$PREFIX/surd.tmp"; then
  echo "[setup] download failed — does the server publish ${asset}?" >&2
  exit 1
fi
chmod +x "$PREFIX/surd.tmp"
mv -f "$PREFIX/surd.tmp" "$PREFIX/surd"

# Ensure PREFIX is on PATH.  We append a single line, idempotently.
if ! echo ":$PATH:" | grep -q ":$PREFIX:"; then
  for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
    [ -f "$rc" ] || continue
    if ! grep -q "$PREFIX" "$rc"; then
      printf '\n# added by surd setup.sh\nexport PATH="%s:$PATH"\n' "$PREFIX" >> "$rc"
    fi
  done
  echo "[setup] added $PREFIX to PATH in your shell rcs — open a new shell or:"
  echo "        export PATH=\"$PREFIX:\$PATH\""
fi

# Pre-seed SURD_SERVER so `surd login` doesn't need --server on first run.
echo "[setup] SURD_SERVER=$DIST_SERVER"
export SURD_SERVER="$DIST_SERVER"

echo
echo "  ✓ installed: $PREFIX/surd"
echo
echo "  next:"
echo "    surd login"
echo "    surd connect${POOL:+ --pool $POOL}${INVITE:+ --invite $INVITE}"
echo
