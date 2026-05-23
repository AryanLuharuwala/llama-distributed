#!/usr/bin/env zsh
# setup.zsh — zsh-flavoured installer for the SURD CLI.
#
#   curl -fsSL https://<host>/setup.zsh | zsh
#
# Functionally identical to setup.sh but runs natively under zsh, writes
# its PATH snippet only to ~/.zshrc, and tolerates zsh's stricter parsing.
# Use setup.sh for bash/sh, setup.ps1 for Windows.

emulate -L zsh
setopt err_exit no_unset pipe_fail

DIST_SERVER="${DIST_SERVER:-}"
PREFIX="${PREFIX:-$HOME/.local/bin}"
POOL=""
INVITE=""

while (( $# > 0 )); do
  case "$1" in
    --pool)   POOL="$2"; shift 2 ;;
    --invite) INVITE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --server) DIST_SERVER="$2"; shift 2 ;;
    *) print -ru2 "[setup] unknown arg: $1 (ignored)"; shift ;;
  esac
done

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
case "$os" in
  linux)  tos="linux" ;;
  darwin) tos="darwin" ;;
  *) print -ru2 "unsupported OS: $os"; exit 1 ;;
esac
case "$arch" in
  x86_64|amd64)   tarch="amd64" ;;
  arm64|aarch64)  tarch="arm64" ;;
  *) print -ru2 "unsupported arch: $arch"; exit 1 ;;
esac

if [[ -z "$DIST_SERVER" ]]; then
  print -ru2 "[setup] DIST_SERVER not set — re-run via the dashboard's /setup.zsh URL."
  exit 2
fi

mkdir -p "$PREFIX"
asset="surd-${tos}-${tarch}"
print -r -- "[setup] downloading ${asset} from ${DIST_SERVER}"
if ! curl -fSL "${DIST_SERVER%/}/releases/${asset}" -o "$PREFIX/surd.tmp"; then
  print -ru2 "[setup] download failed — does the server publish ${asset}?"
  exit 1
fi
chmod +x "$PREFIX/surd.tmp"
mv -f "$PREFIX/surd.tmp" "$PREFIX/surd"

# Ensure PREFIX is on PATH via ~/.zshrc, idempotently.
if [[ ":$PATH:" != *":$PREFIX:"* ]]; then
  rc="$HOME/.zshrc"
  if [[ ! -f "$rc" || -z "$(grep -F "$PREFIX" "$rc" 2>/dev/null || true)" ]]; then
    print -r -- "" >> "$rc"
    print -r -- "# added by surd setup.zsh" >> "$rc"
    print -r -- "export PATH=\"$PREFIX:\$PATH\"" >> "$rc"
  fi
  print -r -- "[setup] added $PREFIX to PATH in ~/.zshrc — open a new shell, or:"
  print -r -- "        export PATH=\"$PREFIX:\$PATH\""
fi

# Pre-seed SURD_SERVER.
print -r -- "[setup] SURD_SERVER=$DIST_SERVER"
export SURD_SERVER="$DIST_SERVER"

print -r -- ""
print -r -- "  ✓ installed: $PREFIX/surd"
print -r -- ""
print -r -- "  next:"
print -r -- "    surd login"
next_line="    surd connect"
[[ -n "$POOL"   ]] && next_line+=" --pool $POOL"
[[ -n "$INVITE" ]] && next_line+=" --invite $INVITE"
print -r -- "$next_line"
print -r -- ""
