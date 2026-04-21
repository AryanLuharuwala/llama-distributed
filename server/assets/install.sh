#!/usr/bin/env sh
# install.sh — one-shot installer, hosted by the dashboard.
#
# Usage:
#   curl -fsSL https://<your-dashboard>/install.sh | sh -s -- --pair <distpool://url>
#
# Or the legacy two-arg form (--token ... --server wss://...) which the
# dashboard still mints for older one-liners.
#
# Detects OS + arch, downloads the matching release tarball from GitHub,
# extracts it, and runs the platform installer.  Windows users run
# install.ps1 instead.

set -eu

GITHUB_REPO="${GITHUB_REPO:-AryanLuharuwala/llama-distributed}"
VERSION="${VERSION:-latest}"
PAIR=""
TOKEN=""
SERVER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --pair)    PAIR="$2"; shift 2 ;;
    --token)   TOKEN="$2"; shift 2 ;;
    --server)  SERVER="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --repo)    GITHUB_REPO="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PAIR" ] && [ -z "$TOKEN" ]; then
  echo "error: --pair <distpool://url>   (or legacy --token <t> [--server <ws-url>])" >&2
  exit 2
fi

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"

case "$os" in
  linux)  target_os="linux" ;;
  darwin) target_os="macos" ;;
  *) echo "unsupported OS: $os (use windows-install.ps1 on Windows)" >&2; exit 1 ;;
esac

case "$arch" in
  x86_64|amd64)   target_arch="x86_64" ;;
  arm64|aarch64)  target_arch="arm64"  ;;
  *) echo "unsupported arch: $arch" >&2; exit 1 ;;
esac

# Pick the right variant: default to CPU, upgrade if an accelerator is present.
variant="cpu"
if [ "$target_os" = "linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  variant="cuda"
fi
if [ "$target_os" = "macos" ] && [ "$target_arch" = "arm64" ]; then
  variant="metal"
fi

target="${target_os}-${target_arch}-${variant}"
echo "[install] detected target: $target"

# Resolve download URL.
if [ "$VERSION" = "latest" ]; then
  # /releases/latest skips prereleases — try it first, then fall back to the
  # newest release (including prereleases) so dev builds still install.
  tag="$(curl -fsSL "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" 2>/dev/null |
         sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n1)"
  if [ -z "$tag" ]; then
    tag="$(curl -fsSL "https://api.github.com/repos/${GITHUB_REPO}/releases?per_page=1" 2>/dev/null |
           sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n1)"
  fi
  [ -n "$tag" ] || { echo "could not resolve latest release" >&2; exit 1; }
  VERSION="$tag"
fi

asset="llama-distributed-${VERSION}-${target}.tar.gz"
url="https://github.com/${GITHUB_REPO}/releases/download/${VERSION}/${asset}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "[install] fetching $url"
curl -fSL "$url" -o "$tmp/$asset"

echo "[install] extracting"
tar -xzf "$tmp/$asset" -C "$tmp"
pkg="$tmp/llama-distributed-${VERSION}-${target}"

echo "[install] running platform installer"
if [ -n "$PAIR" ]; then
  sh "$pkg/scripts/install/${target_os}-install.sh" install --pair "$PAIR"
else
  args="install --token $TOKEN"
  [ -n "$SERVER" ] && args="$args --server $SERVER"
  sh "$pkg/scripts/install/${target_os}-install.sh" $args
fi

echo "[install] done."
