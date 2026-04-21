#!/usr/bin/env bash
# build.sh — source-build installer for llama-distributed agents.
#
# Used when no prebuilt tarball matches the host (unusual accelerator, CPU
# arch, or a pinned server commit that's newer than the latest release).
# Detects the toolchain, prompts to install missing deps via the distro
# package manager, clones the repo at a pinned SHA, builds `dist-node`,
# and hands off to the per-OS installer to register the service and pair.
#
# Usage:
#   curl -fsSL http://<dashboard>/build.sh | bash -s -- \
#     --pair 'distpool://pair?token=…&server=ws://…' \
#     --accel cuda \
#     --ref   <git-sha>
#
# Flags:
#   --pair   <url>    distpool:// deep link (required)
#   --accel  <name>   cpu | cuda | rocm | metal | vulkan  (default: auto)
#   --ref    <sha>    commit/tag/branch to build (default: current default branch)
#   --repo   <slug>   owner/name (default: AryanLuharuwala/llama-distributed)
#   --yes             accept dep-install prompts non-interactively
#   --prefix <dir>    install prefix for `make install` (default: ~/.local)

set -euo pipefail

PAIR=""
ACCEL="${ACCEL:-}"
# REF is the git commit to build.  The server injects its own SHA here when
# serving build.sh so `curl … | bash` always builds the exact revision the
# control plane runs.  Overridden by --ref on the command line.
REF="${REF:-}"
REPO="${REPO:-AryanLuharuwala/llama-distributed}"
YES="${YES:-0}"
PREFIX="${PREFIX:-$HOME/.local}"

while [ $# -gt 0 ]; do
  case "$1" in
    --pair)   PAIR="$2";   shift 2 ;;
    --accel)  ACCEL="$2";  shift 2 ;;
    --ref)    REF="$2";    shift 2 ;;
    --repo)   REPO="$2";   shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --yes|-y) YES=1;       shift   ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "$PAIR" ] || { echo "error: --pair <distpool://…> is required" >&2; exit 2; }

# ─── Platform + accelerator detection ──────────────────────────────────────

os_raw="$(uname -s)"
case "$os_raw" in
  Linux)  OS="linux" ;;
  Darwin) OS="macos" ;;
  *) echo "build.sh: unsupported OS $os_raw (use build.ps1 on Windows)" >&2; exit 1 ;;
esac

arch_raw="$(uname -m)"
case "$arch_raw" in
  x86_64|amd64)   ARCH="x86_64" ;;
  arm64|aarch64)  ARCH="arm64" ;;
  *) echo "build.sh: unsupported arch $arch_raw" >&2; exit 1 ;;
esac

detect_accel() {
  if [ "$OS" = "macos" ] && [ "$ARCH" = "arm64" ]; then echo "metal"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1; then echo "cuda"; return; fi
  if command -v rocminfo   >/dev/null 2>&1; then echo "rocm"; return; fi
  if command -v vulkaninfo >/dev/null 2>&1; then echo "vulkan"; return; fi
  echo "cpu"
}

[ -n "$ACCEL" ] || ACCEL="$(detect_accel)"

case "$ACCEL" in cpu|cuda|rocm|metal|vulkan) ;;
  *) echo "build.sh: unknown --accel '$ACCEL' (cpu|cuda|rocm|metal|vulkan)" >&2; exit 2 ;;
esac

if [ "$OS" = "macos" ] && [ "$ACCEL" != "metal" ] && [ "$ACCEL" != "cpu" ]; then
  echo "build.sh: on macOS only --accel metal or cpu are supported" >&2; exit 2
fi
if [ "$OS" = "linux" ] && [ "$ACCEL" = "metal" ]; then
  echo "build.sh: metal is macOS-only" >&2; exit 2
fi

echo "[build] platform:    $OS/$ARCH"
echo "[build] accelerator: $ACCEL"
echo "[build] repo:        $REPO"
[ -n "$REF" ] && echo "[build] ref:         $REF"

# ─── Toolchain detection ───────────────────────────────────────────────────

need_tools=()
has() { command -v "$1" >/dev/null 2>&1; }

require_tool() { has "$1" || need_tools+=("$1"); }

# Always required.
require_tool git
require_tool cmake
# Either ninja or make is fine; prefer ninja for speed.
has ninja || has make || need_tools+=("ninja")
# A C++ toolchain.
if [ "$OS" = "macos" ]; then
  has clang++ || need_tools+=("xcode-clt")
else
  has g++ || has clang++ || need_tools+=("gcc-c++")
fi
case "$ACCEL" in
  cuda)
    has nvcc || need_tools+=("cuda-toolkit")
    ;;
  rocm)
    has hipcc || need_tools+=("rocm")
    ;;
  vulkan)
    has glslc      || need_tools+=("glslc")
    has vulkaninfo || need_tools+=("vulkan-sdk")
    ;;
esac

# ─── Package-manager mapping ───────────────────────────────────────────────

PM=""
PM_INSTALL=""
if   has dnf;     then PM="dnf";     PM_INSTALL="sudo dnf install -y"
elif has apt-get; then PM="apt";     PM_INSTALL="sudo apt-get install -y"
elif has pacman;  then PM="pacman";  PM_INSTALL="sudo pacman -S --noconfirm"
elif has zypper;  then PM="zypper";  PM_INSTALL="sudo zypper install -y"
elif has apk;     then PM="apk";     PM_INSTALL="sudo apk add"
elif has brew;    then PM="brew";    PM_INSTALL="brew install"
fi

# Maps our generic tool names to per-package-manager names.
pkgs_for() {
  local tool="$1"
  case "$PM:$tool" in
    dnf:git|apt:git|pacman:git|zypper:git|apk:git|brew:git) echo "git" ;;
    dnf:cmake|apt:cmake|pacman:cmake|zypper:cmake|apk:cmake|brew:cmake) echo "cmake" ;;
    dnf:ninja)   echo "ninja-build" ;;
    apt:ninja)   echo "ninja-build" ;;
    pacman:ninja|zypper:ninja|apk:ninja|brew:ninja) echo "ninja" ;;
    dnf:gcc-c++) echo "gcc-c++" ;;
    apt:gcc-c++) echo "g++ build-essential" ;;
    pacman:gcc-c++|zypper:gcc-c++) echo "gcc" ;;
    apk:gcc-c++) echo "g++" ;;
    brew:xcode-clt) echo "" ;;  # handled separately via xcode-select
    dnf:cuda-toolkit|apt:cuda-toolkit) echo "cuda" ;;
    pacman:cuda-toolkit) echo "cuda" ;;
    zypper:cuda-toolkit) echo "cuda" ;;
    dnf:rocm)    echo "rocm-hip-devel" ;;
    apt:rocm)    echo "rocm-hip-sdk" ;;
    pacman:rocm) echo "rocm-hip-sdk" ;;
    dnf:glslc|apt:glslc|pacman:glslc|zypper:glslc|apk:glslc|brew:glslc)
      # glslc is shipped with the Vulkan SDK / glslang on most distros.
      echo "" ;;
    dnf:vulkan-sdk)   echo "vulkan-headers vulkan-loader-devel glslang" ;;
    apt:vulkan-sdk)   echo "libvulkan-dev vulkan-tools glslang-tools" ;;
    pacman:vulkan-sdk) echo "vulkan-devel shaderc" ;;
    zypper:vulkan-sdk) echo "vulkan-devel shaderc" ;;
    apk:vulkan-sdk)   echo "vulkan-headers vulkan-loader-dev glslang-dev" ;;
    brew:vulkan-sdk)  echo "molten-vk vulkan-headers shaderc" ;;
    *) echo "" ;;
  esac
}

# ─── Dep-prompt flow ───────────────────────────────────────────────────────

if [ "${#need_tools[@]}" -gt 0 ]; then
  echo
  echo "[build] the following tools are missing:"
  printf '         - %s\n' "${need_tools[@]}"
  echo

  if [ "$OS" = "macos" ]; then
    need_xclt=0
    for t in "${need_tools[@]}"; do [ "$t" = "xcode-clt" ] && need_xclt=1; done
    if [ "$need_xclt" = "1" ]; then
      echo "[build] Xcode Command Line Tools are required."
      echo "        Install them with:  xcode-select --install"
      echo "        Then re-run this script."
      exit 1
    fi
    if ! has brew; then
      echo "[build] Homebrew not found. Install from https://brew.sh then re-run."
      exit 1
    fi
  fi

  if [ -z "$PM" ]; then
    echo "[build] No supported package manager detected. Install the tools"
    echo "        above manually and re-run with --accel $ACCEL."
    exit 1
  fi

  pkgs=()
  unresolved=()
  for t in "${need_tools[@]}"; do
    p="$(pkgs_for "$t")"
    if [ -n "$p" ]; then
      for one in $p; do pkgs+=("$one"); done
    else
      unresolved+=("$t")
    fi
  done

  if [ "${#unresolved[@]}" -gt 0 ]; then
    echo "[build] Don't know the $PM package name for:"
    printf '         - %s\n' "${unresolved[@]}"
    echo "        Please install these manually, then re-run."
    [ "${#pkgs[@]}" -eq 0 ] && exit 1
  fi

  if [ "${#pkgs[@]}" -gt 0 ]; then
    cmd="$PM_INSTALL ${pkgs[*]}"
    echo "[build] Proposed install command:"
    echo "         $cmd"
    echo
    if [ "$YES" = "1" ]; then
      echo "[build] --yes given; running it."
    else
      printf "[build] Run it now? [y/N] "
      read -r reply </dev/tty || reply=""
      case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "[build] aborted; install deps yourself and re-run."; exit 1 ;;
      esac
    fi
    eval "$cmd"
  fi
fi

# ─── Sanity re-check after install ─────────────────────────────────────────

missing_after=()
has git   || missing_after+=("git")
has cmake || missing_after+=("cmake")
has ninja || has make || missing_after+=("ninja/make")
if [ "$OS" = "macos" ]; then
  has clang++ || missing_after+=("clang++")
else
  has g++ || has clang++ || missing_after+=("g++")
fi
if [ "${#missing_after[@]}" -gt 0 ]; then
  echo "[build] still missing after install: ${missing_after[*]}" >&2
  exit 1
fi

echo "[build] toolchain ok."

# ─── Clone or update source tree ───────────────────────────────────────────

SRC_ROOT="${SRC_ROOT:-$HOME/.local/share/llama-distributed/src}"
mkdir -p "$(dirname "$SRC_ROOT")"

if [ ! -d "$SRC_ROOT/.git" ]; then
  echo "[build] cloning https://github.com/$REPO into $SRC_ROOT"
  git clone --recurse-submodules "https://github.com/$REPO.git" "$SRC_ROOT"
else
  echo "[build] updating existing clone at $SRC_ROOT"
  git -C "$SRC_ROOT" fetch --tags origin
fi

if [ -n "$REF" ]; then
  echo "[build] checking out $REF"
  git -C "$SRC_ROOT" checkout --detach "$REF"
  git -C "$SRC_ROOT" submodule update --init --recursive
fi

# ─── Configure ─────────────────────────────────────────────────────────────

BUILD_DIR="$SRC_ROOT/build"
CMAKE_ACCEL_FLAGS=()
case "$ACCEL" in
  cpu)    ;;  # defaults are fine
  cuda)   CMAKE_ACCEL_FLAGS+=("-DDIST_USE_CUDA=ON") ;;
  rocm)   CMAKE_ACCEL_FLAGS+=("-DDIST_USE_HIP=ON") ;;  # ROCm → HIP backend in ggml
  metal)  CMAKE_ACCEL_FLAGS+=("-DDIST_USE_METAL=ON") ;;
  vulkan) CMAKE_ACCEL_FLAGS+=("-DDIST_USE_VULKAN=ON") ;;
esac

echo "[build] configuring ($ACCEL)"
cmake -S "$SRC_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDIST_BUILD_TESTS=OFF \
  "${CMAKE_ACCEL_FLAGS[@]}"

# ─── Compile ───────────────────────────────────────────────────────────────

# Parallelism: cap to RAM/2GiB to avoid OOM on small boxes.  nvcc is
# especially hungry; clamp harder when building CUDA.
nproc_hw="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 4194304)"
max_by_mem=$(( mem_kb / (2 * 1024 * 1024) ))
[ "$max_by_mem" -lt 1 ] && max_by_mem=1
jobs="$nproc_hw"
[ "$jobs" -gt "$max_by_mem" ] && jobs="$max_by_mem"
[ "$ACCEL" = "cuda" ] && [ "$jobs" -gt 2 ] && jobs=2

echo "[build] compiling (target dist-node, -j$jobs)"
cmake --build "$BUILD_DIR" --target dist-node --parallel "$jobs"

# ─── Stage pkg dir matching the release tarball layout ─────────────────────

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PKG_DIR="$STAGE/llama-distributed-src-$ACCEL"
mkdir -p "$PKG_DIR/bin" "$PKG_DIR/scripts/install"

for b in dist-node dist-coordinator dist-client dist-join; do
  if [ -x "$BUILD_DIR/$b" ]; then
    install -m 0755 "$BUILD_DIR/$b" "$PKG_DIR/bin/$b"
  fi
done

# Use the per-OS installer from the freshly-cloned tree so service-unit
# changes ship with the build.
cp -r "$SRC_ROOT/scripts/install/." "$PKG_DIR/scripts/install/"

# ─── Hand off to per-OS installer ──────────────────────────────────────────

echo "[build] registering service and pairing"
sh "$PKG_DIR/scripts/install/${OS}-install.sh" install --pair "$PAIR"

echo "[build] done."
