#!/usr/bin/env bash
# scripts/build.sh — one-command build for llama-distributed
#
# Usage:
#   ./scripts/build.sh                  # CPU-only, release
#   ./scripts/build.sh --cuda           # CUDA
#   ./scripts/build.sh --metal          # Metal (macOS)
#   ./scripts/build.sh --vulkan         # Vulkan
#   ./scripts/build.sh --debug          # debug symbols
#   ./scripts/build.sh --jobs 4         # parallel jobs
#   ./scripts/build.sh --prefix /opt/ld # install prefix
#   ./scripts/build.sh --llama /path    # external llama.cpp source
#
# After a successful build, binaries are in ./build/bin/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ─────────────────────────────────────────────────────────────────
BUILD_TYPE="Release"
USE_CUDA=OFF
USE_METAL=OFF
USE_VULKAN=OFF
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
PREFIX="${ROOT_DIR}/install"
LLAMA_SOURCE="AUTO"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)         USE_CUDA=ON           ; shift ;;
        --metal)        USE_METAL=ON          ; shift ;;
        --vulkan)       USE_VULKAN=ON         ; shift ;;
        --debug)        BUILD_TYPE=Debug      ; shift ;;
        --jobs|-j)      JOBS="$2"             ; shift 2 ;;
        --prefix)       PREFIX="$2"           ; shift 2 ;;
        --llama)        LLAMA_SOURCE="$2"     ; shift 2 ;;
        -h|--help)
            head -20 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)  echo "Unknown flag: $1"; exit 1 ;;
    esac
done

BUILD_DIR="${ROOT_DIR}/build"

# ── Submodule init ────────────────────────────────────────────────────────────
if [[ "${LLAMA_SOURCE}" == "AUTO" ]]; then
    SUBMOD="${ROOT_DIR}/third_party/llama.cpp"
    if [[ ! -f "${SUBMOD}/CMakeLists.txt" ]]; then
        echo "⟳  Initialising llama.cpp submodule..."
        git -C "${ROOT_DIR}" submodule update --init --recursive third_party/llama.cpp
    fi
fi

# ── Configure ─────────────────────────────────────────────────────────────────
echo "⟳  Configuring (${BUILD_TYPE}, CUDA=${USE_CUDA}, Metal=${USE_METAL}, Vulkan=${USE_VULKAN})..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"       \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"       \
    -DDIST_USE_CUDA="${USE_CUDA}"            \
    -DDIST_USE_METAL="${USE_METAL}"          \
    -DDIST_USE_VULKAN="${USE_VULKAN}"        \
    -DDIST_LLAMA_SOURCE="${LLAMA_SOURCE}"    \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# ── Build ─────────────────────────────────────────────────────────────────────
echo "⟳  Building with ${JOBS} jobs..."
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "✓  Build complete. Binaries:"
for bin in dist-coordinator dist-node dist-client dist-vm-coordinator dist-vm-node; do
    path="${BUILD_DIR}/${bin}"
    [[ -f "$path" ]] && echo "     ${path}"
done
echo ""
echo "To install to ${PREFIX}:"
echo "  cmake --install ${BUILD_DIR}"
