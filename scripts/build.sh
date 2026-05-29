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
#   ./scripts/build.sh --with-comfyui   # also init third_party/ComfyUI + venv
#   ./scripts/build.sh --gpunet-turn-only        # build only the Go relay binary
#   ./scripts/build.sh --static-gpunet-turn      # CGO_ENABLED=0 portable gpunet-turn
#   ./scripts/build.sh --static-gpunet-turn-all  # cross-compile linux/{amd64,arm64,arm},
#                                              # darwin/{amd64,arm64}, windows/amd64
#                                              # into build/gpunet-turn-dist/
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
WITH_COMFYUI="${DIST_WITH_COMFYUI:-OFF}"
DIST_TURN_ONLY=OFF
STATIC_DIST_TURN=OFF
STATIC_DIST_TURN_ALL=OFF

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
        --with-comfyui) WITH_COMFYUI=ON       ; shift ;;
        --gpunet-turn-only)       DIST_TURN_ONLY=ON       ; shift ;;
        --static-gpunet-turn)     STATIC_DIST_TURN=ON     ; shift ;;
        --static-gpunet-turn-all) STATIC_DIST_TURN_ALL=ON ; shift ;;
        -h|--help)
            head -20 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)  echo "Unknown flag: $1"; exit 1 ;;
    esac
done

BUILD_DIR="${ROOT_DIR}/build"
DIST_TURN_SRC="${ROOT_DIR}/cmd/gpunet-turn"

# ── gpunet-turn build helpers ──────────────────────────────────────────────────
# Single source of truth so the host build, --gpunet-turn-only, --static-gpunet-turn,
# and --static-gpunet-turn-all all share flags + output naming.

build_dist_turn_host() {
    # Native build using the host's default Go toolchain.  Dynamic-glibc on
    # Linux; works as the sidecar that dist-node spawns.
    local out="${BUILD_DIR}/gpunet-turn"
    echo "⟳  Building gpunet-turn (Go, native, dynamic)..."
    if ( cd "${DIST_TURN_SRC}" && go build -o "${out}" . ); then
        echo "✓  gpunet-turn → ${out}"
        return 0
    else
        echo "⚠  gpunet-turn build failed"
        return 1
    fi
}

build_dist_turn_static() {
    # CGO_ENABLED=0 → fully static pure-Go binary suitable for Alpine/musl,
    # scratch containers, and "drop on a server and run" provisioning.  Adds
    # -s -w to strip the symbol table + DWARF (~30% smaller).  Uses
    # netgo+osusergo build tags so the binary never needs nsswitch/libc.
    local goos="${1:-}"     # empty = host
    local goarch="${2:-}"
    local outdir="${3:-${BUILD_DIR}}"
    local suffix=""
    [[ -n "${goos}" || -n "${goarch}" ]] && suffix="-${goos:-host}-${goarch:-host}"
    [[ "${goos}" == "windows" ]] && suffix="${suffix}.exe"
    local out="${outdir}/gpunet-turn${suffix}"
    mkdir -p "${outdir}"
    echo "⟳  Building gpunet-turn (static, goos=${goos:-host} goarch=${goarch:-host})..."
    if ( cd "${DIST_TURN_SRC}" \
            && CGO_ENABLED=0 \
               GOOS="${goos}" GOARCH="${goarch}" \
               go build -trimpath \
                        -tags 'netgo osusergo' \
                        -ldflags '-s -w -extldflags "-static"' \
                        -o "${out}" . ); then
        local sz
        sz=$(du -h "${out}" 2>/dev/null | awk '{print $1}')
        echo "✓  gpunet-turn → ${out} (${sz})"
        return 0
    else
        echo "⚠  gpunet-turn static build failed (goos=${goos:-host} goarch=${goarch:-host})"
        return 1
    fi
}

build_dist_turn_all_targets() {
    # Cross-compile to the canonical set of OS/arch combos for distribution.
    # Pure-Go means no cross-toolchain setup required — Go's own runtime
    # handles every target.
    local outdir="${BUILD_DIR}/gpunet-turn-dist"
    rm -rf "${outdir}"
    mkdir -p "${outdir}"
    local targets=(
        "linux   amd64"
        "linux   arm64"
        "linux   arm"
        "darwin  amd64"
        "darwin  arm64"
        "windows amd64"
    )
    local failed=0
    for t in "${targets[@]}"; do
        # shellcheck disable=SC2086
        set -- $t
        build_dist_turn_static "$1" "$2" "${outdir}" || failed=$((failed + 1))
    done
    if [[ ${failed} -eq 0 ]]; then
        echo ""
        echo "✓  All targets built in ${outdir}:"
        ls -lah "${outdir}" | awk 'NR>1 {printf "     %-32s %s\n", $NF, $5}'
        # Generate SHA-256 sums for distribution.
        if command -v sha256sum >/dev/null 2>&1; then
            ( cd "${outdir}" && sha256sum gpunet-turn-* > SHA256SUMS )
            echo "✓  SHA256SUMS written to ${outdir}/SHA256SUMS"
        elif command -v shasum >/dev/null 2>&1; then
            ( cd "${outdir}" && shasum -a 256 gpunet-turn-* > SHA256SUMS )
            echo "✓  SHA256SUMS written to ${outdir}/SHA256SUMS"
        fi
    else
        echo "⚠  ${failed} target(s) failed"
        return 1
    fi
}

# ── --gpunet-turn-only / --static-gpunet-turn / --static-gpunet-turn-all ───────────
# Short-circuit paths.  Skip the cmake + C++ build entirely; useful for CI
# pipelines and release packaging.
if [[ "${DIST_TURN_ONLY}" == "ON" || "${STATIC_DIST_TURN}" == "ON" || "${STATIC_DIST_TURN_ALL}" == "ON" ]]; then
    if ! command -v go >/dev/null 2>&1; then
        echo "✗  go toolchain required for gpunet-turn build" >&2
        exit 1
    fi
    mkdir -p "${BUILD_DIR}"
    if [[ "${STATIC_DIST_TURN_ALL}" == "ON" ]]; then
        build_dist_turn_all_targets
    elif [[ "${STATIC_DIST_TURN}" == "ON" ]]; then
        build_dist_turn_static
    else
        build_dist_turn_host
    fi
    exit 0
fi

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

# ── Optional: ComfyUI submodule + venv ──────────────────────────────────────
#
# Off by default — the C++ agent ships with the comfy adapter compiled in and
# can talk to *any* ComfyUI install on localhost.  This block exists so a
# rig operator can stand up the whole stack with one command.
if [[ "${WITH_COMFYUI}" == "ON" ]]; then
    COMFY_DIR="${ROOT_DIR}/third_party/ComfyUI"
    if [[ ! -f "${COMFY_DIR}/main.py" ]]; then
        echo "⟳  Initialising ComfyUI submodule..."
        git -C "${ROOT_DIR}" submodule update --init --recursive third_party/ComfyUI
    fi
    if [[ ! -d "${COMFY_DIR}/.venv" ]]; then
        echo "⟳  Creating ComfyUI venv at ${COMFY_DIR}/.venv..."
        python3 -m venv "${COMFY_DIR}/.venv"
    fi
    # shellcheck disable=SC1091
    source "${COMFY_DIR}/.venv/bin/activate"
    # Pinned requirements — leave it to ComfyUI's own requirements.txt so we
    # don't drift.  --quiet keeps the build log readable; -U bumps things in
    # an existing venv when the submodule moves forward.
    pip install --quiet --upgrade pip
    pip install --quiet -U -r "${COMFY_DIR}/requirements.txt"
    deactivate
    echo "✓  ComfyUI ready at ${COMFY_DIR} (.venv populated)"
    echo "    Start it with:  python ${COMFY_DIR}/main.py --listen 127.0.0.1 --port 8188"
    echo "    The agent advertises comfy_caps automatically once ComfyUI is up."
fi

# ── gpunet-turn (Go) ───────────────────────────────────────────────────────────
# Dual-mode binary:
#   • sidecar  — spawned by dist-node when DIST_WITH_TURN=1 (uses --auth-secret
#                derived per-rig from the server's welcome frame).
#   • standalone — provision a relay-only node with `gpunet-turn --server URL
#                  --token PAIR_TOKEN`; it registers with the dist-server,
#                  auto-discovers its public IP via STUN, and receives traffic
#                  through pickPeerRelays just like a compute rig would.
# Failure here is non-fatal — relay-capable rigs fall back to the legacy
# peer:<agent_id> sentinel (rig sees plaintext) instead of true TURN.
if command -v go >/dev/null 2>&1 && [[ -d "${DIST_TURN_SRC}" ]]; then
    build_dist_turn_host || {
        echo "    peer-relay rigs will fall back to peer:<agent_id> (plaintext)."
    }
else
    echo "⚠  skipping gpunet-turn (no 'go' on PATH or cmd/gpunet-turn missing)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "✓  Build complete. Binaries:"
for bin in dist-coordinator dist-node dist-client dist-vm-coordinator dist-vm-node gpunet-turn; do
    path="${BUILD_DIR}/${bin}"
    [[ -f "$path" ]] && echo "     ${path}"
done
echo ""
echo "To install to ${PREFIX}:"
echo "  cmake --install ${BUILD_DIR}"
