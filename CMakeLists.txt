cmake_minimum_required(VERSION 3.14)
project(llama-distributed VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Options ────────────────────────────────────────────────────────────────

option(DIST_USE_CUDA     "Enable CUDA acceleration"               OFF)
option(DIST_USE_HIP      "Enable HIP/ROCm acceleration"           OFF)
option(DIST_USE_METAL    "Enable Metal acceleration (macOS)"      OFF)
option(DIST_USE_VULKAN   "Enable Vulkan acceleration"             OFF)
option(DIST_BUILD_TESTS  "Build unit tests"                       OFF)
option(DIST_BUILD_VM     "Build the experimental VM layer"        OFF)
option(DIST_USE_P2P      "Build with libdatachannel-backed P2P ACTV transport (WebRTC + ICE)" OFF)
option(DIST_USE_SDCPP    "Vendor stable-diffusion.cpp and build gpunet-sdcpp-worker (whole-pipeline C++ diffusion backend for rigs without python+diffusers)" OFF)

# Where to find llama.cpp.
#   AUTO    — use third_party/llama.cpp submodule (default)
#   SYSTEM  — find via FindLlamaCpp.cmake (pre-installed)
#   <path>  — explicit path to llama.cpp source tree
set(DIST_LLAMA_SOURCE "AUTO" CACHE STRING
    "llama.cpp source: AUTO | SYSTEM | /path/to/llama.cpp")

# ─── Locate llama.cpp ────────────────────────────────────────────────────────

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(DIST_LLAMA_SOURCE STREQUAL "SYSTEM")
    # ── Mode 1: system-installed llama.cpp ──────────────────────────────────
    find_package(LlamaCpp REQUIRED)
    set(_LLAMA_LIBS    LlamaCpp::llama)
    set(_LLAMA_INCDIRS "${LlamaCpp_INCLUDE_DIRS}")
    message(STATUS "[llama-distributed] using system llama.cpp")

elseif(DIST_LLAMA_SOURCE STREQUAL "AUTO")
    # ── Mode 2: submodule at third_party/llama.cpp ───────────────────────────
    set(_SUBMOD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")
    if(NOT EXISTS "${_SUBMOD_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "third_party/llama.cpp is empty.\n"
            "Run:  git submodule update --init --recursive")
    endif()
    # Apply our partial-layer-load patch if it hasn't already been applied.
    # Idempotent: git apply --check succeeds only when the patch still needs
    # to be applied; once applied, --reverse --check succeeds and we skip.
    set(_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patches/partial-layer-load.patch")
    if(EXISTS "${_PATCH_FILE}")
        find_program(_GIT_EXE git)
        if(_GIT_EXE)
            execute_process(
                COMMAND ${_GIT_EXE} apply --reverse --check "${_PATCH_FILE}"
                WORKING_DIRECTORY "${_SUBMOD_DIR}"
                RESULT_VARIABLE _patch_reverse_rc
                OUTPUT_QUIET ERROR_QUIET)
            if(NOT _patch_reverse_rc EQUAL 0)
                execute_process(
                    COMMAND ${_GIT_EXE} apply --check "${_PATCH_FILE}"
                    WORKING_DIRECTORY "${_SUBMOD_DIR}"
                    RESULT_VARIABLE _patch_check_rc
                    OUTPUT_QUIET ERROR_QUIET)
                if(_patch_check_rc EQUAL 0)
                    message(STATUS "[llama-distributed] applying partial-layer-load.patch")
                    execute_process(
                        COMMAND ${_GIT_EXE} apply "${_PATCH_FILE}"
                        WORKING_DIRECTORY "${_SUBMOD_DIR}"
                        RESULT_VARIABLE _patch_apply_rc)
                    if(NOT _patch_apply_rc EQUAL 0)
                        message(FATAL_ERROR "failed to apply partial-layer-load.patch")
                    endif()
                else()
                    message(WARNING "[llama-distributed] patch neither applied nor applicable — submodule may be out of sync")
                endif()
            else()
                message(STATUS "[llama-distributed] partial-layer-load.patch already applied")
            endif()
        else()
            message(WARNING "git not found — cannot auto-apply partial-layer-load.patch")
        endif()
    endif()
    set(_LLAMA_SRC_DIR "${_SUBMOD_DIR}")
    message(STATUS "[llama-distributed] using submodule llama.cpp @ ${_LLAMA_SRC_DIR}")

else()
    # ── Mode 3: user-supplied path ───────────────────────────────────────────
    set(_LLAMA_SRC_DIR "${DIST_LLAMA_SOURCE}")
    if(NOT EXISTS "${_LLAMA_SRC_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "DIST_LLAMA_SOURCE='${DIST_LLAMA_SOURCE}' is not a valid llama.cpp source tree")
    endif()
    message(STATUS "[llama-distributed] using llama.cpp @ ${_LLAMA_SRC_DIR}")
endif()

# ── Build llama.cpp from source (modes AUTO and explicit path) ───────────────
if(DEFINED _LLAMA_SRC_DIR)
    # Suppress llama.cpp's own tests/examples/tools
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

    # Forward acceleration options
    if(DIST_USE_CUDA)
        set(GGML_CUDA   ON CACHE BOOL "" FORCE)
    endif()
    if(DIST_USE_HIP)
        set(GGML_HIP    ON CACHE BOOL "" FORCE)
    endif()
    if(DIST_USE_METAL)
        set(GGML_METAL  ON CACHE BOOL "" FORCE)
    endif()
    if(DIST_USE_VULKAN)
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    endif()

    add_subdirectory("${_LLAMA_SRC_DIR}" "third_party/llama.cpp" EXCLUDE_FROM_ALL)

    set(_LLAMA_LIBS    llama ggml)
    set(_LLAMA_INCDIRS
        "${_LLAMA_SRC_DIR}/include"
        "${_LLAMA_SRC_DIR}/ggml/include"
    )
endif()

# ─── Platform link deps ──────────────────────────────────────────────────────

set(_PLATFORM_LIBS)
if(UNIX)
    list(APPEND _PLATFORM_LIBS pthread)
endif()
if(WIN32)
    list(APPEND _PLATFORM_LIBS ws2_32)
endif()

# ─── dist_common — base pipeline layer ──────────────────────────────────────
#
# Provides: NodeAgent, Coordinator, PipelineStage, protocol helpers.

add_library(dist_common STATIC
    src/coordinator.cpp
    src/node_agent.cpp
    src/pipeline.cpp
    src/cluster_monitor.cpp
    src/dashboard_server.cpp
    src/topology.cpp
    src/auth.cpp
    src/dist_ws_client.cpp
    src/pp_engine.cpp
    src/shard_download.cpp
    src/gpu_lock.cpp
    src/comfy_adapter.cpp
    src/diffusion_pp_adapter.cpp
    src/agent_identity.cpp
    src/actv_p2p.cpp
    src/runtime_adapter.cpp
    src/vllm_adapter.cpp
    src/sglang_adapter.cpp
    src/trtllm_adapter.cpp
    src/nat_pmp.cpp
)

target_include_directories(dist_common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${_LLAMA_INCDIRS}
)

find_package(OpenSSL REQUIRED)

# Optional libdatachannel — when present (and DIST_USE_P2P=ON), wires WebRTC
# data channels as the ACTV transport so rigs can talk directly instead of
# relaying every frame through the dist-server.  Without it, gpunet-node falls
# back to the legacy server-relay path (still fully functional).
set(_DIST_P2P_LIBS)
if(DIST_USE_P2P)
    find_package(LibDataChannel CONFIG QUIET)
    if(LibDataChannel_FOUND)
        message(STATUS "[llama-distributed] using system libdatachannel for P2P ACTV")
    else()
        include(FetchContent)
        FetchContent_Declare(
            LibDataChannel
            GIT_REPOSITORY https://github.com/paullouisageneau/libdatachannel.git
            GIT_TAG        v0.21.2
        )
        # Keep the FetchContent build lean — we only need data channels.
        set(NO_MEDIA    ON  CACHE BOOL "" FORCE)
        set(NO_WEBSOCKET ON CACHE BOOL "" FORCE)
        set(NO_EXAMPLES ON  CACHE BOOL "" FORCE)
        set(NO_TESTS    ON  CACHE BOOL "" FORCE)
        set(USE_GNUTLS  OFF CACHE BOOL "" FORCE)  # reuse our OpenSSL
        set(USE_NICE    OFF CACHE BOOL "" FORCE)  # use bundled libjuice
        FetchContent_MakeAvailable(LibDataChannel)
        message(STATUS "[llama-distributed] vendored libdatachannel via FetchContent")
    endif()
    target_compile_definitions(dist_common PUBLIC DIST_HAVE_P2P=1)
    list(APPEND _DIST_P2P_LIBS LibDataChannel::LibDataChannel)
else()
    message(STATUS "[llama-distributed] P2P ACTV disabled (DIST_USE_P2P=OFF) — server-relay only")
endif()

target_link_libraries(dist_common
    PUBLIC  ${_LLAMA_LIBS} ${_PLATFORM_LIBS} OpenSSL::SSL OpenSSL::Crypto ${_DIST_P2P_LIBS}
)

target_compile_options(dist_common PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)

# ─── dist_vm — experimental VM / unified-computer layer ─────────────────────
#
# Provides: VmTensorRegistry, VmScheduler, CollectiveEngine, FaultManager,
#           VmCoordinator, VmNode, VmContext.  Gated behind DIST_BUILD_VM
# because the VM sources currently lag the base protocol and don't compile.

if(DIST_BUILD_VM)
    add_library(dist_vm STATIC
        src/vm_tensor_registry.cpp
        src/vm_scheduler.cpp
        src/vm_collective.cpp
        src/vm_fault.cpp
        src/vm_coordinator.cpp
        src/vm_node.cpp
        src/vm_context.cpp
    )

    target_include_directories(dist_vm
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${_LLAMA_INCDIRS}
    )

    target_link_libraries(dist_vm
        PUBLIC  dist_common
    )

    target_compile_options(dist_vm PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
    )
endif()

# ─── Executables ─────────────────────────────────────────────────────────────

# Base pipeline (no VM layer)
add_executable(gpunet-coordinator src/dist_coordinator_main.cpp)
target_link_libraries(gpunet-coordinator PRIVATE dist_common)

add_executable(gpunet-node         src/dist_node_main.cpp)
target_link_libraries(gpunet-node  PRIVATE dist_common)

add_executable(gpunet-client       src/dist_client_main.cpp)
target_link_libraries(gpunet-client PRIVATE dist_common ${_PLATFORM_LIBS})

# VM layer (unified-computer mode) — off by default, see DIST_BUILD_VM above.
if(DIST_BUILD_VM)
    add_executable(gpunet-vm-coordinator src/dist_vm_coordinator_main.cpp)
    target_link_libraries(gpunet-vm-coordinator PRIVATE dist_vm)

    add_executable(gpunet-vm-node        src/dist_vm_node_main.cpp)
    target_link_libraries(gpunet-vm-node PRIVATE dist_vm)
endif()

# gpunet-join: zero-friction pool membership — queries dashboard, exec's gpunet-node
add_executable(gpunet-join src/dist_join_main.cpp)
target_include_directories(gpunet-join PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpunet-join PRIVATE ${_PLATFORM_LIBS})

# nat-pingpong: standalone UDP reachability tester for diagnosing symmetric
# NAT inbound-blocking.  Two modes: `listen` (STUNs its external mapping and
# echoes) and `probe` (sends pings to a listener's printed ip:port).
add_executable(nat-pingpong src/nat_pingpong.cpp)

# nat-portmap: probe NAT-PMP / PCP on the upstream router to obtain
# an external (ip:port) mapping for a UDP port.  P17.
add_executable(nat-portmap src/nat_portmap_main.cpp)
target_link_libraries(nat-portmap PRIVATE dist_common)

# ─── gpunet-cli — operator CLI ────────────────────────────────────────────────
#
# Single binary that mirrors the dashboard's surface: pools, models, rigs,
# inference logs, plus a live TUI dashboard (`gpunet-cli top`) built on FTXUI.
# Auth is read from the same on-disk state gpunet-node maintains so users who
# have run `gpunet-node login` are immediately authenticated for CLI ops.
option(DIST_BUILD_CLI     "Build gpunet-cli operator CLI"               ON)
option(DIST_BUILD_CLI_TUI "Vendor FTXUI to enable `gpunet-cli top` TUI" ON)

if(DIST_BUILD_CLI)
    set(_DIST_CLI_LIBS OpenSSL::SSL OpenSSL::Crypto ${_PLATFORM_LIBS})

    if(DIST_BUILD_CLI_TUI)
        include(FetchContent)
        FetchContent_Declare(
            ftxui
            GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
            GIT_TAG        v5.0.0
        )
        set(FTXUI_BUILD_DOCS    OFF CACHE BOOL "" FORCE)
        set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(FTXUI_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
        set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(ftxui)
        message(STATUS "[llama-distributed] vendored FTXUI for gpunet-cli top")
        list(APPEND _DIST_CLI_LIBS
            ftxui::screen ftxui::dom ftxui::component)
    endif()

    add_executable(gpunet-cli
        src/cli_common.cpp
        src/dist_cli_main.cpp
        src/dist_cli_top.cpp
        src/dist_cli_split.cpp
    )
    target_include_directories(gpunet-cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(gpunet-cli PRIVATE ${_DIST_CLI_LIBS})

    if(DIST_BUILD_CLI_TUI)
        target_compile_definitions(gpunet-cli PRIVATE DIST_HAVE_FTXUI=1)
    endif()
endif()

# ─── gpunet-sdcpp-worker — C++ diffusion backend ──────────────────────────────
#
# Vendored stable-diffusion.cpp wrapped as a standalone worker binary.  This
# is the fallback path for rigs that can't run python+diffusers (e.g.
# paramshakti cn01 — conda py3.7, no sudo, stripped image).  The DppAdapter
# forks this binary the same way it forks `python -m dpp_runtime`, so the
# WS / wire surface in gpunet-node is unchanged.
#
# Built as a *separate executable* on purpose: sd.cpp ships its own ggml fork
# (leejet/ggml) which would symbol-collide with the llama.cpp ggml linked
# into dist_common.  Keeping them in different processes sidesteps the issue
# and matches the existing python launch pattern.
#
# Phase A (this build): worker handles role="full" — single-rig whole-pipeline
# generation (text-encode + unet + vae all local).  Phase B (task #254) adds
# per-role entrypoints so sd.cpp can fill a stage in a DPP chain.
if(DIST_USE_SDCPP)
    set(_SDCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp")
    if(NOT EXISTS "${_SDCPP_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "DIST_USE_SDCPP=ON but third_party/stable-diffusion.cpp is empty.\n"
            "Run:  git submodule update --init --recursive third_party/stable-diffusion.cpp")
    endif()

    # Apply our CF12-W6a block-split patch idempotently — adds the
    # sd_compute_unet_split_step C surface so the role bridge can split a
    # UNet forward pass across rigs.  Phase-1 bodies return SD_SPLIT_ENOTSUP;
    # the follow-up patch swaps in the real forward_half implementation.
    # Same pattern as the llama.cpp partial-layer-load patch above.
    set(_SDCPP_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/patches/sdcpp-block-split.patch")
    if(EXISTS "${_SDCPP_PATCH}")
        find_program(_GIT_EXE git)
        if(_GIT_EXE)
            execute_process(
                COMMAND ${_GIT_EXE} apply --reverse --check "${_SDCPP_PATCH}"
                WORKING_DIRECTORY "${_SDCPP_DIR}"
                RESULT_VARIABLE _sdcpp_patch_reverse_rc
                OUTPUT_QUIET ERROR_QUIET)
            if(NOT _sdcpp_patch_reverse_rc EQUAL 0)
                execute_process(
                    COMMAND ${_GIT_EXE} apply --check "${_SDCPP_PATCH}"
                    WORKING_DIRECTORY "${_SDCPP_DIR}"
                    RESULT_VARIABLE _sdcpp_patch_check_rc
                    OUTPUT_QUIET ERROR_QUIET)
                if(_sdcpp_patch_check_rc EQUAL 0)
                    message(STATUS "[llama-distributed] applying sdcpp-block-split.patch")
                    execute_process(
                        COMMAND ${_GIT_EXE} apply "${_SDCPP_PATCH}"
                        WORKING_DIRECTORY "${_SDCPP_DIR}"
                        RESULT_VARIABLE _sdcpp_patch_apply_rc)
                    if(NOT _sdcpp_patch_apply_rc EQUAL 0)
                        message(FATAL_ERROR "failed to apply sdcpp-block-split.patch")
                    endif()
                else()
                    message(WARNING "[llama-distributed] sdcpp-block-split.patch neither applied nor applicable — submodule may be out of sync")
                endif()
            else()
                message(STATUS "[llama-distributed] sdcpp-block-split.patch already applied")
            endif()
        else()
            message(WARNING "git not found — cannot auto-apply sdcpp-block-split.patch")
        endif()
    endif()

    # Suppress sd.cpp's own examples; we only want the library.
    set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    # Backend mirrors dist's selection — sd.cpp owns the ggml fork inside its
    # own subtree, so its SD_CUDA / SD_VULKAN flags drive a *separate* ggml
    # build that won't link into dist_common.
    if(DIST_USE_CUDA)
        set(SD_CUDA ON CACHE BOOL "" FORCE)
    endif()
    if(DIST_USE_HIP)
        set(SD_HIPBLAS ON CACHE BOOL "" FORCE)
    endif()
    if(DIST_USE_METAL)
        set(SD_METAL ON CACHE BOOL "" FORCE)
    endif()
    if(DIST_USE_VULKAN)
        set(SD_VULKAN ON CACHE BOOL "" FORCE)
    endif()

    add_subdirectory(third_party/stable-diffusion.cpp EXCLUDE_FROM_ALL)

    add_executable(gpunet-sdcpp-worker
        src/sdcpp_worker.cpp
        src/sdcpp_roles.cpp
        src/sdt_codec.cpp
    )
    target_include_directories(gpunet-sdcpp-worker PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${_SDCPP_DIR}/include
        ${_SDCPP_DIR}/thirdparty
    )
    target_link_libraries(gpunet-sdcpp-worker PRIVATE
        stable-diffusion
        ${_PLATFORM_LIBS}
    )
    target_compile_definitions(gpunet-sdcpp-worker PRIVATE DIST_HAVE_SDCPP=1)

    # CF12-W6a: when the block-split patch is present, expose the symbol so
    # the role bridge calls into the new C surface instead of guessing from
    # the model filename.
    if(EXISTS "${_SDCPP_PATCH}")
        target_compile_definitions(gpunet-sdcpp-worker PRIVATE DIST_HAVE_SDCPP_SPLIT=1)

        # Smoke test for the CF12-W6a C surface — no model load, just exercises
        # the sd_split_state_t setter/getter/serialize/SDCD wire roundtrip.
        add_executable(test_sdcpp_split
            src/test_sdcpp_split.cpp
            src/sdt_codec.cpp
        )
        target_include_directories(test_sdcpp_split PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${_SDCPP_DIR}/include
            ${_SDCPP_DIR}/thirdparty
        )
        target_link_libraries(test_sdcpp_split PRIVATE
            stable-diffusion
            ${_PLATFORM_LIBS}
        )
        target_compile_definitions(test_sdcpp_split PRIVATE
            DIST_HAVE_SDCPP=1
            DIST_HAVE_SDCPP_SPLIT=1
        )
    endif()

    # Expose to dist_common via compile definition — used by capability probe.
    target_compile_definitions(dist_common PUBLIC DIST_HAVE_SDCPP=1)

    message(STATUS "[llama-distributed] gpunet-sdcpp-worker enabled (vendored sd.cpp)")
else()
    message(STATUS "[llama-distributed] gpunet-sdcpp-worker disabled (DIST_USE_SDCPP=OFF)")
endif()

# ─── Install ─────────────────────────────────────────────────────────────────

include(GNUInstallDirs)

set(_DIST_INSTALL_TARGETS
    dist_common
    gpunet-coordinator gpunet-node gpunet-client gpunet-join
)
if(DIST_BUILD_VM)
    list(APPEND _DIST_INSTALL_TARGETS
        dist_vm gpunet-vm-coordinator gpunet-vm-node)
endif()
if(DIST_USE_SDCPP)
    list(APPEND _DIST_INSTALL_TARGETS gpunet-sdcpp-worker)
endif()

install(TARGETS ${_DIST_INSTALL_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/llama-distributed
    FILES_MATCHING PATTERN "*.h"
)

# ─── Summary ─────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════╗")
message(STATUS "║       llama-distributed ${PROJECT_VERSION}                  ║")
message(STATUS "╠══════════════════════════════════════════════╣")
message(STATUS "║  llama.cpp source  : ${DIST_LLAMA_SOURCE}")
message(STATUS "║  CUDA              : ${DIST_USE_CUDA}")
message(STATUS "║  HIP/ROCm          : ${DIST_USE_HIP}")
message(STATUS "║  Metal             : ${DIST_USE_METAL}")
message(STATUS "║  Vulkan            : ${DIST_USE_VULKAN}")
message(STATUS "║  Tests             : ${DIST_BUILD_TESTS}")
message(STATUS "║  sd.cpp worker     : ${DIST_USE_SDCPP}")
message(STATUS "╠══════════════════════════════════════════════╣")
message(STATUS "║  Targets built:                              ║")
message(STATUS "║    gpunet-coordinator   (base pipeline)        ║")
message(STATUS "║    gpunet-node          (base pipeline)        ║")
message(STATUS "║    gpunet-client        (CLI client)           ║")
message(STATUS "║    gpunet-vm-coordinator(unified VM mode)      ║")
message(STATUS "║    gpunet-vm-node       (unified VM mode)      ║")
message(STATUS "║    gpunet-join          (one-command pool join)║")
message(STATUS "╚══════════════════════════════════════════════╝")
message(STATUS "")
