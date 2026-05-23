cmake_minimum_required(VERSION 3.14)
project(llama-distributed VERSION 0.1.0 LANGUAGES CXX)

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
# relaying every frame through the dist-server.  Without it, dist-node falls
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
add_executable(dist-coordinator src/dist_coordinator_main.cpp)
target_link_libraries(dist-coordinator PRIVATE dist_common)

add_executable(dist-node         src/dist_node_main.cpp)
target_link_libraries(dist-node  PRIVATE dist_common)

add_executable(dist-client       src/dist_client_main.cpp)
target_link_libraries(dist-client PRIVATE dist_common ${_PLATFORM_LIBS})

# VM layer (unified-computer mode) — off by default, see DIST_BUILD_VM above.
if(DIST_BUILD_VM)
    add_executable(dist-vm-coordinator src/dist_vm_coordinator_main.cpp)
    target_link_libraries(dist-vm-coordinator PRIVATE dist_vm)

    add_executable(dist-vm-node        src/dist_vm_node_main.cpp)
    target_link_libraries(dist-vm-node PRIVATE dist_vm)
endif()

# dist-join: zero-friction pool membership — queries dashboard, exec's dist-node
add_executable(dist-join src/dist_join_main.cpp)
target_include_directories(dist-join PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dist-join PRIVATE ${_PLATFORM_LIBS})

# nat-pingpong: standalone UDP reachability tester for diagnosing symmetric
# NAT inbound-blocking.  Two modes: `listen` (STUNs its external mapping and
# echoes) and `probe` (sends pings to a listener's printed ip:port).
add_executable(nat-pingpong src/nat_pingpong.cpp)

# ─── dist-cli — operator CLI ────────────────────────────────────────────────
#
# Single binary that mirrors the dashboard's surface: pools, models, rigs,
# inference logs, plus a live TUI dashboard (`dist-cli top`) built on FTXUI.
# Auth is read from the same on-disk state dist-node maintains so users who
# have run `dist-node login` are immediately authenticated for CLI ops.
option(DIST_BUILD_CLI     "Build dist-cli operator CLI"               ON)
option(DIST_BUILD_CLI_TUI "Vendor FTXUI to enable `dist-cli top` TUI" ON)

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
        message(STATUS "[llama-distributed] vendored FTXUI for dist-cli top")
        list(APPEND _DIST_CLI_LIBS
            ftxui::screen ftxui::dom ftxui::component)
    endif()

    add_executable(dist-cli
        src/cli_common.cpp
        src/dist_cli_main.cpp
        src/dist_cli_top.cpp
        src/dist_cli_split.cpp
    )
    target_include_directories(dist-cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(dist-cli PRIVATE ${_DIST_CLI_LIBS})

    if(DIST_BUILD_CLI_TUI)
        target_compile_definitions(dist-cli PRIVATE DIST_HAVE_FTXUI=1)
    endif()
endif()

# ─── Install ─────────────────────────────────────────────────────────────────

include(GNUInstallDirs)

set(_DIST_INSTALL_TARGETS
    dist_common
    dist-coordinator dist-node dist-client dist-join
)
if(DIST_BUILD_VM)
    list(APPEND _DIST_INSTALL_TARGETS
        dist_vm dist-vm-coordinator dist-vm-node)
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
message(STATUS "╠══════════════════════════════════════════════╣")
message(STATUS "║  Targets built:                              ║")
message(STATUS "║    dist-coordinator   (base pipeline)        ║")
message(STATUS "║    dist-node          (base pipeline)        ║")
message(STATUS "║    dist-client        (CLI client)           ║")
message(STATUS "║    dist-vm-coordinator(unified VM mode)      ║")
message(STATUS "║    dist-vm-node       (unified VM mode)      ║")
message(STATUS "║    dist-join          (one-command pool join)║")
message(STATUS "╚══════════════════════════════════════════════╝")
message(STATUS "")
