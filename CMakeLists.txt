cmake_minimum_required(VERSION 3.14)
project(llama-distributed CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ────────────────────────────────────────────────────────────────

option(DIST_USE_CUDA    "Build with CUDA support (requires llama.cpp CUDA build)" OFF)
option(DIST_BUILD_TESTS "Build unit tests" OFF)

# Path to llama.cpp source tree (default: sibling directory)
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp"
    CACHE PATH "Path to llama.cpp source tree")

# ─── llama.cpp integration ───────────────────────────────────────────────────
#
# We build llama.cpp as a subdirectory so we get llama and ggml targets.
# Users can alternatively point to a pre-built install via CMAKE_PREFIX_PATH.

if(EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(STATUS "Using llama.cpp from: ${LLAMA_CPP_DIR}")

    # Only build the library targets, not all tools/examples
    set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)

    # Forward CUDA option
    if(DIST_USE_CUDA)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
    endif()

    add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)

    set(LLAMA_LIBS llama ggml)
    set(LLAMA_INCLUDE_DIRS
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/ggml/include
        ${LLAMA_CPP_DIR}/common
    )
else()
    # Try finding a pre-installed llama.cpp
    find_package(llama REQUIRED)
    set(LLAMA_LIBS llama::llama)
    set(LLAMA_INCLUDE_DIRS "")
    message(STATUS "Using installed llama.cpp")
endif()

# ─── Common library (protocol + connection + queue + pipeline) ───────────────

add_library(dist_common STATIC
    src/node_agent.cpp
    src/coordinator.cpp
    src/pipeline.cpp
)

# ─── VM layer library ────────────────────────────────────────────────────────

add_library(dist_vm STATIC
    src/vm_tensor_registry.cpp
    src/vm_scheduler.cpp
    src/vm_collective.cpp
    src/vm_fault.cpp
    src/vm_coordinator.cpp
    src/vm_node.cpp
    src/vm_context.cpp
)

target_include_directories(dist_vm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLAMA_INCLUDE_DIRS}
)

target_link_libraries(dist_vm PUBLIC dist_common ${LLAMA_LIBS})

if(UNIX)
    target_link_libraries(dist_vm PUBLIC pthread)
endif()
if(WIN32)
    target_link_libraries(dist_vm PUBLIC ws2_32)
endif()

target_compile_options(dist_vm PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)

target_include_directories(dist_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLAMA_INCLUDE_DIRS}
)

target_link_libraries(dist_common PUBLIC
    ${LLAMA_LIBS}
)

if(UNIX)
    target_link_libraries(dist_common PUBLIC pthread)
endif()

if(WIN32)
    target_link_libraries(dist_common PUBLIC ws2_32)
endif()

target_compile_options(dist_common PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)

# ─── Executables ────────────────────────────────────────────────────────────

# Base pipeline executables
add_executable(dist-coordinator src/dist_coordinator_main.cpp)
target_link_libraries(dist-coordinator PRIVATE dist_common)

add_executable(dist-node src/dist_node_main.cpp)
target_link_libraries(dist-node PRIVATE dist_common)

add_executable(dist-client src/dist_client_main.cpp)
target_include_directories(dist-client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(dist-client PRIVATE ws2_32)
endif()
if(UNIX)
    target_link_libraries(dist-client PRIVATE pthread)
endif()

# VM-layer executables
add_executable(dist-vm-coordinator src/dist_vm_coordinator_main.cpp)
target_link_libraries(dist-vm-coordinator PRIVATE dist_vm)

add_executable(dist-vm-node src/dist_vm_node_main.cpp)
target_link_libraries(dist-vm-node PRIVATE dist_vm)

# ─── Install ────────────────────────────────────────────────────────────────

install(TARGETS
        dist-coordinator dist-node dist-client
        dist-vm-coordinator dist-vm-node
        RUNTIME DESTINATION bin)

# ─── Summary ────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== llama-distributed build configuration ===")
message(STATUS "  CUDA support     : ${DIST_USE_CUDA}")
message(STATUS "  llama.cpp        : ${LLAMA_CPP_DIR}")
message(STATUS "  Build tests      : ${DIST_BUILD_TESTS}")
message(STATUS "  Executables      : dist-coordinator dist-node dist-client")
message(STATUS "                     dist-vm-coordinator dist-vm-node")
message(STATUS "")
