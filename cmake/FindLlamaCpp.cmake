# FindLlamaCpp.cmake
#
# Finds a pre-installed llama.cpp (built and installed via `cmake --install`).
#
# Imported targets:
#   LlamaCpp::llama   — the main inference library
#   LlamaCpp::ggml    — the ggml tensor library
#
# Variables set:
#   LlamaCpp_FOUND
#   LlamaCpp_INCLUDE_DIRS
#   LlamaCpp_LIBRARIES
#
# Search order:
#   1. LLAMA_INSTALL_PREFIX (cmake cache variable or env var)
#   2. Standard system paths (/usr/local, /usr, homebrew prefixes)

set(_search_hints)

if(DEFINED LLAMA_INSTALL_PREFIX)
    list(APPEND _search_hints "${LLAMA_INSTALL_PREFIX}")
endif()
if(DEFINED ENV{LLAMA_INSTALL_PREFIX})
    list(APPEND _search_hints "$ENV{LLAMA_INSTALL_PREFIX}")
endif()

find_path(LlamaCpp_INCLUDE_DIR
    NAMES llama.h
    HINTS ${_search_hints}
    PATH_SUFFIXES include
    PATHS /usr/local/include /usr/include
)

find_library(LlamaCpp_LLAMA_LIB
    NAMES llama
    HINTS ${_search_hints}
    PATH_SUFFIXES lib lib64
    PATHS /usr/local/lib /usr/lib
)

find_library(LlamaCpp_GGML_LIB
    NAMES ggml
    HINTS ${_search_hints}
    PATH_SUFFIXES lib lib64
    PATHS /usr/local/lib /usr/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LlamaCpp
    REQUIRED_VARS LlamaCpp_INCLUDE_DIR LlamaCpp_LLAMA_LIB
)

if(LlamaCpp_FOUND)
    set(LlamaCpp_INCLUDE_DIRS "${LlamaCpp_INCLUDE_DIR}")
    set(LlamaCpp_LIBRARIES    "${LlamaCpp_LLAMA_LIB}")
    if(LlamaCpp_GGML_LIB)
        list(APPEND LlamaCpp_LIBRARIES "${LlamaCpp_GGML_LIB}")
    endif()

    if(NOT TARGET LlamaCpp::llama)
        add_library(LlamaCpp::llama UNKNOWN IMPORTED)
        set_target_properties(LlamaCpp::llama PROPERTIES
            IMPORTED_LOCATION             "${LlamaCpp_LLAMA_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${LlamaCpp_INCLUDE_DIR}"
        )
    endif()

    if(LlamaCpp_GGML_LIB AND NOT TARGET LlamaCpp::ggml)
        add_library(LlamaCpp::ggml UNKNOWN IMPORTED)
        set_target_properties(LlamaCpp::ggml PROPERTIES
            IMPORTED_LOCATION "${LlamaCpp_GGML_LIB}"
        )
        target_link_libraries(LlamaCpp::llama INTERFACE LlamaCpp::ggml)
    endif()
endif()
