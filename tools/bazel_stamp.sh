#!/usr/bin/env bash
# workspace_status_command for Bazel — emits key=value pairs Bazel
# substitutes into stamped binaries via `-X` linker flags.
#
# `STABLE_` prefix → invalidates cache when the value changes.
# Without the prefix → values change without re-running cached steps
# (useful for unimportant metadata like build host).
#
# Wired in from .bazelrc:
#     build --stamp
#     build --workspace_status_command=tools/bazel_stamp.sh
#
# Consumed by go_binary rules via `x_defs`, see server/BUILD.bazel.

set -euo pipefail

# Git SHA — fall back to "unknown" so out-of-tree builds (CI tarball
# extract, sandbox without .git) still produce a binary.
GIT_SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=""
if ! git diff --quiet 2>/dev/null; then GIT_DIRTY="-dirty"; fi
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# Deterministic build date: 0 if SOURCE_DATE_EPOCH is set (CI release
# builds set this), otherwise current UTC time (dev builds).
if [ -n "${SOURCE_DATE_EPOCH:-}" ]; then
    BUILD_DATE=$(date -u -d "@${SOURCE_DATE_EPOCH}" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null \
                 || date -u +"%Y-%m-%dT%H:%M:%SZ")
else
    BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
fi

# STABLE_ keys participate in cache key — change them and Go binaries
# get re-linked.  This is what we want for SHA + branch.
echo "STABLE_GIT_SHA ${GIT_SHA}${GIT_DIRTY}"
echo "STABLE_GIT_BRANCH ${GIT_BRANCH}"
echo "STABLE_BUILD_DATE ${BUILD_DATE}"

# Non-stable keys: change them freely without triggering re-link.
echo "BUILD_HOST $(hostname 2>/dev/null || echo unknown)"
echo "BUILD_USER ${USER:-unknown}"
