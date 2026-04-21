# build.ps1 — source-build installer for llama-distributed on Windows.
#
# PLACEHOLDER: full Windows source-build support is still under
# construction.  Prefer the prebuilt Windows tarball for now.

param(
    [Parameter(Mandatory=$true)][string]$Pair,
    [string]$Accel = "auto",
    [string]$Ref,
    [string]$Repo = "AryanLuharuwala/llama-distributed",
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

Write-Error @"
build.ps1 is not implemented yet.

For now, please use the prebuilt installer:
  iwr -useb https://<dashboard>/install.ps1 | iex

or run build.sh under WSL.  Tracking in docs/FLOW.md.
"@
exit 1
