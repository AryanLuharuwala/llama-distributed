# setup.ps1 — minimal one-line installer for the SURD CLI on Windows.
#
#   iwr -useb https://<host>/setup.ps1 | iex
#
# Downloads the `gpunet.exe` binary into %LOCALAPPDATA%\Surd and prepends it
# to the user's PATH.  Legacy installer (with -Pair, etc.) still lives at
# /install.ps1.

[CmdletBinding()]
param(
  [string]$Pool   = "",
  [string]$Invite = "",
  [string]$Server = "",
  [string]$Prefix = ""
)

$ErrorActionPreference = "Stop"

# Fallback to env vars when called via `iwr | iex` (which can't take argv).
# The configurator sets $env:SURD_POOL / $env:SURD_INVITE before iex.
if (-not $Pool   -and $env:SURD_POOL)   { $Pool   = $env:SURD_POOL }
if (-not $Invite -and $env:SURD_INVITE) { $Invite = $env:SURD_INVITE }

# DIST_SERVER is injected by the server when serving this script; falls back
# to whatever the caller passed via -Server.
$DistServer = "${env:DIST_SERVER}"
if ([string]::IsNullOrEmpty($DistServer) -and -not [string]::IsNullOrEmpty($Server)) {
  $DistServer = $Server
}
if ([string]::IsNullOrEmpty($DistServer)) {
  Write-Error "[setup] DIST_SERVER not set — re-run via the dashboard's /setup.ps1 URL."
  exit 2
}

if ([string]::IsNullOrEmpty($Prefix)) {
  $Prefix = Join-Path $env:LOCALAPPDATA "Surd"
}

# Architecture detection.  PROCESSOR_ARCHITECTURE is what the current shell
# sees (could be x86 inside a 32-bit shell on 64-bit Windows); fall back to
# PROCESSOR_ARCHITEW6432 for that case.
$archRaw = $env:PROCESSOR_ARCHITECTURE
if ($env:PROCESSOR_ARCHITEW6432) { $archRaw = $env:PROCESSOR_ARCHITEW6432 }
switch -Regex ($archRaw) {
  "ARM64"            { $tarch = "arm64"; break }
  "AMD64|x86_64|x64" { $tarch = "amd64"; break }
  default            { Write-Error "[setup] unsupported arch: $archRaw"; exit 1 }
}

$asset   = "gpunet-windows-$tarch.exe"
$baseUrl = $DistServer.TrimEnd("/")
$dlUrl   = "$baseUrl/releases/$asset"

New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
$destTmp = Join-Path $Prefix "gpunet.exe.tmp"
$dest    = Join-Path $Prefix "gpunet.exe"

Write-Host "[setup] downloading $asset from $baseUrl"
try {
  Invoke-WebRequest -Uri $dlUrl -OutFile $destTmp -UseBasicParsing
} catch {
  Write-Error "[setup] download failed — does the server publish ${asset}? ($_)"
  exit 1
}
Move-Item -Force $destTmp $dest

# Prepend $Prefix to the user PATH if not already there.
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($null -eq $userPath) { $userPath = "" }
$paths = $userPath -split ";" | Where-Object { $_ -ne "" }
if ($paths -notcontains $Prefix) {
  $newPath = if ($userPath) { "$Prefix;$userPath" } else { $Prefix }
  [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
  Write-Host "[setup] added $Prefix to user PATH — open a new shell, or:"
  Write-Host "        `$env:Path = `"$Prefix;`" + `$env:Path"
}

# Pre-seed SURD_SERVER so `gpunet login` doesn't need --server on first run.
[Environment]::SetEnvironmentVariable("SURD_SERVER", $DistServer, "User")
$env:SURD_SERVER = $DistServer
Write-Host "[setup] SURD_SERVER=$DistServer"

Write-Host ""
Write-Host "  ✓ installed: $dest"
Write-Host ""
Write-Host "  next:"
$next = "    gpunet login`r`n    gpunet connect"
if ($Pool)   { $next += " --pool $Pool" }
if ($Invite) { $next += " --invite $Invite" }
Write-Host $next
Write-Host ""
