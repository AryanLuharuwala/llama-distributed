# install.ps1 — one-shot Windows installer, hosted by the dashboard.
#
# Usage (from an admin PowerShell):
#   iwr -useb https://<dashboard>/install.ps1 | iex
#   # or, with explicit args:
#   powershell -ExecutionPolicy Bypass -Command `
#     "& { iwr -useb https://<dashboard>/install.ps1 | iex } -Token <t>"
#
# Detects arch, fetches the matching release zip from GitHub, extracts it,
# and hands off to scripts\install\windows-install.ps1.

param(
    [string]$Pair       = $env:DIST_PAIR,
    [string]$Token      = $env:DIST_TOKEN,
    [string]$Server     = $env:DIST_SERVER,
    [string]$Version    = "latest",
    [string]$GithubRepo = $(if ($env:GITHUB_REPO) { $env:GITHUB_REPO } else { "AryanLuharuwala/llama-distributed" }),
    [int]   $GpuLayers  = 999
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrEmpty($Pair) -and -not [string]::IsNullOrEmpty($Token)) {
    if ([string]::IsNullOrEmpty($Server)) { $Server = "wss://pool.llamadist.dev/ws/agent" }
    $Pair = "distpool://pair?token=$Token&server=$Server"
}

if ([string]::IsNullOrEmpty($Pair)) {
    Write-Error "-Pair <distpool://url> is required (or -Token <t> [-Server <ws-url>])"
    exit 2
}

# Detect arch — only x86_64 is shipped right now; arm64 Windows is future work.
$arch = (Get-CimInstance Win32_Processor).Architecture
switch ($arch) {
    9 { $targetArch = "x86_64" }   # AMD64
    12 { Write-Error "Windows on ARM64 is not yet supported"; exit 1 }
    default { Write-Error "unsupported arch code: $arch"; exit 1 }
}

# CPU-only build for Windows at present; CUDA-on-Windows artifact can be added later.
$target = "windows-$targetArch-cpu"
Write-Host "[install] detected target: $target"

# Resolve version — GitHub's "latest" alias redirects, but we need the concrete tag
# for the filename so we fetch the release JSON.
if ($Version -eq "latest") {
    # /releases/latest skips prereleases — try it first, then fall back to the
    # newest release (including prereleases) so dev builds install too.
    $Version = $null
    try {
        $rel = Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$GithubRepo/releases/latest"
        if ($rel.tag_name) { $Version = $rel.tag_name }
    } catch { }
    if ([string]::IsNullOrEmpty($Version)) {
        try {
            $rel = Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$GithubRepo/releases?per_page=1"
            if ($rel -and $rel.Count -gt 0) { $Version = $rel[0].tag_name }
        } catch { }
    }
    if ([string]::IsNullOrEmpty($Version)) {
        Write-Error "could not resolve latest release for $GithubRepo"
        exit 1
    }
}

$asset = "llama-distributed-$Version-$target.zip"
$url   = "https://github.com/$GithubRepo/releases/download/$Version/$asset"

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("llama-dist-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    $zipPath = Join-Path $tmp $asset
    Write-Host "[install] fetching $url"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zipPath

    Write-Host "[install] extracting"
    Expand-Archive -Path $zipPath -DestinationPath $tmp -Force

    $pkgDir = Join-Path $tmp "llama-distributed-$Version-$target"
    $inner  = Join-Path $pkgDir "scripts\install\windows-install.ps1"
    if (-not (Test-Path $inner)) {
        Write-Error "bundled installer missing: $inner"
        exit 1
    }

    Write-Host "[install] running platform installer"
    $argsList = @("-Pair", $Pair, "-GpuLayers", $GpuLayers)
    & powershell -ExecutionPolicy Bypass -File $inner @argsList
    if ($LASTEXITCODE -ne 0) {
        Write-Error "platform installer exited with code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Write-Host "[install] done."
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
