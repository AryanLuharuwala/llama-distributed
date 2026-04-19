# windows-install.ps1 — installs the agent under %ProgramFiles% and registers
# a per-user scheduled task that runs at login and restarts on failure.
#
# Usage (from an extracted release zip):
#   powershell -ExecutionPolicy Bypass -File scripts\install\windows-install.ps1 `
#     -Token <pool-token> [-Server host:port]
#
# Uninstall:
#   powershell -ExecutionPolicy Bypass -File scripts\install\windows-install.ps1 -Uninstall

param(
    [string]$Pair    = "",
    [string]$Token   = "",
    [string]$Server  = "wss://pool.llamadist.dev/ws/agent",
    [int]   $GpuLayers = 999,
    [switch]$Uninstall
)

if ([string]::IsNullOrEmpty($Pair) -and -not [string]::IsNullOrEmpty($Token)) {
    $Pair = "distpool://pair?token=$Token&server=$Server"
}

$ErrorActionPreference = "Stop"

$InstallDir = Join-Path $env:ProgramFiles "llama-distributed"
$ConfigDir  = Join-Path $env:APPDATA     "llama-distributed"
$BinDir     = Join-Path $InstallDir "bin"
$TaskName   = "llama-distributed-agent"
$PkgRoot    = Split-Path -Parent $PSScriptRoot  # the extracted zip root

function Install-Bins {
    Write-Host "[install] copying binaries into $BinDir"
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    Get-ChildItem -Path (Join-Path $PkgRoot "bin") -File |
        Copy-Item -Destination $BinDir -Force
}

function Write-Config {
    New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
    $cfg = @(
      "DIST_PAIR=$Pair",
      "DIST_GPU_LAYERS=$GpuLayers"
    ) -join "`r`n"
    Set-Content -Path (Join-Path $ConfigDir "agent.env") -Value $cfg -Encoding UTF8
}

function Register-AgentTask {
    $exe = Join-Path $BinDir "dist-node.exe"
    if (-not (Test-Path $exe)) { throw "dist-node.exe not found in $BinDir" }

    $argsLine = "--pair `"$Pair`" -g $GpuLayers"
    $action   = New-ScheduledTaskAction -Execute $exe -Argument $argsLine
    $trigger  = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Seconds 30)

    # Replace any existing task
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $TaskName `
        -Action $action -Trigger $trigger -Settings $settings `
        -Description "llama-distributed pool agent"
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "[install] task '$TaskName' registered and started"
}

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $ConfigDir  -ErrorAction SilentlyContinue
    Write-Host "[uninstall] done"
    exit 0
}

Install-Bins
Write-Config

if ([string]::IsNullOrEmpty($Pair)) {
    Write-Host "[install] no -Pair URL supplied — binaries installed but task not registered"
    Write-Host "         edit $ConfigDir\agent.env then re-run this script with -Pair"
    exit 0
}

Register-AgentTask
