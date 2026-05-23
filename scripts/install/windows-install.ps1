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
    [switch]$WithComfyUI,
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
$ComfyTask  = "llama-distributed-comfyui"
$ComfyDir   = Join-Path $env:LOCALAPPDATA "llama-distributed\ComfyUI"
$PkgRoot    = Split-Path -Parent $PSScriptRoot  # the extracted zip root

function Install-Bins {
    Write-Host "[install] copying binaries into $BinDir"
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    Get-ChildItem -Path (Join-Path $PkgRoot "bin") -File |
        Copy-Item -Destination $BinDir -Force
}

function Write-Config {
    New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
    $lines = @(
      "DIST_PAIR=$Pair",
      "DIST_GPU_LAYERS=$GpuLayers"
    )
    if ($WithComfyUI) {
        $lines += "DIST_WITH_COMFYUI=1"
        $lines += "DIST_COMFY_URL=http://127.0.0.1:8188"
    }
    $cfg = $lines -join "`r`n"
    Set-Content -Path (Join-Path $ConfigDir "agent.env") -Value $cfg -Encoding UTF8
}

function Resolve-Python {
    foreach ($cmd in @("py","python","python3")) {
        $p = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($p) { return $p.Source }
    }
    return $null
}

function Install-ComfyUI {
    if (-not $WithComfyUI) { return }
    Write-Host "[install] setting up ComfyUI at $ComfyDir"
    $py = Resolve-Python
    if (-not $py) {
        Write-Host "[install] python not found — install Python 3.10+ from python.org then re-run with -WithComfyUI"
        return
    }
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        Write-Host "[install] git not found — install Git for Windows then re-run with -WithComfyUI"
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ComfyDir) | Out-Null
    if (Test-Path (Join-Path $ComfyDir ".git")) {
        Write-Host "[install] ComfyUI already cloned — pulling latest"
        & git -C $ComfyDir pull --ff-only 2>$null
    } else {
        & git clone --depth=1 https://github.com/comfyanonymous/ComfyUI $ComfyDir
    }
    $venv = Join-Path $ComfyDir ".venv"
    if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
        Write-Host "[install] creating ComfyUI venv"
        & $py -m venv $venv
    }
    $venvPy = Join-Path $venv "Scripts\python.exe"
    & $venvPy -m pip install --quiet --upgrade pip
    & $venvPy -m pip install --quiet -U -r (Join-Path $ComfyDir "requirements.txt")

    # Companion scheduled task to run ComfyUI on login
    $argsLine = "-s `"$ComfyDir\main.py`" --listen 127.0.0.1 --port 8188"
    $action   = New-ScheduledTaskAction -Execute $venvPy -Argument $argsLine -WorkingDirectory $ComfyDir
    $trigger  = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Seconds 30)
    Unregister-ScheduledTask -TaskName $ComfyTask -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $ComfyTask `
        -Action $action -Trigger $trigger -Settings $settings `
        -Description "ComfyUI for llama-distributed agent"
    Start-ScheduledTask -TaskName $ComfyTask
    Write-Host "[install] ComfyUI task '$ComfyTask' registered and started"
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
    Unregister-ScheduledTask -TaskName $TaskName  -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $ComfyTask -Confirm:$false -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $ConfigDir  -ErrorAction SilentlyContinue
    # Leave $ComfyDir intact — it may hold model checkpoints the user downloaded.
    Write-Host "[uninstall] done (ComfyUI dir at $ComfyDir left in place)"
    exit 0
}

Install-Bins
Write-Config
Install-ComfyUI

if ([string]::IsNullOrEmpty($Pair)) {
    Write-Host "[install] no -Pair URL supplied — binaries installed but task not registered"
    Write-Host "         edit $ConfigDir\agent.env then re-run this script with -Pair"
    exit 0
}

Register-AgentTask
