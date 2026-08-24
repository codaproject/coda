# One-command setup for the CODA Ryzen AI benchmark (Windows / PowerShell).
# Creates an isolated venv and installs the Python deps. Lemonade itself is
# assumed already installed; this script only checks that its server responds.
# The whisper-small weights download on the first STT run.

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = if ($env:CODA_INBENCH_VENV) { $env:CODA_INBENCH_VENV } else { "$HOME\.virtualenvs\coda-inbench" }

Write-Host "== CODA Ryzen AI inference benchmark setup =="

try {
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://localhost:13305/api/v1/models" | Out-Null
    Write-Host "Lemonade server is up on port 13305."
} catch {
    Write-Host "Lemonade not reachable on port 13305."
    Write-Host "Start it with:  lemonade-server serve"
    Write-Host "then re-run this script (setup still continues)."
}

if (-not (Test-Path $Venv)) {
    Write-Host "Creating venv at $Venv"
    python -m venv $Venv
}
Write-Host "Installing dependencies..."
& "$Venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
& "$Venv\Scripts\python.exe" -m pip install --quiet `
    openai whisperlivekit faster-whisper numpy psutil py-cpuinfo

Write-Host ""
Write-Host "Setup complete. Confirm your model ids first:"
Write-Host "  lemonade-server list"
Write-Host "then edit MODELS at the top of run_bench.py to match, and run:"
Write-Host "  $Venv\Scripts\python.exe $Here\run_bench.py"
