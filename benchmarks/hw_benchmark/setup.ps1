# One-command setup for the CODA hardware benchmark on Ryzen AI (Windows / PowerShell).
# Creates an isolated venv and installs the Python deps. Lemonade itself is
# assumed already installed; this script only checks that its server responds.
# The whisper-small weights download on the first STT run. For Apple Silicon,
# use setup.sh instead.

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = if ($env:CODA_INBENCH_VENV) { $env:CODA_INBENCH_VENV } else { "$HOME\.virtualenvs\coda-inbench" }

Write-Host "== CODA hardware benchmark setup (ryzen config) =="

try {
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://localhost:13305/api/v1/models" | Out-Null
    Write-Host "Lemonade server is up on port 13305."
} catch {
    Write-Host "Lemonade not reachable on port 13305."
    Write-Host "Start it with:  lemonade serve   (CLI may be lemonade / lemonade-server / lemonade-server-dev)"
    Write-Host "then re-run this script (setup still continues)."
}

if (-not (Test-Path $Venv)) {
    Write-Host "Creating venv at $Venv"
    python -m venv $Venv
}
Write-Host "Installing dependencies..."
& "$Venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
& "$Venv\Scripts\python.exe" -m pip install --quiet `
    openai whisperlivekit faster-whisper numpy psutil py-cpuinfo matplotlib

Write-Host ""
Write-Host "Setup complete. Confirm your model ids first (CLI may be named"
Write-Host "lemonade / lemonade-server / lemonade-server-dev on your install):"
Write-Host "  lemonade list"
Write-Host "then edit MODELS in configs\ryzen.py to match, and run:"
Write-Host "  $Venv\Scripts\python.exe $Here\run_bench.py"
Write-Host "(config auto-detects as 'ryzen'; override with --config)"
Write-Host "After collecting results, chart them with:"
Write-Host "  $Venv\Scripts\python.exe $Here\plot_results.py"
