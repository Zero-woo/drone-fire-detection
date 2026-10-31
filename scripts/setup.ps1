param([switch]$SkipModels)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    function Invoke-Checked {
        param([string]$Executable, [string[]]$Arguments)
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Command failed: $Executable $Arguments" }
    }
    $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        Invoke-Checked 'py' @('-3.10', '-m', 'venv', '.venv')
    }
    Invoke-Checked $taskPython @('-m', 'pip', 'install', 'torch==2.10.0', 'torchvision==0.25.0',
        '--index-url', 'https://download.pytorch.org/whl/cu128')
    Invoke-Checked $taskPython @('-m', 'pip', 'install', '-r', 'requirements.txt')
    Invoke-Checked $taskPython @('-m', 'pip', 'check')
    Invoke-Checked $taskPython @('-m', 'unittest', 'discover', '-s', 'tests', '-v')
    Invoke-Checked $taskPython @('firelab.py', 'doctor', '--require-gpu')
    if (-not $SkipModels) { Invoke-Checked $taskPython @('firelab.py', 'fetch-models') }
    Write-Host 'Setup complete. Follow docs/USAGE.md for staged validation and training.'
} finally {
    Pop-Location
}
