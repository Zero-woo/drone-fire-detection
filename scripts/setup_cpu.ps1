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
        Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.venv')
    }
    Invoke-Checked $taskPython @('-m', 'pip', 'install', 'torch==2.10.0+cpu',
        'torchvision==0.25.0+cpu', '--index-url', 'https://download.pytorch.org/whl/cpu')
    Invoke-Checked $taskPython @('-m', 'pip', 'install', '-r', 'requirements-cpu.txt')
    Invoke-Checked $taskPython @('-m', 'pip', 'check')
    Invoke-Checked $taskPython @('-m', 'unittest', 'discover', '-s', 'tests', '-v')
    Write-Host 'CPU setup complete. See docs/USAGE.md for DroidCam and live inference.'
} finally {
    Pop-Location
}
