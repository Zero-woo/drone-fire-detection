param([switch]$SkipModel)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    function Invoke-Checked {
        param([string]$Executable, [string[]]$Arguments)
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed (exit $LASTEXITCODE): $Executable $Arguments"
        }
    }
    $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.venv')
        } else {
            Invoke-Checked 'python' @('-m', 'venv', '.venv')
        }
    }
    Invoke-Checked $taskPython @('-m', 'pip', 'install', '--index-url',
        'https://download.pytorch.org/whl/cpu', 'torch==2.10.0', 'torchvision==0.25.0')
    Invoke-Checked $taskPython @('-m', 'pip', 'install', '-r', 'requirements.txt')
    Invoke-Checked $taskPython @('-m', 'pip', 'check')
    New-Item -ItemType Directory -Path 'work' -Force | Out-Null
    $taskFreeze = & $taskPython -m pip freeze
    if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed.' }
    [System.IO.File]::WriteAllLines(
        (Join-Path $taskRoot 'work\installed-requirements.txt'),
        [string[]]$taskFreeze, [System.Text.UTF8Encoding]::new($false)
    )
    Invoke-Checked $taskPython @('-m', 'unittest', 'discover', '-s', 'tests', '-v')
    if (-not $SkipModel) {
        Invoke-Checked $taskPython @('firelab.py', 'fetch-model')
        Invoke-Checked $taskPython @('firelab.py', 'doctor')
    }
    Write-Host 'Setup complete. Place an image in data/raw and run firelab.py run.'
} finally {
    Pop-Location
}
