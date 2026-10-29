param([switch]$Push)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskRemote = 'https://github.com/Zero-woo/drone-fire-detection.git'
Push-Location $taskRoot
try {
    function Invoke-GitChecked {
        param([string[]]$Arguments)
        & git @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Git failed: $Arguments" }
    }
    if (-not (Test-Path -LiteralPath '.git')) {
        Invoke-GitChecked @('init', '-b', 'main')
    }
    $taskBranch = & git symbolic-ref --short HEAD
    if ($LASTEXITCODE -ne 0 -or $taskBranch -ne 'main') {
        throw 'Expected main branch. Inspect the current repository before proceeding.'
    }
    $taskName = & git config user.name
    $taskEmail = & git config user.email
    if (-not $taskName -or -not $taskEmail) {
        throw 'Configure your Git user.name and user.email before committing.'
    }
    $taskStaged = & git diff --cached --name-only
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the Git index.' }
    if ($taskStaged) { throw 'Existing staged changes found. Review/commit them before running this script.' }
    $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        throw 'Project .venv is missing. Run scripts/setup.ps1 -SkipModel first.'
    }
    & $taskPython -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed; no commits or push performed.' }

    $taskStages = @(
        @{
            Message = 'chore: initialize CPU fire-detection experiment workspace'
            Paths = @('.gitignore', '.gitattributes', 'AGENTS.md', 'README.md',
                'data', 'models/README.md', 'experiments/README.md',
                'docs/USAGE.md', 'experiments/ROADMAP.md',
                'experiments/001_pretrained_cpu.md', 'experiments/runs/.gitkeep',
                'outputs/.gitkeep', 'reports/experiment_template.md')
        },
        @{
            Message = 'feat: add pinned pretrained model and reproducible offline inference'
            Paths = @('requirements.txt', 'configs', 'src', 'firelab.py', 'scripts')
        },
        @{
            Message = 'test: verify experiment logging and document environment limits'
            Paths = @('tests', 'reports/000_environment.md', 'docs/PROGRESS.md')
        }
    )
    foreach ($taskStage in $taskStages) {
        Invoke-GitChecked (@('add', '--') + $taskStage.Paths)
        & git diff --cached --quiet
        $taskDiffExit = $LASTEXITCODE
        if ($taskDiffExit -eq 1) {
            Invoke-GitChecked @('diff', '--cached', '--stat')
            Invoke-GitChecked @('commit', '-m', $taskStage.Message)
        } elseif ($taskDiffExit -ne 0) {
            throw 'Cannot inspect staged differences.'
        }
    }
    $taskRemotes = & git remote
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect remotes.' }
    if (@($taskRemotes) -contains 'origin') {
        $taskOrigin = & git remote get-url origin
        if ($LASTEXITCODE -ne 0) { throw 'Cannot read origin URL.' }
        if ($taskOrigin -ne $taskRemote) { throw 'Existing origin differs; it was not changed.' }
    } else {
        Invoke-GitChecked @('remote', 'add', 'origin', $taskRemote)
    }
    if ($Push) {
        $taskHeads = & git ls-remote --heads origin
        if ($LASTEXITCODE -ne 0) { throw 'Cannot reach GitHub. Local commits are preserved.' }
        $taskHead = & git rev-parse HEAD
        if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve local HEAD.' }
        if ($taskHeads) {
            $taskExpected = "$taskHead" + [char]9 + 'refs/heads/main'
            if (@($taskHeads).Count -ne 1 -or $taskHeads -ne $taskExpected) {
                throw 'Remote already has history. Stop and inspect; no force-push was attempted.'
            }
        }
        Invoke-GitChecked @('push', '-u', 'origin', 'main')
    }
    Invoke-GitChecked @('log', '--oneline', '-3')
    Invoke-GitChecked @('status', '--short')
    Write-Host 'Milestone commits are ready.'
} finally {
    Pop-Location
}
