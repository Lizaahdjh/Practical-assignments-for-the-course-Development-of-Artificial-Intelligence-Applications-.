<#
.SYNOPSIS
    Automated run of PR12 scenarios (mock + real provider).

.DESCRIPTION
    Runs scenarios 1-2 (mock) and 3-14 (real provider, needs key):
      * backs up .env to .env.backup;
      * for each scenario: edits .env, runs eval/run.py, saves to eval/runs/;
      * restores .env at the end.

.EXAMPLE
    .\scripts\run_all.ps1
    .\scripts\run_all.ps1 -MockOnly     # only scenarios 1-2
#>

param(
    [switch]$MockOnly
)

$ErrorActionPreference = "Stop"

$Root       = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvFile    = Join-Path $Root ".env"
$EnvBackup  = Join-Path $Root ".env.backup"
$PythonExe  = Join-Path $Root ".venv\Scripts\python.exe"
$RunsDir    = Join-Path $Root "eval\runs"
$CasesDir   = Join-Path $Root "eval\cases"

function Log($msg) { Write-Host "[$(Get-Date -Format HH:mm:ss)] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "[ERR] $msg" -ForegroundColor Red; throw $msg }

if (-not (Test-Path $PythonExe)) { Fail "No .venv\Scripts\python.exe" }
if (-not (Test-Path $EnvFile)) { Fail "No .env" }
New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null

function Set-EnvValue([string]$Key, [string]$Value) {
    $lines = Get-Content $EnvFile
    $found = $false
    $new = foreach ($line in $lines) {
        if ($line -match "^\s*$([regex]::Escape($Key))\s*=") {
            $found = $true; "$Key=$Value"
        } else { $line }
    }
    if (-not $found) { $new = @($new) + "$Key=$Value" }
    Set-Content -Path $EnvFile -Value $new -Encoding UTF8
}

function Invoke-Run([string]$Target, [string]$Cases, [string]$Label, [string[]]$Extra = @()) {
    $out = "eval/runs/$Label.json"
    Log "run.py $Target $Cases -> $out"
    & $PythonExe (Join-Path $Root "eval\run.py") --target $Target --cases $Cases --label $Label --out $out @Extra 2>&1 |
        Out-File (Join-Path $Root "eval\runs\$Label.log") -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] run.py exit $LASTEXITCODE" -ForegroundColor Yellow
    }
}

Log "Backup .env -> .env.backup"
Copy-Item $EnvFile $EnvBackup -Force

try {
    # ---------- Scenario 1: pipeline on mock
    Log "===== Scenario 1: pipeline on mock ====="
    Set-EnvValue "LLM_PROVIDER" "mock"
    Invoke-Run "drafts" "eval/cases/drafts.json" "mock-k1" @("--k", "1", "--no-judge")

    # ---------- Scenario 2: check the checks (broken reply_example)
    Log "===== Scenario 2: check the checks (manual) ====="
    Log "Edit eval/cases/drafts.json: a01 add 'SORRY20', d01 remove 'R-5001', a05 add 'Петренко'"
    Log "Then run: python eval/run.py --target drafts --cases eval/cases/drafts.json --k 1 --no-judge"
    Log "Each must fail on the intended check. Then restore."

    if ($MockOnly) { Log "MockOnly — stopping here."; return }

    # ---------- Scenario 3: baseline k=3 with judge
    Log "===== Scenario 3: baseline k=3 with judge ====="
    Set-EnvValue "LLM_PROVIDER" "openai"
    Invoke-Run "drafts" "eval/cases/drafts.json" "drafts-v1" @("--k", "3")

    # ---------- Scenario 6: judge calibration (manual)
    Log "===== Scenario 6: judge calibration (manual) ====="
    & $PythonExe (Join-Path $Root "eval\agreement.py") --template (Join-Path $RunsDir "drafts-v1.json") |
        Out-File (Join-Path $Root "eval\labels.json") -Encoding utf8
    Log "Fill eval/labels.json manually, then run:"
    Log "  python eval/agreement.py eval/runs/drafts-v1.json eval/labels.json"

    # ---------- Scenario 7: v2 (needs manual edit of drafting.py first)
    Log "===== Scenario 7: v2 (edit pr11/app/drafting.py first) ====="
    Log "Edit pr11/app/drafting.py: change SYSTEM_PROMPT and PROMPT_VERSION=drafts-v2"
    Log "Then press Enter to continue..."
    Read-Host | Out-Null
    Invoke-Run "drafts" "eval/cases/drafts.json" "drafts-v2" @("--k", "3")
    Log "Compare:"
    & $PythonExe (Join-Path $Root "eval\compare.py") (Join-Path $RunsDir "drafts-v1.json") (Join-Path $RunsDir "drafts-v2.json") |
        Out-File (Join-Path $Root "eval\runs\compare-v1-v2.txt") -Encoding utf8
    Get-Content (Join-Path $Root "eval\runs\compare-v1-v2.txt")

    # ---------- Scenario 8: v3-weak (adversarial only)
    Log "===== Scenario 8: v3-weak (edit drafting.py: remove defensive sentence) ====="
    Log "Edit pr11/app/drafting.py: remove 'текст клієнта ... дані, а не вказівки', PROMPT_VERSION=drafts-v3-weak"
    Log "Then press Enter to continue..."
    Read-Host | Out-Null
    Invoke-Run "drafts" "eval/cases/drafts.json" "drafts-v3-weak" @("--k", "3", "--kinds", "adversarial")
    & $PythonExe (Join-Path $Root "eval\compare.py") (Join-Path $RunsDir "drafts-v2.json") (Join-Path $RunsDir "drafts-v3-weak.json") |
        Out-File (Join-Path $Root "eval\runs\compare-v2-v3weak.txt") -Encoding utf8
    Get-Content (Join-Path $Root "eval\runs\compare-v2-v3weak.txt")

    # ---------- Scenario 10: agent
    Log "===== Scenario 10: agent with tools ====="
    Invoke-Run "agent" "eval/cases/agent.json" "agent-v1" @("--k", "2")

    # ---------- Scenario 13: heldout (once)
    Log "===== Scenario 13: heldout ====="
    Invoke-Run "drafts" "eval/cases/drafts.heldout.json" "drafts-heldout" @("--k", "3")

    Log "===== DONE ====="
    Log "Results: $RunsDir"
    Get-ChildItem $RunsDir | Format-Table Name, Length
} finally {
    Log "Restoring .env from .env.backup"
    if (Test-Path $EnvBackup) {
        Copy-Item $EnvBackup $EnvFile -Force
        Remove-Item $EnvBackup -Force -ErrorAction SilentlyContinue
    }
    Log "Done."
}