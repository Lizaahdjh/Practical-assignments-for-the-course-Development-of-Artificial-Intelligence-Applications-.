<#
.SYNOPSIS
    Сценарій 17: справжній провайдер (Gemini) з фактичним usage.

.DESCRIPTION
    Не входить до run_all.ps1, бо витрачає гроші й час (~9 с на запит).
    Запуск — коли є ключ і залишок бюджету ≥ 0.01 $.

.EXAMPLE
    .\scripts\run_openai.ps1
    .\scripts\run_openai.ps1 -Orders 5   # менше замовлень — швидше
#>

param(
    [int]$Orders = 10,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$Root      = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvFile   = Join-Path $Root ".env"
$EnvBackup = Join-Path $Root ".env.backup"
$NotesDir  = Join-Path $Root "eval\notes"
$LogFile   = Join-Path $Root "logs\app.jsonl"
$StateDir  = Join-Path $Root "state"
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
$BaseUrl   = "http://127.0.0.1:8000"

if (-not (Test-Path $PythonExe)) { throw "Немає .venv\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path $NotesDir | Out-Null

Write-Host "Бекап .env → .env.backup" -ForegroundColor Cyan
Copy-Item $EnvFile $EnvBackup -Force

$uvicorn = $null
try {
    # ---- Перевіряємо, що ключ є
    $env_text = Get-Content $EnvFile -Raw
    if ($env_text -notmatch "LLM_API_KEY=\S+") {
        throw "LLM_API_KEY порожній у .env — заповніть і повторіть."
    }

    # ---- Перемикаємо .env на openai
    $new = $env_text -replace '(?m)^LLM_PROVIDER=.*$', 'LLM_PROVIDER=openai'
    $new = $new -replace '(?m)^CACHE_ENABLED=.*$', 'CACHE_ENABLED=1'
    $new = $new -replace '(?m)^BUDGET_DAILY_USD=.*$', 'BUDGET_DAILY_USD=1.00'
    $new = $new -replace '(?m)^BUDGET_WARN_AT=.*$', 'BUDGET_WARN_AT=0.8'
    $new = $new -replace '(?m)^LLM_IDLE_TIMEOUT=.*$', 'LLM_IDLE_TIMEOUT=30'
    $new = $new -replace '(?m)^LLM_TIMEOUT=.*$', 'LLM_TIMEOUT=60'
    $new = $new -replace '(?m)^LLM_TOTAL_TIMEOUT=.*$', 'LLM_TOTAL_TIMEOUT=120'
    $new = $new -replace '(?m)^RATE_LIMIT_PER_MINUTE=.*$', 'RATE_LIMIT_PER_MINUTE=100'
    Set-Content $EnvFile $new -Encoding UTF8

    # ---- Чистимо logs/state
    if (Test-Path $LogFile) { Remove-Item $LogFile -Force }
    if (Test-Path $StateDir) {
        Get-ChildItem $StateDir -Filter "*.json" -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 300

    # ---- uvicorn
    Write-Host "Запуск uvicorn ..." -ForegroundColor Cyan
    $uvicorn = Start-Process -FilePath $PythonExe `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
        -WorkingDirectory $Root -PassThru -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "$BaseUrl/api/health" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { break }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $uvicorn -or $uvicorn.HasExited) { throw "uvicorn не піднявся." }

    # ---- replay
    Write-Host "Сценарій 17: $Orders замовлень, seed=$Seed, fresh" -ForegroundColor Cyan
    $outJson = Join-Path $Root "eval\s17-openai.json"
    & $PythonExe (Join-Path $Root "eval\replay.py") `
        "--orders" "$Orders" "--seed" "$Seed" "--fresh" "--out" $outJson `
        2>&1 | Tee-Object -FilePath (Join-Path $NotesDir "17-openai-replay.txt")

    # ---- аналіз
    Write-Host "Аналіз ..." -ForegroundColor Cyan
    & $PythonExe (Join-Path $Root "eval\analyze.py") "--section" "spend" `
        2>&1 | Tee-Object -FilePath (Join-Path $NotesDir "17-openai-spend.txt")
    & $PythonExe (Join-Path $Root "eval\analyze.py") "--section" "slow" `
        2>&1 | Tee-Object -FilePath (Join-Path $NotesDir "17-openai-slow.txt")
    & $PythonExe (Join-Path $Root "eval\analyze.py") "--section" "summary" `
        2>&1 | Tee-Object -FilePath (Join-Path $NotesDir "17-openai-summary.txt")

    Write-Host "Готово. Результати: $NotesDir\17-*" -ForegroundColor Green
} finally {
    if ($uvicorn -and -not $uvicorn.HasExited) {
        Stop-Process -Id $uvicorn.Id -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $EnvBackup) {
        Copy-Item $EnvBackup $EnvFile -Force
        Remove-Item $EnvBackup -Force -ErrorAction SilentlyContinue
        Write-Host ".env відновлено." -ForegroundColor Cyan
    }
}