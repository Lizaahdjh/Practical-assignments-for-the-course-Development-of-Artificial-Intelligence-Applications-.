<#
.SYNOPSIS
    Автоматичний прогін сценаріїв ПР11 (mock-провайдер).

.DESCRIPTION
    Проганяє сценарії 2, 3, 7, 8, 13–14, 18 з eval/README.md:
      • зберігає .env як .env.backup;
      • для кожного сценарію: змінює .env, перезапускає uvicorn,
        очищає logs/ і state/, запускає дії, збирає analyze.py;
      • усе пише в eval/notes/;
      • в кінці відновлює .env з .env.backup.

    Сценарій 17 (справжній Gemini) — окремо, у run_openai.ps1.

.EXAMPLE
    # з папки pr11
    .\scripts\run_all.ps1
#>

$ErrorActionPreference = "Stop"

# ---- Шляхи
$Root        = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvFile     = Join-Path $Root ".env"
$EnvBackup   = Join-Path $Root ".env.backup"
$NotesDir    = Join-Path $Root "eval\notes"
$LogFile     = Join-Path $Root "logs\app.jsonl"
$StateDir    = Join-Path $Root "state"
$PythonExe   = Join-Path $Root ".venv\Scripts\python.exe"
$UvicornPort = 8000
$BaseUrl     = "http://127.0.0.1:$UvicornPort"

function Log($msg) {
    $stamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$stamp] $msg" -ForegroundColor Cyan
}

function Fail($msg) {
    Write-Host "[ERR] $msg" -ForegroundColor Red
    throw $msg
}

# ---- Перевірки на старті
if (-not (Test-Path $PythonExe)) {
    Fail "Не знайдено .venv\Scripts\python.exe. Активуйте venv."
}
if (-not (Test-Path $EnvFile)) {
    Fail "Не знайдено .env. Скопіюйте .env.example у .env."
}
New-Item -ItemType Directory -Force -Path $NotesDir | Out-Null

# ---- Управління uvicorn
$script:UvicornProc = $null

function Stop-Uvicorn {
    if ($script:UvicornProc -and -not $script:UvicornProc.HasExited) {
        try {
            Stop-Process -Id $script:UvicornProc.Id -Force -ErrorAction SilentlyContinue
            $script:UvicornProc.WaitForExit(5000) | Out-Null
        } catch {}
    }
    # На всяк випадок — прибираємо «висячі» python на порту
    $conn = Get-NetTCPConnection -LocalPort $UvicornPort -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $conn | ForEach-Object {
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
    $script:UvicornProc = $null
}

function Start-Uvicorn {
    Stop-Uvicorn
    Log "Запускаю uvicorn на $BaseUrl ..."
    $script:UvicornProc = Start-Process -FilePath $PythonExe `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$UvicornPort" `
        -WorkingDirectory $Root -PassThru -WindowStyle Hidden

    # Чекаємо, поки /api/health відповість (до 20 с)
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "$BaseUrl/api/health" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) {
                Log "uvicorn готовий."
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    Fail "uvicorn не піднявся за 20 с. Перевірте логи в консолі."
}

# ---- .env: читання/запис окремих ключів
function Set-EnvValue([string]$Key, [string]$Value) {
    $lines = Get-Content $EnvFile
    $found = $false
    $new = foreach ($line in $lines) {
        if ($line -match "^\s*$([regex]::Escape($Key))\s*=") {
            $found = $true
            "$Key=$Value"
        } else {
            $line
        }
    }
    if (-not $found) {
        $new = @($new) + "$Key=$Value"
    }
    Set-Content -Path $EnvFile -Value $new -Encoding UTF8
}

function Set-EnvValues([hashtable]$Values) {
    foreach ($k in $Values.Keys) {
        Set-EnvValue $k $Values[$k]
    }
}

# ---- Скидання стану
function Reset-State {
    if (Test-Path $LogFile) {
        Remove-Item $LogFile -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $StateDir) {
        Get-ChildItem $StateDir -Filter "*.json" -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }
    # Дочекатися, щоб файл журналу точно зник
    Start-Sleep -Milliseconds 300
}

# ---- Аналіз журналу в файл
function Save-Analyze([string]$Name, [string[]]$ExtraArgs = @()) {
    $out = Join-Path $NotesDir $Name
    Log "analyze.py → $Name"
    & $PythonExe (Join-Path $Root "eval\analyze.py") @ExtraArgs 2>&1 |
        Out-File -FilePath $out -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] analyze.py завершився з кодом $LASTEXITCODE" -ForegroundColor Yellow
    }
}

# ---- replay.py з опціями
function Invoke-Replay([string]$Name, [string[]]$ExtraArgs = @()) {
    $outJson = Join-Path $Root "eval\$Name.json"
    $outTxt  = Join-Path $NotesDir "$Name.txt"
    Log "replay.py → $Name.json"
    & $PythonExe (Join-Path $Root "eval\replay.py") @ExtraArgs `
        "--out" $outJson 2>&1 | Tee-Object -FilePath $outTxt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] replay.py завершився з кодом $LASTEXITCODE" -ForegroundColor Yellow
    }
}

# ---- «Траса» — усі записи за draft_id
function Save-Trace([string]$DraftId, [string]$Name) {
    $out = Join-Path $NotesDir $Name
    Log "Траса $DraftId → $Name"
    if (Test-Path $LogFile) {
        Select-String -Path $LogFile -Pattern $DraftId -SimpleMatch |
            ForEach-Object { $_.Line } |
            Out-File -FilePath $out -Encoding utf8
    } else {
        "(журналу немає)" | Out-File -FilePath $out -Encoding utf8
    }
}

# ---- «grep» на секрети
function Save-Grep([string]$Name) {
    $out = Join-Path $NotesDir $Name
    $pattern = "Бондар|Коваль|Мельник|10412|10421|10458|10463|10377|" +
               "20450012345678|R-5001|R-5002|AIza|sk-|Bearer"
    Log "grep секретів → $Name"
    $hits = @()
    if (Test-Path $LogFile) {
        $hits = Select-String -Path $LogFile -Pattern $pattern
    }
    if ($hits.Count -eq 0) {
        "(порожньо — жодного збігу)" | Out-File -FilePath $out -Encoding utf8
    } else {
        $hits | ForEach-Object { $_.Line } | Out-File -FilePath $out -Encoding utf8
        Write-Host "[WARN] Знайдено $($hits.Count) збігів — див. $Name" -ForegroundColor Yellow
    }
}

# ---- Отримати останню чернетку за request_id з трьома спробами
function Find-ThreeAttemptDraft {
    if (-not (Test-Path $LogFile)) { return $null }
    $lines = Get-Content $LogFile | Where-Object { $_ -match '"event": "draft.done"' }
    foreach ($l in $lines) {
        $j = $l | ConvertFrom-Json
        if ($j.attempts -ge 3) { return $j.draft_id }
    }
    return $null
}

# ================================================================ MAIN
Log "Початок прогону. Бекап .env → .env.backup"
Copy-Item $EnvFile $EnvBackup -Force

try {
    # ---------- Сценарій 2: три спроби
    Log "===== Сценарій 2: три спроби (mock, failure_rate=0.6) ====="
    Set-EnvValues @{
        LLM_PROVIDER         = "mock"
        LLM_FALLBACK         = "none"
        MOCK_FAILURE_RATE    = "0.6"
        MOCK_FAILURE_KIND    = "unavailable"
        MOCK_BILLABLE        = "1"
        CACHE_ENABLED        = "0"
        RATE_LIMIT_PER_MINUTE= "100"
        BUDGET_DAILY_USD     = "1.00"
        BUDGET_WARN_AT       = "0.8"
    }
    Reset-State
    Start-Uvicorn

    # Замовляємо 10 чернеток — статистично 2-3 матимуть 3 спроби
    Invoke-Replay "s02-retries" @("--orders", "10", "--seed", "7", "--fresh")
    Save-Analyze "02-retries-failed.txt"  @("--section", "failed")
    Save-Analyze "02-retries-repeated.txt" @("--section", "repeated")
    Save-Analyze "02-retries-slow.txt"     @("--section", "slow")

    $threeAttemptDraft = Find-ThreeAttemptDraft
    if ($threeAttemptDraft) {
        Save-Trace $threeAttemptDraft "02-trace.txt"
        Log "Траса трьох спроб збережена (draft=$threeAttemptDraft)."
    } else {
        Log "Не знайдено чернетки з 3 спробами — можливо, seed інший. Пропускаю трасу."
        "(не знайдено чернетки з attempts>=3)" |
            Out-File (Join-Path $NotesDir "02-trace.txt") -Encoding utf8
    }

    # ---------- Сценарій 3: секрети
    Log "===== Сценарій 3: перевірка на секрети ====="
    Save-Grep "03-grep.txt"

    # ---------- Сценарій 7: без кешу
    Log "===== Сценарій 7: без кешу ====="
    Set-EnvValues @{
        LLM_PROVIDER         = "mock"
        MOCK_FAILURE_RATE    = "0"
        CACHE_ENABLED        = "0"
        RATE_LIMIT_PER_MINUTE= "100"
        BUDGET_DAILY_USD     = "1.00"
    }
    Reset-State
    Start-Uvicorn
    Invoke-Replay "s07-nocache" @("--orders", "12", "--seed", "1")
    Save-Analyze "07-nocache-analyze.txt"

    # ---------- Сценарій 8: з кешем
    Log "===== Сценарій 8: з кешем ====="
    Set-EnvValues @{
        CACHE_ENABLED = "1"
    }
    Reset-State
    Start-Uvicorn
    Invoke-Replay "s08-cache" @("--orders", "12", "--seed", "1")
    Save-Analyze "08-cache-analyze.txt" @("--section", "cache")

    Log "Порівняння 7 vs 8:"
    & $PythonExe (Join-Path $Root "eval\replay.py") "--compare" `
        (Join-Path $Root "eval\s07-nocache.json") `
        (Join-Path $Root "eval\s08-cache.json") 2>&1 |
        Tee-Object -FilePath (Join-Path $NotesDir "07-vs-08.txt")

    # ---------- Сценарії 13–14: бюджет
    Log "===== Сценарії 13–14: бюджет ====="
    Set-EnvValues @{
        LLM_PROVIDER         = "mock"
        MOCK_FAILURE_RATE    = "0"
        MOCK_BILLABLE        = "1"
        CACHE_ENABLED        = "0"
        RATE_LIMIT_PER_MINUTE= "100"
        BUDGET_DAILY_USD     = "0.003"
        BUDGET_WARN_AT       = "0.8"
    }
    Reset-State
    Start-Uvicorn
    Invoke-Replay "s13-budget" @("--orders", "20", "--seed", "2", "--fresh")
    Save-Analyze "13-14-budget-analyze.txt"
    if (Test-Path (Join-Path $StateDir "budget.json")) {
        Log "Стан бюджету після прогону:"
        Get-Content (Join-Path $StateDir "budget.json") |
            Out-File (Join-Path $NotesDir "13-14-budget-state-before-restart.txt") -Encoding utf8
        Get-Content (Join-Path $StateDir "budget.json")
    }

    Log "Перезапуск — бюджет має зберегтися."
    Start-Uvicorn
    Invoke-Replay "s14-restart" @("--orders", "3", "--seed", "3", "--fresh")
    Save-Analyze "14-restart-analyze.txt"
    if (Test-Path (Join-Path $StateDir "budget.json")) {
        Get-Content (Join-Path $StateDir "budget.json") |
            Out-File (Join-Path $NotesDir "14-budget-state-after-restart.txt") -Encoding utf8
    }

    # ---------- Сценарій 18: вартість без usage (обрив)
    Log "===== Сценарій 18: обрив потоку (mid_stream) ====="
    Set-EnvValues @{
        LLM_PROVIDER         = "mock"
        MOCK_FAILURE_RATE    = "1"
        MOCK_FAILURE_KIND    = "mid_stream"
        CACHE_ENABLED        = "0"
        BUDGET_DAILY_USD     = "1.00"
        RATE_LIMIT_PER_MINUTE= "100"
    }
    Reset-State
    Start-Uvicorn
    Invoke-Replay "s18-midstream" @("--orders", "3", "--seed", "4", "--fresh")
    Save-Analyze "18-midstream-analyze.txt"

    Log "===== ГОТОВО ====="
    Log "Результати: $NotesDir"
    Log "JSON прогонів: $Root\eval\*.json"
    Get-ChildItem $NotesDir | Format-Table Name, Length
} finally {
    Log "Зупиняю uvicorn."
    Stop-Uvicorn
    Log "Відновлюю .env з .env.backup"
    if (Test-Path $EnvBackup) {
        Copy-Item $EnvBackup $EnvFile -Force
        Remove-Item $EnvBackup -Force -ErrorAction SilentlyContinue
    }
    Log "Готово. .env відновлено."
}