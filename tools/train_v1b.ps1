param(
    [string]$AdapterName = "adapter_v1b",
    [int]$BenchN = 20,
    [string]$Functions = "add,divide,fibonacci",
    [int]$HealthTimeoutSec = 120
)

# --- Resolve repo root & venv ---
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) { throw "venv not found at .\\venv\\Scripts\\Activate.ps1" }
& .\venv\Scripts\Activate.ps1

# --- Stop running server(s) ---
Write-Host "[1/7] Stopping existing python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# --- Build dataset from success logs ---
Write-Host "[2/7] Building dataset from success_logs..." -ForegroundColor Yellow
python tools\build_dataset_from_logs.py

# --- Sanitize JSONL (ensure string types for input/output) ---
if (Test-Path "tools\sanitize_sft_jsonl.py") {
  Write-Host "[3/7] Sanitizing train/val JSONL..." -ForegroundColor Yellow
  python tools\sanitize_sft_jsonl.py training\data\train.jsonl training\data\val.jsonl
} else {
  Write-Host "[3/7] Skipping sanitize (tools\\sanitize_sft_jsonl.py not found)" -ForegroundColor DarkYellow
}

# --- Train (QLoRA 1 epoch) ---
Write-Host "[4/7] Training QLoRA adapter (v1b)..." -ForegroundColor Yellow
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$trainScript = if (Test-Path "training\train_sft_qwen3_8b.py") { "training\train_sft_qwen3_8b.py" } elseif (Test-Path "training\train_sft_qwen3.py") { "training\train_sft_qwen3.py" } else { $null }
if (-not $trainScript) { throw "train script not found (training\\train_sft_qwen3_8b.py or training\\train_sft_qwen3.py)" }
python $trainScript training\configs\qlora_qwen3_8b.json

# --- Version the adapter output ---
Write-Host "[5/7] Versioning adapter output..." -ForegroundColor Yellow
$baseAdapter = "training\qlora-out\adapter"
if (-not (Test-Path $baseAdapter)) { throw "Adapter output not found at $baseAdapter" }
$dest = "training\qlora-out\$AdapterName"
if (Test-Path $dest) { $dest = "training\qlora-out\${AdapterName}_$(Get-Date -Format yyyyMMdd_HHmmss)" }
Copy-Item $baseAdapter $dest -Recurse -Force
Write-Host (" -> saved as {0}" -f $dest) -ForegroundColor Green

# --- Start server with v1b adapter (background) ---
Write-Host "[6/7] Starting server on :8765 with ADAPTER_PATH=$dest ..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
$env:ADAPTER_PATH = $dest
$env:QWEN_8BIT = "1"; $env:QWEN_4BIT = "0"
$env:TORCH_DTYPE = "float16"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$log = "server_8765.log"
Start-Process python -ArgumentList "-m uvicorn server.app:app --host 127.0.0.1 --port 8765 --workers 1 --log-level info" -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError $log | Out-Null

# Wait for health
$ok=$false
for($i=0;$i -lt $HealthTimeoutSec;$i++){
  Start-Sleep -Seconds 1
  try {
    $h = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 3
    if ($h.status -eq "healthy") { $ok=$true; break }
  } catch {}
}
if (-not $ok) {
  Write-Host "Health check failed; printing last 40 log lines:" -ForegroundColor Red
  if (Test-Path $log) { Get-Content $log -Tail 40 }
  throw "Server did not become healthy in $HealthTimeoutSec s"
}
Write-Host "Health OK:" -ForegroundColor Green
(Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 5 | ConvertTo-Json -Depth 6)

# --- Bench v1b ---
Write-Host "[7/7] Benchmarking v1b..." -ForegroundColor Yellow
$fnList = $Functions.Split(",") | % { $_.Trim() } | Where-Object { $_ }
foreach($fn in $fnList){
  Write-Host (" -> {0} x{1}" -f $fn, $BenchN)
  .\tools\bench.ps1 -N $BenchN -Path "examples\sample_py\app.py" -FunctionName $fn | Out-Null
  if (Test-Path "bench_results.csv") {
    $target = "bench_results_v1b_${fn}.csv"
    Move-Item -Force bench_results.csv $target
    # Quick p50 summary
    $rows = Import-Csv $target
    $plan = $rows | % { [double]$_.'plan_ms' }
    $patch = $rows | % { [double]$_.'patch_ms' }
    function Get-P50($arr){ if($arr.Count -eq 0){ return $null }; $a = $arr | Sort-Object; if($a.Count % 2 -eq 1){ return [double]$a[[int](($a.Count-1)/2)] } else { return ([double]$a[($a.Count/2 - 1)] + [double]$a[($a.Count/2)]) / 2.0 } }
    $p50p = Get-P50 $plan
    $p50q = Get-P50 $patch
    $okCnt = ($rows | Where-Object { $_.ok -eq 'True' }).Count
    $tot = $rows.Count
    Write-Host ("    p50(plan)={0} ms, p50(patch)={1} ms, ok={2}/{3}" -f [int]$p50p,[int]$p50q,$okCnt,$tot) -ForegroundColor Cyan
  }
}

Write-Host "`nAll done. Adapter: $dest" -ForegroundColor Green
Write-Host "Tip: Compare against v0 by restarting with ADAPTER_PATH=__none__ and rerunning bench.ps1." -ForegroundColor DarkGray
