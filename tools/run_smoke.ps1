param(
  [int]$Port = 8765,
  [string]$AdapterPath = "__none__",
  [int]$WaitSeconds = 420,
  [switch]$NoKill
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root\..

function Write-Head($msg){ Write-Host $msg -ForegroundColor Yellow }
function Tail-IfExists($p,$n=60){ if(Test-Path $p){ Write-Host "`n===== tail $p =====" -ForegroundColor DarkCyan; Get-Content $p -Tail $n } }

Write-Head "Stopping existing python..."
if(-not $NoKill){ Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue }

Write-Head "Activating venv..."
$venvPy = Join-Path $PWD "venv\Scripts\python.exe"
if(!(Test-Path $venvPy)){ throw "venv python not found: $venvPy" }

# envs
$env:ADAPTER_PATH = $AdapterPath
$env:QWEN_4BIT = "0"
$env:QWEN_8BIT = "1"
$env:TORCH_DTYPE = "float16"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CLEAN_RESP = "1"

# logs
$logDir = "logs"; if(!(Test-Path $logDir)){ New-Item -ItemType Directory $logDir | Out-Null }
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "uvicorn_out_$ts.log"
$errLog = Join-Path $logDir "uvicorn_err_$ts.log"

Write-Head "Starting uvicorn..."
$args = @("-u","-m","uvicorn","server.app:app","--host","127.0.0.1","--port",$Port.ToString(),"--workers","1","--log-level","info","--no-server-header")
$proc = Start-Process -FilePath $venvPy -ArgumentList $args -WorkingDirectory $PWD -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru

Write-Head "Health check (waiting up to $WaitSeconds s)..."
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$healthy = $false
do {
  Start-Sleep -Seconds 2
  if ($proc.HasExited) {
    Write-Error "Uvicorn exited early with code $($proc.ExitCode)."
    Tail-IfExists $errLog 120; Tail-IfExists $outLog 80
    throw "Server failed to start."
  }
  try {
    $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2
    if($h.status -eq "healthy"){ $healthy = $true; break }
  } catch { }
} while ((Get-Date) -lt $deadline)

if(-not $healthy){
  Write-Error "Health did not reach 'healthy' within $WaitSeconds s."
  Tail-IfExists $errLog 120; Tail-IfExists $outLog 80
  throw "Start failed (timeout)."
}

$h | ConvertTo-Json -Depth 8 | Write-Output

Write-Head "Smoke: /plan"
$planBody = @{ intent="smoke"; paths=@("examples\sample_py"); code_paste="def add(a,b): return a+b" } | ConvertTo-Json -Depth 20
$plan = Invoke-RestMethod "http://127.0.0.1:$Port/plan" -Method Post -ContentType 'application/json' -Body $planBody -TimeoutSec 30
$plan | ConvertTo-Json -Depth 10 | Write-Output

Write-Head "Smoke: /patch_smart"
$fb = @{ hint="Return ONLY items of the edits array. No markdown."; reason="smoke" }
$patchBody = @{ plan=@{ edits=@() }; feedback=$fb } | ConvertTo-Json -Depth 40
$patch = Invoke-RestMethod "http://127.0.0.1:$Port/patch_smart" -Method Post -ContentType 'application/json' -Body $patchBody -TimeoutSec 30
$patch | ConvertTo-Json -Depth 10 | Write-Output

Write-Head "Smoke: Ollama-like /api/generate"
$gen = @{ prompt="Hello"; stream=$false; options=@{ num_predict=16; temperature=0.0 } } | ConvertTo-Json -Depth 10
$g = Invoke-RestMethod "http://127.0.0.1:$Port/api/generate" -Method Post -ContentType application/json -Body $gen -TimeoutSec 30
$g | ConvertTo-Json -Depth 6 | Write-Output

Write-Host "`nAll smoke checks passed." -ForegroundColor Green
Write-Host "Logs: $outLog / $errLog"


