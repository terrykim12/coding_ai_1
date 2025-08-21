param(
  [string]$Port = "8765"
)
$ErrorActionPreference = "Stop"

Write-Host "Stopping existing python..." -ForegroundColor DarkGray
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "Activating venv..." -ForegroundColor DarkGray
.\n+venv\Scripts\Activate.ps1

$env:ADAPTER_PATH="__none__"
$env:QWEN_8BIT="1"; $env:QWEN_4BIT="0"; $env:TORCH_DTYPE="float16"

Write-Host "Starting uvicorn..." -ForegroundColor DarkGray
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList @("-m","uvicorn","server.app:app","--host","127.0.0.1","--port",$Port,"--workers","1","--log-level","info")

Start-Sleep -Seconds 2

Write-Host "Health check..." -ForegroundColor Cyan
$h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 15
$h | ConvertTo-Json -Depth 6 | Write-Output

Write-Host "PLAN smoke..." -ForegroundColor Cyan
$body = @{ intent="smoke"; paths=@("examples\sample_py"); code_paste="def add(a,b): return a+b" } | ConvertTo-Json -Depth 20
Invoke-RestMethod "http://127.0.0.1:$Port/plan" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 25 | ConvertTo-Json -Depth 10 | Write-Output

Write-Host "PATCH_SMART smoke..." -ForegroundColor Cyan
$fb = @{ hint="Return ONLY items of the edits array. No markdown."; reason="smoke" }
$payload = @{ plan=@{ edits=@() }; feedback=$fb } | ConvertTo-Json -Depth 50
Invoke-RestMethod "http://127.0.0.1:$Port/patch_smart" -Method Post -ContentType 'application/json' -Body $payload -TimeoutSec 25 | ConvertTo-Json -Depth 10 | Write-Output

Write-Host "Ollama-like generate smoke..." -ForegroundColor Cyan
$g = @{ prompt="Hello"; stream=$false; options=@{ num_predict=16; temperature=0.0 } } | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://127.0.0.1:$Port/api/generate" -Method Post -ContentType application/json -Body $g -TimeoutSec 30 | ConvertTo-Json -Depth 10 | Write-Output

Write-Host "Done." -ForegroundColor Green


