Set-Location C:\Ai\coding_AI
.\venv\Scripts\Activate.ps1
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
$env:ADAPTER_PATH="__none__"
$env:QWEN_8BIT="1"; $env:QWEN_4BIT="0"; $env:TORCH_DTYPE="float16"
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
Write-Host "🚀 Ollama API 포함 v0 서버 시작..." -ForegroundColor Green
python -m uvicorn server.app:app --host 127.0.0.1 --port 8765 --workers 1 --log-level info
