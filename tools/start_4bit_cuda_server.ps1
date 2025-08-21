# 4비트 양자화 + CUDA GPU 서버 시작 스크립트
# start_4bit_cuda_server.ps1

# UTF-8 설정
. .\tools\utf8.ps1

# 4비트 양자화 환경변수 설정
$env:QWEN_4BIT = 'true'
$env:QWEN_FORCE_4BIT = 'true'
$env:TORCH_DTYPE = 'float16'
$env:CUDA_VISIBLE_DEVICES = '0'

# 환경변수 확인
Write-Host "=== 4비트 양자화 + CUDA GPU 서버 설정 ===" -ForegroundColor Cyan
Write-Host "QWEN_4BIT: $env:QWEN_4BIT" -ForegroundColor Green
Write-Host "QWEN_FORCE_4BIT: $env:QWEN_FORCE_4BIT" -ForegroundColor Green
Write-Host "TORCH_DTYPE: $env:TORCH_DTYPE" -ForegroundColor Green
Write-Host "CUDA_VISIBLE_DEVICES: $env:CUDA_VISIBLE_DEVICES" -ForegroundColor Green

# 기존 서버 프로세스 종료
Write-Host "기존 서버 프로세스 확인 중..." -ForegroundColor Yellow
$existingProcesses = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($existingProcesses) {
    Write-Host "포트 8765에서 실행 중인 프로세스 발견, 종료 중..." -ForegroundColor Yellow
    foreach ($pid in $existingProcesses) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Write-Host "프로세스 $pid 종료 완료" -ForegroundColor Green
        } catch {
            Write-Host "프로세스 $pid 종료 실패: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    Start-Sleep -Seconds 2
}

# 서버 시작
Write-Host "4비트 양자화 + CUDA GPU 서버 시작 중..." -ForegroundColor Green
Write-Host "서버 URL: http://127.0.0.1:8765" -ForegroundColor Cyan
Write-Host "Health 체크: http://127.0.0.1:8765/health" -ForegroundColor Cyan
Write-Host "종료하려면 Ctrl+C를 누르세요" -ForegroundColor Red
Write-Host "=" * 60 -ForegroundColor Gray

try {
    # 서버 시작
    python run_server.py --host 127.0.0.1 --port 8765
} catch {
    Write-Host "서버 시작 실패: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
