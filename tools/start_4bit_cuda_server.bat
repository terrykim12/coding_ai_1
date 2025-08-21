@echo off
REM 4비트 양자화 + CUDA GPU 서버 시작 배치 파일
REM start_4bit_cuda_server.bat

echo === 4비트 양자화 + CUDA GPU 서버 설정 ===

REM 4비트 양자화 환경변수 설정
set QWEN_4BIT=true
set QWEN_FORCE_4BIT=true
set TORCH_DTYPE=float16
set CUDA_VISIBLE_DEVICES=0

REM 환경변수 확인
echo QWEN_4BIT: %QWEN_4BIT%
echo QWEN_FORCE_4BIT: %QWEN_FORCE_4BIT%
echo TORCH_DTYPE: %TORCH_DTYPE%
echo CUDA_VISIBLE_DEVICES: %CUDA_VISIBLE_DEVICES%

REM 기존 서버 프로세스 종료
echo 기존 서버 프로세스 확인 중...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765') do (
    echo 포트 8765에서 실행 중인 프로세스 발견, 종료 중... PID: %%a
    taskkill /PID %%a /F >nul 2>&1
)

REM 잠시 대기
timeout /t 2 /nobreak >nul

REM 서버 시작
echo 4비트 양자화 + CUDA GPU 서버 시작 중...
echo 서버 URL: http://127.0.0.1:8765
echo Health 체크: http://127.0.0.1:8765/health
echo 종료하려면 Ctrl+C를 누르세요
echo ============================================================

REM 가상환경 활성화 및 서버 시작
call venv\Scripts\activate.bat
python run_server.py --host 127.0.0.1 --port 8765

pause
