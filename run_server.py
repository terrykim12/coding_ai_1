#!/usr/bin/env python3
"""
Qwen3-8B Local Coding AI 서버 실행 스크립트

사용법:
    python run_server.py [--host HOST] [--port PORT] [--reload]

예시:
    python run_server.py
    python run_server.py --host 0.0.0.0 --port 8765
    python run_server.py --reload
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_environment():
    """환경 설정"""
    # 환경 변수 설정
    os.environ.setdefault("QWEN_BASE_MODEL", "Qwen/Qwen3-8B")
    os.environ.setdefault("QWEN_4BIT", "true")
    # Windows에서 4-bit 강제 사용 (기본 비활성화 가드 우회)
    os.environ.setdefault("QWEN_FORCE_4BIT", "true")
    # 안정적 dtype
    os.environ.setdefault("TORCH_DTYPE", "float16")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 서버 디렉토리 확인
    server_dir = project_root / "server"
    if not server_dir.exists():
        print(f"Error: Server directory not found: {server_dir}")
        sys.exit(1)
    
    # 필요한 파일들 확인
    required_files = [
        "app.py",
        "model.py",
        "context.py",
        "patch_schema.py",
        "patch_apply.py",
        "test_runner.py",
        "debug_runtime.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (server_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        sys.exit(1)

def check_dependencies():
    """의존성 확인"""
    try:
        import fastapi
        import uvicorn
        print("✓ FastAPI and Uvicorn available")
    except ImportError:
        print("✗ FastAPI or Uvicorn not available")
        print("Install with: pip install fastapi uvicorn")
        return False
    
    try:
        import transformers
        print("✓ Transformers available")
    except ImportError:
        print("✗ Transformers not available")
        print("Install with: pip install transformers")
        return False
    
    try:
        import peft
        print("✓ PEFT available")
    except ImportError:
        print("✗ PEFT not available")
        print("Install with: pip install peft")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch available (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        print("✗ PyTorch not available")
        print("Install with: pip install torch")
        return False
    
    return True

def run_server(host: str, port: int, reload: bool):
    """서버 실행"""
    # uvicorn 모듈 자체 import 오류와 내부 ImportError를 구분해 진짜 원인을 보여줌
    try:
        import uvicorn  # noqa: F401
    except ImportError as e:
        print(f"Error: Uvicorn not available: {e}")
        sys.exit(1)

    try:
        import uvicorn

        app_path = "server.app:app"

        print(f"Starting server at http://{host}:{port}")
        print(f"App path: {app_path}")
        print(f"Reload mode: {'enabled' if reload else 'disabled'}")
        print("-" * 50)

        # 서버 시작
        uvicorn.run(app_path, host=host, port=port, reload=reload, log_level="info")

    except Exception as e:
        import traceback
        print(f"Error starting server: {e}")
        traceback.print_exc()
        sys.exit(1)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Qwen3-8B Local Coding AI 서버 실행"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="서버 호스트 (기본값: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="서버 포트 (기본값: 8765)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="자동 리로드 활성화"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="의존성만 확인하고 종료"
    )
    
    args = parser.parse_args()
    
    print("Qwen3-8B Local Coding AI Server")
    print("=" * 40)
    
    # 환경 설정
    setup_environment()
    
    # 의존성 확인
    if not check_dependencies():
        print("\nDependency check failed. Please install missing packages.")
        sys.exit(1)
    
    if args.check_deps:
        print("\n✓ All dependencies are available!")
        return
    
    # 서버 실행
    run_server(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()

