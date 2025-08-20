# 🚀 CUDA 설정 가이드 - Qwen3-8B Local Coding AI

## 📋 시스템 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU (RTX 4000 시리즈 이상 권장)
- **VRAM**: 최소 8GB, 권장 12GB 이상
- **RAM**: 최소 16GB, 권장 32GB 이상

### 소프트웨어
- **OS**: Windows 10/11 (64비트)
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8 - 12.1

## 🔧 CUDA 설치 단계

### 1. NVIDIA 드라이버 설치
```bash
# 현재 드라이버 버전 확인
nvidia-smi

# 최신 드라이버 다운로드 (권장)
# https://www.nvidia.com/Download/index.aspx
```

### 2. CUDA Toolkit 설치
```bash
# CUDA 12.1 설치 (권장)
# https://developer.nvidia.com/cuda-12-1-0-download-archive

# 설치 후 환경 변수 확인
echo $CUDA_HOME
echo $PATH
```

### 3. cuDNN 설치
```bash
# cuDNN 8.9.7 설치 (CUDA 12.1용)
# https://developer.nvidia.com/cudnn

# 압축 해제 후 CUDA 폴더에 복사
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\
```

## 🐍 Python 환경 설정

### 1. 가상환경 생성
```bash
# Conda 사용 (권장)
conda create -n qwen-ai python=3.11
conda activate qwen-ai

# 또는 venv 사용
python -m venv qwen-ai
qwen-ai\Scripts\activate
```

### 2. PyTorch 설치 (CUDA 지원)
```bash
# CUDA 12.1용 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 설치 확인
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 3. 기타 의존성 설치
```bash
# 기본 패키지들
pip install -r requirements.txt

# 또는 단계별 설치
pip install fastapi uvicorn pydantic
pip install transformers accelerate peft
pip install bitsandbytes scikit-learn
pip install pytest debugpy requests
```

## ⚙️ 환경 변수 설정

### 1. .env 파일 생성
```bash
# 프로젝트 루트에 .env 파일 생성
cp env_example.txt .env
```

### 2. CUDA 관련 설정
```env
# CUDA 설정
DEVICE=cuda
TORCH_DTYPE=float16
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=8.9

# 모델 설정
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
ADAPTER_PATH=training/qlora-out/export

# 성능 최적화
MAX_LENGTH=2048
TEMPERATURE=0.7
TOP_P=0.9
DO_SAMPLE=True
```

## 🧪 CUDA 설치 확인

### 1. 기본 CUDA 확인
```bash
# CUDA 컴파일러 확인
nvcc --version

# GPU 정보 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 2. 성능 테스트
```bash
# GPU 성능 벤치마크 실행
python test_cuda_api.py

# 또는 간단한 테스트
python -c "
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.mm(x, y)
    print('GPU 연산 성공!')
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('CUDA 사용 불가')
"
```

## 🚀 서버 실행

### 1. 서버 시작
```bash
# CUDA 최적화된 서버 실행
python run_server.py --host 127.0.0.1 --port 8765

# 또는 직접 실행
python -m uvicorn server.app:app --host 127.0.0.1 --port 8765 --reload
```

### 2. CUDA 상태 확인
```bash
# 서버 상태 확인 (CUDA 정보 포함)
curl http://127.0.0.1:8765/health

# 또는 브라우저에서
# http://127.0.0.1:8765/health
```

## 🔍 문제 해결

### 1. 일반적인 CUDA 오류

#### CUDA out of memory
```bash
# GPU 메모리 정리
python -c "import torch; torch.cuda.empty_cache()"

# 모델 양자화 설정 조정
# .env 파일에서 TORCH_DTYPE=float16 설정
```

#### CUDA driver version is insufficient
```bash
# NVIDIA 드라이버 업데이트 필요
# https://www.nvidia.com/Download/index.aspx
```

#### PyTorch CUDA 버전 불일치
```bash
# 기존 PyTorch 제거
pip uninstall torch torchvision torchaudio

# 올바른 버전 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 성능 최적화

#### 메모리 사용량 최적화
```env
# .env 파일에서
TORCH_DTYPE=float16  # 16비트 정밀도
MAX_LENGTH=1024      # 컨텍스트 길이 제한
```

#### GPU 메모리 모니터링
```bash
# 실시간 GPU 메모리 모니터링
watch -n 1 nvidia-smi

# 또는 Windows에서
# nvidia-smi -l 1
```

## 📊 성능 벤치마크

### 1. CPU vs GPU 비교
```bash
# CPU 모드 테스트
DEVICE=cpu python test_cuda_api.py

# GPU 모드 테스트
DEVICE=cuda python test_cuda_api.py
```

### 2. 예상 성능 향상
- **CPU 모드**: PLAN 생성 30-60초, PATCH 생성 60-120초
- **GPU 모드**: PLAN 생성 5-15초, PATCH 생성 15-30초
- **성능 향상**: **3-5배 빠름** 🚀

## 🎯 최적화 팁

### 1. 모델 선택
- **Qwen2.5-7B**: 빠른 응답, 적은 메모리 (권장)
- **Qwen2.5-14B**: 더 정확한 응답, 더 많은 메모리
- **Qwen2.5-32B**: 최고 품질, 대용량 GPU 필요

### 2. 메모리 최적화
- **4-bit 양자화**: 메모리 사용량 75% 감소
- **Flash Attention 2**: 메모리 효율성 향상
- **Gradient Checkpointing**: 학습 시 메모리 절약

### 3. 배치 처리
- 여러 요청을 배치로 처리하여 GPU 활용도 향상
- 비동기 처리로 응답 시간 단축

## 🔗 유용한 링크

- [NVIDIA CUDA 다운로드](https://developer.nvidia.com/cuda-downloads)
- [PyTorch CUDA 설치](https://pytorch.org/get-started/locally/)
- [CUDA 아키텍처 호환성](https://developer.nvidia.com/cuda-gpus)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

## 📝 체크리스트

- [ ] NVIDIA 드라이버 설치
- [ ] CUDA Toolkit 설치
- [ ] cuDNN 설치
- [ ] Python 가상환경 생성
- [ ] PyTorch (CUDA) 설치
- [ ] 의존성 패키지 설치
- [ ] 환경 변수 설정
- [ ] CUDA 설치 확인
- [ ] 성능 테스트 실행
- [ ] 서버 실행 및 테스트

---

**💡 팁**: CUDA 설정이 완료되면 `python test_cuda_api.py`를 실행하여 전체 시스템을 테스트하세요!
