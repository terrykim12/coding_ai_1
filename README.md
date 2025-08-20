# Qwen3-8B 로컬 코딩 보조 AI

> 목표: **Qwen3-8B**를 QLoRA로 파인튜닝해, Cursor/Continue처럼 *정확한 위치에 코드 수정 패치*를 생성·적용하고, 테스트/디버깅 루프를 자동화하는 로컬 코딩 보조 AI

## 🚀 주요 기능

- **정확한 위치 지정**: 앵커/라인-범위/AST 기반 코드 수정
- **자동 테스트**: pytest 실행 및 결과 분석
- **실시간 디버깅**: debugpy를 통한 VS Code 연동
- **컨텍스트 인텔리전스**: TF-IDF 기반 관련 코드 검색
- **견고한 패치 적용**: fuzz 매칭으로 안정적인 코드 수정

## 📁 프로젝트 구조

```
local-coding-ai/
├─ server/           # FastAPI 서버 (핵심 로직)
├─ training/         # 모델 학습 스크립트
├─ ui/              # 웹 UI (선택사항)
├─ examples/        # 샘플 프로젝트
└─ requirements.txt
```

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
python -m venv venv
.\\venv\\Scripts\\Activate.ps1  # Windows
pip install -r requirements.txt
# CUDA 12.1용 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 최신 bitsandbytes (Windows 4bit)
pip install bitsandbytes==0.43.3
```

### 2. 모델/양자화 고정 실행
- 환경 변수(권장):
```powershell
$env:MODEL_PATH="Qwen/Qwen3-8B"
$env:QWEN_4BIT="true"
$env:QWEN_FORCE_4BIT="true"
$env:CUDA_VISIBLE_DEVICES="0"
```
- 혹은 `.env` 사용: `env.lock` 내용을 참고해 동일 키를 `.env`에 작성

### 3. 서버 실행
```powershell
python run_server.py --host 127.0.0.1 --port 8765
```

### 4. 상태 확인
```powershell
Invoke-RestMethod http://127.0.0.1:8765/health | ConvertTo-Json -Depth 5
```

## 📊 Patch JSON 스키마

```json
{
  "version": "1",
  "edits": [
    {
      "path": "file.py",
      "loc": {
        "type": "anchor",
        "before": "def function():",
        "after": "return value"
      },
      "action": "replace_range",
      "range": {"start": {"line": 1, "col": 0}, "end": {"line": 5, "col": 0}},
      "code": "def function():\n    return new_value\n"
    }
  ]
}
```

## 🚨 주의사항
- Windows에서 4bit(bitsandbytes) 사용 시 0.43.2+ 권장
- VRAM 12GB 기준 Qwen3-8B 4bit 가능. 필요시 `bfloat16` 유지
- 환경 변수로 모델/양자화 고정 권장

## 🔗 VS Code 연동

1. `debugpy` 포트(5678)로 프로젝트 실행
2. VS Code에서 "Python: Attach using Port" 선택
3. 실시간 디버깅 및 변수 검사 가능

## 📄 라이선스

MIT License

