## Qwen3-8B 코드 패치 서버(로컬)

> FastAPI 기반 PLAN/PATCH API + QLoRA 어댑터 자동 로드 + 8-bit 양자화.
> 목표: 코드 스니펫을 받아 수정 계획(PLAN)과 실제 패치(PATCH)를 안정적·빠르게 생성.

---

## TL;DR (운영 순서)

1. venv 활성화 후 단일 워커로 서버 실행(8-bit):

```powershell
Set-Location C:\Ai\coding_AI
.\venv\Scripts\Activate.ps1
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:QWEN_4BIT="0"; $env:QWEN_8BIT="1"; $env:TORCH_DTYPE="float16"
python -m uvicorn server.app:app --host 127.0.0.1 --port 8765 --workers 1 --log-level info
```

2. /health 확인 → model_loaded=true, quantization=8bit(기본), adapter_path 노출.
3. PLAN/ PATCH 호출(스니펫 위주, 컨텍스트 1200자 컷).
4. 성공 로그 40~100건 적립 → 변환 → 1epoch QLoRA → 서버 재기동.
5. 벤치 20회로 p50/실패율 기록 → 필요 시 파라미터 미세 튜닝.

---

## 기능 요약

- PLAN: 수정 계획 생성(조기 종료: 균형 중괄호/완결 시 stop)
- PATCH_SMART: 편집 JSON 생성·검증·드라이런(AST)·아이템포턴시(ledger)·롤백
- 성공 로그 파이프라인: `/patch_smart` 성공 케이스 자동 적립 → SFT 변환 → QLoRA
- 어댑터 자동 로드: `ADAPTER_PATH` 없으면 `training/qlora-out/adapter`
- 양자화: 기본 8-bit(int8), 4-bit(NF4) A/B 가능
- 운영 보일러플레이트: 단일 워커, 1회 로드, `/health` 노출, 벤치 스크립트

---

## 요구 사항

- Windows 10/11, PowerShell 7 권장(PS5.1은 `tools/utf8.ps1`로 UTF-8 강제)
- NVIDIA GPU ≥ 12GB (예: RTX 4070 SUPER)
- Python 3.10+
- CUDA 12.x, PyTorch, transformers, peft, bitsandbytes

---

## 서버/모델 로딩 안정화 (적용됨)

- 단일 워커: `--workers 1` (리로드 금지) → 요청마다 모델 재로딩 방지
- 앱 시작 시 1회 로드 → `app.state` 공유
- 8-bit 강제: `BitsAndBytesConfig(load_in_8bit=True)`
- 디바이스 확정: `device_map={"":0}` 로 CPU offloading 방지
- 메모리 단편화 완화: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- /health 필드: `model_loaded`, `quantization`, `use_4bit`, `adapter_path`, `adapter_version`, `clean_response`, `stop_think_default`

---

## PLAN/PATCH 기본값 (권장 고정)

- PLAN
  - `max_new_tokens=16`, `repetition_penalty=1.10`, `do_sample=False`
  - `max_time=10~12s`, `CTX_LIMIT=1200`, tokenizer `truncation=True, max_length=1024`
  - 샘플링 파라미터(temperature/top_k/top_p) 전달 금지
- PATCH_SMART
  - `max_new_tokens=192`(필요 시 160/224 A/B), `repetition_penalty=1.03`
  - `max_time=25s`, JsonEditsClosed + WallClockBudget
- 스니펫 다이어트: 함수 ±20~30줄 위주, `paths=@()`(디렉터리 스캔 OFF) 옵션 제공

---

## API 사용 예 (PowerShell)

```powershell
$port=8765
# 스니펫 예시(함수 ±20줄)
$src = "def add(a,b): return a+b"
$planBody = @{ intent="add() 음수 방지"; paths=@(); code_paste=$src } | ConvertTo-Json -Depth 40 -Compress
$plan = irm "http://127.0.0.1:$port/plan" -Method Post -ContentType 'application/json; charset=utf-8' -Body $planBody -TimeoutSec 25

$planObj = if ($plan.plan -is [string]) { $plan.plan | ConvertFrom-Json } else { $plan.plan }
$fb = @{ hint="Return ONLY items of the edits array. No markdown."; reason="smoke" }
$patchBody = @{ plan=$planObj; feedback=$fb } | ConvertTo-Json -Depth 100
$patch = irm "http://127.0.0.1:$port/patch_smart" -Method Post -ContentType 'application/json; charset=utf-8' -Body $patchBody -TimeoutSec 40
```

---

## Ollama-호환 API (간이)

- 엔드포인트
  - `POST /api/generate` — 단발 프롬프트
  - `POST /api/chat` — 대화 메시지 배열

- 요청 예시

```powershell
# 1) 단발
$body = @{ prompt="Hello"; stream=$false; options=@{ num_predict=24; temperature=0.0 } } | ConvertTo-Json -Depth 10
irm http://127.0.0.1:8765/api/generate -Method Post -ContentType application/json -Body $body -TimeoutSec 40

# 2) 채팅
$chat = @{ messages = @(@{ role="user"; content="Say hello" }); stream = $false; options = @{ num_predict = 24; temperature = 0.0 } } | ConvertTo-Json -Depth 10
irm http://127.0.0.1:8765/api/chat -Method Post -ContentType application/json -Body $chat -TimeoutSec 40
```

- options
  - `num_predict`(기본 24), `max_time`(기본 20), `stop_think`(기본 false), `system`(기본 no-think 시스템)

- 운영 플래그(환경변수)
  - `CLEAN_RESP=1|0` — 응답 클린업 토글(기본 1)
  - `OLLAMA_CTX_CHARS` — 프롬프트 길이 컷(기본 2000)
  - `ADAPTER_PATH` — PEFT 어댑터 경로(`__none__`이면 미적용)

- 롤백/안전장치
  - 클린 후 빈 응답이면 약한 클린 → 인용구 → 원문 순 폴백
  - 필요 시 클린 비활성: `CLEAN_RESP=0`

---

## 어댑터 자동 로드 / A·B 토글

- 자동 로드: `ADAPTER_PATH` 미설정 시 `training/qlora-out/adapter`
- 명시 지정:

```powershell
$env:ADAPTER_PATH="training\qlora-out\adapter_v1"
python -m uvicorn server.app:app --host 127.0.0.1 --port 8765 --workers 1 --log-level info
```

- `/health`에 어댑터 경로/버전 노출 권장

---

## 벤치마크(bench)

- 스크립트: `tools/bench.ps1` (Stopwatch 계측 포함)
- 주의: 테스트 대상이 이미 고쳐져 있으면 `edits=[]` → ok=False → 벤치 무의미. 반복 전 고장난 베이스라인 리셋 또는 별도 케이스 사용.
- 예시 실행:

```powershell
.\tools\bench.ps1 -N 20 -Path "examples\sample_py\app.py" -FunctionName add
```

- 산출물: `bench_results.csv`(반복 결과), `bench_history.csv`(요약 p50/율 라인)
- 튜닝 후 대표 수치: PLAN p50 ≈ 7.8s, PATCH p50 ≈ 0.66s, JSON 실패 0%, 에러 0%

---

## 성공 로그 → SFT → QLoRA 학습

1) 로그 적립

```powershell
1..40 | % { .\tools\log_only.ps1 }
# 누적 개수
(Get-ChildItem training\success_logs\*.jsonl | % { (Get-Content $_ -Encoding UTF8 | Measure-Object -Line).Lines } | Measure-Object -Sum).Sum
```

2) SFT 변환

```powershell
python tools\build_dataset_from_logs.py
# 출력: training\data\train_from_logs.jsonl / val_from_logs.jsonl
```

3) (옵션) 병합

```powershell
Copy-Item training\data\train.jsonl training\data\train.base.jsonl
Copy-Item training\data\val.jsonl   training\data\val.base.jsonl
Get-Content training\data\train.base.jsonl, training\data\train_from_logs.jsonl | Set-Content -Encoding UTF8 training\data\train.jsonl
Get-Content training\data\val.base.jsonl,   training\data\val_from_logs.jsonl   | Set-Content -Encoding UTF8 training\data\val.jsonl
```

4) 스키마 정규화(중요) — PyArrow 타입 혼합 에러 방지

- `tools/sanitize_sft_jsonl.py`: `input/output`이 dict/list면 `json.dumps()`로 항상 문자열화

```powershell
python tools\sanitize_sft_jsonl.py training\data\train.jsonl training\data\val.jsonl
```

5) 학습(1epoch, QLoRA)

```powershell
# 단일 GPU에서는 학습 시 서버 종료 권장
python training\train_sft_qwen3_8b.py training\configs\qlora_qwen3_8b.json
```

6) 서버 재기동(어댑터 로드)

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
.\venv\Scripts\Activate.ps1
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:QWEN_4BIT="0"; $env:QWEN_8BIT="1"; $env:TORCH_DTYPE="float16"
python -m uvicorn server.app:app --host 127.0.0.1 --port 8765 --workers 1 --log-level info
```

과적합 방지 팁

- 성공 로그 10~20%를 holdout으로 분리(학습 제외)
- 중복 샘플 제거(해시)
- `lora_r=8~16`, `lora_dropout≈0.1`, `epoch=1` 유지

---

## 문제 → 원인 → 해결 (레퍼런스)

### /plan 타임아웃·500
- 원인: 양자화 미적용(bf16 풀가중치) OOM, 요청마다 재로딩, 컨텍스트 과대
- 해결: 8bit 강제 + `/health.quantization` 확인, 앱 시작 1회 로드 공유, `CTX_LIMIT` + `paths=@()` + `max_time` 하드컷

### generation flags not valid
- 원인: `do_sample=False`에서 `temperature/top_k/top_p` 전달
- 해결: 샘플링 파라미터 제거

### PATCH JSON 파싱 실패/엉뚱 액션
- 원인: 스키마 미정, 앵커/프리컨디션 붕괴
- 해결: 스키마 고정, AST 검사+드라이런, `_pre_match` 보정, 경로 보정, 실패 시 422/409

### 벤치 p50=0, edits=0
- 원인: ms 미기록, 테스트 대상 이미 정상
- 해결: Stopwatch 계측 추가, 반복 전 고장난 베이스라인 리셋

### SFT 로딩(PyArrow) 에러
- 원인: `output` 타입 혼합(dict/list/str)
- 해결: `sanitize_sft_jsonl.py`로 문자열화(+ 변환 파이프라인에도 동일 처리)

### 인코딩(모지바케)
- 원인: PS5.1 기본 CP949
- 해결: PowerShell 7 권장, PS5.1이면 `tools/utf8.ps1`

---

## 운영 플레이북 (Day‑2)

1. venv 활성화 → 서버(8bit, 단일 워커) 기동
2. `/health` 확인 → 스모크 1회
3. 벤치 10~20회(p50/율 기록)
4. 성공 로그 40~100 적립 → 변환 → 1epoch → 재벤치
5. 필요 시 `GEN_PLAN/GEN_PATCH` 미세 튜닝, A/B로 v1 채택/롤백

---

## Git 힌트

```powershell
# 대용량 산출물 제외
@"
training/qlora-out/
out/
*.bin
*.pt
*.safetensors
"@ | Out-File .gitignore -Encoding utf8 -Append

git add server app tools training .gitignore
git commit -m "feat(server): 8bit + single-worker; PLAN=16t/12s CTX=1200; PATCH defaults; adapter auto-load; bench metrics"
git push origin main
```

---

## 라이선스

(원하는 라이선스로 교체)

