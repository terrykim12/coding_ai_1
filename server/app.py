#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-8B Local Coding AI - FastAPI 서버 (CUDA 최적화)
"""

import os
import logging
import uuid
import json
import atexit
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
import anyio
import torch
from contextlib import asynccontextmanager

import re, os
from pathlib import Path
from .model import Model, load_model_once, get_model
from .context import build_context
from .patch_apply import apply_patch_json
from .test_runner import run_pytest, get_test_summary
from .debug_runtime import DebugRuntime
from .path_resolver import PathResolver
import re

# 로깅 설정 개선
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# .env 로드
load_dotenv()

app = FastAPI(
    title="Qwen3-8B Local Coding AI",
    description="로컬에서 실행되는 AI 코딩 어시스턴트",
    version="1.0.0"
)

@app.on_event("startup")
def _startup():
    global model, debug_runtime
    logger.info("🚀 서버 시작")
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name()}")
        logger.info(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        logger.info(f"   CUDA 버전: {torch.version.cuda}")
    else:
        logger.warning("⚠️ CUDA 사용 불가, CPU 모드로 실행")

    # 시작 시 1회 로드 후 app.state에 보관
    m, t, q = load_model_once()
    app.state.model = m
    app.state.tok = t
    app.state.quant = q
    model = Model()  # 기존 코드 의존성 호환 목적 (get_device_info 등)
    debug_runtime = DebugRuntime()
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info("🎉 서버 시작 완료!")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JSON 파싱 개선을 위한 미들웨어
@app.middleware("http")
async def json_error_handler(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        return JSONResponse(
            status_code=400,
            content={"detail": f"JSON 파싱 실패: {str(e)}"}
        )
    except UnicodeDecodeError as e:
        logger.error(f"인코딩 오류: {e}")
        return JSONResponse(
            status_code=400,
            content={"detail": f"인코딩 오류: {str(e)}"}
        )

# DebugRuntime 클래스 수정
class DebugRuntime:
    def __init__(self):
        self.resources = []
    
    def add_resource(self, resource):
        self.resources.append(resource)
    
    def cleanup_all(self):
        """모든 리소스 정리"""
        try:
            for resource in self.resources:
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
            self.resources.clear()
            logger.info("✅ 모든 리소스 정리 완료")
        except Exception as e:
            logger.error(f"❌ 리소스 정리 실패: {e}")

# 전역 변수
WORKSPACE_ROOT = str(Path(__file__).resolve().parents[1])  # repo 루트로 조정
RESOLVER = PathResolver(WORKSPACE_ROOT)

# 함수 스니펫 추출
def extract_func_snippet(src: str, func: str, ctx_lines: int = 40):
    m = re.search(rf'(?m)^[ \t]*def[ \t]+{re.escape(func)}\s*\(.*\):', src)
    if not m: 
        return src[:2000]
    start = m.start()
    m2 = re.search(r'(?m)^(?=\S)', src[m.end():])
    end = m.end() + (m2.start() if m2 else len(src))
    pre = src.rfind("\n", 0, max(0, start-1))
    pre = src.rfind("\n", 0, max(0, pre-ctx_lines)) if pre != -1 else 0
    post = src.find("\n", end)
    post = src.find("\n", post+1 if post != -1 else end)
    return src[pre:post]
model: Optional[Model] = None
debug_runtime: Optional[DebugRuntime] = None

def _pre_match(text: str, pat: str, use_regex: bool) -> bool:
    """안전한 매칭: 기본은 리터럴. regex 요청 시 컴파일 실패하면 리터럴 fallback."""
    if not use_regex:
        return pat in text
    try:
        return re.search(pat, text, re.DOTALL | re.MULTILINE) is not None
    except re.error:
        return pat in text

def _tighten_loc_for_func(edit: dict, func_name: str):
    """함수 블록 정규식으로 loc 자동 보정"""
    pat = rf"^def {re.escape(func_name)}\([^\)]*\):[\s\S]*?(?=^\S)"
    loc = edit.get("loc", {})
    if loc.get("type") != "regex":
        edit["loc"] = {"type": "regex", "pattern": pat}
    else:
        edit["loc"]["pattern"] = pat

def _sanitize_preconditions(patch: dict) -> tuple[dict, list[dict]]:
    """
    새 코드에만 있는 패턴이 must_contain에 있으면 must_not_contain로 이동.
    pre.regex가 없으면 기본 리터럴 매칭.
    """
    changes = []
    for e in patch.get("edits", []):
        path = e.get("path"); code = e.get("code", "")
        pre  = e.get("pre", {}) or {}
        mc   = list(pre.get("must_contain", []) or [])
        mnc  = list(pre.get("must_not_contain", []) or [])
        use_regex = bool(pre.get("regex", False))

        abs_path = RESOLVER.resolve(path) if path else None
        try:
            src = open(abs_path, "r", encoding="utf-8").read() if abs_path else ""
        except Exception:
            src = ""

        moved = []
        for pat in list(mc):
            in_src = _pre_match(src, pat, use_regex)
            in_new = _pre_match(code, pat, use_regex)
            # 파일에는 없고 새 코드에는 있으면 → must_not_contain으로 이동
            if (not in_src) and in_new:
                mc.remove(pat)
                if pat not in mnc:
                    mnc.append(pat)
                moved.append(pat)

        if moved:
            pre["must_contain"] = mc
            pre["must_not_contain"] = mnc
            e["pre"] = pre
            changes.append({"path": path, "moved": moved})
            
            # 함수 블록 앵커 자동-조이기
            for pat in moved:
                m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", pat)
                if m:
                    _tighten_loc_for_func(e, m.group(1))
                    
    return patch, changes

# 요청 모델 정의 (Pydantic 모델)
class PlanRequest(BaseModel):
    intent: str = Field(..., description="계획의 의도")
    paths: List[str] = Field(..., description="파일 경로들")
    code_paste: Optional[str] = Field(None, description="붙여넣은 코드")

class FeedbackRequest(BaseModel):
    hint: Optional[str] = Field(None, description="힌트")
    reason: Optional[str] = Field(None, description="이유")

class PatchRequest(BaseModel):
    plan: Union[Dict[str, Any], str] = Field(..., description="플랜 데이터")
    feedback: Optional[FeedbackRequest] = Field(None, description="피드백")

class PlanResponse(BaseModel):
    plan_id: str
    plan: Dict[str, Any]
    raw_response: str

class PatchResponse(BaseModel):
    patch_id: str
    patch: Dict[str, Any]
    raw_response: str

class ApplyRequest(BaseModel):
    patch: Dict[str, Any] = Field(..., description="적용할 패치")
    allowed_paths: List[str] = Field(..., description="허용된 파일 경로들")
    dry_run: bool = Field(False, description="실제 적용 여부")

class ApplyResponse(BaseModel):
    applied: List[str]
    skipped: List[Dict[str, Any]] = Field(default_factory=list, description="건너뛴 파일들")
    failed: List[Dict[str, Any]] = Field(default_factory=list, description="실패한 파일들")
    dry_run: bool
    details: List[Dict[str, Any]] = Field(default_factory=list, description="상세 정보")

class TestRequest(BaseModel):
    paths: Optional[List[str]] = Field(None, description="테스트할 경로들")
    coverage: bool = Field(False, description="커버리지 포함 여부")

class TestResponse(BaseModel):
    summary: Dict[str, Any]
    output: str

class WorkflowRequest(BaseModel):
    intent: str = Field(..., description="사용자 의도")
    paths: List[str] = Field(..., description="분석할 파일 경로들")
    code_paste: str = Field("", description="코드 스니펫")
    auto_apply: bool = Field(False, description="자동 적용 여부")
    auto_test: bool = Field(False, description="자동 테스트 여부")

class WorkflowResponse(BaseModel):
    workflow_id: str
    plan: Dict[str, Any]
    patch: Dict[str, Any]
    applied: bool
    test_results: Optional[Dict[str, Any]]

class DebugRequest(BaseModel):
    command: str = Field(..., description="디버그 명령어")
    script_path: Optional[str] = Field(None, description="스크립트 경로")
    port: Optional[int] = Field(None, description="디버그 포트")

class DebugResponse(BaseModel):
    success: bool
    message: str
    debug_port: Optional[int]
    processes: List[Dict[str, Any]]

# startup_event 제거 - lifespan에서 처리

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    global model, debug_runtime
    
    try:
        if model:
            del model
        if debug_runtime:
            debug_runtime.cleanup_all()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ 서버 정리 완료")
        
    except Exception as e:
        logger.error(f"❌ 서버 정리 실패: {e}")

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    try:
        # 모델 로드 상태 및 양자화 모드
        model_loaded = hasattr(app.state, "model")
        quant = getattr(app.state, "quant", "unknown")
        
        # 기본 상태 정보
        status = {
            "status": "healthy",
            "model_loaded": model_loaded,
            "quantization": quant,
            "use_4bit": quant == "4bit",
            "timestamp": datetime.now().isoformat()
        }
        
        # CUDA 정보 추가 (기존 로직 유지)
        if torch.cuda.is_available():
            status.update({
                "cuda_available": True,
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.1f}GB",
                "cuda_version": torch.version.cuda
            })
        else:
            status["cuda_available"] = False
        
        # 모델 디바이스 정보
        if model:
            try:
                device_info = model.get_device_info()
                status.update(device_info)
            except Exception:
                pass
        
        return status
    except Exception as e:
        logger.error(f"헬스 체크 실패: {e}")
        raise HTTPException(status_code=500, detail=f"서버 상태 불안정: {str(e)}")

@app.post("/plan", response_model=PlanResponse)
async def create_plan(request: PlanRequest):
    try:
        logger.info(f"플랜 요청 받음: intent={request.intent}, paths={request.paths}")
        
        # context 변수 초기화
        context = ""
        
        # 컨텍스트 빌드
        try:
            context = await run_in_threadpool(build_context, request.paths)
            logger.info(f"컨텍스트 빌드 성공: {len(context)} 문자")
        except Exception as e:
            logger.warning(f"컨텍스트 빌드 실패: {e}")
            context = f"컨텍스트 빌드 실패: {str(e)}"
        
        # 플랜 생성
        plan_data = {
            "intent": request.intent,
            "context": context,
            "code_paste": request.code_paste or "",
            "paths": request.paths,
            "timestamp": str(datetime.now())
        }
        
        # 실제 플랜 생성 (여기에 AI 모델 호출 로직 구현)
        plan_result = await generate_plan_with_ai(plan_data)
        
        return PlanResponse(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            plan=plan_result,
            raw_response="Generated with 4-bit quantization"
        )
        
    except Exception as e:
        logger.error(f"플랜 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"플랜 생성 실패: {str(e)}")

@app.post("/patch", response_model=PatchResponse)
async def create_patch(request: PatchRequest):
    try:
        logger.info("패치 요청 받음")
        
        # plan이 문자열인 경우 JSON으로 파싱
        if isinstance(request.plan, str):
            try:
                plan_data = json.loads(request.plan)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"플랜 JSON 파싱 실패: {str(e)}")
        else:
            plan_data = request.plan
        
        # 패치 생성 로직
        patch_result = await generate_patch_with_ai(plan_data, request.feedback)
        
        return {
            "patch": patch_result,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"패치 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"패치 생성 실패: {str(e)}")

@app.post("/patch_smart")
async def create_smart_patch(request: PatchRequest):
    try:
        logger.info("스마트 패치 요청 받음")
        
        # plan 데이터 처리
        if isinstance(request.plan, str):
            try:
                plan_data = json.loads(request.plan)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {e}")
                raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {str(e)}")
        else:
            plan_data = request.plan
        
        # 스마트 패치 생성
        patch_result = await generate_smart_patch(plan_data, request.feedback)
        
        # 성공 로그 기록
        try:
            os.makedirs("training/success_logs", exist_ok=True)
            rec = {
                "when": datetime.now().isoformat(),
                "role": "patch_success",
                "plan": plan_data,
                "feedback": request.feedback.dict() if request.feedback else {},
                "patch": patch_result
            }
            with open(f"training/success_logs/patch_{datetime.now():%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("✅ 성공 로그 기록 완료")
        except Exception as e:
            logger.warning(f"⚠️ 성공 로그 기록 실패: {e}")
        
        # feedback의 hint가 "Return ONLY the items of the edits array"인 경우
        if (request.feedback and 
            request.feedback.hint and 
            "Return ONLY the items of the edits array" in request.feedback.hint):
            
            # edits 배열만 반환
            if isinstance(patch_result, dict) and "edits" in patch_result:
                return patch_result["edits"]
            
        return patch_result
        
    except Exception as e:
        logger.error(f"스마트 패치 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"스마트 패치 생성 실패: {str(e)}")

@app.post("/apply", response_model=ApplyResponse)
async def apply_patch(request: ApplyRequest):
    """패치 적용"""
    try:
        # 패치 적용
        result = apply_patch_json(
            patch=request.patch,
            allowed_paths=request.allowed_paths,
            dry_run=request.dry_run
        )
        
        return ApplyResponse(
            applied=result.get("applied", []),
            skipped=result.get("skipped", []),
            failed=result.get("failed", []),
            dry_run=request.dry_run,
            details=result.get("details", [])
        )
        
    except Exception as e:
        logger.error(f"패치 적용 실패: {e}")
        raise HTTPException(status_code=500, detail=f"패치 적용 실패: {str(e)}")

@app.post("/test", response_model=TestResponse)
async def run_tests(request: TestRequest):
    """테스트 실행"""
    try:
        # 테스트 실행
        if request.coverage:
            test_path = (request.paths or ["."])[0] if request.paths else "."
            output = run_pytest(test_path, coverage=True)
        else:
            test_path = (request.paths or ["."])[0] if request.paths else "."
            output = run_pytest(test_path)
        
        # 테스트 요약
        summary = get_test_summary(output)
        
        return TestResponse(summary=summary, output=output.output)
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")
        raise HTTPException(status_code=500, detail=f"테스트 실행 실패: {str(e)}")

@app.post("/workflow", response_model=WorkflowResponse)
async def run_workflow(request: WorkflowRequest):
    """전체 워크플로우 실행"""
    if not model:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않음")
    
    try:
        workflow_id = str(uuid.uuid4())
        logger.info(f"워크플로우 시작: {workflow_id}")
        
        # 1. 컨텍스트 빌드
        context = await run_in_threadpool(build_context, request.paths)
        
        # 2. 계획 생성 (타임아웃 적용)
        with anyio.move_on_after(30) as cs:
            plan_result = await run_in_threadpool(
                model.plan,
                context,
                request.intent,
                request.code_paste
            )
        
        if cs.cancel_called:
            raise TimeoutError("PLAN generation timeout")
        
        # 3. 패치 생성 (타임아웃 적용)
        with anyio.move_on_after(90) as cs:   # 90초로 증가
            patch_result = await run_in_threadpool(model.patch, plan_result)
        
        if cs.cancel_called:
            raise TimeoutError("PATCH generation timeout")
        
        # 4. 패치 적용 (선택사항)
        applied = False
        if request.auto_apply:
            apply_result = apply_patch_json(
                patch=patch_result,
                allowed_paths=request.paths,
                dry_run=False
            )
            applied = len(apply_result.get("applied", [])) > 0
        
        # 5. 테스트 실행 (선택사항)
        test_results = None
        if request.auto_test:
            try:
                test_path = request.paths[0] if request.paths else "."
                output = run_pytest(test_path)
                test_results = get_test_summary(output)
            except Exception as e:
                logger.warning(f"테스트 실행 실패: {e}")
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            plan=plan_result,
            patch=patch_result,
            applied=applied,
            test_results=test_results
        )
        
    except TimeoutError as te:
        logger.error(f"워크플로우 타임아웃: {te}")
        raise HTTPException(status_code=504, detail=str(te))
    except Exception as e:
        logger.error(f"워크플로우 실행 실패: {e}")
        raise HTTPException(status_code=500, detail=f"워크플로우 실행 실패: {str(e)}")

@app.post("/debug", response_model=DebugResponse)
async def debug_control(request: DebugRequest):
    """디버그 런타임 제어"""
    if not debug_runtime:
        raise HTTPException(status_code=503, detail="디버그 런타임이 초기화되지 않음")
    
    try:
        if request.command == "start":
            if not request.script_path:
                raise HTTPException(status_code=400, detail="스크립트 경로가 필요합니다")
            
            debug_port = debug_runtime.run_with_debugpy(
                script_path=request.script_path,
                port=request.port
            )
            
            return DebugResponse(
                success=True,
                message=f"디버그 프로세스 시작됨 (포트: {debug_port})",
                debug_port=debug_port,
                processes=debug_runtime.list_processes()
            )
            
        elif request.command == "stop":
            if not request.port:
                raise HTTPException(status_code=400, detail="포트 번호가 필요합니다")
            
            debug_runtime.stop_process(request.port)
            
            return DebugResponse(
                success=True,
                message=f"포트 {request.port}의 프로세스 중지됨",
                debug_port=None,
                processes=debug_runtime.list_processes()
            )
            
        elif request.command == "list":
            return DebugResponse(
                success=True,
                message="디버그 프로세스 목록",
                debug_port=None,
                processes=debug_runtime.list_processes()
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"알 수 없는 명령어: {request.command}")
            
    except Exception as e:
        logger.error(f"디버그 제어 실패: {e}")
        raise HTTPException(status_code=500, detail=f"디버그 제어 실패: {str(e)}")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Qwen3-8B Local Coding AI 서버",
        "version": "1.0.0",
        "endpoints": [
            "/health - 서버 상태 확인",
            "/plan - 코드 수정 계획 생성",
            "/patch - 코드 패치 생성",
            "/apply - 패치 적용",
            "/test - 테스트 실행",
            "/workflow - 전체 워크플로우 실행",
            "/debug - 디버그 런타임 제어"
        ]
    }

# AI 생성 함수들
import os, json, logging
from server.model import get_model

log = logging.getLogger(__name__)
USE_DUMMY = os.getenv("USE_DUMMY_AI", "0") == "1"

async def generate_plan_with_ai(plan_data: dict) -> dict:
    """AI를 사용하여 플랜 생성"""
    if USE_DUMMY:
        # 임시 더미가 필요하면 남겨두되 기본값은 비활성화
        return {"files":[{"path": plan_data["paths"][0], "reason": plan_data["intent"], "strategy":"anchor"}], "notes":""}

    # 앱 상태의 단일 모델/토크나이저 사용
    m = app.state.model
    tok = app.state.tok
    # 입력 컷(1500자) + 컨텍스트 제거 → 프리필 시간 단축
    code = (plan_data.get("code_paste") or "")[:1500]
    sys = "You must output STRICT JSON plan with keys 'files' and 'notes'."
    prompt = f"{sys}\n\n[CODE]\n{code}\n"
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=1024).to(m.device)
    from transformers import StoppingCriteriaList
    class _Wall:
        def __init__(self, s=12):
            import time
            self.t0=time.monotonic(); self.S=s; self._time=time
        def __call__(self, input_ids, scores=None, **kw):
            return (self._time.monotonic()-self.t0) >= self.S
    gen_kw = dict(
        max_new_tokens=24,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        use_cache=True,
        repetition_penalty=1.05,
        max_time=12,
        stopping_criteria=StoppingCriteriaList([_Wall(12)]),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    out = m.generate(**inputs, **gen_kw)
    text = tok.decode(out[0], skip_special_tokens=True)
    # 간략 파서: 최소 스키마 보정
    try:
        import json as _json
        obj = _json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError
        if "files" not in obj:
            obj["files"] = [{"path": plan_data["paths"][0] if plan_data.get("paths") else "", "reason": plan_data["intent"], "strategy":"regex", "tests": []}]
        if "notes" not in obj:
            obj["notes"] = ""
        return obj
    except Exception:
        return {"files":[{"path": plan_data["paths"][0] if plan_data.get("paths") else "", "reason": plan_data["intent"], "strategy":"regex", "tests": []}], "notes":""}
    if isinstance(plan, str):
        plan = json.loads(plan)
    return plan

async def generate_patch_with_ai(plan_data: dict, feedback: Optional[FeedbackRequest]) -> dict:
    """AI를 사용하여 패치 생성"""
    if USE_DUMMY:
        return {"version":"1", "edits":[]}

    m = get_model()
    patch = m.patch(plan=plan_data, feedback=feedback, max_new_tokens=256, budget_s=90)
    if isinstance(patch, str):
        patch = json.loads(patch)
    return patch

async def generate_smart_patch(plan_data: dict, feedback: Optional[FeedbackRequest]) -> dict:
    """스마트 패치 생성"""
    if USE_DUMMY:
        return {"version":"1", "edits":[]}

    m = get_model()
    patch = m.patch(plan=plan_data, feedback=feedback, max_new_tokens=256, budget_s=90)
    if isinstance(patch, str):
        patch = json.loads(patch)
    return patch

# 전역 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"전역 예외 발생: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"서버 내부 오류: {str(exc)}"}
    )

# 또는 기존 방식을 사용한다면
@app.on_event("shutdown")
async def shutdown_event():
    try:
        if hasattr(debug_runtime, 'cleanup_all'):
            debug_runtime.cleanup_all()
        logger.info("✅ 서버 정리 완료")
    except Exception as e:
        logger.error(f"❌ 서버 정리 실패: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)

