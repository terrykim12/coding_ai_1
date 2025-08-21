from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: add startup logic if needed
    yield
    # TODO: add shutdown logic if needed

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single entry that re-exports the FastAPI app from server.app
"""

from server.app import app  # noqa: F401
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-8B Local Coding AI - FastAPI 서버 (CUDA 최적화)
"""

import os
import logging
import uuid
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import anyio
import torch

import re, os
from pathlib import Path
from .model import Model
from .context import build_context
from .patch_apply import apply_patch_json
from .test_runner import run_pytest, get_test_summary
from .debug_runtime import DebugRuntime
from .path_resolver import PathResolver

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Qwen3-8B Local Coding AI",
    description="로컬에서 실행되는 AI 코딩 어시스턴트",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
WORKSPACE_ROOT = str(Path(__file__).resolve().parents[1])  # repo 루트로 조정
RESOLVER = PathResolver(WORKSPACE_ROOT)
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

# Pydantic 모델들
class PlanRequest(BaseModel):
    intent: str = Field(..., description="사용자 의도")
    paths: List[str] = Field(..., description="분석할 파일 경로들")
    code_paste: str = Field("", description="코드 스니펫")

class PlanResponse(BaseModel):
    plan_id: str
    plan: Dict[str, Any]
    raw_response: str

class PatchRequest(BaseModel):
    plan: Dict[str, Any] = Field(..., description="수정 계획")

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
async def startup_event():
    """서버 시작 시 모델 로딩"""
    global model, debug_runtime
    
    try:
        logger.info("🚀 Qwen3-8B Local Coding AI 서버 시작...")
        
        # CUDA 환경 확인
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name()}")
            logger.info(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            logger.info(f"   CUDA 버전: {torch.version.cuda}")
        else:
            logger.warning("⚠️ CUDA 사용 불가, CPU 모드로 실행")
        
        # 모델 로딩
        model = Model()
        logger.info("✅ 모델 로딩 완료")
        
        # 디버그 런타임 초기화
        debug_runtime = DebugRuntime()
        logger.info("✅ 디버그 런타임 초기화 완료")
        
        logger.info("🎉 서버 시작 완료!")
        
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")
        raise
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
    """서버 상태 확인 (CUDA 정보 포함)"""
    if not model:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않음")
    
    # 기본 상태
    status = {
        "status": "healthy",
        "model_loaded": model.is_loaded(),
        "debug_processes": len(debug_runtime.list_processes()) if debug_runtime else 0
    }
    
    # CUDA 정보 추가
    if torch.cuda.is_available():
        status.update({
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.1f}GB",
            "cuda_version": torch.version.cuda
        })
    else:
        status["cuda_available"] = False
    
    # 모델 디바이스 정보
    if model:
        device_info = model.get_device_info()
        status.update(device_info)
    
    return status

@app.post("/plan", response_model=PlanResponse)
async def create_plan(request: PlanRequest):
    """코드 수정 계획 생성"""
    if not model:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않음")
    
    try:
        # 컨텍스트 빌드도 무거우면 스레드로...
        context = await run_in_threadpool(build_context, request.paths)
        
        # 추론은 반드시 스레드 + 타임아웃
        with anyio.fail_after(90):   # 90초 타임아웃 (모델 시간예산보다 여유롭게)
            plan_result = await run_in_threadpool(
                model.plan,
                context,
                request.intent,
                request.code_paste
            )
        
        # 응답 형식 맞추기
        return PlanResponse(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            plan=plan_result,
            raw_response="Generated with prefix forcing"
        )
        
    except TimeoutError as te:
        logger.error(f"PLAN 타임아웃: {te}")
        raise HTTPException(status_code=504, detail=str(te))
    except Exception as e:
        logger.error(f"PLAN 생성 실패 상세: {e}")
        logger.error(f"입력 컨텍스트 길이: {len(context)}")
        logger.error(f"사용자 의도: {request.intent}")
        logger.error(f"코드 스니펫 길이: {len(request.code_paste)}")
        raise HTTPException(status_code=500, detail=f"PLAN 생성 실패: {str(e)}")

@app.post("/patch", response_model=PatchResponse)
async def create_patch(request: PatchRequest):
    """코드 패치 생성 (기존)"""
    if not model:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않음")
    
    try:
        # 패치 생성도 스레드 + 타임아웃
        with anyio.move_on_after(90) as cs:   # 90초 타임아웃 (45초 → 90초로 증가)
            patch_result = await run_in_threadpool(model.patch, request.plan)
        
        if cs.cancel_called:
            raise TimeoutError("PATCH generation timeout")
        
        # 응답 형식 맞추기
        return PatchResponse(
            patch_id=f"patch_{uuid.uuid4().hex[:8]}",
            patch=patch_result,
            raw_response="Generated with prefix forcing"
        )
        
    except TimeoutError as te:
        logger.error(f"PATCH 타임아웃: {te}")
        raise HTTPException(status_code=504, detail=str(te))
    except Exception as e:
        logger.error(f"PATCH 생성 실패 상세: {e}")
        logger.error(f"PLAN 데이터: {json.dumps(request.plan, ensure_ascii=False)[:200]}...")
        raise HTTPException(status_code=500, detail=f"패치 생성 실패: {str(e)}")

@app.post("/patch_smart", response_model=PatchResponse)
async def create_patch_smart(request: PatchRequest):
    """코드 패치 생성 (자체 복구 루프 포함)"""
    if not model:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않음")
    
    RETRY = 2
    plan = request.plan
    feedback = ""
    
    for attempt in range(1, RETRY + 2):
        try:
            logger.info(f"PATCH 생성 시도 {attempt}/{RETRY + 1}")
            
            with anyio.fail_after(70):
                patch = await run_in_threadpool(model.patch, plan, feedback)
                
            # --- after: patch = await run_in_threadpool(model.patch, plan, feedback)
            import json, re

            _FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.M)
            def _strip_fences(s: str) -> str:
                return _FENCE_RE.sub("", s).strip()

            def _balanced_obj(s: str) -> str | None:
                s = s.strip()
                i = s.find("{")
                if i < 0: return None
                depth = 0; j0 = None; in_str = False; esc = False
                for j, ch in enumerate(s[i:], start=i):
                    if in_str:
                        if esc: esc = False
                        elif ch == "\\": esc = True
                        elif ch == '"': in_str = False
                        continue
                    if ch == '"': in_str = True
                    elif ch == "{":
                        depth += 1
                        if j0 is None: j0 = j
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and j0 is not None:
                            return s[j0:j+1]
                return None

            def _coerce_edit_item(x):
                if isinstance(x, dict):
                    return x
                if isinstance(x, str):
                    t = _strip_fences(x).strip().strip(",")
                    obj = _balanced_obj(t)
                    if obj:
                        try:
                            return json.loads(obj)
                        except Exception:
                            pass
                return None

            # normalize edits
            if not isinstance(patch, dict) or not isinstance(patch.get("edits"), list):
                raise HTTPException(status_code=422, detail="patch missing edits[]")

            bad_idx = []
            norm = []
            for i, it in enumerate(patch["edits"]):
                ed = _coerce_edit_item(it)
                if ed is None:
                    bad_idx.append(i)
                else:
                    norm.append(ed)

            if bad_idx:
                if not norm:
                    raise HTTPException(status_code=422, detail=f"invalid edit type at indices {bad_idx[:10]}")
                # 부분만 유효하면 유효한 것만 사용 (선택)
                patch["edits"] = norm
                
        except TimeoutError as te:
            if attempt <= RETRY:
                feedback += f"\n[ERROR] generation-timeout: {te}"
                logger.warning(f"PATCH 생성 타임아웃, 재시도 {attempt}/{RETRY}")
                continue
            raise HTTPException(504, f"PATCH timeout: {te}")
        except ValueError as e:
            # 예: "PATCH: JSON parse failed" 등
            if attempt <= RETRY:
                feedback = f"[ERROR] {str(e)}"
                logger.warning(f"PATCH ValueError, 재시도 {attempt}/{RETRY}: {e}")
                continue
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            # 미상 예외도 절대 500으로 터뜨리지 말고 메시지 전달
            if attempt <= RETRY:
                feedback = f"[ERROR] unexpected: {str(e)}"
                logger.warning(f"PATCH 예외, 재시도 {attempt}/{RETRY}: {e}")
                continue
            raise HTTPException(status_code=422, detail=f"patch failed: {e}")
        
        # 1) 스키마 점검
        edits = patch.get("edits", [])
        if not edits or not isinstance(edits, list):
            if attempt <= RETRY:
                feedback = "[ERROR] empty-edits: produce at least one edit targeting the function in PLAN."
                logger.warning(f"PATCH empty edits, 재시도 {attempt}/{RETRY}")
                continue
            raise HTTPException(422, "PATCH empty edits")
        
        # edits 배열 유효성 검사
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                if attempt <= RETRY:
                    feedback = f"[ERROR] invalid-edit-{i}: edit must be a JSON object, got {type(edit).__name__}"
                    logger.warning(f"PATCH invalid edit type at index {i}: {type(edit).__name__}")
                    continue
                raise HTTPException(422, f"PATCH invalid edit at index {i}")
            
            if not edit.get("path"):
                if attempt <= RETRY:
                    feedback = f"[ERROR] missing-path-{i}: edit must have 'path' field"
                    logger.warning(f"PATCH missing path at index {i}")
                    continue
                raise HTTPException(422, f"PATCH missing path at index {i}")
        
        # 2) 경로 자동 보정
        patch, remap, not_found = RESOLVER.fix_patch_paths(patch)
        
        if remap:
            logger.info(f"PATCH path remap: {remap}")  # /home/user/... -> examples/sample_py/app.py
        if not_found:
            # 후보 자체가 없으면 바로 409로 반환 (모델 재시도 전에 사용자/플랜 확인)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=409,
                content={"error":"file-not-found","paths":not_found}
            )
        
        # ★ 프리컨디션 자동 교정
        patch, moved = _sanitize_preconditions(patch)
        if moved:
            logger.info(f"PATCH preconditions sanitized: {moved}")
        
        # 3) dry-run 적용 검사
        try:
            dry = apply_patch_json(patch, dry_run=True)
            if dry["failed"]:
                if attempt <= RETRY:
                    # pre.must_contain miss 오류 특별 처리
                    fails = dry.get("failed", [])
                    misses = [f.get("error","") for f in fails if "pre.must_contain miss:" in f.get("error","")]
                    if misses:
                        feedback = (
                            "Use pre.must_contain ONLY for patterns that EXIST in the current file (BEFORE patch). "
                            "Move new patterns like 'def main(' to pre.must_not_contain. "
                            "Return the edits array again, minimal and anchored."
                        )
                    else:
                        feedback = "[ERROR] dry-run-failed: tighten loc with regex/ast; add pre.must_not_contain to avoid duplicates."
                    
                    logger.warning(f"PATCH dry-run 실패, 재시도 {attempt}/{RETRY}: {dry['failed']}")
                    continue
                raise HTTPException(409, f"apply failed: {dry['failed']}")
            
            # 성공 시 응답 반환
            logger.info(f"PATCH 생성 성공 (시도 {attempt})")
            return PatchResponse(
                patch_id=f"patch_{uuid.uuid4().hex[:8]}",
                patch=patch,
                raw_response=f"Generated with autorepair (attempt {attempt})"
            )
            
        except Exception as e:
            if attempt <= RETRY:
                feedback += f"\n[ERROR] apply-check-failed: {str(e)}"
                logger.warning(f"PATCH 검증 실패, 재시도 {attempt}/{RETRY}: {e}")
                continue
            raise HTTPException(500, f"PATCH validation failed: {str(e)}")
    
    # 모든 시도 실패
    raise HTTPException(500, f"PATCH generation failed after {RETRY + 1} attempts. Final feedback: {feedback}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)


